// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/parameter/dual_ekf.hpp — Dual extended Kalman filter: two coupled EKFs,
//  one for the state x and one for the parameter vector theta.
//
//  References
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs — Part 3. State and parameter
//      estimation. J. Power Sources 134(2), 277-292.       (Section 4, dual EKF)
//    * Wan, E. A. & Nelson, A. T. (2001). Dual extended Kalman filter methods.
//      In S. Haykin (Ed.), Kalman Filtering and Neural Networks, Ch. 5,
//      pp. 123-173. Wiley, New York.
//    * Ljung, L. (1979). Asymptotic behavior of the extended Kalman filter as a
//      parameter estimator for linear systems. IEEE Trans. Autom. Control 24(1),
//      36-50.
//
//  Model
//        x_{k+1} = f(x_k, u_k; theta) + w_k,        cov(w) = Q
//        theta_{k+1} = theta_k + r_k,               cov(r) = Sigma_r = diag(q_theta)
//        y_k = h(x_k, u_k; theta) + v_k,            cov(v) = R
//
//  The parameter filter treats y_k as its measurement and the *state filter's*
//  output prediction as its measurement function.  Because the state estimate
//  itself depends on theta, the parameter-filter output Jacobian is a TOTAL
//  derivative (Plett 2004 Part 3, Eqs. (26)-(28); Wan & Nelson 2001 §5.3):
//
//    C^theta_k = d yhat_k / d theta
//              = dh/dtheta |_(xhat^-_k, u_k)  +  dh/dx |_(xhat^-_k,u_k) * dxhat^-_k/dtheta
//    dxhat^-_k / dtheta
//              = df/dtheta |_(xhat^+_{k-1}, u_{k-1}) + df/dx * dxhat^+_{k-1}/dtheta
//    dxhat^+_{k-1}/dtheta
//              = dxhat^-_{k-1}/dtheta - K_{k-1} * C^theta_{k-1}
//
//  (the dependence of the state-filter gain K on theta is neglected, as in both
//  references).  With Psi^- = dxhat^-/dtheta and Psi^+ = dxhat^+/dtheta the
//  parameter filter is an ordinary EKF:
//
//    Sigma^-_theta = Sigma^+_theta + Sigma_r
//    S_theta = C^theta Sigma^-_theta (C^theta)^T + R_theta
//    L_theta = Sigma^-_theta (C^theta)^T S_theta^{-1}
//    thetahat^+ = thetahat^- + L_theta (y_k - h(xhat^-_k,u_k))
//    Sigma^+_theta = (I - L_theta C^theta) Sigma^-_theta (I - L_theta C^theta)^T
//                     + L_theta R_theta L_theta^T                    (Joseph form)
//
//  R_theta defaults to the *state filter's innovation covariance*
//  H P^- H^T + R, because the residual y_k - h(xhat^-_k) contains the state
//  estimation error as well as the sensor noise; using R alone makes the
//  parameter filter over-confident (see Implementation notes in the chapter).
//
//  Sequencing follows Plett 2004 Part 3, Table 4:
//      parameter time update -> state time update
//      -> parameter measurement update -> state measurement update.
//
//  df/dtheta and dh/dtheta are obtained by central differences through
//  Model::set_params(), so no analytic parameter Jacobian is required.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../filters/ekf.hpp"

namespace estkit {

template <class Model>
class DualEkf {
public:
    static_assert(has_params<Model>::value, "DualEkf requires Model::NP, params(), set_params()");
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY, NP = Model::NP;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>; using Theta = Vec<NP>;

    struct Options {
        Theta q_theta;      // per-step random-walk variance Sigma_r; <= 0 -> (q_rel*|theta_0|)^2
        Theta p0_theta;     // initial variance;                      <= 0 -> (p0_rel*|theta_0|)^2
        Real q_rel  = Real(1e-5);
        Real p0_rel = Real(0.1);
        Theta theta_lo, theta_hi;          // box constraints (inactive when hi <= lo)
        Real fd_eps = Real(1e-4);          // relative step of the central differences in theta
        bool innovation_covariance = true; // R_theta = H P^- H^T + R  (else R only)
        Real r_theta_scale = Real(1);      // extra inflation of R_theta
        // Innovation-based adaptive measurement noise for the PARAMETER filter
        // (covariance matching, Mehra 1970; Mohamed & Schwartz 1999).  R_theta is
        // floored at an exponentially weighted estimate of the actual squared
        // residual, so the parameter filter can never claim the residual is
        // smaller than it demonstrably is.  This is what makes the estimator
        // degrade gracefully when the residual is structural MODEL error that no
        // value of theta can remove (see the chapter, "Behaviour under model
        // mismatch"); on a well-modelled plant the measured residual is below the
        // sensor model and the floor is inactive, so nothing changes.
        bool adapt_r_theta = true;
        Real r_theta_forget = Real(0.999); // EWMA factor (~1000 samples = 100 s at 10 Hz)
        // Never let the parameter covariance grow beyond the commissioning prior:
        // the prior is a physical statement ("capacity within 10 % of nameplate")
        // and a filter that becomes less certain than that has lost the plot.
        bool p_theta_ceiling = true;
        Real innov_gate = Real(0);         // skip the parameter update when
                                           // |e| > innov_gate * sqrt(R_theta); 0 = off
        bool joseph = true;
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    explicit DualEkf(const Model& m) : xf_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        xf_.opts.joseph = opts.joseph;
        xf_.opts.apply_constraints = opts.apply_constraints;
        xf_.opts.q_scale = opts.q_scale;
        xf_.opts.r_scale = opts.r_scale;
        xf_.init(x0, P0);
        theta_ = xf_.model().params();
        Ptheta_ = Mat<NP, NP>();
        Qtheta_ = Mat<NP, NP>();
        for (int k = 0; k < NP; ++k) {
            const Real scale = std::fabs(theta_[k]);
            Qtheta_(k, k) = (opts.q_theta[k]  > Real(0)) ? opts.q_theta[k]  : sq(opts.q_rel  * scale);
            Ptheta_(k, k) = (opts.p0_theta[k] > Real(0)) ? opts.p0_theta[k] : sq(opts.p0_rel * scale);
            P0theta_[k] = Ptheta_(k, k);
        }
        Psi_m_ = Mat<NX, NP>();
        Psi_p_ = Mat<NX, NP>();
        C_ = Mat<NY, NP>();
        Rhat_ = Mat<NY, NY>();
        have_rhat_ = false;
    }

    // --- time update: parameter random walk, sensitivity propagation, state EKF ---
    void predict(const U& u) {
        Ptheta_ += Qtheta_;                        // Sigma^-_theta = Sigma^+_theta + Sigma_r
        const Mat<NX, NX> A   = jac_F(xf_.model(), xf_.x(), u);
        const Mat<NX, NP> Fth = dfdtheta_(xf_.x(), u);
        Psi_m_ = Fth + A * Psi_p_;                 // dxhat^-/dtheta
        xf_.predict(u);
    }

    Y predict_measurement(const U& u) const { return xf_.predict_measurement(u); }

    // --- measurement update: parameter EKF first, then state EKF (Plett Table 4) ---
    void update(const Y& y, const U& u) {
        const Mat<NY, NX> Hx  = jac_H(xf_.model(), xf_.x(), u);
        const Mat<NY, NP> Hth = dhdtheta_(xf_.x(), u);
        C_ = Hth + Hx * Psi_m_;                    // total derivative dyhat/dtheta
        const Y e = y - xf_.model().h(xf_.x(), u);

        Mat<NY, NY> Rth = xf_.model().R(xf_.x(), u) * (opts.r_scale * opts.r_theta_scale);
        if (opts.innovation_covariance) Rth += Hx * xf_.P() * Hx.t();
        if (opts.adapt_r_theta) {                  // covariance matching (Mehra 1970)
            if (!have_rhat_) { Rhat_ = Rth; have_rhat_ = true; }
            for (int j = 0; j < NY; ++j) Rth(j, j) = std::max(Rth(j, j), Rhat_(j, j));
        }
        const Mat<NY, NY> Sth = C_ * Ptheta_ * C_.t() + Rth;
        const Mat<NY, NY> Sinv = inverse(Sth);
        // Outlier test against the MEASUREMENT covariance R_theta (not S_theta,
        // which collapses once the parameters are identified and would then gate
        // out perfectly ordinary residuals).
        bool gated = false;
        if (opts.innov_gate > Real(0))
            for (int j = 0; j < NY; ++j)
                if (std::fabs(e[j]) > opts.innov_gate * std::sqrt(std::max(Rth(j, j), Real(0)))) gated = true;
        if (Sinv.is_finite() && !gated) {
            const Mat<NP, NY> L = Ptheta_ * C_.t() * Sinv;
            const Theta th_new = theta_ + L * e;
            if (th_new.is_finite()) {
                theta_ = th_new;
                project_();
                const Mat<NP, NP> ILC = Mat<NP, NP>::identity() - L * C_;
                Ptheta_ = ILC * Ptheta_ * ILC.t() + L * Rth * L.t();
                Ptheta_.symmetrize();
                if (opts.p_theta_ceiling)
                    for (int k = 0; k < NP; ++k)
                        if (Ptheta_(k, k) > P0theta_[k]) {
                            const Real s = std::sqrt(P0theta_[k] / Ptheta_(k, k));
                            for (int j = 0; j < NP; ++j) { Ptheta_(k, j) *= s; Ptheta_(j, k) *= s; }
                        }
            }
        }
        // update the residual statistic AFTER it has been used, so that a genuine
        // parameter error is not allowed to inflate its own noise floor instantly
        if (opts.adapt_r_theta)
            for (int j = 0; j < NY; ++j)
                Rhat_(j, j) = opts.r_theta_forget * Rhat_(j, j) + (Real(1) - opts.r_theta_forget) * sq(e[j]);

        xf_.update(y, u);                          // state measurement update (theta^- is used)
        Psi_p_ = Psi_m_ - xf_.K() * C_;            // dxhat^+/dtheta
        xf_.model().set_params(theta_);            // theta^+ takes effect from step k+1 on
        theta_ = xf_.model().params();             // read back (the model may clamp)
    }

    // --- generic accessors ---
    const X& x() const { return xf_.x(); }
    const Mat<NX, NX>& P() const { return xf_.P(); }
    const Y& innovation() const { return xf_.innovation(); }
    Model& model() { return xf_.model(); }
    const Model& model() const { return xf_.model(); }
    Ekf<Model>& state_filter() { return xf_; }
    const Ekf<Model>& state_filter() const { return xf_; }

    // --- identified parameters ---
    const Theta& params() const { return theta_; }
    Real param(int k) const { return theta_[k]; }
    Real param_std(int k) const { const Real p = Ptheta_(k, k); return p > Real(0) ? std::sqrt(p) : Real(0); }
    const Mat<NP, NP>& param_cov() const { return Ptheta_; }
    // Convenience alias for param(0); for the estkit ECM theta = [Q_Ah, R0].
    Real capacity() const { return theta_[0]; }
    const Mat<NX, NP>& sensitivity() const { return Psi_p_; }   // dxhat^+/dtheta

private:
    void project_() {
        for (int k = 0; k < NP; ++k)
            if (opts.theta_hi[k] > opts.theta_lo[k])
                theta_[k] = clampr(theta_[k], opts.theta_lo[k], opts.theta_hi[k]);
    }
    Mat<NX, NP> dfdtheta_(const X& x, const U& u) const {
        Mat<NX, NP> J;
        for (int k = 0; k < NP; ++k) {
            const Real d = opts.fd_eps * (Real(1) + std::fabs(theta_[k]));
            Theta tp = theta_, tm = theta_; tp[k] += d; tm[k] -= d;
            Model mp = xf_.model(); mp.set_params(tp);
            Model mm = xf_.model(); mm.set_params(tm);
            const X fp = mp.f(x, u), fm = mm.f(x, u);
            for (int i = 0; i < NX; ++i) J(i, k) = (fp[i] - fm[i]) / (Real(2) * d);
        }
        return J;
    }
    Mat<NY, NP> dhdtheta_(const X& x, const U& u) const {
        Mat<NY, NP> J;
        for (int k = 0; k < NP; ++k) {
            const Real d = opts.fd_eps * (Real(1) + std::fabs(theta_[k]));
            Theta tp = theta_, tm = theta_; tp[k] += d; tm[k] -= d;
            Model mp = xf_.model(); mp.set_params(tp);
            Model mm = xf_.model(); mm.set_params(tm);
            const Y hp = mp.h(x, u), hm = mm.h(x, u);
            for (int i = 0; i < NY; ++i) J(i, k) = (hp[i] - hm[i]) / (Real(2) * d);
        }
        return J;
    }

    Ekf<Model> xf_;                 // state filter (its model carries thetahat)
    Theta theta_;
    Mat<NP, NP> Ptheta_, Qtheta_;
    Theta P0theta_;                 // commissioning prior variances (covariance ceiling)
    Mat<NX, NP> Psi_m_, Psi_p_;     // dxhat^-/dtheta, dxhat^+/dtheta
    Mat<NY, NP> C_;                 // total output derivative
    Mat<NY, NY> Rhat_;              // EWMA of the squared residual (covariance matching)
    bool have_rhat_ = false;
public:
    // measured residual variance driving the adaptive R_theta (diagnostics)
    const Mat<NY, NY>& residual_cov() const { return Rhat_; }
private:
};

}  // namespace estkit
