// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/parameter/dual_ukf.hpp — Dual sigma-point (unscented) Kalman filter:
//  a state UKF plus a parameter SPKF whose "measurement function" is the state
//  filter's output prediction seen as a function of theta.
//
//  References
//    * Plett, G. L. (2006). Sigma-point Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs — Part 2: Simultaneous state and
//      parameter estimation. J. Power Sources 161(2), 1369-1384.  (dual SPKF)
//    * Wan, E. A. & Nelson, A. T. (2001). Dual extended Kalman filter methods.
//      In S. Haykin (Ed.), Kalman Filtering and Neural Networks, Ch. 5,
//      pp. 123-173. Wiley, New York.
//    * Plett, G. L. (2004). Extended Kalman filtering ... Part 3. J. Power
//      Sources 134(2), 277-292.           (the total-derivative recursion)
//    * Julier, S. J. & Uhlmann, J. K. (2004). Unscented filtering and nonlinear
//      estimation. Proc. IEEE 92(3), 401-422.
//
//  Parameter filter (Plett 2006 Part 2, §3.2).  With theta in R^{n_p},
//  2*n_p+1 sigma points W^(i) drawn from N(thetahat^-, Sigma^-_theta),
//
//      d^(i)_k = h( f(xhat^(i)+_{k-1}, u_{k-1}; W^(i)), u_k ; W^(i) )        (1)
//      yhat_k  = sum_i w_m^(i) d^(i)_k
//      P_dd    = sum_i w_c^(i) (d^(i)-yhat)(d^(i)-yhat)^T + R_theta
//      P_td    = sum_i w_c^(i) (W^(i)-thetahat^-)(d^(i)-yhat)^T
//      L_theta = P_td P_dd^{-1}
//      thetahat^+ = thetahat^- + L_theta (y_k - yhat_k)
//      Sigma^+_theta = Sigma^-_theta - L_theta P_dd L_theta^T
//
//  STATE REPLICAS.  In (1) each parameter sigma point propagates its OWN state
//  xhat^(i), corrected after every measurement with the state filter's gain,
//
//      xhat^(i)-_k = f(xhat^(i)+_{k-1}, u_{k-1}; W^(i))
//      xhat^(i)+_k = xhat^(i)-_k + K_k ( y_k - d^(i)_k ).                    (2)
//
//  (2) is the sigma-point image of Plett's total-derivative recursion
//  dxhat^+/dtheta = dxhat^-/dtheta - K dyhat/dtheta used by the dual EKF, and it
//  is essential: with a single shared xhat the one-step sensitivity of the
//  output to R0 is the full -i, whereas the true TOTAL sensitivity is nearly
//  zero because the state filter has already re-absorbed the change through the
//  SOC.  A dual SPKF without (2) therefore over-corrects R0 by roughly a factor
//  of i/sigma_v per step and saturates it against its bound within a few seconds
//  (measured on the ECM plant: SOC RMSE 0.10 instead of 0.002).
//  Because K_k e_k = xhat^+_k - xhat^-_k and n_y = 1 here, K_k is never formed
//  explicitly: the replica correction is (xhat^+_k - xhat^-_k)(y_k-d^(i)_k)/e_k.
//  The bank is re-centred on the state filter's estimate after every update so
//  it cannot drift away from it.
//
//  R_theta.  The residual y_k - yhat_k also contains the state estimation error,
//  so R_theta = R + H P^- H^T by default (the same choice as in dual_ekf.hpp);
//  without it the parameter filter is grossly over-confident during the initial
//  transient.
//
//  Sequencing (as in the dual EKF): parameter time update -> state time update
//  -> parameter measurement update -> state measurement update.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../filters/ukf.hpp"

namespace estkit {

template <class Model>
class DualUkf {
public:
    static_assert(has_params<Model>::value, "DualUkf requires Model::NP, params(), set_params()");
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY, NP = Model::NP;
    static constexpr int NSP = 2 * NP + 1;          // parameter-filter sigma points
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>; using Theta = Vec<NP>;

    struct Options {
        Theta q_theta;      // per-step random-walk variance; <= 0 -> (q_rel*|theta_0|)^2
        Theta p0_theta;     // initial variance;              <= 0 -> (p0_rel*|theta_0|)^2
        Real q_rel  = Real(1e-5);
        Real p0_rel = Real(0.1);
        Theta theta_lo, theta_hi;    // box constraints (inactive when hi <= lo)
        // scaled UT of the STATE filter
        Real alpha = Real(1e-1), beta = Real(2), kappa = Real(0);
        // scaled UT of the PARAMETER filter: alpha = 1, kappa = 0 puts the points
        // at +/- sqrt(n_p) sigma with w_0^m = 0, a well-spread and stable choice
        // for the low-dimensional, nearly linear parameter subproblem
        Real alpha_theta = Real(1), beta_theta = Real(2), kappa_theta = Real(0);
        Real r_theta_scale = Real(1);      // inflation of R_theta
        bool innovation_covariance = true; // R_theta = R + H P^- H^T (else R only)
        bool state_replicas = true;        // Eq. (2); switching this off reproduces
                                           // the naive one-step dual SPKF
        Real innov_gate = Real(0);         // skip the parameter update when
                                           // |e| > innov_gate*sqrt(P_dd); 0 = off
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    explicit DualUkf(const Model& m) : xf_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        xf_.opts.alpha = opts.alpha; xf_.opts.beta = opts.beta; xf_.opts.kappa = opts.kappa;
        xf_.opts.apply_constraints = opts.apply_constraints;
        xf_.opts.q_scale = opts.q_scale; xf_.opts.r_scale = opts.r_scale;
        xf_.init(x0, P0);
        theta_ = xf_.model().params();
        Ptheta_ = Mat<NP, NP>(); Qtheta_ = Mat<NP, NP>();
        for (int k = 0; k < NP; ++k) {
            const Real scale = std::fabs(theta_[k]);
            Qtheta_(k, k) = (opts.q_theta[k]  > Real(0)) ? opts.q_theta[k]  : sq(opts.q_rel  * scale);
            Ptheta_(k, k) = (opts.p0_theta[k] > Real(0)) ? opts.p0_theta[k] : sq(opts.p0_rel * scale);
        }
        for (int s = 0; s < NSP; ++s) { xrep_[s] = x0; xrep_m_[s] = x0; }
        make_sigma_();
    }

    void predict(const U& u) {
        Ptheta_ += Qtheta_;                       // parameter time update
        make_sigma_();
        for (int s = 0; s < NSP; ++s) {
            Model m = xf_.model(); m.set_params(W_[s]);
            xrep_m_[s] = opts.state_replicas ? m.f(xrep_[s], u) : m.f(xf_.x(), u);
        }
        xf_.predict(u);                           // state time update
    }

    Y predict_measurement(const U& u) const { return xf_.predict_measurement(u); }

    void update(const Y& y, const U& u) {
        // --- output of every parameter sigma point, Eq. (1) ---
        Y D[NSP]; Y ym;
        for (int s = 0; s < NSP; ++s) {
            Model m = xf_.model(); m.set_params(W_[s]);
            D[s] = m.h(xrep_m_[s], u);
            ym += D[s] * wm_[s];
        }
        param_update_(y, u, D, ym);

        // --- state measurement update ---
        const X xminus = xf_.x();
        xf_.update(y, u);

        // --- replica correction with the state filter's gain, Eq. (2) ---
        const X dx = xf_.x() - xminus;                     // = K_k e_k
        const Real es = xf_.innovation()[0];
        X mean;
        for (int s = 0; s < NSP; ++s) {
            X xs = xrep_m_[s];
            if (opts.state_replicas && std::fabs(es) > Real(1e-12))
                xs = xs + dx * ((y[0] - D[s][0]) / es);
            else
                xs = xs + dx;
            if (opts.apply_constraints) xs = constrain(xf_.model(), xs);
            xrep_[s] = xs;
            mean += xs * (Real(1) / Real(NSP));
        }
        const X shift = xf_.x() - mean;                    // re-centre the bank
        for (int s = 0; s < NSP; ++s) xrep_[s] = xrep_[s] + shift;

        xf_.model().set_params(theta_);
        theta_ = xf_.model().params();                     // read back (model may clamp)
    }

    // --- generic accessors ---
    const X& x() const { return xf_.x(); }
    const Mat<NX, NX>& P() const { return xf_.P(); }
    const Y& innovation() const { return xf_.innovation(); }
    Model& model() { return xf_.model(); }
    const Model& model() const { return xf_.model(); }
    Ukf<Model>& state_filter() { return xf_; }
    const Ukf<Model>& state_filter() const { return xf_; }

    // --- identified parameters ---
    const Theta& params() const { return theta_; }
    Real param(int k) const { return theta_[k]; }
    Real param_std(int k) const { const Real p = Ptheta_(k, k); return p > Real(0) ? std::sqrt(p) : Real(0); }
    const Mat<NP, NP>& param_cov() const { return Ptheta_; }
    Real capacity() const { return theta_[0]; }

private:
    void project_(Theta& th) const {
        for (int k = 0; k < NP; ++k)
            if (opts.theta_hi[k] > opts.theta_lo[k])
                th[k] = clampr(th[k], opts.theta_lo[k], opts.theta_hi[k]);
    }

    void make_sigma_() {
        const Real n = Real(NP);
        const Real lambda = opts.alpha_theta * opts.alpha_theta * (n + opts.kappa_theta) - n;
        Real c = n + lambda;
        if (!(c > Real(0))) c = n;
        const Mat<NP, NP> Lc = cholesky_safe(Ptheta_ * c);
        W_[0] = theta_;
        wm_[0] = lambda / c;
        wc_[0] = lambda / c + (Real(1) - opts.alpha_theta * opts.alpha_theta + opts.beta_theta);
        for (int i = 0; i < NP; ++i) {
            const Theta col = Lc.col(i);
            W_[1 + i] = theta_ + col; W_[1 + NP + i] = theta_ - col;
            wm_[1 + i] = wm_[1 + NP + i] = wc_[1 + i] = wc_[1 + NP + i] = Real(1) / (Real(2) * c);
        }
        for (int s = 0; s < NSP; ++s) project_(W_[s]);
    }

    void param_update_(const Y& y, const U& u, const Y (&D)[NSP], const Y& ym) {
        Mat<NY, NY> Pdd = xf_.model().R(xf_.x(), u) * (opts.r_scale * opts.r_theta_scale);
        if (opts.innovation_covariance) {
            const Mat<NY, NX> Hx = jac_H(xf_.model(), xf_.x(), u);
            Pdd += Hx * xf_.P() * Hx.t();
        }
        const Mat<NY, NY> Rth = Pdd;     // measurement part only, used by the outlier gate
        Mat<NP, NY> Ptd;
        for (int s = 0; s < NSP; ++s) {
            const Y dy = D[s] - ym;
            const Theta dth = W_[s] - theta_;
            Pdd += outer(dy, dy) * wc_[s];
            Ptd += outer(dth, dy) * wc_[s];
        }
        const Mat<NY, NY> Pinv = inverse(Pdd);
        if (!Pinv.is_finite()) return;
        const Y e = y - ym;
        if (opts.innov_gate > Real(0))
            for (int j = 0; j < NY; ++j)
                if (std::fabs(e[j]) > opts.innov_gate * std::sqrt(std::max(Rth(j, j), Real(0)))) return;
        const Mat<NP, NY> L = Ptd * Pinv;
        const Theta th_new = theta_ + L * e;
        if (!th_new.is_finite()) return;
        theta_ = th_new;
        project_(theta_);
        Ptheta_ = Ptheta_ - L * Pdd * L.t();
        Ptheta_.symmetrize();
        for (int k = 0; k < NP; ++k) if (!(Ptheta_(k, k) > Real(0))) Ptheta_(k, k) = Qtheta_(k, k);
    }

    Ukf<Model> xf_;
    Theta theta_;
    Mat<NP, NP> Ptheta_, Qtheta_;
    Theta W_[NSP]; Real wm_[NSP], wc_[NSP];
    X xrep_[NSP], xrep_m_[NSP];
};

}  // namespace estkit
