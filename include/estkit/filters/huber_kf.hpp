// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/huber_kf.hpp — Huber-based robust extended Kalman filter
//
//  References
//    * Huber, P. J. (1964). Robust estimation of a location parameter.
//      Ann. Math. Statist. 35(1), 73-101.                (M-estimation, psi/rho)
//    * Karlgaard, C. D. & Schaub, H. (2007). Huber-based divided difference
//      filtering. J. Guidance, Control, and Dynamics 30(3), 885-891.
//      (Huber M-estimation cast as a linear regression over the stacked
//       prior/measurement residual, iteratively reweighted least squares)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, Ch. 8.                                (EKF linearisation)
//    * Bucy, R. S. & Joseph, P. D. (1968). Filtering for Stochastic Processes
//      with Applications to Guidance. Wiley.                  (Joseph form)
//
//  Derivation sketch ------------------------------------------------------
//  The Kalman measurement update is the solution of the quadratic problem
//
//      min_x  (x - x^-)^T (P^-)^{-1} (x - x^-) + (ytil - H x)^T R^{-1} (ytil - H x),
//      ytil = y - h(x^-,u) + H x^- .
//
//  Replacing the quadratic measurement term by Huber's rho function of the
//  standardised residual makes the estimator an M-estimator with bounded
//  influence:
//
//      rho(e) = e^2/2                 |e| <= gamma
//               gamma|e| - gamma^2/2  |e| >  gamma,
//      psi(e) = rho'(e) = e                for |e| <= gamma
//                       = gamma sgn(e)     otherwise,
//      w(e)   = psi(e)/e = min(1, gamma/|e|)  in (0,1].
//
//  Iteratively reweighted least squares (IRLS) then solves a sequence of
//  ordinary Kalman updates in which R is replaced by the reweighted
//
//      Rtil_{jl} = R_{jl} / sqrt(w_j w_l)      (exactly R_jj / w_j for diagonal R),
//      K   = P^- H^T (H P^- H^T + Rtil)^{-1},
//      x^(t+1) = x^- + K (y - h(x^-,u)),
//
//  where the weights at iteration t are computed from the standardised residual
//  of the current iterate,
//
//      zeta_j^(t) = [ (y - h(x^-,u)) - H (x^(t) - x^-) ]_j / sqrt(R_jj).
//
//  For a gross outlier |zeta| -> infinity the correction
//  K e = P H^T e / (H P H^T + sigma |e|/gamma) -> P H^T gamma sgn(e)/sigma
//  stays BOUNDED - this is exactly Huber's bounded-influence property, and it is
//  what distinguishes the filter from the EKF (whose correction grows linearly
//  with the outlier).  gamma = 1.345 gives 95 % asymptotic efficiency at the
//  nominal Gaussian model (Huber 1964).
//
//  The posterior covariance uses the Joseph form with the SAME Rtil that
//  produced the gain, so that a rejected measurement correctly leaves P nearly
//  unchanged (Karlgaard & Schaub 2007: P^+ = (M^T Psi M)^{-1}).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class HuberKf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real gamma  = Real(1.345);   // Huber tuning constant (95 % Gaussian efficiency)
        int  n_iter = 3;             // IRLS iterations
        Real w_floor = Real(1e-4);   // lower bound on the weight (numerical guard)
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    explicit HuberKf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; for (int j = 0; j < NY; ++j) w_[j] = Real(1); }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        P_ = F * P_ * F.t() + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> R = m_.R(x_, u) * opts.r_scale;
        const X xprior = x_;
        innovation_ = y - m_.h(x_, u);

        Mat<NY, NY> Reff = R;
        Mat<NX, NY> K;
        Mat<NY, NY> S = H * P_ * H.t() + R;
        X xit = xprior;
        const int nit = (opts.n_iter < 1) ? 1 : opts.n_iter;
        for (int it = 0; it < nit; ++it) {
            const Y dlin = H * (xit - xprior);
            for (int j = 0; j < NY; ++j) {
                const Real sj = std::sqrt(std::max(R(j, j), Real(1e-30)));
                const Real zeta = (innovation_[j] - dlin[j]) / sj;
                const Real a = std::fabs(zeta);
                Real w = (a <= opts.gamma) ? Real(1) : opts.gamma / a;
                if (w < opts.w_floor) w = opts.w_floor;
                w_[j] = w;
            }
            for (int j = 0; j < NY; ++j)
                for (int l = 0; l < NY; ++l) Reff(j, l) = R(j, l) / std::sqrt(w_[j] * w_[l]);
            S = H * P_ * H.t() + Reff; S.symmetrize();
            const Mat<NY, NY> Sinv = inverse(S);
            if (!Sinv.is_finite()) return;
            K = P_ * H.t() * Sinv;
            xit = xprior + K * innovation_;
        }
        if (!xit.is_finite() || !K.is_finite()) return;
        x_ = xit; S_ = S; K_ = K;
        const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
        P_ = IKH * P_ * IKH.t() + K * Reff * K.t();
        P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    Real weight(int j) const { return w_[j]; }     // last Huber weight (1 = Gaussian regime)
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    Real w_[NY] = {};
};

}  // namespace estkit
