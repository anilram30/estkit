// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/ukf.hpp — Unscented Kalman filter (scaled unscented transform)
//
//  References
//    * Julier, S. J., Uhlmann, J. K. & Durrant-Whyte, H. F. (2000). A new method
//      for the nonlinear transformation of means and covariances in filters and
//      estimators. IEEE Trans. Autom. Control 45(3), 477-482.
//    * Wan, E. A. & van der Merwe, R. (2000). The unscented Kalman filter for
//      nonlinear estimation. Proc. IEEE AS-SPCC, 153-158.  (scaled UT, alpha/beta/kappa)
//    * Julier, S. J. & Uhlmann, J. K. (2004). Unscented filtering and nonlinear
//      estimation. Proc. IEEE 92(3), 401-422.
//    * Plett, G. L. (2006). Sigma-point Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs — Part 1. J. Power Sources 161, 1356-1368.
//
//  Sigma points (2n+1):  X_0 = x,  X_i = x +/- (sqrt((n+lambda) P))_i,
//  lambda = alpha^2 (n+kappa) - n,  W_0^m = lambda/(n+lambda),
//  W_0^c = W_0^m + 1 - alpha^2 + beta,  W_i = 1/(2(n+lambda)).
//  Additive-noise form: Q added after propagation, R added to P_yy.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class Ukf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NS = 2 * NX + 1;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        Real alpha = Real(1e-1), beta = Real(2), kappa = Real(0);
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
        // true : redraw the sigma points from (x^-, P^-) before the measurement update
        //        (Sarkka 2013, Alg. 5.14; Plett 2006) so that the additive Q is reflected in P_yy
        // false: reuse the propagated sigma points X_f for the measurement (Wan & van der Merwe
        //        2000, Alg. 3.1, as implemented in FilterPy) — used for cross-validation
        bool redraw_sigma_points = true;
    } opts;

    explicit Ukf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; have_propagated_ = false; }

    void predict(const U& u) {
        sigma_points(x_, P_);
        X xm;
        for (int s = 0; s < NS; ++s) { Xs_[s] = m_.f(Xs_[s], u); xm += Xs_[s] * wm_[s]; }
        Mat<NX, NX> Pm;
        for (int s = 0; s < NS; ++s) { const X d = Xs_[s] - xm; Pm += outer(d, d) * wc_[s]; }
        x_ = xm; P_ = Pm + m_.Q(x_, u) * opts.q_scale; P_.symmetrize();
        have_propagated_ = true;
    }
    Y predict_measurement(const U& u) const {
        // mean of h over the prior sigma set (redrawn into local storage, or the propagated set)
        Y ym;
        if (!opts.redraw_sigma_points && have_propagated_) { for (int s = 0; s < NS; ++s) ym += m_.h(Xs_[s], u) * wm_[s]; return ym; }
        X Xs[NS]; Real wm[NS], wc[NS];
        sigma_points_into(x_, P_, Xs, wm, wc);
        for (int s = 0; s < NS; ++s) ym += m_.h(Xs[s], u) * wm[s];
        return ym;
    }
    void update(const Y& y, const U& u) {
        if (opts.redraw_sigma_points || !have_propagated_) sigma_points(x_, P_);
        have_propagated_ = false;
        Y Ys[NS]; Y ym;
        for (int s = 0; s < NS; ++s) { Ys[s] = m_.h(Xs_[s], u); ym += Ys[s] * wm_[s]; }
        Mat<NY, NY> Pyy = m_.R(x_, u) * opts.r_scale; Mat<NX, NY> Pxy;
        for (int s = 0; s < NS; ++s) {
            const Y dy = Ys[s] - ym; const X dx = Xs_[s] - x_;
            Pyy += outer(dy, dy) * wc_[s]; Pxy += outer(dx, dy) * wc_[s];
        }
        const Mat<NX, NY> K = Pxy * inverse(Pyy);
        innovation_ = y - ym; S_ = Pyy;
        x_ = x_ + K * innovation_;
        P_ = P_ - K * Pyy * K.t(); P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }
    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

    // expose the sigma-point machinery for derived filters (SR-UKF, UPF, URTS ...)
    void sigma_points(const X& x, const Mat<NX, NX>& P) { sigma_points_into(x, P, Xs_, wm_, wc_); }
    void sigma_points_into(const X& x, const Mat<NX, NX>& P, X* Xs, Real* wm, Real* wc) const {
        const Real n = Real(NX);
        const Real lambda = opts.alpha * opts.alpha * (n + opts.kappa) - n;
        const Real c = n + lambda;
        const Mat<NX, NX> L = cholesky_safe(P * c);
        Xs[0] = x; wm[0] = lambda / c; wc[0] = lambda / c + (Real(1) - opts.alpha * opts.alpha + opts.beta);
        for (int i = 0; i < NX; ++i) {
            const X col = L.col(i);
            Xs[1 + i] = x + col; Xs[1 + NX + i] = x - col;
            wm[1 + i] = wm[1 + NX + i] = wc[1 + i] = wc[1 + NX + i] = Real(1) / (Real(2) * c);
        }
    }
    const X* sigma() const { return Xs_; }
    Real wm(int s) const { return wm_[s]; }
    Real wc(int s) const { return wc_[s]; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    X Xs_[NS];
    Real wm_[NS], wc_[NS];
    Y innovation_;
    Mat<NY, NY> S_;
    bool have_propagated_ = false;
};

}  // namespace estkit
