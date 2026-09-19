// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/iplf.hpp — Iterated posterior linearisation filter (IPLF)
//
//  References
//    * Garcia-Fernandez, A. F., Svensson, L., Morelande, M. R. & Sarkka, S. (2015).
//      Posterior linearization filter: principles and implementation using sigma
//      points. IEEE Transactions on Signal Processing 63(20), 5561-5573. (primary)
//    * Garcia-Fernandez, A. F., Svensson, L. & Sarkka, S. (2015). Iterated posterior
//      linearization smoother. IEEE Transactions on Automatic Control 62(4),
//      2056-2063.
//    * Julier, S. J., Uhlmann, J. K. & Durrant-Whyte, H. F. (2000). A new method for
//      the nonlinear transformation of means and covariances in filters and
//      estimators. IEEE Transactions on Automatic Control 45(3), 477-482.
//      (the sigma-point rule used to evaluate the regression moments)
//    * Bell, B. M. & Cathey, F. W. (1993). The iterated Kalman filter update as a
//      Gauss-Newton method. IEEE Transactions on Automatic Control 38(2), 294-297.
//      (the IEKF, which the IPLF replaces by a statistical linearisation)
//
//  Idea
//    Sigma-point filters (UKF/CKF) perform a statistical linear regression (SLR) of
//    h with respect to the PRIOR density; the IEKF performs an analytic Taylor
//    linearisation at the posterior mode.  The IPLF combines the two ideas: it
//    iterates an SLR taken with respect to the CURRENT POSTERIOR approximation,
//    which is the linearisation that minimises the mean square error where the
//    state actually is, and is shown in the primary reference to be a fixed-point
//    iteration for the Kullback-Leibler-optimal Gaussian posterior.
//
//    SLR of h w.r.t. p_j = N(x_j, P_j):  with sigma points X_i and weights w_i,
//        ybar = sum_i w_i h(X_i)
//        Psi  = sum_i w_i (X_i - x_j)(h(X_i) - ybar)^T          (n x n_y)
//        Phi  = sum_i w_i (h(X_i) - ybar)(h(X_i) - ybar)^T      (n_y x n_y)
//        A    = Psi^T P_j^{-1},   b = ybar - A x_j,   Omega = Phi - A P_j A^T   (1)
//    (A,b,Omega) is the minimiser of E[||h(x) - (Ax+b)||^2] under p_j together with
//    the covariance of the residual.  The enabled (linear) update against the PRIOR
//    N(x^-,P^-) is then
//        S     = A P^- A^T + Omega + R
//        K     = P^- A^T S^{-1}
//        x_{j+1} = x^- + K (y - A x^- - b),   P_{j+1} = P^- - K S K^T             (2)
//    Starting from p_0 = N(x^-,P^-), the first iteration reproduces the UKF update
//    exactly; further iterations move the linearisation towards the posterior.
//
//  Time update: standard additive-noise sigma-point (unscented) prediction.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class Iplf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NS = 2 * NX + 1;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        Real alpha = Real(1e-1), beta = Real(2), kappa = Real(0);   // scaled UT parameters
        int  max_iter = 5;                 // posterior-linearisation iterations (1 == UKF)
        Real tol = Real(1e-6);             // relative stopping tolerance on the mean
        bool apply_constraints = true;     // project with Model::constrain (if defined)
        Real q_scale = Real(1);            // multiplicative tuning of Q
        Real r_scale = Real(1);            // multiplicative tuning of R
    } opts;

    explicit Iplf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; iters_ = 0; }

    void predict(const U& u) {
        sigma_points(x_, P_);
        X xm;
        for (int s = 0; s < NS; ++s) { Xs_[s] = m_.f(Xs_[s], u); xm += Xs_[s] * wm_[s]; }
        Mat<NX, NX> Pm;
        for (int s = 0; s < NS; ++s) { const X d = Xs_[s] - xm; Pm += outer(d, d) * wc_[s]; }
        x_ = xm;
        P_ = Pm + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const {
        sigma_points(x_, P_);
        Y ym; for (int s = 0; s < NS; ++s) ym += m_.h(Xs_[s], u) * wm_[s];
        return ym;
    }

    void update(const Y& y, const U& u) {
        const X xprior = x_;
        const Mat<NX, NX> Pprior = P_;
        const Mat<NY, NY> R = m_.R(x_, u) * opts.r_scale;
        X xj = xprior; Mat<NX, NX> Pj = Pprior;
        const int it_max = opts.max_iter < 1 ? 1 : opts.max_iter;
        iters_ = 0;
        for (int j = 0; j < it_max; ++j) {
            // --- statistical linear regression of h w.r.t. N(xj, Pj) ---
            sigma_points(xj, Pj);
            Y Zs[NS]; Y ybar;
            for (int s = 0; s < NS; ++s) { Zs[s] = m_.h(Xs_[s], u); ybar += Zs[s] * wm_[s]; }
            Mat<NX, NY> Psi; Mat<NY, NY> Phi;
            for (int s = 0; s < NS; ++s) {
                const Y dy = Zs[s] - ybar; const X dx = Xs_[s] - xj;
                Psi += outer(dx, dy) * wc_[s]; Phi += outer(dy, dy) * wc_[s];
            }
            const Mat<NX, NY> PinvPsi = solve_spd(Pj, Psi);
            if (!PinvPsi.is_finite()) break;
            const Mat<NY, NX> A = PinvPsi.t();              // A = Psi^T Pj^{-1}
            const Y b = ybar - A * xj;
            Mat<NY, NY> Om = Phi - A * Pj * A.t();          // residual covariance of the regression
            for (int i = 0; i < NY; ++i) if (!(Om(i, i) > Real(0))) Om(i, i) = Real(0);
            Om.symmetrize();
            // --- linear update against the prior ---
            const Mat<NY, NY> S = A * Pprior * A.t() + Om + R;
            const Mat<NX, NY> K = Pprior * A.t() * inverse(S);
            if (!K.is_finite()) break;
            const X xn = xprior + K * (y - A * xprior - b);
            Mat<NX, NX> Pn = Pprior - K * S * K.t();
            Pn.symmetrize();
            if (!xn.is_finite() || !Pn.is_finite()) break;
            const X d = xn - xj;
            xj = xn; Pj = Pn; S_ = S; innovation_ = y - ybar;
            ++iters_;
            if (d.norm() <= opts.tol * (Real(1) + xj.norm())) break;
        }
        x_ = xj; P_ = Pj; P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    int iterations() const { return iters_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

    // scaled unscented transform (Julier et al. 2000; Wan & van der Merwe 2000)
    void sigma_points(const X& x, const Mat<NX, NX>& P) const {
        const Real n = Real(NX);
        const Real lambda = opts.alpha * opts.alpha * (n + opts.kappa) - n;
        const Real c = n + lambda;
        const Mat<NX, NX> L = cholesky_safe(P * c);
        Xs_[0] = x; wm_[0] = lambda / c; wc_[0] = lambda / c + (Real(1) - opts.alpha * opts.alpha + opts.beta);
        for (int i = 0; i < NX; ++i) {
            const X col = L.col(i);
            Xs_[1 + i] = x + col; Xs_[1 + NX + i] = x - col;
            wm_[1 + i] = wm_[1 + NX + i] = wc_[1 + i] = wc_[1 + NX + i] = Real(1) / (Real(2) * c);
        }
    }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    mutable X Xs_[NS];
    mutable Real wm_[NS], wc_[NS];
    Y innovation_;
    Mat<NY, NY> S_;
    int iters_ = 0;
};

}  // namespace estkit
