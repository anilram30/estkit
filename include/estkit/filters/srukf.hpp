// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/srukf.hpp — Square-root unscented Kalman filter (SR-UKF)
//
//  References
//    * van der Merwe, R. & Wan, E. A. (2001). The square-root unscented Kalman
//      filter for state and parameter-estimation. Proc. IEEE International
//      Conference on Acoustics, Speech, and Signal Processing (ICASSP), vol. 6,
//      3461-3464.                                                    (primary)
//    * Wan, E. A. & van der Merwe, R. (2000). The unscented Kalman filter for
//      nonlinear estimation. Proc. IEEE AS-SPCC, 153-158.        (scaled UT)
//    * Julier, S. J. & Uhlmann, J. K. (2004). Unscented filtering and nonlinear
//      estimation. Proceedings of the IEEE 92(3), 401-422.
//    * Gill, P. E., Golub, G. H., Murray, W. & Saunders, M. A. (1974). Methods for
//      modifying matrix factorizations. Mathematics of Computation 28(126),
//      505-535.                              (rank-one Cholesky update/downdate)
//    * Plett, G. L. (2006). Sigma-point Kalman filtering for battery management
//      systems - Part 1. Journal of Power Sources 161(2), 1356-1368.  (SPKF for SOC)
//
//  The SR-UKF replaces the covariance P by a Cholesky factor S (P = S S^T) and
//  never forms P.  Two primitives are needed:
//    * Tria(A): the lower-triangular factor of A A^T from the QR of A^T - used for
//      the 2n sigma points, whose weights W_i = 1/(2(n+lambda)) are positive;
//    * a rank-one Cholesky update/downdate for the centre point, whose weight
//      W_0^c = lambda/(n+lambda) + 1 - alpha^2 + beta is NEGATIVE for the usual
//      small alpha, so the centre contributes a DOWNdate.
//
//  Cycle (additive noise, gamma = sqrt(n+lambda)):
//    sigma points     X_0 = x,  X_i = x +/- gamma S_i                         (1)
//    time update      X*_i = f(X_i,u),  x^- = sum_i W_i^m X*_i
//                     S^- = Tria([ sqrt(W_1^c)(X*_{1:2n} - x^-) , S_Q ])
//                     S^- = cholupdate(S^-, X*_0 - x^-, sign W_0^c)           (2)
//    measurement      Z_i = h(X_i,u),  yhat = sum_i W_i^m Z_i
//                     S_y = Tria([ sqrt(W_1^c)(Z_{1:2n} - yhat) , S_R ])
//                     S_y = cholupdate(S_y, Z_0 - yhat, sign W_0^c)
//                     P_xy = sum_i W_i^c (X_i - x^-)(Z_i - yhat)^T
//                     K = (P_xy / S_y^T) / S_y                                (3)
//                     x^+ = x^- + K (y - yhat)
//                     S^+ = cholupdate(S^-, columns of K S_y, -1)             (4)
//  (4) implements P^+ = P^- - K P_yy K^T as n_y successive downdates.  A downdate
//  can fail when the argument is not positive definite in finite precision; the
//  implementation then falls back to an explicit re-factorisation with jitter
//  (cholesky_safe), so the filter never aborts and never returns NaN.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "sckf.hpp"    // kalman_detail::tria

namespace estkit {

namespace kalman_detail {
// Rank-one Cholesky update/downdate with a guaranteed fall-back: on return
// L L^T = L_old L_old^T + sign * v v^T (re-factorised with jitter if the in-place
// update would lose positive definiteness).
template <int N>
inline void safe_chol_update(Mat<N, N>& L, const Vec<N>& v, Real sign) {
    const Mat<N, N> L0 = L;
    if (chol_update(L, v, sign) && L.is_finite()) return;
    Mat<N, N> A = L0 * L0.t() + outer(v, v) * sign;
    A.symmetrize();
    L = cholesky_safe(A);
}
}  // namespace kalman_detail

template <class Model>
class SqrtUkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NS = 2 * NX + 1;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        Real alpha = Real(1e-1), beta = Real(2), kappa = Real(0);   // scaled UT parameters
        bool apply_constraints = true;     // project with Model::constrain (if defined)
        Real q_scale = Real(1);            // multiplicative tuning of Q
        Real r_scale = Real(1);            // multiplicative tuning of R
    } opts;

    explicit SqrtUkf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; Sx_ = cholesky_safe(P0); p_dirty_ = true; }

    void predict(const U& u) {
        sigma_points(x_, Sx_);
        X xm;
        for (int s = 0; s < NS; ++s) { Xs_[s] = m_.f(Xs_[s], u); xm += Xs_[s] * wm_[s]; }
        const Real rw = std::sqrt(wc_[1]);                       // W_i^c > 0 for i >= 1
        Mat<NX, 2 * NX + NX> pre;
        for (int i = 0; i < 2 * NX; ++i) pre.set_col(i, X((Xs_[1 + i] - xm) * rw));
        const Mat<NX, NX> SQ = cholesky_safe(m_.Q(xm, u) * opts.q_scale);
        for (int j = 0; j < NX; ++j) pre.set_col(2 * NX + j, SQ.col(j));
        Sx_ = kalman_detail::tria(pre);
        kalman_detail::safe_chol_update(Sx_, X((Xs_[0] - xm) * std::sqrt(std::fabs(wc_[0]))), sgn(wc_[0]));
        x_ = xm; p_dirty_ = true;
    }

    Y predict_measurement(const U& u) const {
        sigma_points(x_, Sx_);
        Y ym; for (int s = 0; s < NS; ++s) ym += m_.h(Xs_[s], u) * wm_[s];
        return ym;
    }

    void update(const Y& y, const U& u) {
        sigma_points(x_, Sx_);
        Y Zs[NS]; Y ym;
        for (int s = 0; s < NS; ++s) { Zs[s] = m_.h(Xs_[s], u); ym += Zs[s] * wm_[s]; }
        const Real rw = std::sqrt(wc_[1]);
        Mat<NY, 2 * NX + NY> pz;
        for (int i = 0; i < 2 * NX; ++i) pz.set_col(i, Y((Zs[1 + i] - ym) * rw));
        const Mat<NY, NY> SR = cholesky_safe(m_.R(x_, u) * opts.r_scale);
        for (int j = 0; j < NY; ++j) pz.set_col(2 * NX + j, SR.col(j));
        Mat<NY, NY> Sy = kalman_detail::tria(pz);
        kalman_detail::safe_chol_update(Sy, Y((Zs[0] - ym) * std::sqrt(std::fabs(wc_[0]))), sgn(wc_[0]));
        Mat<NX, NY> Pxy;
        for (int s = 0; s < NS; ++s) Pxy += outer(X(Xs_[s] - x_), Y(Zs[s] - ym)) * wc_[s];
        // K = Pxy S_y^{-T} S_y^{-1}   (two triangular solves)
        const Mat<NY, NX> t1 = solve_lower(Sy, Pxy.t());
        const Mat<NY, NX> t2 = solve_upper(Sy.t(), t1);
        const Mat<NX, NY> K = t2.t();
        if (!K.is_finite()) return;                              // keep the prior rather than produce NaN
        innovation_ = y - ym; Syy_ = Sy * Sy.t();
        x_ = x_ + K * innovation_;
        const Mat<NX, NY> KSy = K * Sy;                          // P^+ = P^- - (K S_y)(K S_y)^T
        for (int j = 0; j < NY; ++j) kalman_detail::safe_chol_update(Sx_, X(KSy.col(j)), Real(-1));
        p_dirty_ = true;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const {                               // reconstructed on demand, cached
        if (p_dirty_) { P_ = Sx_ * Sx_.t(); P_.symmetrize(); p_dirty_ = false; }
        return P_;
    }
    const Mat<NX, NX>& S_factor() const { return Sx_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return Syy_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

    // scaled unscented transform directly from the square root: X_i = x +/- gamma S_i
    void sigma_points(const X& x, const Mat<NX, NX>& S) const {
        const Real n = Real(NX);
        const Real lambda = opts.alpha * opts.alpha * (n + opts.kappa) - n;
        const Real c = n + lambda;
        const Real gamma = std::sqrt(std::fabs(c));
        Xs_[0] = x; wm_[0] = lambda / c; wc_[0] = lambda / c + (Real(1) - opts.alpha * opts.alpha + opts.beta);
        for (int i = 0; i < NX; ++i) {
            const X col = S.col(i) * gamma;
            Xs_[1 + i] = x + col; Xs_[1 + NX + i] = x - col;
            wm_[1 + i] = wm_[1 + NX + i] = wc_[1 + i] = wc_[1 + NX + i] = Real(1) / (Real(2) * c);
        }
    }

private:
    Model m_;
    X x_;
    Mat<NX, NX> Sx_;                 // P = Sx Sx^T
    mutable Mat<NX, NX> P_;
    mutable bool p_dirty_ = true;
    mutable X Xs_[NS];
    mutable Real wm_[NS], wc_[NS];
    Y innovation_;
    Mat<NY, NY> Syy_;
};

}  // namespace estkit
