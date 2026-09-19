// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/srekf.hpp — Square-root (covariance square-root) extended Kalman
//  filter using the QR "array algorithm".
//
//  References
//    * Kaminski, P. G., Bryson, A. E. & Schmidt, S. F. (1971). Discrete square root
//      filtering: a survey of current techniques. IEEE Transactions on Automatic
//      Control 16(6), 727-736.                                        (primary)
//    * Kailath, T., Sayed, A. H. & Hassibi, B. (2000). Linear Estimation.
//      Prentice Hall, Ch. 12 (array algorithms, pre-array / post-array).
//    * Potter, J. E. & Stern, R. G. (1963). Statistical filtering of space
//      navigation measurements. Proc. AIAA Guidance and Control Conference.
//      (the original scalar square-root measurement update flown on Apollo)
//    * Verhaegen, M. & Van Dooren, P. (1986). Numerical aspects of different Kalman
//      filter implementations. IEEE Transactions on Automatic Control 31(10),
//      907-917.                                   (error analysis of the variants)
//    * Golub, G. H. & Van Loan, C. F. (2013). Matrix Computations, 4th ed.,
//      Alg. 5.2.1 (Householder QR).
//
//  Idea.  The conventional covariance recursion can produce an indefinite P under
//  round-off because it subtracts (P^- - K S K^T).  A square-root filter stores a
//  factor S with P = S S^T, so P is positive semi-definite by construction, and the
//  condition number of the propagated array is the square root of cond(P): the
//  filter behaves like a covariance filter run with twice the mantissa length
//  (Kaminski et al. 1971), which is decisive in single precision.
//
//  The whole cycle is expressed as the orthogonal triangularisation of a pre-array,
//      Tria(A) := R^T with R = qr(A^T) upper triangular  =>  Tria(A)Tria(A)^T = A A^T.
//
//  Time update (n x 2n pre-array):
//      S^- = Tria( [ F S^+ ,  S_Q ] ),        Q = S_Q S_Q^T                  (1)
//  Measurement update ((n_y+n) x (n_y+n) pre-array, Kailath et al. Ch. 12):
//
//      [ S_R   H S^- ]         [ S_e     0   ]
//      [  0     S^-  ]  --->   [ K_p    S^+  ]  = Tria(.)                    (2)
//
//  Multiplying out the post-array gives
//      S_e S_e^T = H P^- H^T + R   (innovation covariance factor),
//      K_p       = P^- H^T S_e^{-T}   (normalised gain),  K = K_p S_e^{-1},
//      S^+ S^+^T = P^- - K (H P^- H^T + R) K^T,
//  i.e. one QR produces the gain, the innovation covariance and the updated factor
//  simultaneously, with no explicit inverse.  The state update uses one forward
//  substitution:  x^+ = x^- + K_p (S_e^{-1} nu),  nu = y - h(x^-,u).
//
//  Potter's original 1963 form treats a scalar measurement with a rank-one
//  symmetric-square-root update S^+ = S^-(I - a phi phi^T), phi = S^{-T} H^T; it is
//  cheaper (no QR) but restricted to scalar updates and to a symmetric (non
//  triangular) factor.  The array form above is preferred here because it handles
//  vector measurements and process noise in one orthogonal transformation.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "sckf.hpp"    // kalman_detail::tria

namespace estkit {

template <class Model>
class SqrtEkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        bool apply_constraints = true;     // project with Model::constrain (if defined)
        Real q_scale = Real(1);            // multiplicative tuning of Q
        Real r_scale = Real(1);            // multiplicative tuning of R
    } opts;

    explicit SqrtEkf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; S_ = cholesky_safe(P0); p_dirty_ = true; }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        const Mat<NX, NX> SQ = cholesky_safe(m_.Q(x_, u) * opts.q_scale);
        Mat<NX, 2 * NX> pre;                                   // [ F S , S_Q ]
        const Mat<NX, NX> FS = F * S_;
        for (int j = 0; j < NX; ++j) { pre.set_col(j, FS.col(j)); pre.set_col(NX + j, SQ.col(j)); }
        S_ = kalman_detail::tria(pre);
        p_dirty_ = true;
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> SR = cholesky_safe(m_.R(x_, u) * opts.r_scale);
        Mat<NY + NX, NY + NX> pre;                             // [ S_R , H S ; 0 , S ]
        pre.set_block(0, 0, SR);
        pre.set_block(0, NY, Mat<NY, NX>(H * S_));
        pre.set_block(NY, NY, S_);
        const Mat<NY + NX, NY + NX> post = kalman_detail::tria(pre);
        const Mat<NY, NY> Se = post.template block<NY, NY>(0, 0);
        const Mat<NX, NY> Kp = post.template block<NX, NY>(NY, 0);
        const Mat<NX, NX> Sp = post.template block<NX, NX>(NY, NY);
        innovation_ = y - m_.h(x_, u);
        Syy_ = Se * Se.t();
        const Y w = solve_lower(Se, innovation_);              // w = S_e^{-1} nu
        if (!w.is_finite() || !Sp.is_finite()) return;         // keep the prior rather than produce NaN
        x_ = x_ + Kp * w;
        S_ = Sp;
        p_dirty_ = true;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const {                             // reconstructed on demand, cached
        if (p_dirty_) { P_ = S_ * S_.t(); P_.symmetrize(); p_dirty_ = false; }
        return P_;
    }
    const Mat<NX, NX>& S_factor() const { return S_; }         // P = S S^T
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return Syy_; }              // innovation covariance
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> S_;
    mutable Mat<NX, NX> P_;
    mutable bool p_dirty_ = true;
    Y innovation_;
    Mat<NY, NY> Syy_;
};

}  // namespace estkit
