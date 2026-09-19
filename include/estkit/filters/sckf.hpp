// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/sckf.hpp — Square-root cubature Kalman filter (SCKF)
//
//  References
//    * Arasaratnam, I. & Haykin, S. (2009). Cubature Kalman filters. IEEE
//      Transactions on Automatic Control 54(6), 1254-1269, Section VI
//      ("Square-root cubature Kalman filter", Algorithms 3-4).
//    * Kailath, T., Sayed, A. H. & Hassibi, B. (2000). Linear Estimation.
//      Prentice Hall, Ch. 12 (array algorithms; the pre-array/post-array idea).
//    * Golub, G. H. & Van Loan, C. F. (2013). Matrix Computations, 4th ed.,
//      Alg. 5.2.1 (Householder QR, used for the triangularisation).
//
//  The SCKF propagates a Cholesky factor S with P = S S^T instead of P itself.
//  Every covariance sum is written as a matrix product A A^T and the factor is
//  obtained by triangularising the "pre-array" A:
//
//      Tria(A) := R^T  where  R = qr(A^T)  is upper triangular,             (1)
//
//  so that Tria(A) is lower triangular and Tria(A) Tria(A)^T = A A^T.  Because the
//  Householder reflections are orthogonal, the condition number of the propagated
//  quantity is the square root of that of P: a square-root filter needs roughly
//  half the mantissa bits of the covariance form (Kaminski et al. 1971), which is
//  what makes it the standard choice in single-precision avionics software.
//
//  Algorithm (one cycle, additive noise; m = 2n cubature points, weight 1/m)
//    time update:    X_i   = S^+ xi_i + x^+,   X*_i = f(X_i, u_{k-1})
//                    x^-   = (1/m) sum_i X*_i
//                    chi*  = (1/sqrt(m)) [ X*_1 - x^-, ... , X*_m - x^- ]
//                    S^-   = Tria( [ chi* , S_Q ] ),        Q = S_Q S_Q^T
//    measurement:    X_i   = S^- xi_i + x^-,   Z_i  = h(X_i, u_k)
//                    yhat  = (1/m) sum_i Z_i
//                    Z     = (1/sqrt(m)) [ Z_1 - yhat, ... ],  chi likewise
//                    S_zz  = Tria( [ Z , S_R ] ),            R = S_R S_R^T
//                    P_xz  = chi Z^T,   W = (P_xz / S_zz^T) / S_zz
//                    x^+   = x^- + W (y - yhat)
//                    S^+   = Tria( [ chi - W Z , W S_R ] )
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

namespace kalman_detail {
// Tria(): lower-triangular factor L of A A^T, obtained from the QR of A^T
// (Arasaratnam & Haykin 2009, Eq. 46; Kailath et al. 2000, Ch. 12).
template <int N, int M>
inline Mat<N, N> tria(const Mat<N, M>& A) {
    static_assert(M >= N, "tria() needs at least as many columns as rows");
    return qr_r(A.t()).t();
}
}  // namespace kalman_detail

template <class Model>
class SqrtCkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NS = 2 * NX;      // number of cubature points
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        bool apply_constraints = true;     // project the estimate with Model::constrain (if defined)
        Real q_scale = Real(1);            // multiplicative tuning of Q
        Real r_scale = Real(1);            // multiplicative tuning of R
    } opts;

    explicit SqrtCkf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; Sx_ = cholesky_safe(P0); p_dirty_ = true;
    }

    void predict(const U& u) {
        cubature_points(x_, Sx_);
        X xm;
        for (int s = 0; s < NS; ++s) { Xs_[s] = m_.f(Xs_[s], u); xm += Xs_[s]; }
        xm *= kW;
        Mat<NX, NS + NX> pre;                             // [ chi* , S_Q ]
        for (int s = 0; s < NS; ++s) { const X d = (Xs_[s] - xm) * root_w(); pre.set_col(s, d); }
        const Mat<NX, NX> SQ = cholesky_safe(m_.Q(xm, u) * opts.q_scale);
        for (int j = 0; j < NX; ++j) pre.set_col(NS + j, SQ.col(j));
        x_ = xm; Sx_ = kalman_detail::tria(pre); p_dirty_ = true;
    }

    Y predict_measurement(const U& u) const {
        cubature_points(x_, Sx_);
        Y ym; for (int s = 0; s < NS; ++s) ym += m_.h(Xs_[s], u);
        return ym * kW;
    }

    void update(const Y& y, const U& u) {
        cubature_points(x_, Sx_);
        Y Zs[NS]; Y ym;
        for (int s = 0; s < NS; ++s) { Zs[s] = m_.h(Xs_[s], u); ym += Zs[s]; }
        ym *= kW;
        Mat<NY, NS + NY> pz;                              // [ Z , S_R ]
        Mat<NX, NS> chi;                                  // weighted state deviations
        for (int s = 0; s < NS; ++s) {
            pz.set_col(s, Y((Zs[s] - ym) * root_w()));
            chi.set_col(s, X((Xs_[s] - x_) * root_w()));
        }
        const Mat<NY, NY> SR = cholesky_safe(m_.R(x_, u) * opts.r_scale);
        for (int j = 0; j < NY; ++j) pz.set_col(NS + j, SR.col(j));
        const Mat<NY, NY> Szz = kalman_detail::tria(pz);
        const Mat<NX, NY> Pxz = chi * pz.template block<NY, NS>(0, 0).t();
        // W = Pxz Szz^{-T} Szz^{-1}   (two triangular solves, never forms Pzz^{-1})
        const Mat<NY, NX> t1 = solve_lower(Szz, Pxz.t());
        const Mat<NY, NX> t2 = solve_upper(Szz.t(), t1);
        const Mat<NX, NY> W = t2.t();
        if (!W.is_finite()) return;                       // keep the prior rather than produce NaN
        innovation_ = y - ym; Syy_ = Szz * Szz.t();
        x_ = x_ + W * innovation_;
        Mat<NX, NS + NY> post;                            // [ chi - W Z , W S_R ]
        const Mat<NX, NS> WZ = W * pz.template block<NY, NS>(0, 0);
        for (int s = 0; s < NS; ++s) post.set_col(s, X(chi.col(s) - WZ.col(s)));
        const Mat<NX, NY> WSR = W * SR;
        for (int j = 0; j < NY; ++j) post.set_col(NS + j, WSR.col(j));
        Sx_ = kalman_detail::tria(post); p_dirty_ = true;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const {                        // reconstructed on demand, cached
        if (p_dirty_) { P_ = Sx_ * Sx_.t(); P_.symmetrize(); p_dirty_ = false; }
        return P_;
    }
    const Mat<NX, NX>& S_factor() const { return Sx_; }   // the propagated square root
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return Syy_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

    void cubature_points(const X& x, const Mat<NX, NX>& S) const {
        const Real g = std::sqrt(Real(NX));
        for (int i = 0; i < NX; ++i) {
            const X col = S.col(i) * g;
            Xs_[i] = x + col; Xs_[NX + i] = x - col;
        }
    }

private:
    static constexpr Real kW = Real(1) / Real(NS);
    static Real root_w() { return Real(1) / std::sqrt(Real(NS)); }   // sqrt(1/(2n))
    Model m_;
    X x_;
    Mat<NX, NX> Sx_;                 // P = Sx Sx^T
    mutable Mat<NX, NX> P_;          // cache for P()
    mutable bool p_dirty_ = true;
    mutable X Xs_[NS];
    Y innovation_;
    Mat<NY, NY> Syy_;
};

}  // namespace estkit
