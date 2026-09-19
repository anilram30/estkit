// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/ckf_high_degree.hpp — Fifth-degree cubature Kalman filter (CKF5)
//
//  References
//    * Jia, B., Xin, M. & Cheng, Y. (2013). High-degree cubature Kalman filter.
//      Automatica 49(2), 510-518.                                   (primary)
//    * Arasaratnam, I. & Haykin, S. (2009). Cubature Kalman filters. IEEE
//      Transactions on Automatic Control 54(6), 1254-1269.  (third-degree parent)
//    * Mysovskikh, I. P. (1980). The approximation of multiple integrals by using
//      interpolatory cubature formulae. In: Quantitative Approximation (Devore &
//      Scherer, eds.), Academic Press, 217-243.  (the fifth-degree invariant rule)
//    * Stroud, A. H. (1971). Approximate Calculation of Multiple Integrals.
//      Prentice-Hall.                          (classification of the rules)
//
//  The third-degree spherical-radial rule of the CKF integrates polynomials up to
//  total degree three exactly; the residual error is therefore governed by the
//  fourth-order term of the Taylor expansion of f and h.  Jia, Xin & Cheng (2013)
//  construct arbitrary odd-degree spherical-radial rules by combining a
//  Gauss-Chebyshev radial rule with a degree-matched spherical rule.  The
//  fifth-degree member uses 2n^2+1 points:
//
//    xi_0 = 0                                        w_0 = 2/(n+2)
//    xi   = sqrt(n+2) * (+/- e_i),      2n points,   w_1 = (4-n)/(2 (n+2)^2)
//    xi   = sqrt(n+2) * (+/- e_k +/- e_l)/sqrt(2),
//                                  2n(n-1) points,   w_2 = 1/(n+2)^2
//
//  with 1 <= i <= n and 1 <= k < l <= n.  Direct verification of the moments gives
//    sum w = 1,   sum w xi_a^2 = 1,   sum w xi_a^4 = 3,   sum w xi_a^2 xi_b^2 = 1,
//  and all odd moments vanish by central symmetry, i.e. the rule is exact for every
//  monomial of total degree <= 5.  For n = 4 (the cell model) w_1 = 0 exactly, so
//  the 8 axial points carry no weight; for n > 4 the axial weight becomes negative
//  and the sample covariance is no longer guaranteed positive semi-definite - this
//  is the known price of high-degree rules and is handled here by the jittered
//  Cholesky (`cholesky_safe`) and by adding Q (resp. R) after the sum.
//
//  Algorithm (one cycle, additive noise) - identical in structure to the CKF, only
//  the point set and the weights differ:
//    time update:    X_i = S^+ xi_i + x^+,  X*_i = f(X_i,u),  x^- = sum w_i X*_i
//                    P^- = sum w_i (X*_i - x^-)(X*_i - x^-)^T + Q
//    measurement:    X_i = S^- xi_i + x^-,  Z_i = h(X_i,u),   yhat = sum w_i Z_i
//                    P_zz = sum w_i (Z_i-yhat)(.)^T + R,  P_xz = sum w_i (X_i-x^-)(.)^T
//                    W = P_xz P_zz^{-1},  x^+ = x^- + W(y-yhat),  P^+ = P^- - W P_zz W^T
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class Ckf5 {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NS = 2 * NX * NX + 1;     // 2n^2+1 cubature points
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        bool apply_constraints = true;             // project with Model::constrain (if defined)
        Real q_scale = Real(1);                    // multiplicative tuning of Q
        Real r_scale = Real(1);                    // multiplicative tuning of R
    } opts;

    explicit Ckf5(const Model& m) : m_(m) {
        const Real n = Real(NX), c = Real(NX) + Real(2);
        const Real w0 = Real(2) / c;
        const Real w1 = (Real(4) - n) / (Real(2) * c * c);
        const Real w2 = Real(1) / (c * c);
        int s = 0;
        w_[s++] = w0;
        for (int i = 0; i < 2 * NX; ++i) w_[s++] = w1;
        for (; s < NS; ++s) w_[s] = w2;
    }
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; }

    void predict(const U& u) {
        cubature_points(x_, P_);
        X xm;
        for (int s = 0; s < NS; ++s) { Xs_[s] = m_.f(Xs_[s], u); xm += Xs_[s] * w_[s]; }
        Mat<NX, NX> Pm;
        for (int s = 0; s < NS; ++s) { const X d = Xs_[s] - xm; Pm += outer(d, d) * w_[s]; }
        x_ = xm;
        P_ = Pm + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const {
        cubature_points(x_, P_);
        Y ym; for (int s = 0; s < NS; ++s) ym += m_.h(Xs_[s], u) * w_[s];
        return ym;
    }

    void update(const Y& y, const U& u) {
        cubature_points(x_, P_);
        Y Zs[NS]; Y ym;
        for (int s = 0; s < NS; ++s) { Zs[s] = m_.h(Xs_[s], u); ym += Zs[s] * w_[s]; }
        Mat<NY, NY> Pzz = m_.R(x_, u) * opts.r_scale; Mat<NX, NY> Pxz;
        for (int s = 0; s < NS; ++s) {
            const Y dz = Zs[s] - ym; const X dx = Xs_[s] - x_;
            Pzz += outer(dz, dz) * w_[s]; Pxz += outer(dx, dz) * w_[s];
        }
        const Mat<NX, NY> W = Pxz * inverse(Pzz);
        if (!W.is_finite()) return;                // keep the prior rather than produce NaN
        innovation_ = y - ym; S_ = Pzz;
        x_ = x_ + W * innovation_;
        P_ = P_ - W * Pzz * W.t();
        P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    Real weight(int s) const { return w_[s]; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

    // point set of the fifth-degree rule, mapped through S with S S^T = P
    void cubature_points(const X& x, const Mat<NX, NX>& P) const {
        const Mat<NX, NX> S = cholesky_safe(P);
        const Real g1 = std::sqrt(Real(NX) + Real(2));
        const Real g2 = g1 / std::sqrt(Real(2));
        int s = 0;
        Xs_[s++] = x;
        for (int i = 0; i < NX; ++i) {
            const X c = S.col(i) * g1;
            Xs_[s++] = x + c; Xs_[s++] = x - c;
        }
        for (int k = 0; k < NX; ++k)
            for (int l = k + 1; l < NX; ++l) {
                const X ck = S.col(k) * g2, cl = S.col(l) * g2;
                Xs_[s++] = x + ck + cl; Xs_[s++] = x + ck - cl;
                Xs_[s++] = x - ck + cl; Xs_[s++] = x - ck - cl;
            }
    }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    mutable X Xs_[NS];
    Real w_[NS];
    Y innovation_;
    Mat<NY, NY> S_;
};

}  // namespace estkit
