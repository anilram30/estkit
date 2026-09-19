// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/ckf.hpp — Cubature Kalman filter (third-degree spherical-radial
//  rule), additive-noise form.
//
//  References
//    * Arasaratnam, I. & Haykin, S. (2009). Cubature Kalman filters. IEEE
//      Transactions on Automatic Control 54(6), 1254-1269.       (primary)
//    * Ito, K. & Xiong, K. (2000). Gaussian filters for nonlinear filtering
//      problems. IEEE Transactions on Automatic Control 45(5), 910-927.
//      (the Gaussian-filter framework the rule is embedded in)
//    * Sarkka, S. (2013). Bayesian Filtering and Smoothing. Cambridge University
//      Press, Ch. 6.
//
//  The Bayesian filter for a nonlinear model with additive Gaussian noise reduces,
//  under the Gaussian assumption, to a sequence of integrals of the form
//      I[g] = int_{R^n} g(x) N(x; xhat, P) dx .
//  Writing x = xhat + S z with S S^T = P and z ~ N(0,I) and using the
//  spherical-radial decomposition z = r y, ||y|| = 1, Arasaratnam & Haykin derive
//  the third-degree rule with m = 2n equally weighted points
//
//      xi_i = sqrt(n) [1]_i,    w_i = 1/(2n),    i = 1..2n,                    (1)
//
//  where [1]_i runs over the 2n columns of [ +I_n , -I_n ].  The rule integrates
//  every polynomial of total degree <= 3 exactly, has strictly positive weights
//  for every n (unlike the scaled unscented transform, whose centre weight becomes
//  negative for n > 3) and needs no tuning parameters.
//
//  Algorithm (one cycle, additive noise)
//    time update:      X_i = S^+ xi_i + x^+,  X*_i = f(X_i, u_{k-1})
//                      x^- = (1/m) sum X*_i
//                      P^- = (1/m) sum (X*_i - x^-)(X*_i - x^-)^T + Q
//    measurement:      X_i = S^- xi_i + x^-,  Z_i = h(X_i, u_k)
//                      yhat = (1/m) sum Z_i
//                      P_zz = (1/m) sum (Z_i-yhat)(Z_i-yhat)^T + R
//                      P_xz = (1/m) sum (X_i-x^-)(Z_i-yhat)^T
//                      W = P_xz P_zz^{-1},  x^+ = x^- + W (y - yhat),
//                      P^+ = P^- - W P_zz W^T
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class Ckf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NS = 2 * NX;      // number of cubature points
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        bool apply_constraints = true;     // project the estimate with Model::constrain (if defined)
        Real q_scale = Real(1);            // multiplicative tuning of Q
        Real r_scale = Real(1);            // multiplicative tuning of R
    } opts;

    explicit Ckf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; }

    void predict(const U& u) {
        cubature_points(x_, P_);
        X xm;
        for (int s = 0; s < NS; ++s) { Xs_[s] = m_.f(Xs_[s], u); xm += Xs_[s]; }
        xm *= kW;
        Mat<NX, NX> Pm;
        for (int s = 0; s < NS; ++s) { const X d = Xs_[s] - xm; Pm += outer(d, d); }
        Pm *= kW;
        x_ = xm;
        P_ = Pm + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const {
        cubature_points(x_, P_);
        Y ym; for (int s = 0; s < NS; ++s) ym += m_.h(Xs_[s], u);
        return ym * kW;
    }

    void update(const Y& y, const U& u) {
        cubature_points(x_, P_);
        Y Zs[NS]; Y ym;
        for (int s = 0; s < NS; ++s) { Zs[s] = m_.h(Xs_[s], u); ym += Zs[s]; }
        ym *= kW;
        Mat<NY, NY> Pzz; Mat<NX, NY> Pxz;
        for (int s = 0; s < NS; ++s) {
            const Y dz = Zs[s] - ym; const X dx = Xs_[s] - x_;
            Pzz += outer(dz, dz); Pxz += outer(dx, dz);
        }
        Pzz *= kW; Pxz *= kW;
        Pzz += m_.R(x_, u) * opts.r_scale;
        const Mat<NX, NY> W = Pxz * inverse(Pzz);
        if (!W.is_finite()) return;              // keep the prior rather than produce NaN
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
    Model& model() { return m_; }
    const Model& model() const { return m_; }

    // X_i = x + sqrt(n) * (+/- S e_i),  S S^T = P  (Arasaratnam & Haykin 2009, Eq. 33)
    void cubature_points(const X& x, const Mat<NX, NX>& P) const {
        const Mat<NX, NX> S = cholesky_safe(P);
        const Real g = std::sqrt(Real(NX));
        for (int i = 0; i < NX; ++i) {
            const X col = S.col(i) * g;
            Xs_[i] = x + col; Xs_[NX + i] = x - col;
        }
    }
    const X* points() const { return Xs_; }

private:
    static constexpr Real kW = Real(1) / Real(NS);   // equal cubature weight 1/(2n)
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    mutable X Xs_[NS];
    Y innovation_;
    Mat<NY, NY> S_;
};

}  // namespace estkit
