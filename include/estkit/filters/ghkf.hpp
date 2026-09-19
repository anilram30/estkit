// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/ghkf.hpp — Gauss-Hermite quadrature Kalman filter (GHKF)
//
//  References
//    * Ito, K. & Xiong, K. (2000). Gaussian filters for nonlinear filtering
//      problems. IEEE Transactions on Automatic Control 45(5), 910-927. (primary)
//    * Arasaratnam, I., Haykin, S. & Elliott, R. J. (2007). Discrete-time nonlinear
//      filtering algorithms using Gauss-Hermite quadrature. Proceedings of the IEEE
//      95(5), 953-977.
//    * Wu, Y., Hu, D., Wu, M. & Hu, X. (2006). A numerical-integration perspective
//      on Gaussian filters. IEEE Transactions on Signal Processing 54(8), 2910-2921.
//    * Golub, G. H. & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
//      Mathematics of Computation 23(106), 221-230.   (how the nodes are obtained)
//
//  The Gaussian (assumed-density) filter needs the integrals
//      E[g(x)] = int g(x) N(x; xhat, P) dx .
//  With x = xhat + S z (S S^T = P, z ~ N(0,I)) the integral separates into n
//  one-dimensional standard-Gaussian integrals, each of which is approximated by an
//  m-point Gauss-Hermite rule.  The m-point rule integrates polynomials of degree
//  <= 2m-1 exactly, so the m = 3 rule used by default is exact to degree five - one
//  degree better than the unscented transform and equal to the fifth-degree
//  cubature rule, but with a full tensor grid:
//
//      nodes   z in {-sqrt(3), 0, +sqrt(3)},  weights {1/6, 2/3, 1/6}            (1)
//      sum w = 1,  sum w z^2 = 1,  sum w z^4 = 3   (exact),  sum w z^6 = 9 (vs 15)
//
//  The multivariate rule is the tensor product of (1): m^n points with weights
//  equal to the product of the one-dimensional weights.  This is the GHKF's
//  strength (arbitrary accuracy by raising m) and its weakness (the cost grows as
//  m^n, here 3^4 = 81 points for the cell model, and is impractical beyond n ~ 6).
//
//  Algorithm (one cycle, additive noise)
//    time update:    X_i = xhat + S xi_i,  X*_i = f(X_i,u),  x^- = sum w_i X*_i
//                    P^- = sum w_i (X*_i - x^-)(X*_i - x^-)^T + Q
//    measurement:    X_i = x^- + S^- xi_i,  Z_i = h(X_i,u),  yhat = sum w_i Z_i
//                    P_yy = sum w_i (Z_i-yhat)(.)^T + R,  P_xy = sum w_i (X_i-x^-)(.)^T
//                    K = P_xy P_yy^{-1},  x^+ = x^- + K(y-yhat),  P^+ = P^- - K P_yy K^T
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

namespace kalman_detail {
// Nodes and weights of the m-point Gauss-Hermite rule for the standard normal
// density (probabilists' Hermite polynomials He_m; Golub & Welsch 1969).
template <int M> struct GaussHermite;
template <> struct GaussHermite<2> {                 // He_2 = z^2 - 1, exact to degree 3
    static Real node(int i) { return i == 0 ? Real(-1) : Real(1); }
    static Real weight(int)  { return Real(0.5); }
};
template <> struct GaussHermite<3> {                 // He_3 = z^3 - 3z, exact to degree 5
    static Real node(int i) { return i == 0 ? -std::sqrt(Real(3)) : (i == 1 ? Real(0) : std::sqrt(Real(3))); }
    static Real weight(int i) { return i == 1 ? Real(2) / Real(3) : Real(1) / Real(6); }
};
constexpr int ipow_ct(int b, int e) { int r = 1; for (int k = 0; k < e; ++k) r *= b; return r; }
}  // namespace kalman_detail

template <class Model, int MPTS = 3>
class Ghkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NS = kalman_detail::ipow_ct(MPTS, NX);   // m^n quadrature points
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    static_assert(MPTS == 2 || MPTS == 3, "compile-time Gauss-Hermite table available for m = 2, 3");
    static_assert(NS <= 4096, "m^n quadrature grid too large; use a cubature filter instead");
    struct Options {
        bool apply_constraints = true;     // project with Model::constrain (if defined)
        Real q_scale = Real(1);            // multiplicative tuning of Q
        Real r_scale = Real(1);            // multiplicative tuning of R
    } opts;

    explicit Ghkf(const Model& m) : m_(m) {
        using GH = kalman_detail::GaussHermite<MPTS>;
        for (int s = 0; s < NS; ++s) {                 // mixed-radix expansion of the grid index
            int q = s; Real w = Real(1);
            for (int j = 0; j < NX; ++j) {
                const int d = q % MPTS; q /= MPTS;
                xi_[s][j] = GH::node(d); w *= GH::weight(d);
            }
            w_[s] = w;
        }
    }
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; }

    void predict(const U& u) {
        quadrature_points(x_, P_);
        X xm;
        for (int s = 0; s < NS; ++s) { Xs_[s] = m_.f(Xs_[s], u); xm += Xs_[s] * w_[s]; }
        Mat<NX, NX> Pm;
        for (int s = 0; s < NS; ++s) { const X d = Xs_[s] - xm; Pm += outer(d, d) * w_[s]; }
        x_ = xm;
        P_ = Pm + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const {
        quadrature_points(x_, P_);
        Y ym; for (int s = 0; s < NS; ++s) ym += m_.h(Xs_[s], u) * w_[s];
        return ym;
    }

    void update(const Y& y, const U& u) {
        quadrature_points(x_, P_);
        Y ym;
        for (int s = 0; s < NS; ++s) { Zs_[s] = m_.h(Xs_[s], u); ym += Zs_[s] * w_[s]; }
        Mat<NY, NY> Pyy = m_.R(x_, u) * opts.r_scale; Mat<NX, NY> Pxy;
        for (int s = 0; s < NS; ++s) {
            const Y dy = Zs_[s] - ym; const X dx = Xs_[s] - x_;
            Pyy += outer(dy, dy) * w_[s]; Pxy += outer(dx, dy) * w_[s];
        }
        const Mat<NX, NY> K = Pxy * inverse(Pyy);
        if (!K.is_finite()) return;                // keep the prior rather than produce NaN
        innovation_ = y - ym; S_ = Pyy;
        x_ = x_ + K * innovation_;
        P_ = P_ - K * Pyy * K.t();
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

    void quadrature_points(const X& x, const Mat<NX, NX>& P) const {
        const Mat<NX, NX> S = cholesky_safe(P);
        for (int s = 0; s < NS; ++s) {
            X p = x;
            for (int j = 0; j < NX; ++j) {
                const Real c = xi_[s][j];
                if (c != Real(0)) for (int i = 0; i < NX; ++i) p[i] += S(i, j) * c;
            }
            Xs_[s] = p;
        }
    }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Real xi_[NS][NX];        // unit-normal abscissae of the tensor grid
    Real w_[NS];             // product weights
    mutable X Xs_[NS];
    mutable Y Zs_[NS];
    Y innovation_;
    Mat<NY, NY> S_;
};

}  // namespace estkit
