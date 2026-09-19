// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/udkf.hpp — U-D factorised Kalman filter (Bierman-Thornton),
//  extended-Kalman variant.
//
//  References
//    * Bierman, G. J. (1977). Factorization Methods for Discrete Sequential
//      Estimation. Academic Press, New York, Ch. V (the U-D measurement update)
//      and App. III (the U-D factorisation itself).                 (primary)
//    * Thornton, C. L. & Bierman, G. J. (1977). Gram-Schmidt algorithms for
//      covariance propagation. International Journal of Control 25(2), 243-260.
//      (the modified weighted Gram-Schmidt time update)              (primary)
//    * Kaminski, P. G., Bryson, A. E. & Schmidt, S. F. (1971). Discrete square root
//      filtering: a survey of current techniques. IEEE Transactions on Automatic
//      Control 16(6), 727-736.
//    * Grewal, M. S. & Andrews, A. P. (2014). Kalman Filtering: Theory and Practice
//      Using MATLAB, 4th ed. Wiley, Section 6.5 ("Bierman-Thornton UD filter").
//    * Verhaegen, M. & Van Dooren, P. (1986). Numerical aspects of different Kalman
//      filter implementations. IEEE Trans. Autom. Control 31(10), 907-917.
//
//  Parameterisation.  P = U D U^T with U unit upper triangular and D = diag(d)
//  non-negative.  Like a square-root filter it cannot produce a negative-definite
//  covariance and it has the conditioning of a square root, but - unlike a Cholesky
//  square root - it needs NO square roots at all, only multiplies and divides.
//  That is why the U-D filter is the classical choice for fixed-point avionics
//  (Shuttle, Voyager, GPS receivers) and still the reference implementation for
//  certified embedded navigation software.
//
//  Measurement update (Bierman 1977, Ch. V), one SCALAR observation y = a^T x + v,
//  var(v) = r, at a time.  With f = U^T a and v = D f:
//      alpha_0 = r,  alpha_j = alpha_{j-1} + f_j v_j
//      d_j    <- d_j alpha_{j-1} / alpha_j
//      lambda  = -f_j / alpha_{j-1}
//      U_ij   <- U_ij + lambda b_i,   b_i <- b_i + U_ij^{old} v_j   (i < j)
//      b_j     = v_j
//  After the sweep alpha_n = a^T P a + r is the innovation variance and b = P a is
//  the unnormalised gain, so x <- x + (b / alpha_n) (y - a^T x).  Vector
//  measurements are decorrelated first (R = L L^T, a-rows and residual premultiplied
//  by L^{-1}) so that the scalar recursion is exact rather than approximate.
//
//  Time update (Thornton & Bierman 1977).  Write
//      P^- = F U D U^T F^T + G D_q G^T = W diag(D, D_q) W^T,   W = [F U , G]
//  where Q = G D_q G^T is itself U-D factorised (so a fully correlated Q is
//  admissible).  Applying modified weighted Gram-Schmidt to the ROWS of W with the
//  weight vector diag(D, D_q) - orthogonalising from the last row upwards - yields
//  U^- and D^- directly, in O(n^2(n+n_q)) operations and without forming P^-.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

namespace kalman_detail {
// U-D factorisation of a symmetric positive semi-definite matrix:
// M = U D U^T, U unit upper triangular (Bierman 1977, App. III).
template <int N>
inline void ud_factor(const Mat<N, N>& M, Mat<N, N>& U, Vec<N>& d) {
    U = Mat<N, N>(); d = Vec<N>();
    for (int j = N - 1; j >= 0; --j) {
        for (int i = j; i >= 0; --i) {
            Real s = M(i, j);
            for (int k = j + 1; k < N; ++k) s -= U(i, k) * d[k] * U(j, k);
            if (i == j) { d[j] = (s > Real(0)) ? s : Real(0); U(j, j) = Real(1); }
            else        { U(i, j) = (d[j] > Real(0)) ? s / d[j] : Real(0); }
        }
    }
}
}  // namespace kalman_detail

template <class Model>
class UdKf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NW = 2 * NX;        // width of the Gram-Schmidt work array
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        bool decorrelate = true;           // whiten vector measurements with chol(R) before the scalar sweep
        bool apply_constraints = true;     // project with Model::constrain (if defined)
        Real q_scale = Real(1);            // multiplicative tuning of Q
        Real r_scale = Real(1);            // multiplicative tuning of R
        Real d_floor = Real(0);            // lower bound applied to the diagonal factors
    } opts;

    explicit UdKf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; kalman_detail::ud_factor(P0, U_, d_); p_dirty_ = true;
    }

    // --- Thornton modified weighted Gram-Schmidt time update ---------------------
    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        Mat<NX, NX> Uq; Vec<NX> dq;
        kalman_detail::ud_factor(Mat<NX, NX>(m_.Q(x_, u) * opts.q_scale), Uq, dq);
        Mat<NX, NW> W;                                   // [ F U , G ]
        const Mat<NX, NX> FU = F * U_;
        Real dw[NW];
        for (int j = 0; j < NX; ++j) {
            for (int i = 0; i < NX; ++i) { W(i, j) = FU(i, j); W(i, NX + j) = Uq(i, j); }
            dw[j] = d_[j]; dw[NX + j] = dq[j];
        }
        Mat<NX, NX> Un; Vec<NX> dn;
        for (int i = NX - 1; i >= 0; --i) {
            Real s = Real(0);
            for (int k = 0; k < NW; ++k) s += W(i, k) * W(i, k) * dw[k];
            dn[i] = (s > opts.d_floor) ? s : opts.d_floor;
            Un(i, i) = Real(1);
            if (!(dn[i] > Real(0))) continue;            // degenerate row: leave U column at e_i
            for (int j = 0; j < i; ++j) {
                Real c = Real(0);
                for (int k = 0; k < NW; ++k) c += W(i, k) * dw[k] * W(j, k);
                const Real uji = c / dn[i];
                Un(j, i) = uji;
                for (int k = 0; k < NW; ++k) W(j, k) -= uji * W(i, k);
            }
        }
        U_ = Un; d_ = dn; p_dirty_ = true;
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    // --- Bierman scalar measurement updates -------------------------------------
    void update(const Y& y, const U& u) {
        const X xm = x_;                                  // linearisation point x^-
        const Mat<NY, NX> H = jac_H(m_, xm, u);
        const Mat<NY, NY> R = m_.R(xm, u) * opts.r_scale;
        innovation_ = y - m_.h(xm, u);
        Mat<NY, NX> A = H;                                // rows of the (whitened) observation matrix
        Y res = innovation_;                              // (whitened) residual
        Vec<NY> rvar;
        if (opts.decorrelate && NY > 1) {
            const Mat<NY, NY> LR = cholesky_safe(R);      // R = L L^T -> unit-variance components
            A = solve_lower(LR, H);
            res = solve_lower(LR, innovation_);
            for (int k = 0; k < NY; ++k) rvar[k] = Real(1);
        } else {
            for (int k = 0; k < NY; ++k) rvar[k] = R(k, k);
        }
        Real alpha_last = Real(1);
        for (int k = 0; k < NY; ++k) {
            Vec<NX> a; for (int i = 0; i < NX; ++i) a[i] = A(k, i);
            // residual of the k-th component at the CURRENT state (sequential processing)
            Real dz = res[k];
            for (int i = 0; i < NX; ++i) dz -= a[i] * (x_[i] - xm[i]);
            alpha_last = scalar_update_(a, rvar[k] > Real(0) ? rvar[k] : Real(1e-12), dz);
        }
        S_ = Mat<NY, NY>(); S_(0, 0) = alpha_last;        // innovation variance of the last component
        p_dirty_ = true;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const {                        // P = U D U^T, reconstructed on demand
        if (p_dirty_) {
            Mat<NX, NX> UD;
            for (int i = 0; i < NX; ++i) for (int j = 0; j < NX; ++j) UD(i, j) = U_(i, j) * d_[j];
            P_ = UD * U_.t(); P_.symmetrize(); p_dirty_ = false;
        }
        return P_;
    }
    const Mat<NX, NX>& U_factor() const { return U_; }    // unit upper triangular
    const Vec<NX>& D_factor() const { return d_; }        // diagonal of D
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    // Bierman (1977) Ch. V: rank-one U-D update for one scalar observation.
    // Returns alpha = a^T P a + r (the innovation variance).
    Real scalar_update_(const Vec<NX>& a, Real r, Real dz) {
        Real f[NX], b[NX];
        for (int j = 0; j < NX; ++j) {                    // f = U^T a
            Real s = Real(0);
            for (int i = 0; i <= j; ++i) s += U_(i, j) * a[i];
            f[j] = s;
        }
        for (int j = 0; j < NX; ++j) b[j] = d_[j] * f[j];  // b = D f
        Real alpha = r;
        if (!(alpha > Real(0))) alpha = Real(1e-12);
        Real gamma = Real(1) / alpha;
        for (int j = 0; j < NX; ++j) {
            const Real beta = alpha;
            alpha += f[j] * b[j];
            if (!(alpha > Real(0))) { alpha = beta; continue; }
            const Real lambda = -f[j] * gamma;
            gamma = Real(1) / alpha;
            d_[j] *= beta * gamma;
            if (d_[j] < opts.d_floor) d_[j] = opts.d_floor;
            for (int i = 0; i < j; ++i) {
                const Real uij = U_(i, j);
                U_(i, j) = uij + lambda * b[i];
                b[i] += uij * b[j];
            }
        }
        const Real g = dz / alpha;                        // K = b / alpha
        if (std::isfinite(g)) for (int i = 0; i < NX; ++i) x_[i] += g * b[i];
        return alpha;
    }

    Model m_;
    X x_;
    Mat<NX, NX> U_;                  // unit upper triangular, P = U D U^T
    Vec<NX> d_;                      // diagonal of D
    mutable Mat<NX, NX> P_;
    mutable bool p_dirty_ = true;
    Y innovation_;
    Mat<NY, NY> S_;
};

}  // namespace estkit
