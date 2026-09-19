// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/core/linalg.hpp — fixed-size, heap-free linear algebra for embedded
//  state estimators.
//
//  Design goals
//    * Every matrix has compile-time dimensions -> no dynamic memory, no
//      exceptions, deterministic execution time (suitable for Cortex-M targets).
//    * Real is `double` by default; compile with -DESTKIT_REAL=float to obtain a
//      single-precision build (used to study numerical robustness of the
//      square-root / UD filters, cf. Verhaegen & Van Dooren 1986).
//    * Algorithms: Cholesky (Golub & Van Loan 2013, Alg. 4.2.1), Householder QR
//      (GVL Alg. 5.2.1), LU with partial pivoting (GVL Alg. 3.4.1),
//      rank-1 Cholesky update/downdate (GVL §6.5.4), Ackermann pole placement,
//      discrete algebraic Riccati equation by fixed-point iteration.
// =============================================================================
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <initializer_list>

#ifndef ESTKIT_REAL
#define ESTKIT_REAL double
#endif

namespace estkit {

using Real = ESTKIT_REAL;

constexpr Real kNaN = std::numeric_limits<Real>::quiet_NaN();
constexpr Real kEps = std::numeric_limits<Real>::epsilon();
constexpr Real kPi  = Real(3.14159265358979323846);

template <typename T> inline T sq(T x) { return x * x; }
template <typename T> inline T clampr(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline Real sgn(Real x) { return x > Real(0) ? Real(1) : (x < Real(0) ? Real(-1) : Real(0)); }
// saturation function (used by sliding-mode observers to reduce chattering)
inline Real sat(Real x, Real phi) { return (phi <= Real(0)) ? sgn(x) : clampr(x / phi, Real(-1), Real(1)); }

// -----------------------------------------------------------------------------
//  Mat<R,C>: row-major dense matrix with compile-time size
// -----------------------------------------------------------------------------
template <int R, int C>
struct Mat {
    static constexpr int rows = R;
    static constexpr int cols = C;
    Real d[R * C];

    Mat() { for (int k = 0; k < R * C; ++k) d[k] = Real(0); }
    Mat(std::initializer_list<Real> l) {
        int k = 0;
        for (Real v : l) { if (k < R * C) d[k++] = v; }
        for (; k < R * C; ++k) d[k] = Real(0);
    }
    static Mat zeros() { return Mat(); }
    static Mat identity() { Mat m; for (int i = 0; i < (R < C ? R : C); ++i) m(i, i) = Real(1); return m; }
    static Mat constant(Real v) { Mat m; for (int k = 0; k < R * C; ++k) m.d[k] = v; return m; }

    Real& operator()(int i, int j) { return d[i * C + j]; }
    Real operator()(int i, int j) const { return d[i * C + j]; }
    Real& operator[](int k) { return d[k]; }          // linear index (vectors)
    Real operator[](int k) const { return d[k]; }

    Mat<C, R> t() const { Mat<C, R> m; for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) m(j, i) = (*this)(i, j); return m; }

    Mat& operator+=(const Mat& b) { for (int k = 0; k < R * C; ++k) d[k] += b.d[k]; return *this; }
    Mat& operator-=(const Mat& b) { for (int k = 0; k < R * C; ++k) d[k] -= b.d[k]; return *this; }
    Mat& operator*=(Real s) { for (int k = 0; k < R * C; ++k) d[k] *= s; return *this; }
    Mat& operator/=(Real s) { for (int k = 0; k < R * C; ++k) d[k] /= s; return *this; }

    Real trace() const { Real s = 0; for (int i = 0; i < (R < C ? R : C); ++i) s += (*this)(i, i); return s; }
    Real norm() const { Real s = 0; for (int k = 0; k < R * C; ++k) s += d[k] * d[k]; return std::sqrt(s); }   // Frobenius / Euclidean
    Real norm_inf() const { Real s = 0; for (int k = 0; k < R * C; ++k) s = std::max(s, std::fabs(d[k])); return s; }
    bool is_finite() const { for (int k = 0; k < R * C; ++k) if (!std::isfinite(d[k])) return false; return true; }

    // symmetrise in place: P <- (P + P^T)/2  (keeps covariance symmetric under round-off)
    void symmetrize() { static_assert(R == C, "square"); for (int i = 0; i < R; ++i) for (int j = i + 1; j < C; ++j) { Real m = Real(0.5) * ((*this)(i, j) + (*this)(j, i)); (*this)(i, j) = m; (*this)(j, i) = m; } }

    // sub-block extraction / insertion
    template <int BR, int BC> Mat<BR, BC> block(int r0, int c0) const { Mat<BR, BC> b; for (int i = 0; i < BR; ++i) for (int j = 0; j < BC; ++j) b(i, j) = (*this)(r0 + i, c0 + j); return b; }
    template <int BR, int BC> void set_block(int r0, int c0, const Mat<BR, BC>& b) { for (int i = 0; i < BR; ++i) for (int j = 0; j < BC; ++j) (*this)(r0 + i, c0 + j) = b(i, j); }
    Mat<R, 1> col(int j) const { Mat<R, 1> c; for (int i = 0; i < R; ++i) c[i] = (*this)(i, j); return c; }
    Mat<1, C> row(int i) const { Mat<1, C> r; for (int j = 0; j < C; ++j) r[j] = (*this)(i, j); return r; }
    void set_col(int j, const Mat<R, 1>& c) { for (int i = 0; i < R; ++i) (*this)(i, j) = c[i]; }
    void set_row(int i, const Mat<1, C>& r) { for (int j = 0; j < C; ++j) (*this)(i, j) = r[j]; }
};

template <int N> using Vec = Mat<N, 1>;
template <int N> using RowVec = Mat<1, N>;
using Mat1 = Mat<1, 1>;

// ---- arithmetic -------------------------------------------------------------
template <int R, int C> inline Mat<R, C> operator+(Mat<R, C> a, const Mat<R, C>& b) { a += b; return a; }
template <int R, int C> inline Mat<R, C> operator-(Mat<R, C> a, const Mat<R, C>& b) { a -= b; return a; }
template <int R, int C> inline Mat<R, C> operator-(Mat<R, C> a) { a *= Real(-1); return a; }
template <int R, int C> inline Mat<R, C> operator*(Mat<R, C> a, Real s) { a *= s; return a; }
template <int R, int C> inline Mat<R, C> operator*(Real s, Mat<R, C> a) { a *= s; return a; }
template <int R, int C> inline Mat<R, C> operator/(Mat<R, C> a, Real s) { a /= s; return a; }
template <int R, int K, int C>
inline Mat<R, C> operator*(const Mat<R, K>& a, const Mat<K, C>& b) {
    Mat<R, C> m;
    for (int i = 0; i < R; ++i)
        for (int k = 0; k < K; ++k) {
            const Real aik = a(i, k);
            for (int j = 0; j < C; ++j) m(i, j) += aik * b(k, j);
        }
    return m;
}
template <int N> inline Real dot(const Vec<N>& a, const Vec<N>& b) { Real s = 0; for (int k = 0; k < N; ++k) s += a[k] * b[k]; return s; }
template <int R, int C> inline Mat<R, C> outer(const Vec<R>& a, const Vec<C>& b) { Mat<R, C> m; for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) m(i, j) = a[i] * b[j]; return m; }
template <int R, int C> inline Mat<R, C> hadamard(const Mat<R, C>& a, const Mat<R, C>& b) { Mat<R, C> m; for (int k = 0; k < R * C; ++k) m.d[k] = a.d[k] * b.d[k]; return m; }
template <int N> inline Mat<N, N> diag(const Vec<N>& v) { Mat<N, N> m; for (int i = 0; i < N; ++i) m(i, i) = v[i]; return m; }
template <int N> inline Vec<N> diag_of(const Mat<N, N>& m) { Vec<N> v; for (int i = 0; i < N; ++i) v[i] = m(i, i); return v; }
// scalar convenience for 1x1 results
inline Real scalar(const Mat1& m) { return m.d[0]; }
template <int N> inline Real quad_form(const Mat<N, N>& A, const Vec<N>& x) { return dot(x, A * x); }   // x^T A x

// ---- LU with partial pivoting (GVL Alg. 3.4.1) -------------------------------
// Returns false if the matrix is numerically singular.
template <int N>
inline bool lu_decompose(Mat<N, N>& A, int (&piv)[N], int& sign) {
    sign = 1;
    for (int i = 0; i < N; ++i) piv[i] = i;
    for (int k = 0; k < N; ++k) {
        int p = k; Real amax = std::fabs(A(k, k));
        for (int i = k + 1; i < N; ++i) { Real v = std::fabs(A(i, k)); if (v > amax) { amax = v; p = i; } }
        if (amax < Real(1e-300)) return false;
        if (p != k) { for (int j = 0; j < N; ++j) std::swap(A(k, j), A(p, j)); std::swap(piv[k], piv[p]); sign = -sign; }
        for (int i = k + 1; i < N; ++i) {
            A(i, k) /= A(k, k);
            const Real lik = A(i, k);
            for (int j = k + 1; j < N; ++j) A(i, j) -= lik * A(k, j);
        }
    }
    return true;
}
template <int N, int M>
inline void lu_solve(const Mat<N, N>& LU, const int (&piv)[N], const Mat<N, M>& B, Mat<N, M>& X) {
    for (int c = 0; c < M; ++c) {
        Real y[N];
        for (int i = 0; i < N; ++i) { Real s = B(piv[i], c); for (int j = 0; j < i; ++j) s -= LU(i, j) * y[j]; y[i] = s; }
        for (int i = N - 1; i >= 0; --i) { Real s = y[i]; for (int j = i + 1; j < N; ++j) s -= LU(i, j) * X(j, c); X(i, c) = s / LU(i, i); }
    }
}
// General inverse. Returns a NaN-filled matrix if singular (caller may check is_finite()).
template <int N>
inline Mat<N, N> inverse(const Mat<N, N>& A) {
    Mat<N, N> LU = A, X; int piv[N]; int sign;
    if (!lu_decompose(LU, piv, sign)) { return Mat<N, N>::constant(kNaN); }
    lu_solve(LU, piv, Mat<N, N>::identity(), X);
    return X;
}
template <> inline Mat<1, 1> inverse(const Mat<1, 1>& A) {
    if (!(std::fabs(A.d[0]) > Real(0))) return Mat<1, 1>::constant(kNaN);
    Mat<1, 1> m; m.d[0] = Real(1) / A.d[0]; return m;
}
template <> inline Mat<2, 2> inverse(const Mat<2, 2>& A) {
    const Real det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    if (!(std::fabs(det) > Real(0))) return Mat<2, 2>::constant(kNaN);
    Mat<2, 2> m;
    m(0, 0) = A(1, 1) / det; m(0, 1) = -A(0, 1) / det; m(1, 0) = -A(1, 0) / det; m(1, 1) = A(0, 0) / det; return m;
}
template <int N> inline Real det(const Mat<N, N>& A) {
    Mat<N, N> LU = A; int piv[N]; int sign;
    if (!lu_decompose(LU, piv, sign)) return Real(0);
    Real d = Real(sign); for (int i = 0; i < N; ++i) d *= LU(i, i); return d;
}
// Solve A X = B for general square A
template <int N, int M>
inline Mat<N, M> solve(const Mat<N, N>& A, const Mat<N, M>& B) {
    Mat<N, N> LU = A; Mat<N, M> X; int piv[N]; int sign;
    if (!lu_decompose(LU, piv, sign)) return Mat<N, M>::constant(kNaN);
    lu_solve(LU, piv, B, X); return X;
}

// ---- Cholesky (GVL Alg. 4.2.1): A = L L^T, L lower triangular ----------------
// Returns false if A is not (numerically) positive definite.
template <int N>
inline bool cholesky(const Mat<N, N>& A, Mat<N, N>& L) {
    L = Mat<N, N>();
    for (int j = 0; j < N; ++j) {
        Real s = A(j, j);
        for (int k = 0; k < j; ++k) s -= L(j, k) * L(j, k);
        if (!(s > Real(0))) return false;
        const Real ljj = std::sqrt(s); L(j, j) = ljj;
        for (int i = j + 1; i < N; ++i) {
            Real t = A(i, j);
            for (int k = 0; k < j; ++k) t -= L(i, k) * L(j, k);
            L(i, j) = t / ljj;
        }
    }
    return true;
}
// Cholesky with diagonal "jitter" fallback so sigma-point filters never abort.
template <int N>
inline Mat<N, N> cholesky_safe(Mat<N, N> A) {
    Mat<N, N> L; A.symmetrize();
    Real jitter = Real(0);
    for (int attempt = 0; attempt < 8; ++attempt) {
        Mat<N, N> Aj = A; for (int i = 0; i < N; ++i) Aj(i, i) += jitter;
        if (cholesky(Aj, L)) return L;
        jitter = (jitter == Real(0)) ? Real(1e-12) * (Real(1) + A.norm_inf()) : jitter * Real(100);
    }
    return Mat<N, N>::identity() * std::sqrt(Real(1e-6));   // last resort
}
// Solve A X = B with A SPD via Cholesky (forward/back substitution).
template <int N, int M>
inline Mat<N, M> solve_spd(const Mat<N, N>& A, const Mat<N, M>& B) {
    Mat<N, N> L;
    if (!cholesky(A, L)) return solve(A, B);
    Mat<N, M> X;
    for (int c = 0; c < M; ++c) {
        Real y[N];
        for (int i = 0; i < N; ++i) { Real s = B(i, c); for (int j = 0; j < i; ++j) s -= L(i, j) * y[j]; y[i] = s / L(i, i); }
        for (int i = N - 1; i >= 0; --i) { Real s = y[i]; for (int j = i + 1; j < N; ++j) s -= L(j, i) * X(j, c); X(i, c) = s / L(i, i); }
    }
    return X;
}
// Inverse of a lower-triangular matrix (forward substitution)
template <int N>
inline Mat<N, N> inverse_lower(const Mat<N, N>& L) {
    Mat<N, N> X;
    for (int c = 0; c < N; ++c)
        for (int i = 0; i < N; ++i) {
            Real s = (i == c) ? Real(1) : Real(0);
            for (int j = 0; j < i; ++j) s -= L(i, j) * X(j, c);
            X(i, c) = s / L(i, i);
        }
    return X;
}
// Solve L X = B (L lower triangular) and L^T X = B
template <int N, int M>
inline Mat<N, M> solve_lower(const Mat<N, N>& L, const Mat<N, M>& B) {
    Mat<N, M> X;
    for (int c = 0; c < M; ++c) for (int i = 0; i < N; ++i) { Real s = B(i, c); for (int j = 0; j < i; ++j) s -= L(i, j) * X(j, c); X(i, c) = s / L(i, i); }
    return X;
}
template <int N, int M>
inline Mat<N, M> solve_upper(const Mat<N, N>& U, const Mat<N, M>& B) {
    Mat<N, M> X;
    for (int c = 0; c < M; ++c) for (int i = N - 1; i >= 0; --i) { Real s = B(i, c); for (int j = i + 1; j < N; ++j) s -= U(i, j) * X(j, c); X(i, c) = s / U(i, i); }
    return X;
}

// ---- Rank-1 Cholesky update / downdate (GVL §6.5.4; Gill, Golub, Murray & Saunders 1974)
//  On return L satisfies  L L^T = L_old L_old^T + sign * x x^T   (sign = +1 or -1).
//  Returns false if the downdate would destroy positive definiteness.
template <int N>
inline bool chol_update(Mat<N, N>& L, Vec<N> x, Real sign) {
    for (int k = 0; k < N; ++k) {
        const Real lkk = L(k, k);
        const Real r2 = lkk * lkk + sign * x[k] * x[k];
        if (!(r2 > Real(0))) return false;
        const Real r = std::sqrt(r2);
        const Real c = r / lkk, s = x[k] / lkk;
        L(k, k) = r;
        for (int i = k + 1; i < N; ++i) {
            L(i, k) = (L(i, k) + sign * s * x[i]) / c;
            x[i] = c * x[i] - s * L(i, k);
        }
    }
    return true;
}

// ---- Householder QR (GVL Alg. 5.2.1) -----------------------------------------
//  For A (M x N, M >= N) returns the upper-triangular R (N x N) with A = Q R,
//  i.e. R^T R = A^T A. Used by the "array" square-root filters (Kailath, Sayed &
//  Hassibi 2000, Ch. 12): the post-array is the transpose of R of the pre-array.
template <int M, int N>
inline Mat<N, N> qr_r(Mat<M, N> A) {
    static_assert(M >= N, "qr_r requires a tall or square matrix");
    for (int k = 0; k < N; ++k) {
        Real alpha = 0; for (int i = k; i < M; ++i) alpha += A(i, k) * A(i, k);
        alpha = std::sqrt(alpha);
        if (alpha < Real(1e-300)) continue;
        if (A(k, k) > 0) alpha = -alpha;
        Real v[M]; Real vnorm2 = 0;
        for (int i = 0; i < M; ++i) { v[i] = (i < k) ? Real(0) : A(i, k); }
        v[k] -= alpha;
        for (int i = k; i < M; ++i) vnorm2 += v[i] * v[i];
        if (vnorm2 < Real(1e-300)) continue;
        for (int j = k; j < N; ++j) {
            Real s = 0; for (int i = k; i < M; ++i) s += v[i] * A(i, j);
            s = Real(2) * s / vnorm2;
            for (int i = k; i < M; ++i) A(i, j) -= s * v[i];
        }
    }
    Mat<N, N> R;
    for (int i = 0; i < N; ++i) for (int j = i; j < N; ++j) R(i, j) = A(i, j);
    // make the diagonal non-negative (sign convention) so factors are unique
    for (int i = 0; i < N; ++i) if (R(i, i) < 0) for (int j = i; j < N; ++j) R(i, j) = -R(i, j);
    return R;
}

// ---- Matrix square root of an SPD matrix via Cholesky: S S^T = A --------------
template <int N> inline Mat<N, N> sqrt_spd(const Mat<N, N>& A) { return cholesky_safe(A); }

// ---- Matrix power / polynomial helpers ----------------------------------------
template <int N> inline Mat<N, N> mat_pow(const Mat<N, N>& A, int p) { Mat<N, N> R = Mat<N, N>::identity(); for (int k = 0; k < p; ++k) R = R * A; return R; }

// ---- Ackermann's formula for observer gain (Ackermann 1972; Ogata 2010 §10-6)
//  Given (A, C) observable (C is 1 x N) and desired characteristic polynomial
//  phi(z) = z^N + a_{N-1} z^{N-1} + ... + a_0  (coefficients a[0..N-1] = a_0..a_{N-1}),
//  the observer gain L (N x 1) such that eig(A - L C) = roots of phi is
//      L = phi(A) * O^{-1} * e_N,   O = [C; CA; ...; CA^{N-1}].
//  Returns false if the observability matrix is singular.
template <int N>
inline bool ackermann_observer(const Mat<N, N>& A, const RowVec<N>& C, const Real (&a)[N], Vec<N>& L) {
    Mat<N, N> O; RowVec<N> r = C;
    for (int i = 0; i < N; ++i) { O.set_row(i, r); r = r * A; }
    Mat<N, N> phi = mat_pow(A, N);
    for (int k = 0; k < N; ++k) phi += mat_pow(A, k) * a[k];
    Vec<N> eN; eN[N - 1] = Real(1);
    Vec<N> w = solve(O, eN);
    if (!w.is_finite()) return false;
    L = phi * w;
    return L.is_finite();
}
// Numerically conditioned pole placement (observer gain for the PREDICTOR form
// eig(A - L C) = poles).  Ackermann's formula is ill-conditioned for finely
// sampled plants (all poles near z = 1, det(O) ~ 1e-18 for the cell model at
// 10 Hz).  Shifting and scaling the pair, A' = (A - sigma I)/gamma,
// p'_i = (p_i - sigma)/gamma, keeps phi(A') and O well scaled; the gain of the
// original problem is L = gamma L'.  (Same idea as balancing in pole-placement
// codes; see Kautsky, Nichols & Van Dooren 1985 for the general discussion.)
template <int N>
inline bool ackermann_observer_poles(const Mat<N, N>& A, const RowVec<N>& C, const Real (&poles)[N], Vec<N>& L) {
    Real sigma = Real(0);
    for (int i = 0; i < N; ++i) sigma += poles[i];
    sigma /= Real(N);
    Real gamma = std::fabs(Real(1) - sigma);
    for (int i = 0; i < N; ++i) gamma = std::max(gamma, std::fabs(poles[i] - sigma));
    if (!(gamma > Real(0))) gamma = Real(1);
    Mat<N, N> As = A;
    for (int i = 0; i < N; ++i) As(i, i) -= sigma;
    As /= gamma;
    Real ps[N], a[N];
    for (int i = 0; i < N; ++i) ps[i] = (poles[i] - sigma) / gamma;
    Real c[N + 1]; for (int k = 0; k <= N; ++k) c[k] = Real(0); c[0] = Real(1);
    int deg = 0;
    for (int i = 0; i < N; ++i) { for (int k = deg + 1; k >= 1; --k) c[k] = c[k] - ps[i] * c[k - 1]; ++deg; }
    for (int k = 0; k < N; ++k) a[k] = c[N - k];
    Vec<N> Lp;
    if (!ackermann_observer<N>(As, C, a, Lp)) return false;
    L = Lp * gamma;
    return L.is_finite();
}

// Characteristic polynomial coefficients from N real (discrete-time) poles p_i:
//  prod (z - p_i) = z^N + a_{N-1} z^{N-1} + ... + a_0 ; fills a[0..N-1].
template <int N>
inline void poly_from_roots(const Real (&p)[N], Real (&a)[N]) {
    Real c[N + 1]; for (int k = 0; k <= N; ++k) c[k] = Real(0); c[0] = Real(1);   // c[k] = coefficient of z^(deg-k)
    int deg = 0;
    for (int i = 0; i < N; ++i) {
        for (int k = deg + 1; k >= 1; --k) c[k] = c[k] - p[i] * c[k - 1];
        ++deg;
    }
    for (int k = 0; k < N; ++k) a[k] = c[N - k];
}

// ---- Discrete algebraic Riccati equation (steady-state Kalman/observer gain) ----
//  Filter DARE:  P = A P A^T - A P C^T (C P C^T + R)^{-1} C P A^T + Q.
//  Solved with the structure-preserving doubling algorithm (Anderson 1978,
//  Int. J. Control 28(2) 295-306; Chu, Fan, Lin & Wang 2004): starting from
//  A_0 = A^T, G_0 = C^T R^{-1} C, H_0 = Q,
//     A_{k+1} = A_k (I + G_k H_k)^{-1} A_k
//     G_{k+1} = G_k + A_k (I + G_k H_k)^{-1} G_k A_k^T
//     H_{k+1} = H_k + A_k^T H_k (I + G_k H_k)^{-1} A_k
//  H_k converges quadratically to the stabilising solution P (each step doubles
//  the horizon, so 40-60 steps suffice even for nearly marginally stable modes).
//  A few plain fixed-point refinements polish the result.  Returns the prior
//  covariance P and, optionally, K = P C^T (C P C^T + R)^{-1}
//  (current-estimator form x^+ = x^- + K (y - C x^-)).
template <int N, int M>
inline Mat<N, N> dare_doubling(const Mat<N, N>& A, const Mat<M, N>& C, const Mat<N, N>& Q, const Mat<M, M>& R,
                               Mat<N, M>* K_out = nullptr, int max_iter = 80, Real tol = Real(1e-13)) {
    Mat<N, N> Ak = A.t();
    Mat<N, N> Gk = C.t() * inverse(R) * C;
    Mat<N, N> Hk = Q;
    const Mat<N, N> I = Mat<N, N>::identity();
    for (int it = 0; it < max_iter; ++it) {
        const Mat<N, N> Minv = inverse(I + Gk * Hk);
        if (!Minv.is_finite()) break;
        const Mat<N, N> An = Ak * Minv * Ak;
        Mat<N, N> Gn = Gk + Ak * Minv * Gk * Ak.t();
        Mat<N, N> Hn = Hk + Ak.t() * Hk * Minv * Ak;
        Gn.symmetrize(); Hn.symmetrize();
        const Real delta = (Hn - Hk).norm_inf();
        Ak = An; Gk = Gn; Hk = Hn;
        if (!(delta > tol * (Real(1) + Hk.norm_inf()))) break;
    }
    // polish with the plain recursion
    Mat<N, N> P = Hk;
    for (int it = 0; it < 5; ++it) {
        const Mat<M, M> S = C * P * C.t() + R;
        const Mat<N, M> K = P * C.t() * inverse(S);
        Mat<N, N> Pn = A * (P - K * C * P) * A.t() + Q;
        if (!Pn.is_finite()) break;
        Pn.symmetrize(); P = Pn;
    }
    if (K_out) { const Mat<M, M> S = C * P * C.t() + R; *K_out = P * C.t() * inverse(S); }
    return P;
}
//  Plain fixed-point iteration P <- A (P - K C P) A^T + Q from P_0 = Q + I
//  (Anderson & Moore 1979, §4.4).  Convergence is only linear at rate
//  rho((I-KC)A)^2 — for a 10 Hz battery model ~0.9994 per step — so with a finite
//  max_iter the result is the finite-horizon (time-varying) Kalman gain after
//  max_iter samples rather than the steady state; this is sometimes the intended
//  design (a "soft" start-up gain).  Use dare_doubling for the exact steady state.
template <int N, int M>
inline Mat<N, N> dare_iterate(const Mat<N, N>& A, const Mat<M, N>& C, const Mat<N, N>& Q, const Mat<M, M>& R,
                              Mat<N, M>* K_out = nullptr, int max_iter = 100000, Real tol = Real(1e-12)) {
    Mat<N, N> P = Q + Mat<N, N>::identity();
    for (int it = 0; it < max_iter; ++it) {
        const Mat<M, M> S = C * P * C.t() + R;
        const Mat<N, M> K = P * C.t() * inverse(S);
        const Mat<N, N> Pn = A * (P - K * C * P) * A.t() + Q;
        if (!Pn.is_finite()) break;
        const Real delta = (Pn - P).norm_inf();
        P = Pn; P.symmetrize();
        if (delta < tol * (Real(1) + P.norm_inf())) break;
    }
    if (K_out) { const Mat<M, M> S = C * P * C.t() + R; *K_out = P * C.t() * inverse(S); }
    return P;
}

// ---- Exponential of a scalar with overflow guard (used in Arrhenius laws) -----
inline Real safe_exp(Real x) { return std::exp(clampr(x, Real(-700), Real(700))); }

}  // namespace estkit
