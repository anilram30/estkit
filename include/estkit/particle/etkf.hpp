// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/etkf.hpp — ensemble transform Kalman filter (deterministic
//  square-root ensemble filter), in the ensemble-space form of the LETKF.
//
//  References
//    * Bishop, Craig H., Etherton, Brian J. & Majumdar, Sharanya J. (2001).
//      Adaptive sampling with the ensemble transform Kalman filter. Part I:
//      Theoretical aspects. Monthly Weather Review 129(3), 420-436.
//    * Hunt, Brian R., Kostelich, Eric J. & Szunyogh, Istvan (2007). Efficient
//      data assimilation for spatiotemporal chaos: A local ensemble transform
//      Kalman filter. Physica D 230(1-2), 112-126.   (the form implemented here)
//    * Tippett, Michael K., Anderson, Jeffrey L., Bishop, Craig H., Hamill,
//      Thomas M. & Whitaker, Jeffrey S. (2003). Ensemble square root filters.
//      Monthly Weather Review 131(7), 1485-1490.  (symmetric square root, why)
//    * Golub, Gene H. & Van Loan, Charles F. (2013). Matrix Computations, 4th ed.
//      Johns Hopkins University Press, §8.5.       (cyclic Jacobi eigensolver)
//
//  Motivation.  The stochastic EnKF (enkf.hpp) needs random observation
//  perturbations, which inject sampling noise of order N^{-1/2} into the analysis
//  covariance.  Square-root filters instead compute a DETERMINISTIC linear
//  transform of the forecast perturbations that reproduces the Kalman analysis
//  covariance exactly, with no random numbers in the analysis step.
//
//  Derivation (Hunt et al. 2007 §2.3).  Write the forecast ensemble as
//      x^{f,i} = xbar^f + X^f_{:,i},   X^f in R^{n_x x N},  sum_i X^f_{:,i} = 0,
//  and restrict the analysis to the affine subspace spanned by the ensemble,
//      x = xbar^f + X^f w,   w in R^N.                                          (1)
//  With P^f = X^f (X^f)^T/(N-1), the Gaussian cost function
//      J(x) = (x-xbar^f)^T (P^f)^+ (x-xbar^f) + (y-h(x))^T R^{-1} (y-h(x))
//  becomes, after linearising h on the ensemble (Y^f_{:,i} = h(x^{f,i}) - ybar),
//      Jtilde(w) = (N-1) w^T w + (y - ybar - Y^f w)^T R^{-1} (y - ybar - Y^f w),  (2)
//  a quadratic form whose minimiser and Hessian are
//      Ptilde^a = [ (N-1) I / rho + (Y^f)^T R^{-1} Y^f ]^{-1}   (N x N)          (3)
//      wbar^a   = Ptilde^a (Y^f)^T R^{-1} (y - ybar).                            (4)
//  Mapping back through (1) gives the analysis mean xbar^a = xbar^f + X^f wbar^a,
//  and the analysis ensemble is obtained by adding the columns of
//      W^a = [ (N-1) Ptilde^a ]^{1/2}                                            (5)
//  (SYMMETRIC square root), i.e.
//      x^{a,i} = xbar^f + X^f ( wbar^a + W^a_{:,i} ).                            (6)
//  Then X^a (X^a)^T/(N-1) = X^f Ptilde^a (X^f)^T = (I - KH) P^f exactly, so the
//  analysis covariance is the Kalman one without any Monte-Carlo error.
//  The symmetric square root is the unique choice that (a) keeps the analysis
//  ensemble mean equal to xbar^a — because (Y^f) 1 = 0 makes 1 an eigenvector of
//  (3) with eigenvalue N-1, hence W^a 1 = 1 and X^f W^a 1 = X^f 1 = 0 — and
//  (b) minimises the distance to the forecast ensemble (Tippett et al. 2003).
//  rho >= 1 in (3) is multiplicative covariance inflation.
//
//  The only nontrivial numerical operation is the symmetric square root of the
//  N x N SPD matrix (3), computed here with a cyclic Jacobi eigen-decomposition
//  (GVL §8.5): A = V diag(lambda) V^T, W^a = sqrt(N-1) V diag(lambda^{-1/2}) V^T.
//  Jacobi is used rather than tridiagonal QR because it is short, heap-free,
//  branch-simple and highly accurate for small SPD matrices.
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../core/rng.hpp"
#include "pf_common.hpp"

namespace estkit {
namespace pf {

// ---- cyclic Jacobi eigen-decomposition of a symmetric matrix (GVL Alg. 8.5.2)
//  On exit  A_in = V diag(lam) V^T  with V orthogonal.  No heap, no exceptions.
//  Rotation: t = sgn(theta)/(|theta| + sqrt(theta^2+1)), theta = (a_qq-a_pp)/(2a_pq),
//            c = 1/sqrt(1+t^2), s = t c   (the smaller root keeps |rotation| <= pi/4).
template <int N>
inline void jacobi_eigh(Mat<N, N> A, Mat<N, N>& V, Real (&lam)[N], int max_sweeps = 15) {
    V = Mat<N, N>::identity();
    const Real tol2 = sq(Real(100) * kEps);
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        Real off = Real(0), nrm = Real(0);
        for (int p = 0; p < N; ++p) {
            nrm += A(p, p) * A(p, p);
            for (int q = p + 1; q < N; ++q) off += A(p, q) * A(p, q);
        }
        if (off <= tol2 * nrm) break;
        // first sweeps: skip the elements that contribute least (threshold strategy)
        const Real thresh = (sweep < 3) ? (Real(0.2) * off / Real(N * N)) : Real(0);
        for (int p = 0; p < N - 1; ++p) {
            for (int q = p + 1; q < N; ++q) {
                const Real apq = A(p, q);
                if (apq * apq <= thresh || apq == Real(0)) continue;
                const Real theta = (A(q, q) - A(p, p)) / (Real(2) * apq);
                const Real at = std::fabs(theta);
                const Real t = (theta >= Real(0) ? Real(1) : Real(-1)) / (at + std::sqrt(theta * theta + Real(1)));
                const Real c = Real(1) / std::sqrt(t * t + Real(1));
                const Real s = t * c;
                A(p, p) -= t * apq;
                A(q, q) += t * apq;
                A(p, q) = A(q, p) = Real(0);
                for (int r = 0; r < N; ++r) {
                    if (r == p || r == q) continue;
                    const Real arp = A(r, p), arq = A(r, q);
                    A(r, p) = A(p, r) = c * arp - s * arq;
                    A(r, q) = A(q, r) = s * arp + c * arq;
                }
                for (int r = 0; r < N; ++r) {
                    const Real vrp = V(r, p), vrq = V(r, q);
                    V(r, p) = c * vrp - s * vrq;
                    V(r, q) = s * vrp + c * vrq;
                }
            }
        }
    }
    for (int i = 0; i < N; ++i) lam[i] = A(i, i);
}

}  // namespace pf

template <class Model, int N = 24>
class Etkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NENS = N;
    static_assert(N >= 3, "the ensemble transform needs at least 3 members");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real inflation = Real(1.001);    // rho in eq. (3), multiplicative prior inflation per step
        Real q_scale = Real(1);
        Real r_scale = Real(1);
        bool apply_constraints = true;
    } opts;

    explicit Etkf(const Model& m) : m_(m) {}
    void seed(uint64_t s) { rng_.reseed(s); }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        const Mat<NX, NX> L = cholesky_safe(P0);
        X sum;
        for (int i = 0; i < N; ++i) { e_[i] = L * rng_.normal_vec<NX>(); sum += e_[i]; }
        sum = sum / Real(N);
        for (int i = 0; i < N; ++i) {          // centre the initial ensemble exactly
            X xi = x0 + (e_[i] - sum);
            if (opts.apply_constraints) xi = constrain(m_, xi);
            e_[i] = xi;
        }
        refresh_moments();
    }

    // Forecast: the members are integrated through f and given an independent
    // draw of the process noise (stochastic forecast; the analysis itself stays
    // deterministic).
    void predict(const U& u) {
        const Mat<NX, NX> Lq = cholesky_safe(m_.Q(x_, u) * opts.q_scale);
        for (int i = 0; i < N; ++i) {
            X xn = m_.f(e_[i], u) + Lq * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xn = constrain(m_, xn);
            e_[i] = xn;
        }
        refresh_moments();
    }

    Y predict_measurement(const U& u) const {
        Y ym;
        for (int i = 0; i < N; ++i) ym += m_.h(e_[i], u);
        return ym / Real(N);
    }

    // ---- deterministic ensemble transform, eqs. (3)-(6) ----------------------
    void update(const Y& y, const U& u) {
        const X xb = x_;
        Y yb;
        Mat<NY, N> Yf;
        for (int i = 0; i < N; ++i) { ys_[i] = m_.h(e_[i], u); yb += ys_[i]; }
        yb = yb / Real(N);
        Mat<NX, N> Xf;
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < NX; ++k) Xf(k, i) = e_[i][k] - xb[k];
            for (int k = 0; k < NY; ++k) Yf(k, i) = ys_[i][k] - yb[k];
        }
        const Mat<NY, NY> Rm = m_.R(xb, u) * opts.r_scale;
        const Mat<NY, NY> Rinv = inverse(Rm);
        // C = (Y^f)^T R^{-1}   (N x NY);   A = (N-1)/rho I + C Y^f   (N x N)
        Mat<N, NY> C;
        for (int i = 0; i < N; ++i)
            for (int a = 0; a < NY; ++a) {
                Real s = Real(0);
                for (int b = 0; b < NY; ++b) s += Yf(b, i) * Rinv(b, a);
                C(i, a) = s;
            }
        Mat<N, N> A = C * Yf;
        A.symmetrize();
        const Real rho = (opts.inflation > Real(0)) ? opts.inflation : Real(1);
        for (int i = 0; i < N; ++i) A(i, i) += Real(N - 1) / rho;
        Mat<N, N> V; Real lam[N];
        pf::jacobi_eigh(A, V, lam);
        for (int i = 0; i < N; ++i) if (!(lam[i] > Real(0))) lam[i] = Real(N - 1) * kEps;
        // g = C (y - ybar);  d = V diag(1/lam) V^T g  = wbar^a
        const Vec<N> g = C * (y - yb);
        Vec<N> Vg;
        for (int j = 0; j < N; ++j) { Real s = Real(0); for (int i = 0; i < N; ++i) s += V(i, j) * g[i]; Vg[j] = s / lam[j]; }
        // XV = X^f V ;  Z = XV diag(sqrt((N-1)/lam)) ;  X^f W^a = Z V^T
        const Mat<NX, N> XV = Xf * V;
        Mat<NX, N> Z;
        for (int j = 0; j < N; ++j) {
            const Real sc = std::sqrt(Real(N - 1) / lam[j]);
            for (int k = 0; k < NX; ++k) Z(k, j) = XV(k, j) * sc;
        }
        const Mat<NX, N> XW = Z * V.t();
        const Vec<NX> dmean = XV * Vg;                       // X^f wbar^a
        for (int i = 0; i < N; ++i) {
            X xa = xb + dmean;
            for (int k = 0; k < NX; ++k) xa[k] += XW(k, i);
            if (opts.apply_constraints) xa = constrain(m_, xa);
            e_[i] = xa;
        }
        refresh_moments();
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    const X* ensemble() const { return e_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void refresh_moments() {
        X s;
        for (int i = 0; i < N; ++i) s += e_[i];
        x_ = s / Real(N);
        P_ = Mat<NX, NX>();
        const Real inv = Real(1) / Real(N - 1);
        for (int i = 0; i < N; ++i) { const X d = e_[i] - x_; P_ += outer(d, d) * inv; }
        P_.symmetrize();
    }

    Model m_;
    Rng rng_{1};
    X e_[N];
    Y ys_[N];
    X x_;
    Mat<NX, NX> P_;
};

}  // namespace estkit
