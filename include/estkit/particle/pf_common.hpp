// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/pf_common.hpp — building blocks shared by every sequential
//  Monte-Carlo / ensemble filter in estkit/particle/.
//
//  References
//    * Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993). Novel approach to
//      nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F
//      140(2), 107-113.                      (bootstrap filter, roughening §4.2)
//    * Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian
//      nonlinear state space models. J. Comput. Graph. Stat. 5(1), 1-25.
//                                                   (systematic resampling)
//    * Arulampalam, M. S., Maskell, S., Gordon, N. & Clapp, T. (2002). A tutorial
//      on particle filters for online nonlinear/non-Gaussian Bayesian tracking.
//      IEEE Trans. Signal Processing 50(2), 174-188.   (ESS, SIR pseudo-code)
//    * Doucet, A., Godsill, S. & Andrieu, C. (2000). On sequential Monte Carlo
//      sampling methods for Bayesian filtering. Statistics and Computing 10(3),
//      197-208.                                    (weight recursion, ESS rule)
//    * Gustafsson, F. (2010). Particle filter theory and practice with
//      positioning applications. IEEE AESM 25(7), 53-82. (implementation advice)
//
//  Contents
//    ess()                    effective sample size  N_eff = 1 / sum_i (w^i)^2
//    normalize_log_weights()  float-safe  w^i = exp(l^i - max l) / sum(...)
//    systematic_indices()     Kitagawa's stratified/systematic resampling -> parents
//    systematic_counts()      the same draw expressed as offspring counts n^i
//    inplace_redistribute()   O(N) in-place copy of a resampled cloud (no scratch)
//    weighted_mean/cov()      first two moments of the weighted cloud
//    GaussLogLik<NY>          log N(e;0,R) without a transcendental per particle
//    roughening_sigma()/roughen()  Gordon 1993 §4.2 jitter with a collapse floor
// =============================================================================
#pragma once
#include <cmath>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../core/rng.hpp"

namespace estkit {
namespace pf {

// ---- effective sample size (Doucet et al. 2000, eq. 21; Arulampalam 2002 eq. 51)
//   N_eff = 1 / sum_i (w^i)^2   in [1, N];  resample when N_eff < ratio * N.
template <int N>
inline Real ess(const Real (&w)[N]) {
    Real s = Real(0);
    for (int i = 0; i < N; ++i) s += w[i] * w[i];
    return (s > Real(0)) ? Real(1) / s : Real(0);
}

// ---- normalise log-weights: w^i = exp(l^i - l_max) / sum_j exp(l^j - l_max) ----
//  The shift by l_max is what keeps the recursion usable in single precision
//  (exp(-800) underflows to 0 in double, exp(-90) already in float).
//  Returns log sum_j exp(l^j)  (the incremental log-likelihood up to a constant).
template <int N>
inline Real normalize_log_weights(const Real (&lw)[N], Real (&w)[N]) {
    Real mx = lw[0];
    for (int i = 1; i < N; ++i) if (lw[i] > mx) mx = lw[i];
    if (!std::isfinite(mx)) {                       // total failure: reset to uniform
        for (int i = 0; i < N; ++i) w[i] = Real(1) / Real(N);
        return Real(0);
    }
    Real s = Real(0);
    for (int i = 0; i < N; ++i) { const Real e = std::exp(lw[i] - mx); w[i] = e; s += e; }
    if (!(s > Real(0)) || !std::isfinite(s)) {
        for (int i = 0; i < N; ++i) w[i] = Real(1) / Real(N);
        return Real(0);
    }
    const Real inv = Real(1) / s;
    for (int i = 0; i < N; ++i) w[i] *= inv;
    return mx + std::log(s);
}

// ---- systematic resampling (Kitagawa 1996 §3; Arulampalam 2002 Alg. 2) --------
//  Draw ONE uniform u0 ~ U[0,1) and use the deterministic comb
//      u_j = (j + u0)/N ,  j = 0..N-1 ,
//  selecting parent i(j) = min{ i : sum_{m<=i} w^m >= u_j }.  One pass, O(N),
//  minimum Monte-Carlo variance among the unbiased schemes of this type and the
//  cheapest to implement on a microcontroller (no sorting, one random number).
template <int N>
inline void systematic_indices(const Real (&w)[N], Real u0, int (&idx)[N]) {
    const Real du = Real(1) / Real(N);
    Real u = u0 * du;
    Real c = w[0];
    int i = 0;
    for (int j = 0; j < N; ++j) {
        while (u > c && i < N - 1) { ++i; c += w[i]; }
        idx[j] = i;
        u += du;
    }
}

// ---- the same draw expressed as offspring counts n^i (sum_i n^i = N) ----------
template <int N>
inline void systematic_counts(const Real (&w)[N], Real u0, int (&cnt)[N]) {
    const Real du = Real(1) / Real(N);
    Real u = u0 * du;
    Real c = Real(0);
    int j = 0;
    for (int i = 0; i < N; ++i) {
        c += w[i];
        int n = 0;
        while (j < N && u <= c) { ++n; ++j; u += du; }
        cnt[i] = n;
    }
    if (j < N) cnt[N - 1] += N - j;            // round-off guard: sum_i n^i == N
}

// ---- in-place redistribution of a cloud given the offspring counts -----------
//  Dead slots (n^i = 0) hold particles nobody needs, so they can be overwritten
//  immediately; particles with n^i = 1 never move; particles with n^i > 1 push
//  their n^i - 1 surplus copies into dead slots.  Two monotone cursors => ONE
//  pass, O(N) element copies at most, and NO second particle array.  On exit
//  every slot holds a live particle and cnt[] is all ones.
template <class T, int N>
inline void inplace_redistribute(T (&p)[N], int (&cnt)[N]) {
    int src = 0, dst = 0;
    for (;;) {
        while (src < N && cnt[src] <= 1) ++src;          // a parent with surplus copies
        while (dst < N && cnt[dst] != 0) ++dst;          // a free slot
        if (src >= N || dst >= N) break;
        p[dst] = p[src];
        --cnt[src];
        cnt[dst] = 1;
    }
}

// ---- moments of a weighted cloud ---------------------------------------------
template <int NX, int N>
inline Vec<NX> weighted_mean(const Vec<NX> (&p)[N], const Real (&w)[N]) {
    Vec<NX> mu;
    for (int i = 0; i < N; ++i) {
        const Real wi = w[i];
        for (int k = 0; k < NX; ++k) mu[k] += wi * p[i][k];
    }
    return mu;
}
template <int NX, int N>
inline Mat<NX, NX> weighted_cov(const Vec<NX> (&p)[N], const Real (&w)[N], const Vec<NX>& mu) {
    Mat<NX, NX> P;
    for (int i = 0; i < N; ++i) {
        const Real wi = w[i];
        Real d[NX];
        for (int k = 0; k < NX; ++k) d[k] = p[i][k] - mu[k];
        for (int r = 0; r < NX; ++r) {
            const Real wd = wi * d[r];
            for (int c = r; c < NX; ++c) P(r, c) += wd * d[c];
        }
    }
    for (int r = 0; r < NX; ++r) for (int c = 0; c < r; ++c) P(r, c) = P(c, r);
    return P;
}

// ---- Gaussian log-likelihood with the factorisation done once per step -------
//  log N(e; 0, R) = -0.5 e^T R^{-1} e - 0.5 log det(2 pi R)
//                 = -0.5 |L^{-1} e|^2 - sum_i log L_ii - (NY/2) log 2 pi,
//  with R = L L^T.  Only the quadratic form is evaluated per particle, so the
//  per-particle cost is NY^2 multiply-adds and NO transcendental function.
template <int NY>
struct GaussLogLik {
    Mat<NY, NY> Linv;         // inverse of the lower Cholesky factor of R
    Real c0 = Real(0);        // -sum_i log L_ii - (NY/2) log(2 pi)
    void set(const Mat<NY, NY>& R) {
        const Mat<NY, NY> L = cholesky_safe(R);
        Linv = inverse_lower(L);
        Real ld = Real(0);
        for (int i = 0; i < NY; ++i) ld += std::log(L(i, i));
        c0 = -ld - Real(0.5) * Real(NY) * std::log(Real(2) * kPi);
    }
    Real logpdf(const Vec<NY>& e) const {
        const Vec<NY> z = Linv * e;
        return c0 - Real(0.5) * dot(z, z);
    }
};

// ---- log N(e; 0, Sigma) from a Cholesky factor Sigma = L L^T ------------------
//  Used by the proposal-corrected filters (UPF), where every particle carries its
//  own covariance so the factorisation cannot be hoisted out of the loop.
template <int N>
inline Real gauss_logpdf_chol(const Vec<N>& e, const Mat<N, N>& L) {
    const Vec<N> z = solve_lower(L, e);
    Real ld = Real(0);
    for (int i = 0; i < N; ++i) ld += std::log(L(i, i));
    return -Real(0.5) * dot(z, z) - ld - Real(0.5) * Real(N) * std::log(Real(2) * kPi);
}

// ---- roughening / jitter (Gordon, Salmond & Smith 1993, §4.2) -----------------
//  After resampling, add independent noise  J^i ~ N(0, diag(sigma^2)) with
//      sigma_k = K * E_k * N^{-1/nx},
//  where E_k is the spread (max - min) of the cloud in coordinate k and K ~ 0.2
//  is a tuning constant.  N^{-1/nx} is the mean inter-sample spacing of N points
//  in an nx-dimensional unit hypercube, so the jitter fills exactly the gaps that
//  resampling opened, and the bias it introduces vanishes as N -> infinity.
//
//  Two guards are added here, both expressed as fractions of the PRIOR standard
//  deviation sqrt(P0_kk) so that they stay model-independent:
//
//    FLOOR  sigma_k >= floor_frac sqrt(P0_kk).  When a single particle takes
//      essentially all the weight, E_k -> 0 and the jitter dies with it, after
//      which the cloud is frozen and the filter can never correct a slowly
//      accumulating error (terminal sample impoverishment).  The floor is the
//      "artificial dynamic noise" of Gordon et al. §4.2 in model-free form.
//
//    CAP    sigma_k <= cap_frac sqrt(P0_kk).  The raw rule is a POSITIVE
//      FEEDBACK: the jitter is proportional to the spread, so in any direction
//      the measurement cannot contract fast enough the spread grows
//      geometrically.  Such directions exist whenever states are only jointly
//      observable on a time scale longer than the sample interval (for the cell
//      model, SOC and the RC voltages are nearly indistinguishable within one
//      0.1 s step).  The cap bounds the sampling density by the prior
//      uncertainty and removes the instability; see the PF chapters for the
//      measured divergence of the uncapped rule.
template <int NX, int N>
inline Vec<NX> roughening_sigma(const Vec<NX> (&p)[N], Real K, Real n_pow,
                                const Vec<NX>& floor, const Vec<NX>& cap) {
    Vec<NX> s;
    for (int k = 0; k < NX; ++k) {
        Real lo = p[0][k], hi = p[0][k];
        for (int i = 1; i < N; ++i) { const Real v = p[i][k]; if (v < lo) lo = v; else if (v > hi) hi = v; }
        Real jit = K * (hi - lo) * n_pow;
        if (jit > cap[k]) jit = cap[k];
        if (jit < floor[k]) jit = floor[k];
        s[k] = jit;
    }
    return s;
}
template <int NX, int N>
inline void roughen(Vec<NX> (&p)[N], const Vec<NX>& sigma, Rng& rng) {
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < NX; ++k) p[i][k] += sigma[k] * rng.normal();
}

// N^{-1/nx}: mean spacing of N samples in an nx-dimensional hypercube.
inline Real mean_spacing(int n, int nx) { return std::pow(Real(n), -Real(1) / Real(nx)); }

}  // namespace pf
}  // namespace estkit
