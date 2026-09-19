// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/pf_bootstrap.hpp — bootstrap / SIR particle filter
//
//  References
//    * Gordon, Neil J., Salmond, David J. & Smith, Adrian F. M. (1993). Novel
//      approach to nonlinear/non-Gaussian Bayesian state estimation. IEE
//      Proceedings F (Radar and Signal Processing) 140(2), 107-113.
//    * Arulampalam, M. Sanjeev, Maskell, Simon, Gordon, Neil & Clapp, Tim (2002).
//      A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian
//      tracking. IEEE Transactions on Signal Processing 50(2), 174-188.
//    * Kitagawa, Genshiro (1996). Monte Carlo filter and smoother for
//      non-Gaussian nonlinear state space models. Journal of Computational and
//      Graphical Statistics 5(1), 1-25.          (systematic resampling)
//    * Doucet, Arnaud, Godsill, Simon & Andrieu, Christophe (2000). On sequential
//      Monte Carlo sampling methods for Bayesian filtering. Statistics and
//      Computing 10(3), 197-208.                 (weight recursion, ESS rule)
//
//  Model  x_k = f(x_{k-1}, u_{k-1}) + w_{k-1},  w ~ N(0, Q(x,u))
//         y_k = h(x_k, u_k) + v_k,              v ~ N(0, R(x,u))
//
//  Sequential importance sampling with the transition prior as proposal,
//  q(x_k | x_{k-1}^i, y_k) = p(x_k | x_{k-1}^i), reduces the weight recursion
//
//      w_k^i  =  w_{k-1}^i * p(y_k|x_k^i) p(x_k^i|x_{k-1}^i) / q(x_k^i|x_{k-1}^i,y_k)
//
//  to the likelihood-only form (Gordon 1993 eq. 12; Arulampalam 2002 eq. 63)
//
//      w_k^i  \propto  w_{k-1}^i * p(y_k | x_k^i) ,     sum_i w_k^i = 1 .
//
//  The empirical filtering distribution is  p(x_k|y_{1:k}) ~ sum_i w_k^i d(x - x_k^i)
//  and the MMSE point estimate is the weighted mean.  Resampling is triggered by
//  the effective sample size N_eff = 1/sum_i (w^i)^2 < ratio * N (ratio = 1/2),
//  and is performed with Kitagawa's systematic scheme, followed by the roughening
//  of Gordon et al. §4.2 which restores the diversity that resampling destroys.
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../core/rng.hpp"
#include "pf_common.hpp"

namespace estkit {

template <class Model, int N = 500>
class BootstrapPf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NPART = N;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real ess_ratio = Real(0.5);          // resample when N_eff < ess_ratio * N (Doucet 2000)
        Real roughen_K = Real(0.2);          // Gordon 1993 §4.2 roughening constant
        Real roughen_floor_frac = Real(0.005);  // jitter floor, fraction of sqrt(diag(P0))
        Real roughen_cap_frac = Real(0.02);     // jitter ceiling, fraction of sqrt(diag(P0))
        Real q_scale = Real(1);              // multiplicative tuning of Q
        Real r_scale = Real(1);              // multiplicative tuning of R
        bool apply_constraints = true;       // project every particle with Model::constrain
    } opts;

    explicit BootstrapPf(const Model& m) : m_(m) {}

    // deterministic RNG stream; set from EstimatorConfig::seed before init()
    void seed(uint64_t s) { rng_.reseed(s); }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        const Mat<NX, NX> L = cholesky_safe(P0);
        for (int i = 0; i < N; ++i) {
            X xi = x0 + L * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xi = constrain(m_, xi);
            p_[i] = xi;
            w_[i] = Real(1) / Real(N);
            lw_[i] = -std::log(Real(N));
        }
        for (int k = 0; k < NX; ++k)
            { const Real sd = std::sqrt(P0(k, k) > Real(0) ? P0(k, k) : Real(0));
              floor_[k] = opts.roughen_floor_frac * sd; cap_[k] = opts.roughen_cap_frac * sd; }
        spacing_ = pf::mean_spacing(N, NX);
        refresh_moments();
    }

    // ---- time update: draw x_k^i ~ p(x_k | x_{k-1}^i) ------------------------
    void predict(const U& u) {
        // Q is evaluated once at the cloud mean: for models whose Q depends on the
        // state only weakly this saves N-1 factorisations per step (see chapter).
        const Mat<NX, NX> Lq = cholesky_safe(m_.Q(x_, u) * opts.q_scale);
        for (int i = 0; i < N; ++i) {
            X xn = m_.f(p_[i], u) + Lq * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xn = constrain(m_, xn);
            p_[i] = xn;
        }
        x_ = pf::weighted_mean(p_, w_);
    }

    // ---- prior predictive measurement: E[h(x_k,u_k) | y_{1:k-1}] -------------
    Y predict_measurement(const U& u) const {
        Y ym;
        for (int i = 0; i < N; ++i) ym += m_.h(p_[i], u) * w_[i];
        return ym;
    }

    // ---- measurement update: w^i <- w^i p(y_k|x_k^i), resample if needed -----
    void update(const Y& y, const U& u) {
        lik_.set(m_.R(x_, u) * opts.r_scale);
        for (int i = 0; i < N; ++i) lw_[i] += lik_.logpdf(y - m_.h(p_[i], u));
        const Real lse = pf::normalize_log_weights(lw_, w_);
        if (std::isfinite(lse)) for (int i = 0; i < N; ++i) lw_[i] -= lse;   // keep log-weights O(1)
        else                    for (int i = 0; i < N; ++i) lw_[i] = -std::log(Real(N));
        ess_ = pf::ess(w_);
        if (ess_ < opts.ess_ratio * Real(N)) resample();
        refresh_moments();
    }

    // ---- accessors -----------------------------------------------------------
    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    Real ess() const { return ess_; }
    int resample_count() const { return nresample_; }
    const X* particles() const { return p_; }
    const Real* weights() const { return w_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    // Systematic resampling (Kitagawa 1996) into a scratch buffer, then copy back.
    void resample() {
        pf::systematic_indices(w_, rng_.uniform(), idx_);
        for (int i = 0; i < N; ++i) scratch_[i] = p_[idx_[i]];
        for (int i = 0; i < N; ++i) { p_[i] = scratch_[i]; w_[i] = Real(1) / Real(N); lw_[i] = -std::log(Real(N)); }
        const X s = pf::roughening_sigma(p_, opts.roughen_K, spacing_, floor_, cap_);
        pf::roughen(p_, s, rng_);
        if (opts.apply_constraints) for (int i = 0; i < N; ++i) p_[i] = constrain(m_, p_[i]);
        ++nresample_;
    }
    void refresh_moments() {
        x_ = pf::weighted_mean(p_, w_);
        P_ = pf::weighted_cov(p_, w_, x_);
    }

    Model m_;
    Rng rng_{1};
    X p_[N];              // particle cloud
    X scratch_[N];        // resampling buffer
    Real w_[N] = {};      // normalised weights
    Real lw_[N] = {};     // log-weights (exp(lw_) == w_)
    int idx_[N] = {};     // parent indices
    X x_;                 // weighted mean  (MMSE estimate)
    Mat<NX, NX> P_;       // weighted covariance
    X floor_, cap_;       // per-coordinate roughening floor / ceiling
    Real spacing_ = Real(0);
    Real ess_ = Real(N);
    int nresample_ = 0;
    pf::GaussLogLik<NY> lik_;
};

}  // namespace estkit
