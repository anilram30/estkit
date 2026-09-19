// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/pf_fast.hpp — embedded-oriented bootstrap particle filter
//
//  Same estimator as pf_bootstrap.hpp (sequential importance resampling with the
//  transition prior as proposal) but restructured for a microcontroller:
//    * small cloud (N = 100 by default) with a roughening kernel that keeps the
//      cloud alive at low N;
//    * ONE particle array — systematic resampling is performed in place from the
//      offspring counts, so no N-element scratch copy of the state is needed;
//    * weights carried in linear form and updated by a single exp() per particle
//      applied to the max-shifted log-likelihood: no log(), no division inside
//      the per-particle loop, no dynamic range problem in float32;
//    * measurement likelihood through a pre-factored R (one Cholesky per step).
//
//  References
//    * Gordon, Neil J., Salmond, David J. & Smith, Adrian F. M. (1993). Novel
//      approach to nonlinear/non-Gaussian Bayesian state estimation. IEE
//      Proceedings F (Radar and Signal Processing) 140(2), 107-113.
//      (bootstrap filter; roughening, §4.2)
//    * Kitagawa, Genshiro (1996). Monte Carlo filter and smoother for
//      non-Gaussian nonlinear state space models. Journal of Computational and
//      Graphical Statistics 5(1), 1-25.            (systematic resampling)
//    * Arulampalam, M. Sanjeev, Maskell, Simon, Gordon, Neil & Clapp, Tim (2002).
//      A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian
//      tracking. IEEE Transactions on Signal Processing 50(2), 174-188.
//    * Gustafsson, Fredrik (2010). Particle filter theory and practice with
//      positioning applications. IEEE Aerospace and Electronic Systems Magazine
//      25(7), 53-82.                                (implementation guidelines)
//
//  Weight update actually executed (mathematically identical to w^i <- w^i p(y|x^i)):
//      l^i   = -0.5 |L_R^{-1} (y - h(x^i,u))|^2 + c0        (no transcendental)
//      l_max = max_i l^i
//      w^i  <- w^i * exp(l^i - l_max)                       (one exp per particle)
//      w^i  <- w^i / sum_j w^j
//
//  In-place resampling.  Systematic resampling draws offspring counts n^i with
//  sum_i n^i = N.  Slots with n^i = 0 hold particles that no longer belong to the
//  posterior, so they can be overwritten at once; slots with n^i = 1 keep their
//  particle where it is; slots with n^i > 1 copy their surplus into free slots.
//  Two monotone cursors sweep the array once, giving at most N - #{i: n^i>0}
//  state copies and no auxiliary particle storage (Gustafsson 2010 §IV-C).
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../core/rng.hpp"
#include "pf_common.hpp"

namespace estkit {

template <class Model, int N = 100>
class FastPf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NPART = N;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real ess_ratio = Real(0.5);           // resample when N_eff < ess_ratio * N
        Real roughen_K = Real(0.3);           // larger than the N=500 filter: fewer samples
        Real roughen_floor_frac = Real(0.01);   // jitter floor, fraction of sqrt(diag(P0)); N=100 needs more than N=500
        Real roughen_cap_frac = Real(0.02);     // jitter ceiling, fraction of sqrt(diag(P0))
        Real q_scale = Real(1);
        Real r_scale = Real(1);
        bool apply_constraints = true;
    } opts;

    explicit FastPf(const Model& m) : m_(m) {}
    void seed(uint64_t s) { rng_.reseed(s); }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        const Mat<NX, NX> L = cholesky_safe(P0);
        for (int i = 0; i < N; ++i) {
            X xi = x0 + L * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xi = constrain(m_, xi);
            p_[i] = xi;
            w_[i] = Real(1) / Real(N);
        }
        for (int k = 0; k < NX; ++k)
            { const Real sd = std::sqrt(P0(k, k) > Real(0) ? P0(k, k) : Real(0));
              floor_[k] = opts.roughen_floor_frac * sd; cap_[k] = opts.roughen_cap_frac * sd; }
        spacing_ = pf::mean_spacing(N, NX);
        refresh_moments();
    }

    void predict(const U& u) {
        const Mat<NX, NX> Lq = cholesky_safe(m_.Q(x_, u) * opts.q_scale);
        for (int i = 0; i < N; ++i) {
            X xn = m_.f(p_[i], u) + Lq * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xn = constrain(m_, xn);
            p_[i] = xn;
        }
        x_ = pf::weighted_mean(p_, w_);
    }

    Y predict_measurement(const U& u) const {
        Y ym;
        for (int i = 0; i < N; ++i) ym += m_.h(p_[i], u) * w_[i];
        return ym;
    }

    void update(const Y& y, const U& u) {
        lik_.set(m_.R(x_, u) * opts.r_scale);
        Real lmax = -std::numeric_limits<Real>::infinity();
        for (int i = 0; i < N; ++i) {
            ll_[i] = lik_.logpdf(y - m_.h(p_[i], u));
            if (ll_[i] > lmax) lmax = ll_[i];
        }
        Real sum = Real(0);
        if (std::isfinite(lmax)) {
            for (int i = 0; i < N; ++i) { w_[i] *= std::exp(ll_[i] - lmax); sum += w_[i]; }   // one exp/particle
        }
        if (sum > Real(0) && std::isfinite(sum)) {
            const Real inv = Real(1) / sum;
            for (int i = 0; i < N; ++i) w_[i] *= inv;
        } else {
            for (int i = 0; i < N; ++i) w_[i] = Real(1) / Real(N);
        }
        ess_ = pf::ess(w_);
        if (ess_ < opts.ess_ratio * Real(N)) resample_in_place();
        refresh_moments();
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    Real ess() const { return ess_; }
    const X* particles() const { return p_; }
    const Real* weights() const { return w_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void resample_in_place() {
        pf::systematic_counts(w_, rng_.uniform(), cnt_);
        pf::inplace_redistribute(p_, cnt_);
        for (int i = 0; i < N; ++i) w_[i] = Real(1) / Real(N);
        const X s = pf::roughening_sigma(p_, opts.roughen_K, spacing_, floor_, cap_);
        pf::roughen(p_, s, rng_);
        if (opts.apply_constraints) for (int i = 0; i < N; ++i) p_[i] = constrain(m_, p_[i]);
    }
    void refresh_moments() {
        x_ = pf::weighted_mean(p_, w_);
        P_ = pf::weighted_cov(p_, w_, x_);
    }

    Model m_;
    Rng rng_{1};
    X p_[N];                  // the ONLY particle array
    Real w_[N] = {};          // linear weights
    Real ll_[N] = {};         // per-particle log-likelihood of the current sample
    int cnt_[N] = {};         // offspring counts (resampling scratch: N ints)
    X x_;
    Mat<NX, NX> P_;
    X floor_, cap_;
    Real spacing_ = Real(0);
    Real ess_ = Real(N);
    pf::GaussLogLik<NY> lik_;
};

}  // namespace estkit
