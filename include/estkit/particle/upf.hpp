// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/upf.hpp — unscented particle filter (UPF)
//
//  References
//    * van der Merwe, Rudolph, Doucet, Arnaud, de Freitas, Nando & Wan, Eric
//      (2000). The unscented particle filter. Technical Report
//      CUED/F-INFENG/TR 380, Cambridge University Engineering Department
//      (also in Advances in Neural Information Processing Systems 13, 584-590).
//    * Doucet, Arnaud, Godsill, Simon & Andrieu, Christophe (2000). On sequential
//      Monte Carlo sampling methods for Bayesian filtering. Statistics and
//      Computing 10(3), 197-208.        (optimal importance density, §II-B)
//    * Wan, Eric A. & van der Merwe, Rudolph (2000). The unscented Kalman filter
//      for nonlinear estimation. Proc. IEEE AS-SPCC, 153-158.
//    * Julier, Simon J. & Uhlmann, Jeffrey K. (2004). Unscented filtering and
//      nonlinear estimation. Proceedings of the IEEE 92(3), 401-422.
//
//  Motivation.  The bootstrap filter proposes from the transition prior
//  p(x_k|x_{k-1}), which ignores y_k.  When the likelihood is much sharper than
//  the one-step prior — the usual situation for a battery cell, where 2 mV of
//  voltage noise resolves SOC to ~0.3 % — nearly all particles land in the tails
//  of p(y_k|x_k) and the weights degenerate.  Doucet et al. (2000, eq. 10) show
//  that the importance density minimising the conditional weight variance is
//
//      q_opt(x_k | x_{k-1}^i, y_k) = p(x_k | x_{k-1}^i, y_k)
//                                  \propto p(y_k|x_k) p(x_k|x_{k-1}^i),         (1)
//
//  which is unavailable in closed form for a nonlinear h.  The UPF approximates
//  (1) by running one UNSCENTED KALMAN measurement update per particle:
//
//      prior^i   = N( f(x_{k-1}^i, u_{k-1}) , Qeff )                            (2)
//      (xa^i, Pa^i) = UKF-update( prior^i , y_k )                               (3)
//      q(x_k|x_{k-1}^i,y_k) = N( xa^i , Pa^i ) ,                                (4)
//
//  so the proposal is MEASUREMENT-AWARE: particles are born where the likelihood
//  already is.  The weight then carries the full importance correction
//
//      w_k^i \propto w_{k-1}^i * p(y_k|x_k^i) p(x_k^i|x_{k-1}^i) / q(x_k^i|.).  (5)
//
//  Because x_{k-1}^i is a SAMPLE and not a distribution, the exact one-step
//  predictive of particle i is the transition density itself, eq. (2); the
//  unscented update (3) is then a Gaussian approximation of the optimal density
//  (1) and (5) has low variance.  van der Merwe et al. additionally carry a
//  per-particle covariance P^i propagated by a full UKF (each particle stands for
//  a local Gaussian); that variant is available through
//  Options::carry_covariance.  On the cell model it is a poor choice and is NOT
//  the default: the per-particle UKF covariance converges to the Kalman
//  steady-state value, whose SOC standard deviation is ~1e-3, while the model's
//  own process noise is ~1e-5, so log p(x^i|x_{k-1}^i) in (5) is evaluated far
//  into the tail of a very narrow Gaussian and the weights collapse onto a single
//  particle.  Measured on the nominal eVTOL mission: SOC RMSE_ss 0.0118 with
//  carry_covariance = true against 0.0067 with the default, and 0.035 if the
//  roughening is additionally left out of the transition covariance below.
//
//  Qeff = Q + diag(sigma_rough^2) is the transition covariance actually used, the
//  jitter of Gordon et al. (1993) §4.2 being interpreted as additional artificial
//  process noise so that proposal and transition density stay consistent and (5)
//  remains exact.
//
//  Cost.  One sigma-point set (2 n_x + 1 evaluations of h) and three n_x x n_x
//  Cholesky factorisations per particle per step, i.e. roughly N times the
//  measurement half of a UKF.  N is small by design (30): the UPF buys accuracy
//  per particle, not per flop.
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../core/rng.hpp"
#include "../filters/ukf.hpp"
#include "pf_common.hpp"

namespace estkit {

template <class Model, int N = 30>
class UnscentedPf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NPART = N;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real ess_ratio = Real(0.5);
        Real roughen_K = Real(0.2);             // Gordon 1993 §4.2 (folded into Qeff)
        Real roughen_floor_frac = Real(0.01);   // jitter floor, fraction of sqrt(diag(P0)); N=30 needs a wide kernel
        Real roughen_cap_frac = Real(0.02);     // jitter ceiling, fraction of sqrt(diag(P0))
        Real q_scale = Real(1);
        Real r_scale = Real(1);
        Real alpha = Real(1e-1), beta = Real(2), kappa = Real(0);   // scaled UT parameters
        bool carry_covariance = false;          // van der Merwe's local-Gaussian variant
        bool apply_constraints = true;
    } opts;

    explicit UnscentedPf(const Model& m) : m_(m), ukf_(m) {}
    void seed(uint64_t s) { rng_.reseed(s); }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        ukf_.opts.alpha = opts.alpha; ukf_.opts.beta = opts.beta; ukf_.opts.kappa = opts.kappa;
        ukf_.opts.q_scale = opts.q_scale; ukf_.opts.r_scale = opts.r_scale;
        ukf_.opts.apply_constraints = opts.apply_constraints;
        const Mat<NX, NX> L = cholesky_safe(P0);
        for (int i = 0; i < N; ++i) {
            X xi = x0 + L * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xi = constrain(m_, xi);
            p_[i] = xi; mu_[i] = xi; xm_[i] = xi;
            Pp_[i] = P0; Pm_[i] = P0;
            w_[i] = Real(1) / Real(N);
            lw_[i] = -std::log(Real(N));
        }
        for (int k = 0; k < NX; ++k) {
            const Real sd = std::sqrt(P0(k, k) > Real(0) ? P0(k, k) : Real(0));
            floor_[k] = opts.roughen_floor_frac * sd;
            cap_[k] = opts.roughen_cap_frac * sd;
        }
        spacing_ = pf::mean_spacing(N, NX);
        have_predicted_ = false;
        refresh_moments();
    }

    // ---- time update: transition mean (2) and the proposal prior -------------
    void predict(const U& u) {
        Mat<NX, NX> Qeff = m_.Q(x_, u) * opts.q_scale;
        const X s = pf::roughening_sigma(p_, opts.roughen_K, spacing_, floor_, cap_);
        for (int k = 0; k < NX; ++k) Qeff(k, k) += s[k] * s[k];
        Lq_ = cholesky_safe(Qeff);
        for (int i = 0; i < N; ++i) {
            mu_[i] = m_.f(p_[i], u);
            if (opts.carry_covariance) {               // van der Merwe's original recursion
                ukf_.init(p_[i], Pp_[i]);
                ukf_.predict(u);
                xm_[i] = ukf_.x(); Pm_[i] = ukf_.P();
            } else {                                   // particle = sample: predictive is (2)
                xm_[i] = mu_[i]; Pm_[i] = Qeff;
            }
        }
        have_predicted_ = true;
        x_ = pf::weighted_mean(xm_, w_);
    }

    Y predict_measurement(const U& u) const {
        Y ym;
        for (int i = 0; i < N; ++i) ym += m_.h(xm_[i], u) * w_[i];
        return ym;
    }

    // ---- measurement update: propose from the per-particle UKF posterior (3) -
    void update(const Y& y, const U& u) {
        lik_.set(m_.R(x_, u) * opts.r_scale);
        const Real cprop = -Real(0.5) * Real(NX) * std::log(Real(2) * kPi);
        for (int i = 0; i < N; ++i) {
            if (!have_predicted_) {          // k = 0: no time update yet -> SIS on the prior cloud
                lw_[i] += lik_.logpdf(y - m_.h(p_[i], u));
                continue;
            }
            ukf_.init(xm_[i], Pm_[i]);
            ukf_.update(y, u);
            const X xa = ukf_.x();
            const Mat<NX, NX> Pa = ukf_.P();
            const Mat<NX, NX> La = cholesky_safe(Pa);
            const X xi = rng_.normal_vec<NX>();
            X xs = xa + La * xi;                                        // draw from q, eq. (4)
            if (opts.apply_constraints) xs = constrain(m_, xs);
            // log q(xs) = -0.5 |xi|^2 - sum_k log La_kk - (nx/2) log 2 pi
            Real logq = cprop - Real(0.5) * dot(xi, xi);
            for (int k = 0; k < NX; ++k) logq -= std::log(La(k, k));
            lw_[i] += lik_.logpdf(y - m_.h(xs, u))                      // eq. (5)
                    + pf::gauss_logpdf_chol(xs - mu_[i], Lq_)
                    - logq;
            p_[i] = xs; Pp_[i] = Pa;
        }
        const Real lse = pf::normalize_log_weights(lw_, w_);
        if (std::isfinite(lse)) for (int i = 0; i < N; ++i) lw_[i] -= lse;
        else                    for (int i = 0; i < N; ++i) lw_[i] = -std::log(Real(N));
        ess_ = pf::ess(w_);
        if (ess_ < opts.ess_ratio * Real(N)) resample();
        refresh_moments();
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    Real ess() const { return ess_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void resample() {
        pf::systematic_indices(w_, rng_.uniform(), idx_);
        for (int i = 0; i < N; ++i) { sx_[i] = p_[idx_[i]]; sP_[i] = Pp_[idx_[i]]; }
        for (int i = 0; i < N; ++i) {
            p_[i] = sx_[i]; Pp_[i] = sP_[i];
            w_[i] = Real(1) / Real(N); lw_[i] = -std::log(Real(N));
        }
    }
    void refresh_moments() {
        x_ = pf::weighted_mean(p_, w_);
        P_ = pf::weighted_cov(p_, w_, x_);
    }

    Model m_;
    Ukf<Model> ukf_;          // reused workspace: one UKF instance serves all particles
    Rng rng_{1};
    X p_[N];                  // particles (posterior samples)
    Mat<NX, NX> Pp_[N];       // per-particle covariance (only used by the carry_covariance variant)
    X mu_[N];                 // f(x_{k-1}^i, u_{k-1})
    X xm_[N];                 // mean of the proposal prior
    Mat<NX, NX> Pm_[N];       // covariance of the proposal prior (= Qeff by default)
    X sx_[N]; Mat<NX, NX> sP_[N];   // resampling scratch
    Real w_[N] = {};
    Real lw_[N] = {};
    int idx_[N] = {};
    X x_;
    Mat<NX, NX> P_;
    X floor_, cap_;
    Mat<NX, NX> Lq_;          // Cholesky factor of Qeff
    Real spacing_ = Real(0);
    Real ess_ = Real(N);
    bool have_predicted_ = false;
    pf::GaussLogLik<NY> lik_;
};

}  // namespace estkit
