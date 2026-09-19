// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/pf_auxiliary.hpp — auxiliary particle filter (APF / ASIR)
//
//  References
//    * Pitt, Michael K. & Shephard, Neil (1999). Filtering via simulation:
//      Auxiliary particle filters. Journal of the American Statistical
//      Association 94(446), 590-599.
//    * Arulampalam, M. Sanjeev, Maskell, Simon, Gordon, Neil & Clapp, Tim (2002).
//      A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian
//      tracking. IEEE Transactions on Signal Processing 50(2), 174-188.
//      (§III-C, Algorithm 5: auxiliary SIR)
//    * Gordon, Neil J., Salmond, David J. & Smith, Adrian F. M. (1993). Novel
//      approach to nonlinear/non-Gaussian Bayesian state estimation. IEE
//      Proceedings F 140(2), 107-113.            (roughening, §4.2)
//
//  Idea.  The bootstrap filter propagates BEFORE it looks at y_k, so particles
//  are pushed into regions the new measurement then rejects; with a sharp
//  likelihood almost all of the computation is wasted.  Pitt & Shephard
//  introduce the auxiliary variable i (the parent index) and target the joint
//
//      p(x_k, i | y_{1:k})  \propto  p(y_k|x_k) p(x_k|x_{k-1}^i) w_{k-1}^i .   (1)
//
//  Sampling (1) with the proposal  q(x_k,i|y_{1:k}) \propto p(y_k|mu_k^i)
//  w_{k-1}^i p(x_k|x_{k-1}^i), where mu_k^i is any cheap characterisation of
//  p(x_k|x_{k-1}^i) (here its mean, mu_k^i = f(x_{k-1}^i, u_{k-1})), gives the
//  two-stage algorithm
//
//      first stage :  lambda^i  \propto  w_{k-1}^i p(y_k | mu_k^i)             (2)
//                     resample the parent indices {i_j} from {lambda^i}
//      propagate   :  x_k^j ~ p(x_k | x_{k-1}^{i_j})                           (3)
//      second stage:  w_k^j  \propto  p(y_k|x_k^j) / p(y_k|mu_k^{i_j}) .       (4)
//
//  Resampling therefore happens BEFORE the propagation and is steered by the new
//  measurement ("look-ahead"), so the children are born where the likelihood is
//  high.  When the process noise is small the predictive p(x_k|x_{k-1}^i) is
//  nearly a point mass, p(y_k|mu^i) is nearly exact and the second-stage weights
//  in (4) are nearly uniform: the filter is then close to "fully adapted" and its
//  weight variance is far below that of the bootstrap filter (Pitt & Shephard
//  1999 §3.2).  That is exactly the regime of the battery problem, where the
//  voltage likelihood is much sharper than the one-step state uncertainty.
//
//  Roughening.  The propagation noise used in (3) is Q + diag(sigma_rough^2),
//  i.e. Gordon's jitter is interpreted as additional artificial process noise so
//  that proposal and transition density remain identical and (4) stays exact.
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
class AuxiliaryPf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NPART = N;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real roughen_K = Real(0);             // Gordon 1993 §4.2 spread term (off: the jitter already acts every step)
        Real roughen_floor_frac = Real(0.002);  // jitter floor, fraction of sqrt(diag(P0))
        Real roughen_cap_frac = Real(0.02);     // jitter ceiling, fraction of sqrt(diag(P0))
        Real q_scale = Real(1);
        Real r_scale = Real(1);
        bool apply_constraints = true;
    } opts;

    explicit AuxiliaryPf(const Model& m) : m_(m) {}
    void seed(uint64_t s) { rng_.reseed(s); }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        const Mat<NX, NX> L = cholesky_safe(P0);
        for (int i = 0; i < N; ++i) {
            X xi = x0 + L * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xi = constrain(m_, xi);
            p_[i] = xi; mu_[i] = xi;
            w_[i] = Real(1) / Real(N);
            lw_[i] = -std::log(Real(N));
        }
        for (int k = 0; k < NX; ++k)
            { const Real sd = std::sqrt(P0(k, k) > Real(0) ? P0(k, k) : Real(0));
              floor_[k] = opts.roughen_floor_frac * sd; cap_[k] = opts.roughen_cap_frac * sd; }
        spacing_ = pf::mean_spacing(N, NX);
        Lq_ = Mat<NX, NX>();    // zero until the first predict(): at k=0 there is no time update
        refresh_moments();
    }

    // Time update: only the deterministic part of (3) — the auxiliary stage needs
    // y_k, which arrives with update().
    void predict(const U& u) {
        Mat<NX, NX> Qp = m_.Q(x_, u) * opts.q_scale;
        const X s = pf::roughening_sigma(p_, opts.roughen_K, spacing_, floor_, cap_);
        for (int k = 0; k < NX; ++k) Qp(k, k) += s[k] * s[k];
        Lq_ = cholesky_safe(Qp);
        for (int i = 0; i < N; ++i) mu_[i] = m_.f(p_[i], u);
        x_ = pf::weighted_mean(mu_, w_);
    }

    Y predict_measurement(const U& u) const {
        Y ym;
        for (int i = 0; i < N; ++i) ym += m_.h(mu_[i], u) * w_[i];
        return ym;
    }

    void update(const Y& y, const U& u) {
        lik_.set(m_.R(x_, u) * opts.r_scale);
        // ---- first stage (2): look-ahead weights at the predicted mean particles
        for (int i = 0; i < N; ++i) {
            llmu_[i] = lik_.logpdf(y - m_.h(mu_[i], u));
            lam_[i] = lw_[i] + llmu_[i];
        }
        pf::normalize_log_weights(lam_, w_);
        pf::systematic_indices(w_, rng_.uniform(), idx_);
        // ---- propagation (3) and second-stage weights (4)
        for (int j = 0; j < N; ++j) {
            const int par = idx_[j];
            X xj = mu_[par] + Lq_ * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xj = constrain(m_, xj);
            p_[j] = xj;
            lw_[j] = lik_.logpdf(y - m_.h(xj, u)) - llmu_[par];
        }
        const Real lse = pf::normalize_log_weights(lw_, w_);
        if (std::isfinite(lse)) for (int j = 0; j < N; ++j) lw_[j] -= lse;
        else                    for (int j = 0; j < N; ++j) lw_[j] = -std::log(Real(N));
        ess_ = pf::ess(w_);
        refresh_moments();
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    Real ess() const { return ess_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void refresh_moments() {
        x_ = pf::weighted_mean(p_, w_);
        P_ = pf::weighted_cov(p_, w_, x_);
    }

    Model m_;
    Rng rng_{1};
    X p_[N];            // particles x_{k-1}^i (in) / x_k^j (out)
    X mu_[N];           // predicted mean particles mu_k^i = f(x_{k-1}^i, u_{k-1})
    Real w_[N] = {};
    Real lw_[N] = {};   // log of the second-stage weights
    Real lam_[N] = {};  // log of the first-stage weights
    Real llmu_[N] = {}; // log p(y_k | mu_k^i)
    int idx_[N] = {};
    X x_;
    Mat<NX, NX> P_;
    X floor_, cap_;
    Mat<NX, NX> Lq_;    // Cholesky factor of Q + diag(sigma_rough^2)
    Real spacing_ = Real(0);
    Real ess_ = Real(N);
    pf::GaussLogLik<NY> lik_;
};

}  // namespace estkit
