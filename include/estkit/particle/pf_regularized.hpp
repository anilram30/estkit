// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/pf_regularized.hpp — regularised particle filter (RPF)
//
//  References
//    * Musso, Christian, Oudjane, Nadia & Le Gland, Francois (2001). Improving
//      regularised particle filters. In: Doucet, A., de Freitas, N. & Gordon, N.
//      (eds.) Sequential Monte Carlo Methods in Practice, Springer, New York,
//      pp. 247-271.
//    * Arulampalam, M. Sanjeev, Maskell, Simon, Gordon, Neil & Clapp, Tim (2002).
//      A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian
//      tracking. IEEE Transactions on Signal Processing 50(2), 174-188. (§III-D,
//      Algorithm 6: regularised particle filter)
//    * Silverman, Bernard W. (1986). Density Estimation for Statistics and Data
//      Analysis. Chapman & Hall, London. (optimal kernel bandwidth, §4.3)
//    * Gordon, Neil J., Salmond, David J. & Smith, Adrian F. M. (1993). Novel
//      approach to nonlinear/non-Gaussian Bayesian state estimation. IEE
//      Proceedings F 140(2), 107-113.
//
//  Idea.  Plain SIR resampling draws from the DISCRETE approximation
//      p(x_k|y_{1:k}) ~ sum_i w^i delta(x - x^i),
//  so every child is an exact copy of its parent and the support of the cloud can
//  only shrink (sample impoverishment).  The RPF instead resamples from the
//  continuous kernel-density approximation (Musso et al. 2001, eq. 12)
//
//      p_hat(x) = sum_i w^i  K_h(x - x^i),      K_h(x) = h^{-n_x} K(x/h),      (1)
//
//  with a kernel K rescaled by the empirical covariance S of the weighted cloud:
//
//      K_{h,S}(x) = h^{-n_x} |S|^{-1/2} K( S^{-1/2} x / h ).                   (2)
//
//  Drawing from (1) is trivial: pick a parent index by systematic resampling and
//  add h * D * eps, where D D^T = S and eps ~ K.  With the Gaussian kernel the
//  bandwidth that minimises the mean integrated square error between p_hat and
//  the true density (Silverman 1986 eq. 4.14; Musso et al. 2001 eq. 15) is
//
//      h_opt = A(n_x) N^{-1/(n_x+4)},     A(n_x) = ( 4 / (n_x+2) )^{1/(n_x+4)}. (3)
//
//  The price is a bias: the regularised cloud has covariance (1+h_opt^2) S, i.e.
//  the posterior is smoothed.  Musso et al. show that for small sample sizes the
//  reduction in impoverishment more than compensates for this bias, and that the
//  RPF is asymptotically equivalent to the SIR filter as N -> infinity
//  (h_opt -> 0).
//
//  Collapse floor.  If one particle takes all the weight, S -> 0 and the kernel
//  degenerates.  As in pf_bootstrap.hpp the sampling covariance is therefore
//  floored: Sigma = h_opt^2 S + diag(f_k^2), f_k = floor_frac * sqrt(P0_kk).
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
class RegularizedPf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NPART = N;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real ess_ratio = Real(0.5);            // regularise+resample when N_eff < ratio*N
        Real bandwidth_scale = Real(1);        // multiplies h_opt of eq. (3)
        Real kernel_floor_frac = Real(0.005);  // diagonal floor, fraction of sqrt(diag(P0))
        Real kernel_cap_frac = Real(0.02);     // kernel ceiling, fraction of sqrt(diag(P0))
        Real q_scale = Real(1);
        Real r_scale = Real(1);
        bool apply_constraints = true;
    } opts;

    explicit RegularizedPf(const Model& m) : m_(m) {}
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
        for (int k = 0; k < NX; ++k) {
            const Real sd = std::sqrt(P0(k, k) > Real(0) ? P0(k, k) : Real(0));
            floor_[k] = opts.kernel_floor_frac * sd;
            cap_[k] = opts.kernel_cap_frac * sd;
        }
        // eq. (3): A(n_x) = (4/(n_x+2))^{1/(n_x+4)},  h_opt = A N^{-1/(n_x+4)}
        const Real ex = Real(1) / Real(NX + 4);
        h_opt_ = std::pow(Real(4) / Real(NX + 2), ex) * std::pow(Real(N), -ex);
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
        for (int i = 0; i < N; ++i) lw_[i] += lik_.logpdf(y - m_.h(p_[i], u));
        const Real lse = pf::normalize_log_weights(lw_, w_);
        if (std::isfinite(lse)) for (int i = 0; i < N; ++i) lw_[i] -= lse;
        else                    for (int i = 0; i < N; ++i) lw_[i] = -std::log(Real(N));
        ess_ = pf::ess(w_);
        refresh_moments();                       // S = empirical covariance of the WEIGHTED cloud
        if (ess_ < opts.ess_ratio * Real(N)) regularised_resample();
        refresh_moments();
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    Real ess() const { return ess_; }
    Real bandwidth() const { return h_opt_ * opts.bandwidth_scale; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    // Draw N samples from  sum_i w^i N(x^i, h^2 S + diag(floor^2))  (Musso 2001 Alg. 2)
    void regularised_resample() {
        const Real h = h_opt_ * opts.bandwidth_scale;
        Mat<NX, NX> Sig = P_ * (h * h);
        // Ceiling: the kernel is a positive feedback on the cloud spread (every
        // regularisation step multiplies the covariance by 1+h^2), which diverges
        // in directions the measurement cannot contract within one step.  Scale
        // the whole kernel — preserving its shape and hence the correlations —
        // so that no coordinate exceeds cap_k.
        Real sc = Real(1);
        for (int k = 0; k < NX; ++k) {
            const Real sk = std::sqrt(Sig(k, k) > Real(0) ? Sig(k, k) : Real(0));
            if (sk > cap_[k] && sk > Real(0)) { const Real r = cap_[k] / sk; if (r < sc) sc = r; }
        }
        Sig *= sc * sc;
        for (int k = 0; k < NX; ++k) Sig(k, k) += floor_[k] * floor_[k];
        const Mat<NX, NX> D = cholesky_safe(Sig);
        pf::systematic_indices(w_, rng_.uniform(), idx_);
        for (int i = 0; i < N; ++i) scratch_[i] = p_[idx_[i]] + D * rng_.normal_vec<NX>();
        for (int i = 0; i < N; ++i) {
            p_[i] = opts.apply_constraints ? constrain(m_, scratch_[i]) : scratch_[i];
            w_[i] = Real(1) / Real(N);
            lw_[i] = -std::log(Real(N));
        }
    }
    void refresh_moments() {
        x_ = pf::weighted_mean(p_, w_);
        P_ = pf::weighted_cov(p_, w_, x_);
    }

    Model m_;
    Rng rng_{1};
    X p_[N];
    X scratch_[N];
    Real w_[N] = {};
    Real lw_[N] = {};
    int idx_[N] = {};
    X x_;
    Mat<NX, NX> P_;
    X floor_, cap_;
    Real h_opt_ = Real(0);
    Real ess_ = Real(N);
    pf::GaussLogLik<NY> lik_;
};

}  // namespace estkit
