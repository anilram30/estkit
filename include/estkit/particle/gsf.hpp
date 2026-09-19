// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/gsf.hpp — Gaussian-sum filter (bank of extended Kalman
//  filters with likelihood-driven mixture weights)
//
//  References
//    * Alspach, Daniel L. & Sorenson, Harold W. (1972). Nonlinear Bayesian
//      estimation using Gaussian sum approximations. IEEE Transactions on
//      Automatic Control 17(4), 439-448.
//    * Sorenson, Harold W. & Alspach, Daniel L. (1971). Recursive Bayesian
//      estimation using Gaussian sums. Automatica 7(4), 465-479.
//    * Salmond, David J. (1990). Mixture reduction algorithms for target
//      tracking in clutter. In: Signal and Data Processing of Small Targets 1990,
//      Proc. SPIE 1305, 434-445.                  (merging criterion used here)
//    * Anderson, Brian D. O. & Moore, John B. (1979). Optimal Filtering.
//      Prentice-Hall, §8.4.                       (Gaussian-sum approximations)
//
//  Idea.  Any density can be approximated arbitrarily well in the L1 sense by a
//  finite Gaussian mixture (Alspach & Sorenson 1972, Lemma 1)
//
//      p(x) ~ sum_{j=1}^{M} alpha_j N(x ; mu_j, P_j),   alpha_j >= 0, sum alpha_j = 1. (1)
//
//  If every component is narrow enough that f and h are nearly affine over its
//  support, then propagating each component with an EKF is accurate, and the
//  mixture stays a mixture.  The measurement update follows from Bayes' rule
//  applied term by term: with the component innovation and innovation covariance
//
//      e_j = y_k - h(mu_j^-, u_k),   S_j = H_j P_j^- H_j^T + R,                 (2)
//
//  the posterior weights are (Alspach & Sorenson 1972, eq. 17)
//
//      alpha_j^+ = alpha_j^- N(e_j; 0, S_j) / sum_m alpha_m^- N(e_m; 0, S_m),    (3)
//
//  and each component is updated by its own Kalman gain.  The point estimate and
//  its covariance are the mixture moments
//
//      xbar = sum_j alpha_j mu_j,
//      Pbar = sum_j alpha_j [ P_j + (mu_j - xbar)(mu_j - xbar)^T ].             (4)
//
//  The filter therefore keeps several LOCAL linearisations alive and lets the
//  measurements decide between them, which is exactly what a single EKF cannot
//  do when the prior is broad and h is curved — the situation created by a large
//  initial SOC error on a nonlinear OCV curve.
//
//  Initialisation.  The prior N(x0,P0) is split into NC components displaced
//  along one chosen coordinate j* (the SOC axis for the cell model):
//
//      mu_m = x0 + delta_m sqrt(P0_{j*j*}) e_{j*},  delta_m = delta (2m/(NC-1) - 1),
//      alpha_m = 1/NC,   (P_m)_{j*j*} = P0_{j*j*} (1 - (1/NC) sum_m delta_m^2),  (5)
//
//  which matches the mean and the covariance of the original Gaussian exactly.
//
//  Mixture reduction.  Components whose weight collapses carry no information,
//  and components that have converged to each other are redundant.  Salmond
//  (1990) measures the cost of merging components i and j by the within-cluster
//  scatter they contribute,
//
//      D_ij = [ alpha_i alpha_j / (alpha_i + alpha_j) ]
//             (mu_i - mu_j)^T Pbar^{-1} (mu_i - mu_j),                          (6)
//
//  and merges the cheapest pair.  Because the bank has a compile-time size, a
//  merge is immediately followed by a moment-preserving SPLIT of the heaviest
//  component along j*, so the filter keeps NC live hypotheses at all times
//  without allocating memory.
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../filters/ekf.hpp"
#include "pf_common.hpp"

namespace estkit {

template <class Model, int NC = 3>
class GaussianSumFilter {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NCOMP = NC;
    static_assert(NC >= 1, "at least one component");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        int spread_index = 0;             // coordinate along which the prior is split (SOC for the cell model)
        Real spread_delta = Real(1);      // +/- delta standard deviations, eq. (5)
        bool moment_match = true;         // shrink the components so the mixture reproduces P0 exactly
        Real split_delta = Real(0);       // re-split distance after a merge, in component std devs
        Real w_floor = Real(1e-12);       // numerical floor on the mixture weights
        Real w_min = Real(0.02);          // below this a component is considered collapsed
        Real merge_threshold = Real(0.1); // Salmond's D_ij threshold, eq. (6)
        bool reduce = false;              // enable Salmond merging (see the chapter: it costs diversity here)
        bool joseph = true;
        Real q_scale = Real(1), r_scale = Real(1);
        bool apply_constraints = true;
    } opts;

    explicit GaussianSumFilter(const Model& m) : m_(m), ekf_(m) {}
    void seed(uint64_t) {}   // deterministic filter: nothing to seed (API symmetry)

    void init(const X& x0, const Mat<NX, NX>& P0) {
        ekf_.opts.joseph = opts.joseph;
        ekf_.opts.apply_constraints = opts.apply_constraints;
        ekf_.opts.q_scale = opts.q_scale;
        ekf_.opts.r_scale = opts.r_scale;
        const int js = clampr(opts.spread_index, 0, NX - 1);
        const Real sd = std::sqrt(P0(js, js) > Real(0) ? P0(js, js) : Real(0));
        Real v = Real(0);
        for (int j = 0; j < NC; ++j) v += sq(offset(j)) / Real(NC);
        Real shrink = (Real(1) - v > Real(0.05)) ? (Real(1) - v) : Real(0.05);
        if (!opts.moment_match) shrink = Real(1);   // overlapping ("covering") mixture: prior inflated by 1+v
        for (int j = 0; j < NC; ++j) {
            X xj = x0; xj[js] += offset(j) * sd;
            Mat<NX, NX> Pj = P0; Pj(js, js) = P0(js, js) * shrink;
            mu_[j] = opts.apply_constraints ? constrain(m_, xj) : xj;
            Pj_[j] = Pj;
            w_[j] = Real(1) / Real(NC);
        }
        refresh_moments();
    }

    void predict(const U& u) {
        for (int j = 0; j < NC; ++j) {
            ekf_.init(mu_[j], Pj_[j]);
            ekf_.predict(u);
            mu_[j] = ekf_.x(); Pj_[j] = ekf_.P();
        }
        refresh_moments();
    }

    Y predict_measurement(const U& u) const {
        Y ym;
        for (int j = 0; j < NC; ++j) ym += m_.h(mu_[j], u) * w_[j];
        return ym;
    }

    void update(const Y& y, const U& u) {
        pf::GaussLogLik<NY> lik;
        Real lw[NC];
        for (int j = 0; j < NC; ++j) {
            ekf_.init(mu_[j], Pj_[j]);
            ekf_.update(y, u);                 // S() and innovation() keep the PRIOR values
            lik.set(ekf_.S());
            lw[j] = std::log(w_[j] > opts.w_floor ? w_[j] : opts.w_floor) + lik.logpdf(ekf_.innovation());
            mu_[j] = ekf_.x(); Pj_[j] = ekf_.P();
        }
        pf::normalize_log_weights(lw, w_);          // eq. (3)
        if (opts.reduce) reduce();
        refresh_moments();
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    Real weight(int j) const { return w_[j]; }
    const X& component_mean(int j) const { return mu_[j]; }
    const Mat<NX, NX>& component_cov(int j) const { return Pj_[j]; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Real offset(int j) const {
        if (NC == 1) return Real(0);
        return opts.spread_delta * (Real(2) * Real(j) / Real(NC - 1) - Real(1));
    }

    // Salmond (1990) mixture reduction: merge the cheapest pair, then re-split the
    // heaviest component so that the bank size stays NC.  Both steps preserve the
    // mixture mean and covariance exactly.
    void reduce() {
        if (NC < 2) return;
        int wi = 0, lo = 0;
        for (int j = 1; j < NC; ++j) { if (w_[j] > w_[wi]) wi = j; if (w_[j] < w_[lo]) lo = j; }
        const Mat<NX, NX> Pinv = inverse(P_);
        if (!Pinv.is_finite()) return;
        int bi = -1, bj = -1; Real best = Real(0);
        for (int i = 0; i < NC; ++i)
            for (int j = i + 1; j < NC; ++j) {
                const X d = mu_[i] - mu_[j];
                const Real wsum = w_[i] + w_[j];
                if (!(wsum > Real(0))) continue;
                const Real D = (w_[i] * w_[j] / wsum) * quad_form(Pinv, d);      // eq. (6)
                if (bi < 0 || D < best) { best = D; bi = i; bj = j; }
            }
        if (bi < 0) return;
        if (!(best < opts.merge_threshold || w_[lo] < opts.w_min)) return;
        // --- merge bj into bi ---
        const Real wm = w_[bi] + w_[bj];
        if (!(wm > Real(0))) return;
        const X mi = mu_[bi], mj = mu_[bj];
        const X mu = (mi * w_[bi] + mj * w_[bj]) / wm;
        const X di = mi - mu, dj = mj - mu;
        Mat<NX, NX> Pm = (Pj_[bi] + outer(di, di)) * (w_[bi] / wm)
                       + (Pj_[bj] + outer(dj, dj)) * (w_[bj] / wm);
        Pm.symmetrize();
        mu_[bi] = mu; Pj_[bi] = Pm;
        w_[bi] = wm; w_[bj] = Real(0);
        // --- split the heaviest component into (heaviest, bj) ---
        const int hs = (wi == bj) ? bi : wi;
        const int js = clampr(opts.spread_index, 0, NX - 1);
        const X mh = mu_[hs];
        Mat<NX, NX> Ph = Pj_[hs];
        const Real sd = std::sqrt(Ph(js, js) > Real(0) ? Ph(js, js) : Real(0));
        const Real d = opts.split_delta;
        Ph(js, js) = Ph(js, js) * (Real(1) - d * d);
        X ma = mh, mb = mh;
        ma[js] -= d * sd; mb[js] += d * sd;
        const Real wh = w_[hs];
        mu_[hs] = opts.apply_constraints ? constrain(m_, ma) : ma; Pj_[hs] = Ph;
        mu_[bj] = opts.apply_constraints ? constrain(m_, mb) : mb; Pj_[bj] = Ph;
        w_[hs] = Real(0.5) * wh; w_[bj] = Real(0.5) * wh;
        Real s = Real(0);
        for (int j = 0; j < NC; ++j) s += w_[j];
        if (s > Real(0)) for (int j = 0; j < NC; ++j) w_[j] /= s;
    }

    void refresh_moments() {
        x_ = X();
        for (int j = 0; j < NC; ++j) x_ += mu_[j] * w_[j];
        P_ = Mat<NX, NX>();
        for (int j = 0; j < NC; ++j) {
            const X d = mu_[j] - x_;
            P_ += (Pj_[j] + outer(d, d)) * w_[j];            // eq. (4)
        }
        P_.symmetrize();
    }

    Model m_;
    Ekf<Model> ekf_;            // one EKF instance serves as the workspace of every component
    X mu_[NC];                  // component means
    Mat<NX, NX> Pj_[NC];        // component covariances
    Real w_[NC] = {};           // mixture weights
    X x_;
    Mat<NX, NX> P_;
};

}  // namespace estkit
