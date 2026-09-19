// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/battery/metrics.hpp — accuracy, convergence, divergence and runtime
//  metrics for SOC estimators.
// =============================================================================
#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include "../core/linalg.hpp"

namespace estkit {

struct RunMetrics {
    double rmse = 0;          // SOC RMSE over the whole run [-]
    double rmse_ss = 0;       // SOC RMSE over the second half (steady state)
    double mae = 0;
    double max_err = 0;
    double final_err = 0;
    double conv_time = -1;    // first time after which |e| < 2 % for at least 60 s (-1 = never)
    bool diverged = false;    // NaN/inf, or |e| > 30 % anywhere in the last half of the run
    double rmse_v = 0;        // voltage prediction RMSE [V] (NaN if unavailable)
    double ns_per_step = 0;   // wall-clock time per step
    size_t bytes = 0;
    double capacity_err_end = kNaN;   // (SOH-capable estimators) capacity error at the end [Ah]
    double bound_violation = kNaN;    // (interval observers) fraction of samples with truth outside [lower, upper]
    double bound_width = kNaN;        // mean interval width
};

inline RunMetrics compute_metrics(const std::vector<Real>& soc_true, const std::vector<Real>& soc_est,
                                  const std::vector<Real>& v_true, const std::vector<Real>& v_pred, Real dt) {
    RunMetrics m; const int n = int(soc_true.size());
    double se = 0, se_ss = 0, ae = 0, sev = 0; int nv = 0; int n_ss = 0;
    std::vector<double> e(n);
    for (int k = 0; k < n; ++k) {
        e[k] = double(soc_est[k]) - double(soc_true[k]);
        if (!std::isfinite(e[k])) { m.diverged = true; e[k] = 1.0; }
        se += e[k] * e[k]; ae += std::fabs(e[k]); m.max_err = std::max(m.max_err, std::fabs(e[k]));
        if (k >= n / 2) { se_ss += e[k] * e[k]; ++n_ss; if (std::fabs(e[k]) > 0.30) m.diverged = true; }
        if (k < int(v_pred.size()) && std::isfinite(v_pred[k])) { sev += sq(double(v_pred[k]) - double(v_true[k])); ++nv; }
    }
    m.rmse = std::sqrt(se / std::max(n, 1)); m.rmse_ss = std::sqrt(se_ss / std::max(n_ss, 1)); m.mae = ae / std::max(n, 1);
    m.final_err = n ? e[n - 1] : 0;
    m.rmse_v = nv ? std::sqrt(sev / nv) : kNaN;
    // convergence time: |e| < 0.02 sustained for 60 s
    const int win = int(60 / dt);
    int run = 0;
    for (int k = 0; k < n; ++k) {
        if (std::fabs(e[k]) < 0.02) { ++run; if (run >= win) { m.conv_time = (k - win + 1) * dt; break; } } else run = 0;
    }
    return m;
}

}  // namespace estkit
