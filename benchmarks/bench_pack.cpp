// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  bench_pack — pack-level (12-cell series module) SOC estimator benchmark
//  Usage: bench_pack [--profile evtol] [--scenario NAME|all] [--only NAME] [--seeds K] [--out DIR] [--reps R] [--list]
//  Output: <out>/summary_pack.csv ; <out>/ts/pack__<est>__<profile>__<scenario>__s1.csv
// =============================================================================
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include "estkit/pack/pack_estimator.hpp"

using namespace estkit;
static std::vector<std::string> split(const std::string& s, char sep) { std::vector<std::string> o; std::stringstream ss(s); std::string it; while (std::getline(ss, it, sep)) if (!it.empty()) o.push_back(it); return o; }
static void mkdirp(const std::string& p) { std::error_code ec; std::filesystem::create_directories(p, ec); }
static bool file_empty(const std::string& p) { std::error_code ec; return !std::filesystem::exists(p, ec) || std::filesystem::file_size(p, ec) == 0; }

int main(int argc, char** argv) {
    std::string profile = "evtol", scenario = "all", only, out = "results"; int seeds = 1, reps = 2; bool list_only = false, timeseries = true;
    for (int a = 1; a < argc; ++a) {
        std::string k = argv[a]; auto val = [&]() { return std::string(a + 1 < argc ? argv[++a] : ""); };
        if (k == "--profile") profile = val(); else if (k == "--scenario") scenario = val(); else if (k == "--only") only = val();
        else if (k == "--seeds") seeds = std::stoi(val()); else if (k == "--out") out = val(); else if (k == "--reps") reps = std::stoi(val());
        else if (k == "--timeseries") timeseries = std::stoi(val()) != 0; else if (k == "--list") list_only = true;
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    PackList ests = make_all_pack_estimators();
    if (list_only) { for (auto& e : ests) std::printf("%-28s %s\n", e->name(), e->group()); return 0; }
    std::set<std::string> only_set; for (auto& s : split(only, ',')) only_set.insert(s);
    std::vector<PackScenario> scs; for (auto& s : pack_scenarios()) if (scenario == "all" || s.name == scenario) scs.push_back(s);
    mkdirp(out); if (timeseries) mkdirp(out + "/ts");
    const std::string sp = out + "/summary_pack.csv"; const bool hdr = file_empty(sp);
    std::ofstream summary(sp, std::ios::app);
    if (hdr) summary << "estimator,group,profile,scenario,seed,rmse_cells,rmse_min,rmse_mean,rmse_max,max_cell_err,ss_rmse_cells,diverged,ns_per_step,bytes,messages\n";
    const MissionProfile mp = mission_by_name(profile);
    for (const auto& sc : scs) for (int seed = 1; seed <= seeds; ++seed) {
        const PackDataset d = make_pack_dataset(mp, sc, uint64_t(seed));
        for (auto& e : ests) {
            if (!only_set.empty() && !only_set.count(e->name())) continue;
            std::vector<std::vector<Real>> est(kPackCells, std::vector<Real>(d.n));
            double best_ns = 1e300;
            for (int rep = 0; rep < reps; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                e->reset(d.est_cfg);
                Real v[kPackCells], T[kPackCells];
                for (int k = 0; k < d.n; ++k) {
                    for (int c = 0; c < kPackCells; ++c) { v[c] = d.v_meas[c][k]; T[c] = d.T_meas[c][k]; }
                    e->step(d.i_meas[k], v, T);
                    for (int c = 0; c < kPackCells; ++c) est[c][k] = e->cell_soc(c);
                }
                const auto t1 = std::chrono::steady_clock::now();
                best_ns = std::min(best_ns, std::chrono::duration<double, std::nano>(t1 - t0).count() / d.n);
            }
            // metrics
            double se = 0, se_ss = 0, semin = 0, semean = 0, semax = 0, maxerr = 0; bool div = false; int nss = 0;
            for (int k = 0; k < d.n; ++k) {
                Real tmin = 2, tmax = -1, tmean = 0, emin = 2, emax = -1, emean = 0;
                for (int c = 0; c < kPackCells; ++c) {
                    const double err = double(est[c][k]) - double(d.soc_true[c][k]);
                    if (!std::isfinite(err)) { div = true; continue; }
                    se += err * err; maxerr = std::max(maxerr, std::fabs(err));
                    if (k >= d.n / 2) { se_ss += err * err; ++nss; if (std::fabs(err) > 0.3) div = true; }
                    tmin = std::min(tmin, d.soc_true[c][k]); tmax = std::max(tmax, d.soc_true[c][k]); tmean += d.soc_true[c][k] / kPackCells;
                    emin = std::min(emin, est[c][k]); emax = std::max(emax, est[c][k]); emean += est[c][k] / kPackCells;
                }
                semin += sq(double(emin - tmin)); semax += sq(double(emax - tmax)); semean += sq(double(emean - tmean));
            }
            const double rmse = std::sqrt(se / (double(d.n) * kPackCells)), rmse_ss = std::sqrt(se_ss / std::max(nss, 1));
            summary << e->name() << ',' << e->group() << ',' << profile << ',' << sc.name << ',' << seed << ',' << rmse << ',' << std::sqrt(semin / d.n) << ','
                    << std::sqrt(semean / d.n) << ',' << std::sqrt(semax / d.n) << ',' << maxerr << ',' << rmse_ss << ',' << (div ? 1 : 0) << ',' << best_ns << ',' << e->state_bytes() << ',' << e->messages_per_step() << '\n';
            summary.flush();
            std::printf("[pack|%s|%-15s|s%d] %-22s rmse_cells=%.4f rmse_min=%.4f ss=%.4f max=%.3f div=%d %8.0f ns/step %zu B\n", profile.c_str(), sc.name.c_str(), seed, e->name(), rmse, std::sqrt(semin / d.n), rmse_ss, maxerr, div ? 1 : 0, best_ns, e->state_bytes());
            if (timeseries && seed == 1) {
                std::ofstream ts(out + "/ts/pack__" + e->name() + "__" + profile + "__" + sc.name + "__s1.csv");
                ts << "t,i_meas,soc_true_min,soc_true_mean,soc_true_max,soc_est_min,soc_est_mean,soc_est_max";
                for (int c = 0; c < kPackCells; ++c) ts << ",soc_true_" << c << ",soc_est_" << c;
                ts << '\n';
                for (int k = 0; k < d.n; k += 10) {
                    Real tmin = 2, tmax = -1, tmean = 0, emin = 2, emax = -1, emean = 0;
                    for (int c = 0; c < kPackCells; ++c) { tmin = std::min(tmin, d.soc_true[c][k]); tmax = std::max(tmax, d.soc_true[c][k]); tmean += d.soc_true[c][k] / kPackCells; emin = std::min(emin, est[c][k]); emax = std::max(emax, est[c][k]); emean += est[c][k] / kPackCells; }
                    ts << d.t[k] << ',' << d.i_meas[k] << ',' << tmin << ',' << tmean << ',' << tmax << ',' << emin << ',' << emean << ',' << emax;
                    for (int c = 0; c < kPackCells; ++c) ts << ',' << d.soc_true[c][k] << ',' << est[c][k];
                    ts << '\n';
                }
            }
        }
    }
    return 0;
}
