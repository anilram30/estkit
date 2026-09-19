// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  bench_cell — cell-level SOC estimator benchmark
//
//  Usage:
//    bench_cell [--plant ecm|spm|all] [--profile evtol|fixed_wing|hover_hold|ground_charge|all]
//               [--scenario NAME|all] [--only NAME[,NAME...]] [--group NAME] [--seeds K]
//               [--out DIR] [--timeseries 0|1] [--ts-decimate N] [--ts-seed S|0] [--reps R] [--list]
//
//  Outputs (appended):  <out>/summary_<plant>.csv   one row per (estimator, profile, scenario, seed)
//                       <out>/ts/<plant>__<estimator>__<profile>__<scenario>__s<seed>.csv (decimated time series)
// =============================================================================
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/battery/datasets.hpp"
#include "estkit/battery/metrics.hpp"

using namespace estkit;

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out; std::stringstream ss(s); std::string it;
    while (std::getline(ss, it, sep)) if (!it.empty()) out.push_back(it);
    return out;
}
static void mkdirp(const std::string& p) { std::error_code ec; std::filesystem::create_directories(p, ec); }
static bool file_empty(const std::string& p) { std::error_code ec; return !std::filesystem::exists(p, ec) || std::filesystem::file_size(p, ec) == 0; }

int main(int argc, char** argv) {
    std::string plant = "ecm", profile = "evtol", scenario = "all", only, group, out = "results";
    int seeds = 1, reps = 2, ts_decimate = 10, ts_seed = 1; bool timeseries = true, list_only = false;
    for (int a = 1; a < argc; ++a) {
        std::string k = argv[a]; auto val = [&]() { return std::string(a + 1 < argc ? argv[++a] : ""); };
        if (k == "--plant") plant = val(); else if (k == "--profile") profile = val(); else if (k == "--scenario") scenario = val();
        else if (k == "--only") only = val(); else if (k == "--group") group = val(); else if (k == "--seeds") seeds = std::stoi(val());
        else if (k == "--out") out = val(); else if (k == "--timeseries") timeseries = std::stoi(val()) != 0; else if (k == "--ts-decimate") ts_decimate = std::stoi(val());
        else if (k == "--reps") reps = std::stoi(val()); else if (k == "--list") list_only = true; else if (k == "--ts-seed") ts_seed = std::stoi(val());
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    EstimatorList ests = make_all_estimators();
    if (list_only) { for (auto& e : ests) std::printf("%-28s %s\n", e->name(), e->group()); return 0; }
    std::set<std::string> only_set; for (auto& s : split(only, ',')) only_set.insert(s);

    std::vector<std::string> plants = (plant == "all") ? std::vector<std::string>{"ecm", "spm"} : std::vector<std::string>{plant};
    std::vector<std::string> profiles = (profile == "all") ? std::vector<std::string>{"evtol", "fixed_wing", "hover_hold", "ground_charge"} : std::vector<std::string>{profile};
    std::vector<Scenario> scenarios;
    for (auto& s : standard_scenarios()) if (scenario == "all" || s.name == scenario) scenarios.push_back(s);
    if (scenarios.empty()) { std::fprintf(stderr, "unknown scenario %s\n", scenario.c_str()); return 2; }

    mkdirp(out); if (timeseries) mkdirp(out + "/ts");
    for (const auto& pl : plants) {
        const std::string summary_path = out + "/summary_" + pl + ".csv";
        const bool need_header = file_empty(summary_path);
        std::ofstream summary(summary_path, std::ios::app);
        if (need_header) summary << "plant,estimator,group,profile,scenario,seed,rmse,rmse_ss,mae,max_err,final_err,conv_time,diverged,rmse_v,ns_per_step,bytes,capacity_err_end,bound_violation,bound_width,real_type\n";
        for (const auto& pr : profiles) {
            const MissionProfile mp = mission_by_name(pr);
            for (const auto& sc : scenarios) {
                if (pl == "spm" && (sc.preisach || sc.aged_capacity_loss > 0)) continue;   // not defined for the SPM plant
                for (int seed = 1; seed <= seeds; ++seed) {
                    const Dataset d = make_dataset(pl, mp, sc, uint64_t(seed));
                    for (auto& e : ests) {
                        if (!only_set.empty() && !only_set.count(e->name())) continue;
                        if (!group.empty() && group != e->group()) continue;
                        std::vector<Real> soc_est(d.n), v_pred(d.n), lo(d.n, kNaN), hi(d.n, kNaN), cap(d.n, kNaN), sstd(d.n, kNaN);
                        double best_ns = 1e300;
                        for (int rep = 0; rep < reps; ++rep) {
                            const auto t0 = std::chrono::steady_clock::now();
                            if (e->is_batch()) {
                                auto* b = dynamic_cast<BatchEstimator*>(e.get());
                                b->reset(d.est_cfg);
                                b->run(d.i_meas.data(), d.v_meas.data(), d.T_meas.data(), d.n, soc_est.data(), v_pred.data());
                            } else {
                                e->reset(d.est_cfg);
                                for (int k = 0; k < d.n; ++k) {
                                    e->step(d.i_meas[k], d.v_meas[k], d.T_meas[k]);
                                    soc_est[k] = e->soc(); v_pred[k] = e->voltage_pred();
                                    if (rep == 0) { lo[k] = e->soc_lower(); hi[k] = e->soc_upper(); cap[k] = e->capacity_Ah(); sstd[k] = e->soc_std(); }
                                }
                            }
                            const auto t1 = std::chrono::steady_clock::now();
                            best_ns = std::min(best_ns, std::chrono::duration<double, std::nano>(t1 - t0).count() / d.n);
                        }
                        RunMetrics m = compute_metrics(d.soc_true, soc_est, d.v_true, v_pred, d.dt);
                        m.ns_per_step = best_ns; m.bytes = e->state_bytes();
                        if (std::isfinite(cap[d.n - 1])) {   // mean capacity error over the last 10 % of the run
                            double s = 0; int c = 0;
                            for (int k = d.n - d.n / 10; k < d.n; ++k) if (std::isfinite(cap[k])) { s += double(cap[k]) - double(d.Q_true[k]); ++c; }
                            m.capacity_err_end = c ? s / c : kNaN;
                        }
                        if (std::isfinite(lo[d.n - 1]) && std::isfinite(hi[d.n - 1])) {
                            int viol = 0; double w = 0;
                            for (int k = 0; k < d.n; ++k) { if (d.soc_true[k] < lo[k] || d.soc_true[k] > hi[k]) ++viol; w += double(hi[k] - lo[k]); }
                            m.bound_violation = double(viol) / d.n; m.bound_width = w / d.n;
                        }
                        summary << pl << ',' << e->name() << ',' << e->group() << ',' << pr << ',' << sc.name << ',' << seed << ','
                                << m.rmse << ',' << m.rmse_ss << ',' << m.mae << ',' << m.max_err << ',' << m.final_err << ',' << m.conv_time << ','
                                << (m.diverged ? 1 : 0) << ',' << m.rmse_v << ',' << m.ns_per_step << ',' << m.bytes << ',' << m.capacity_err_end << ','
                                << m.bound_violation << ',' << m.bound_width << ',' << (sizeof(Real) == 4 ? "float" : "double") << '\n';
                        summary.flush();
                        std::printf("[%s|%s|%-14s|s%d] %-26s rmse=%7.4f ss=%7.4f max=%6.3f conv=%7.1fs div=%d  %8.0f ns/step\n",
                                    pl.c_str(), pr.c_str(), sc.name.c_str(), seed, e->name(), m.rmse, m.rmse_ss, m.max_err, m.conv_time, m.diverged ? 1 : 0, m.ns_per_step);
                        if (timeseries && (seed == ts_seed || ts_seed == 0)) {   // --ts-seed 0 writes every seed
                            std::ofstream ts(out + "/ts/" + pl + "__" + e->name() + "__" + pr + "__" + sc.name + "__s" + std::to_string(seed) + ".csv");
                            ts << "t,i_meas,v_meas,T_true,soc_true,soc_est,soc_std,v_pred,soc_lower,soc_upper,capacity_est,capacity_true,v_true\n";
                            for (int k = 0; k < d.n; k += ts_decimate)
                                ts << d.t[k] << ',' << d.i_meas[k] << ',' << d.v_meas[k] << ',' << d.T_true[k] << ',' << d.soc_true[k] << ',' << soc_est[k] << ',' << sstd[k] << ','
                                   << v_pred[k] << ',' << lo[k] << ',' << hi[k] << ',' << cap[k] << ',' << d.Q_true[k] << ',' << d.v_true[k] << '\n';
                        }
                    }
                }
            }
        }
    }
    return 0;
}
