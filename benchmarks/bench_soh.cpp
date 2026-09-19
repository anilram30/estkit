// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  bench_soh — multi-flight state-of-health (capacity) tracking benchmark
//
//  The true cell ages over F consecutive mission cycles (flight -> rest -> CC-CV
//  recharge -> rest, one cycle per day) according to the cycle + calendar ageing
//  law of the plant (cell.hpp); the estimators run CONTINUOUSLY through all
//  cycles and their capacity estimate is compared with the true capacity at the
//  end of every flight.  Ageing can be accelerated (--accel) to compress the
//  study.  Only estimators that report a capacity take part.
//
//  Usage: bench_soh [--flights F] [--accel A] [--only NAME] [--out DIR] [--profile evtol]
//  Output: <out>/soh_flights.csv (per estimator per flight), <out>/summary_soh.csv
// =============================================================================
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/battery/datasets.hpp"

using namespace estkit;
static std::vector<std::string> split(const std::string& s, char sep) { std::vector<std::string> o; std::stringstream ss(s); std::string it; while (std::getline(ss, it, sep)) if (!it.empty()) o.push_back(it); return o; }
static void mkdirp(const std::string& p) { std::error_code ec; std::filesystem::create_directories(p, ec); }
static bool file_empty(const std::string& p) { std::error_code ec; return !std::filesystem::exists(p, ec) || std::filesystem::file_size(p, ec) == 0; }

// One mission cycle of sensor data generated from a live (ageing) plant.
struct CycleData { std::vector<Real> i_meas, v_meas, T_meas, soc_true, Q_true; std::vector<int> phase; };

static CycleData simulate_cycle(EcmPlant& plant, const MissionProfile& mp, Real dt, Rng& rng, Rng& rng_sens, Real accel) {
    CycleData c;
    std::vector<Real> icmd; std::vector<int> seg; generate_current(mp, dt, plant.p.Q_Ah, rng, icmd, seg);
    auto record = [&](Real i, int phase) {
        const Real v = plant.voltage(i);
        c.i_meas.push_back(std::round((Real(1.005) * i + Real(0.02) + Real(0.05) * rng_sens.normal()) / Real(0.01)) * Real(0.01));
        c.v_meas.push_back(std::round((v + Real(0.002) * rng_sens.normal()) / Real(0.001)) * Real(0.001));
        c.T_meas.push_back(plant.temperature() + Real(0.2) * rng_sens.normal());
        c.soc_true.push_back(plant.soc()); c.Q_true.push_back(plant.capacity_Ah()); c.phase.push_back(phase);
        // accelerated ageing: apply the ageing increment 'accel' times per step
        const Real Qloss_before = plant.capacity_loss();
        plant.step(i, dt);
        if (accel > 1) plant.set_capacity_loss(Qloss_before + accel * (plant.capacity_loss() - Qloss_before));
    };
    for (Real i : icmd) record(i, 0);                                          // flight
    for (int k = 0; k < int(1200 / dt); ++k) record(Real(0), 1);              // rest 20 min
    // CC-CV recharge at 1C to 4.2 V, taper to C/20
    for (int k = 0; k < int(5400 / dt); ++k) {
        Real i = -plant.p.Q_Ah;
        if (plant.voltage(i) > Real(4.20)) { Real lo = 0, hi = -i; for (int it = 0; it < 20; ++it) { Real m = Real(0.5) * (lo + hi); if (plant.voltage(-m) > Real(4.20)) hi = m; else lo = m; } i = -lo; }
        if (-i < Real(0.05) * plant.p.Q_Ah || plant.soc() >= Real(0.999)) break;
        record(i, 2);
    }
    for (int k = 0; k < int(1200 / dt); ++k) record(Real(0), 3);              // rest 20 min
    return c;
}

int main(int argc, char** argv) {
    std::string only, out = "results", profile = "evtol"; int flights = 120; Real accel = Real(3);
    for (int a = 1; a < argc; ++a) {
        std::string k = argv[a]; auto val = [&]() { return std::string(a + 1 < argc ? argv[++a] : ""); };
        if (k == "--flights") flights = std::stoi(val()); else if (k == "--accel") accel = Real(std::stod(val())); else if (k == "--only") only = val();
        else if (k == "--out") out = val(); else if (k == "--profile") profile = val();
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    std::set<std::string> only_set; for (auto& s : split(only, ',')) only_set.insert(s);
    EstimatorList all = make_all_estimators();
    // keep only SOH-capable estimators (those reporting a finite capacity after one step)
    EstimatorList ests;
    {
        EstimatorConfig c; c.params = CellParams::nominal();
        for (auto& e : all) {
            if (e->is_batch()) continue;
            if (!only_set.empty() && !only_set.count(e->name())) continue;
            e->reset(c); e->step(Real(1), Real(4.0), kTref); e->step(Real(1), Real(4.0), kTref);
            if (std::isfinite(e->capacity_Ah())) ests.push_back(std::move(e));
        }
    }
    std::printf("SOH-capable estimators: "); for (auto& e : ests) std::printf("%s ", e->name()); std::printf("\n");
    mkdirp(out);
    const MissionProfile mp = mission_by_name(profile);
    const Real dt = Real(0.1);
    std::ofstream fl(out + "/soh_flights.csv"); fl << "estimator,flight,Q_true,Q_est,Q_err,soc_rmse_flight,ah_throughput\n";
    // the plant ages identically for every estimator: generate the cycles once and keep them in memory (~1 GB would be too much),
    // so instead we re-simulate deterministically per estimator with the same seeds.
    std::vector<double> sum_sq(ests.size(), 0), sum_abs(ests.size(), 0), final_err(ests.size(), 0), sum_soc(ests.size(), 0), ns(ests.size(), 0); std::vector<int> cnt(ests.size(), 0);
    for (size_t ei = 0; ei < ests.size(); ++ei) {
        auto& e = ests[ei];
        EcmPlant plant(CellParams::nominal()); plant.enable_aging = true; plant.reset(mp.z0, kTref, kTref);
        Rng rng(12345), rng_sens(777);
        EstimatorConfig cfg; cfg.params = CellParams::nominal(); cfg.dt = dt; cfg.z0 = mp.z0; cfg.sigma_z0 = Real(0.05); cfg.seed = 1;
        e->reset(cfg);
        long steps = 0; const auto t0 = std::chrono::steady_clock::now();
        for (int f = 1; f <= flights; ++f) {
            // full charge before the flight is guaranteed by the previous cycle; start of study: cell at mp.z0
            CycleData c = simulate_cycle(plant, mp, dt, rng, rng_sens, accel);
            double se = 0; int nf = 0; double qsum = 0; int qn = 0;
            size_t n_flight = 0; for (int ph : c.phase) if (ph == 0) ++n_flight;
            for (size_t k = 0; k < c.i_meas.size(); ++k) {
                e->step(c.i_meas[k], c.v_meas[k], c.T_meas[k]);
                if (c.phase[k] == 0) { const double err = double(e->soc()) - double(c.soc_true[k]); se += err * err; ++nf; }
                // capacity estimate averaged over the last 10 % of the flight phase
                if (c.phase[k] == 0 && k >= size_t(0.9 * double(n_flight))) { const Real q = e->capacity_Ah(); if (std::isfinite(q)) { qsum += q; ++qn; } }
            }
            steps += long(c.i_meas.size());
            const double Qest = qn ? qsum / qn : kNaN, Qtrue = c.Q_true[c.Q_true.size() / 4];
            fl << e->name() << ',' << f << ',' << Qtrue << ',' << Qest << ',' << (Qest - Qtrue) << ',' << std::sqrt(se / std::max(nf, 1)) << ',' << plant.ah_throughput() << '\n';
            if (f > 10) { sum_sq[ei] += sq(Qest - Qtrue); sum_abs[ei] += std::fabs(Qest - Qtrue); ++cnt[ei]; }
            final_err[ei] = Qest - Qtrue; sum_soc[ei] += std::sqrt(se / std::max(nf, 1));
            // next day: calendar ageing during storage, cell relaxes to ambient
            plant.age_calendar(Real(1));
            // top-up to the mission start SOC (fleet practice: charged to z0 before departure)
            if (f % 10 == 0 || f == flights) std::printf("  %-18s flight %3d: Q_true=%.4f Ah  Q_est=%.4f Ah  err=%+.4f  soc_rmse=%.4f\n", e->name(), f, Qtrue, Qest, Qest - Qtrue, std::sqrt(se / std::max(nf, 1)));
        }
        const auto t1 = std::chrono::steady_clock::now();
        ns[ei] = std::chrono::duration<double, std::nano>(t1 - t0).count() / std::max(steps, 1L);
    }
    const std::string sp = out + "/summary_soh.csv"; const bool hdr = file_empty(sp);
    std::ofstream summary(sp, std::ios::app);
    if (hdr) summary << "estimator,group,flights,accel,capacity_rmse_Ah,capacity_mae_Ah,final_capacity_err_Ah,mean_soc_rmse,ns_per_step\n";
    for (size_t ei = 0; ei < ests.size(); ++ei) {
        summary << ests[ei]->name() << ',' << ests[ei]->group() << ',' << flights << ',' << accel << ',' << std::sqrt(sum_sq[ei] / std::max(cnt[ei], 1)) << ',' << sum_abs[ei] / std::max(cnt[ei], 1) << ',' << final_err[ei] << ',' << sum_soc[ei] / flights << ',' << ns[ei] << '\n';
        std::printf("%-18s capacity RMSE (flights 11..F) = %.4f Ah, final err = %+.4f Ah, mean flight SOC RMSE = %.4f, %.0f ns/step\n", ests[ei]->name(), std::sqrt(sum_sq[ei] / std::max(cnt[ei], 1)), final_err[ei], sum_soc[ei] / flights, ns[ei]);
    }
    return 0;
}
