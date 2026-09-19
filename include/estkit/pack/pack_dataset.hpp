// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/pack/pack_dataset.hpp — multi-cell (series string) benchmark data.
//
//  A module of NC cells in series shares one current; each cell has its own
//  parameters drawn from a manufacturing spread (capacity, resistance, initial
//  SOC) and its own thermal environment (a gradient across the module), and
//  is measured by its own voltage channel.  The pack-level quantities a BMS
//  must report are the weakest-cell SOC (usable discharge), the strongest-cell
//  SOC (usable charge) and the mean SOC (energy accounting) — Plett 2016 Vol. II
//  Ch. 4; Plett 2009 (bar-delta filtering).
// =============================================================================
#pragma once
#include <string>
#include <vector>
#include "../battery/datasets.hpp"

namespace estkit {

constexpr int kPackCells = 12;

struct PackScenario {
    std::string name = "nominal";
    Real spread_Q = Real(0.02);      // relative std of capacity
    Real spread_R0 = Real(0.08);     // relative std of R0
    Real spread_z0 = Real(0.015);    // std of initial SOC imbalance
    Real spread_T = Real(1.5);       // K, std of per-cell ambient offsets (thermal gradient)
    Real z0_est_offset = Real(0);    // estimator initial SOC error (all cells)
    Real current_bias = Real(0.02), current_gain = Real(1.005);
    Real sigma_v = Real(0.002), sigma_i = Real(0.05), sigma_T = Real(0.2);
    Real sigma_v2 = Real(-1), t_noise_step = Real(600);
    int weak_cell = -1;              // index of a weak cell (extra ageing)
    Real weak_capacity_loss = Real(0.12), weak_R_scale = Real(1.3);
    Real T_amb = kTref;
};

inline std::vector<PackScenario> pack_scenarios() {
    std::vector<PackScenario> s;
    PackScenario a; a.name = "nominal"; s.push_back(a);
    a = PackScenario(); a.name = "init_error"; a.z0_est_offset = Real(-0.35); s.push_back(a);
    a = PackScenario(); a.name = "current_bias"; a.current_bias = Real(0.25); a.current_gain = Real(1.02); s.push_back(a);
    a = PackScenario(); a.name = "weak_cell"; a.weak_cell = 5; s.push_back(a);
    a = PackScenario(); a.name = "noise_step"; a.sigma_v2 = Real(0.02); s.push_back(a);
    a = PackScenario(); a.name = "large_imbalance"; a.spread_z0 = Real(0.05); a.spread_Q = Real(0.05); s.push_back(a);
    return s;
}

struct PackDataset {
    std::string profile, scenario;
    uint64_t seed = 0; Real dt = Real(0.1); int n = 0;
    std::vector<Real> t, i_true, i_meas;
    std::vector<std::vector<Real>> v_true, v_meas, T_true, T_meas, soc_true;   // [cell][k]
    std::vector<Real> Q_true;                                                  // [cell]
    EstimatorConfig est_cfg;   // nominal (pack-average) configuration handed to estimators
};

inline PackDataset make_pack_dataset(const MissionProfile& mp, const PackScenario& sc, uint64_t seed, Real dt = Real(0.1)) {
    PackDataset d; d.profile = mp.name; d.scenario = sc.name; d.seed = seed; d.dt = dt;
    Rng rng(seed * 7919 + 17), rng_sens(seed * 104729 + 3), rng_cell(seed * 31337 + 11);
    std::vector<Real> i_cmd; std::vector<int> seg;
    const CellParams nominal = CellParams::nominal();
    generate_current(mp, dt, nominal.Q_Ah, rng, i_cmd, seg);
    d.n = int(i_cmd.size());
    // build the cells
    std::vector<EcmPlant> cells; std::vector<Real> z0(kPackCells);
    for (int c = 0; c < kPackCells; ++c) {
        CellParams p = nominal;
        p.Q_Ah *= Real(1) + sc.spread_Q * rng_cell.normal();
        const Real rs = Real(1) + sc.spread_R0 * rng_cell.normal();
        p.R0 *= rs; p.R1 *= rs; p.R2 *= rs;
        EcmPlant cell(p); cell.enable_aging = true;
        z0[c] = clampr(mp.z0 + sc.spread_z0 * rng_cell.normal(), Real(0.05), Real(1));
        const Real Tc = sc.T_amb + sc.spread_T * rng_cell.normal();
        cell.reset(z0[c], Tc, Tc);
        if (c == sc.weak_cell) { cell.set_capacity_loss(sc.weak_capacity_loss); cell.p.R0 *= sc.weak_R_scale; cell.p.R1 *= sc.weak_R_scale; }
        cells.push_back(cell);
        d.Q_true.push_back(cell.capacity_Ah());
    }
    d.t.resize(d.n); d.i_true.resize(d.n); d.i_meas.resize(d.n);
    d.v_true.assign(kPackCells, std::vector<Real>(d.n)); d.v_meas = d.v_true; d.T_true = d.v_true; d.T_meas = d.v_true; d.soc_true = d.v_true;
    for (int k = 0; k < d.n; ++k) {
        const Real t = k * dt, i = i_cmd[k];
        d.t[k] = t; d.i_true[k] = i;
        Real im = sc.current_gain * i + sc.current_bias + sc.sigma_i * rng_sens.normal();
        d.i_meas[k] = std::round(im / Real(0.01)) * Real(0.01);
        const Real sv = (sc.sigma_v2 > 0 && t >= sc.t_noise_step) ? sc.sigma_v2 : sc.sigma_v;
        for (int c = 0; c < kPackCells; ++c) {
            const Real v = cells[c].voltage(i);
            d.v_true[c][k] = v; d.soc_true[c][k] = cells[c].soc(); d.T_true[c][k] = cells[c].temperature();
            d.v_meas[c][k] = std::round((v + sv * rng_sens.normal()) / Real(0.001)) * Real(0.001);
            d.T_meas[c][k] = cells[c].temperature() + sc.sigma_T * rng_sens.normal();
            cells[c].step(i, dt);
        }
    }
    EstimatorConfig c; c.params = nominal; c.dt = dt; c.z0 = clampr(mp.z0 + sc.z0_est_offset, Real(0.02), Real(1)); c.sigma_z0 = Real(0.1);
    c.sigma_v = sc.sigma_v; c.sigma_i = sc.sigma_i; c.sigma_T = sc.sigma_T; c.T0 = sc.T_amb; c.seed = seed;
    d.est_cfg = c;
    return d;
}

}  // namespace estkit
