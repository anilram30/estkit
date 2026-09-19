// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/learning/train_data.hpp — OFFLINE training-set generation for the
//  data-driven estimators of the `learning` family.
//
//  This header is NOT part of the deployed (embedded) code path: it pulls in the
//  simulation plant (battery/datasets.hpp) and uses std::vector/std::string, and
//  is only compiled into the host-side registry translation unit where the
//  networks are fitted once, before any estimator runs.  The deployed objects
//  (Mlp, NnDirectSoc, NnEcmModel) contain nothing but fixed-size arrays.
//
//  Benchmark hygiene
//    The TEST mission of the benchmark is `evtol`.  Training is restricted to the
//    `fixed_wing` and `hover_hold` missions; `training_datasets()` refuses to
//    emit an eVTOL dataset (see `kForbiddenProfile`) so that the train/test split
//    cannot be violated by a later edit.  Only the *measured* signals
//    (i_meas, v_meas, T_meas) and the ground-truth SOC are used as
//    inputs/targets, exactly the information a laboratory characterisation
//    campaign would provide (Chemali et al. 2018).
//
//  References
//    * Chemali, E., Kollmeyer, P. J., Preindl, M. & Emadi, A. (2018).
//      State-of-charge estimation of Li-ion batteries using deep neural networks:
//      A machine learning approach. J. Power Sources 400, 242-250.
//    * Yang, X.-G., Liu, T., Ge, S., Rountree, E. & Wang, C.-Y. (2021).
//      Challenges and key requirements of batteries for electric vertical
//      takeoff and landing aircraft. Joule 5(7), 1644-1659.   (mission profiles)
// =============================================================================
#pragma once
#include <string>
#include <vector>
#include "../battery/datasets.hpp"

namespace estkit {

// The mission reserved for testing; never used for training.
inline const char* kForbiddenProfile() { return "evtol"; }

// Look a standard scenario up by name (falls back to `nominal`).
inline Scenario scenario_by_name(const std::string& name) {
    for (const auto& s : standard_scenarios()) if (s.name == name) return s;
    return Scenario();
}

// -----------------------------------------------------------------------------
//  Specification of a training set: the Cartesian product of mission profiles,
//  operating scenarios and seeds.  `preisach` selects which truth plant variants
//  are simulated:
//      0 = take the scenario's own setting,
//      1 = force the Preisach hysteresis operator on,
//      2 = emit both variants (population average: half the runs have the
//          Preisach cell, half the simple one-state-hysteresis cell).
// -----------------------------------------------------------------------------
struct TrainingSetSpec {
    std::vector<std::string> profiles  = {"fixed_wing", "hover_hold"};
    std::vector<std::string> scenarios = {"nominal", "cold", "hot", "aged_cell"};
    int      n_seeds   = 3;
    uint64_t seed0     = 101;
    int      preisach  = 0;
    int      subsample = 10;     // keep every n-th sample (10 -> 1 Hz at dt = 0.1 s)
};

// Generate the training datasets one at a time and hand each to `fn`
// (a callable `void(const Dataset&)`).  Only one dataset is ever in memory.
template <class Fn>
inline void for_each_training_dataset(const TrainingSetSpec& spec, Fn&& fn) {
    for (const auto& pname : spec.profiles) {
        if (pname == kForbiddenProfile()) continue;          // train/test split guard
        const MissionProfile mp = mission_by_name(pname);
        if (mp.name == kForbiddenProfile()) continue;        // unknown name -> evtol fallback
        for (const auto& sname : spec.scenarios) {
            const Scenario base = scenario_by_name(sname);
            const int nvar = (spec.preisach == 2) ? 2 : 1;
            for (int var = 0; var < nvar; ++var) {
                Scenario sc = base;
                if (spec.preisach == 1) sc.preisach = true;
                else if (spec.preisach == 2) sc.preisach = (var == 0);
                for (int s = 0; s < spec.n_seeds; ++s)
                    fn(make_dataset("ecm", mp, sc, spec.seed0 + uint64_t(31 * s + 7 * var)));
            }
        }
    }
}

}  // namespace estkit
