// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  examples/battery_soc_quickstart.cpp — estimate the state of charge of a cell
//  on a synthetic eVTOL mission with three estimators from three families,
//  directly through the generic filter interface (no benchmark harness):
//    * Ekf<EcmModel>            — the workhorse
//    * IaeAkf<EcmModel>         — innovation-based adaptive EKF (rank 2 of the benchmark)
//    * LuenbergerObserver<EcmModel> — fixed-gain observer (pole placement)
//  The dataset generator of the benchmark provides the mission and the plant.
//
//  Run:  ./build/example_battery [init_error|current_bias|combined]
// =============================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include "estkit/battery/datasets.hpp"
#include "estkit/battery/ecm_model.hpp"
#include "estkit/filters/ekf.hpp"
#include "estkit/filters/iae_akf.hpp"
#include "estkit/observers/luenberger.hpp"
using namespace estkit;

template <class Filter>
static void run(const char* name, Filter& flt, const Dataset& d) {
    const EcmModel m(d.est_cfg);
    flt.init(m.x0(d.est_cfg), m.P0(d.est_cfg));
    Real se = 0, se_ss = 0; int n_ss = 0; bool have_prev = false; Vec<2> u_prev;
    for (int k = 0; k < d.n; ++k) {
        const Vec<2> u = EcmModel::u_of(d.i_meas[k], d.T_meas[k]);
        if (have_prev) flt.predict(u_prev);
        Vec<1> y; y[0] = d.v_meas[k];
        flt.update(y, u);
        const Real e = flt.x()[EcmModel::IZ] - d.soc_true[k];
        se += e * e; if (k >= d.n / 2) { se_ss += e * e; ++n_ss; }
        u_prev = u; have_prev = true;
    }
    std::printf("  %-12s SOC RMSE = %5.2f %%   steady-state = %5.2f %%   (%zu bytes)\n",
                name, 100 * std::sqrt(se / d.n), 100 * std::sqrt(se_ss / n_ss), sizeof(flt));
}

int main(int argc, char** argv) {
    const std::string scen = argc > 1 ? argv[1] : "nominal";
    Scenario sc;
    for (const Scenario& s : standard_scenarios()) if (s.name == scen) sc = s;
    const Dataset d = make_dataset("ecm", mission_evtol(), sc, /*seed=*/1);
    std::printf("eVTOL mission, ECM plant, scenario '%s', %d samples at %.0f Hz\n", sc.name.c_str(), d.n, 1 / d.dt);
    Ekf<EcmModel> ekf{EcmModel(d.est_cfg)};
    IaeAkf<EcmModel> akf{EcmModel(d.est_cfg)};
    LuenbergerObserver<EcmModel> lue{EcmModel(d.est_cfg)};
    lue.opts.design = obs::Design::PolePlacement; lue.opts.set_poles_tau(Real(60), d.dt); lue.opts.max_gain = Real(0.3);
    lue.opts.q_extra[EcmModel::IZ] = Real(1e-4);
    run("EKF", ekf, d);
    run("IAE-AKF", akf, d);
    run("Luenberger", lue, d);
    return 0;
}
