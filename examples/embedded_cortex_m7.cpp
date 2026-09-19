// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  examples/embedded_cortex_m7.cpp — the embedded proof.  This file is compiled
//  and linked for a Cortex-M7 (float32, hard FPU, no exceptions, no RTTI, newlib
//  nano, no OS) by CI:
//
//    arm-none-eabi-g++ -std=c++17 -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard
//        -mthumb -Os -fno-exceptions -fno-rtti -ffunction-sections -fdata-sections
//        -DESTKIT_REAL=float -Iinclude examples/embedded_cortex_m7.cpp
//        --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections -o estkit_m7.elf
//
//  It instantiates the per-cell estimator stack recommended by the report on the
//  equivalent-circuit cell model (adaptive EKF, fading-memory EKF, Luenberger
//  observer, Coulomb-counting monitor, joint EKF for capacity) and steps them on
//  a synthetic current profile.  No heap, no exceptions, no I/O: the resulting
//  ELF's .text/.data/.bss sizes are the static footprint of estkit on the target.
//  On a host it also runs as an ordinary program (returns 0 when every estimate
//  stays finite).
// =============================================================================
#include <cmath>
#include "estkit/battery/ecm_model.hpp"
#include "estkit/filters/ekf.hpp"
#include "estkit/filters/iae_akf.hpp"
#include "estkit/filters/fading_kf.hpp"
#include "estkit/observers/luenberger.hpp"
#include "estkit/parameter/joint_ekf.hpp"
using namespace estkit;

namespace {
EstimatorConfig cfg;                       // nominal cell, dt = 0.1 s
Ekf<EcmModel>                ekf{EcmModel(cfg)};
IaeAkf<EcmModel>             akf{EcmModel(cfg)};
FadingMemoryKf<EcmModel>     fkf{EcmModel(cfg)};
LuenbergerObserver<EcmModel> lue{EcmModel(cfg)};
JointEkf<EcmModel>           jekf{EcmModel(cfg)};
volatile Real sink;                        // keeps the estimates alive for the optimiser

template <class F> bool step_all(F& f, const Vec<2>& u_prev, const Vec<2>& u, Real v, bool have_prev) {
    if (have_prev) f.predict(u_prev);
    Vec<1> y; y[0] = v; f.update(y, u);
    sink = f.x()[0];
    return std::isfinite(f.x()[0]);
}
}  // namespace

int main() {
    const EcmModel m(cfg);
    ekf.init(m.x0(cfg), m.P0(cfg)); akf.init(m.x0(cfg), m.P0(cfg)); fkf.init(m.x0(cfg), m.P0(cfg));
    lue.opts.design = obs::Design::PolePlacement; lue.opts.set_poles_tau(Real(60), cfg.dt); lue.opts.max_gain = Real(0.3);
    lue.init(m.x0(cfg), m.P0(cfg));
    jekf.init(m.x0(cfg), m.P0(cfg));
    // synthetic 10-minute profile: 5 A cruise with a 25 A pulse, voltage from the model itself
    Vec<4> x_true = m.x0(cfg); Vec<2> u_prev; bool have_prev = false; bool ok = true;
    for (int k = 0; k < 6000 && ok; ++k) {
        const Real i = (k > 1800 && k < 2400) ? Real(25) : Real(5);
        const Vec<2> u = EcmModel::u_of(i, kTref);
        const Real v = m.h(x_true, u)[0];
        ok = step_all(ekf, u_prev, u, v, have_prev) && step_all(akf, u_prev, u, v, have_prev) &&
             step_all(fkf, u_prev, u, v, have_prev) && step_all(lue, u_prev, u, v, have_prev) &&
             step_all(jekf, u_prev, u, v, have_prev);
        x_true = m.f(x_true, u); u_prev = u; have_prev = true;
    }
    return ok ? 0 : 1;
}
