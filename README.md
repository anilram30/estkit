<!-- Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0 -->
<div align="center">

# estkit

**A header-only C++17 library and benchmark of 86 state estimators and observers
for battery-management systems in electric aircraft — with aerospace attitude-estimation heritage.**

[![ci](https://github.com/anilram30/estkit/actions/workflows/ci.yml/badge.svg)](https://github.com/anilram30/estkit/actions/workflows/ci.yml)
[![cross-validation](https://github.com/anilram30/estkit/actions/workflows/validate.yml/badge.svg)](https://github.com/anilram30/estkit/actions/workflows/validate.yml)
[![pages](https://github.com/anilram30/estkit/actions/workflows/pages.yml/badge.svg)](https://anilram30.github.io/estkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![license](https://img.shields.io/badge/code-Apache--2.0-green.svg)](LICENSE)
[![report](https://img.shields.io/badge/report-958%20pp.-red.svg)](https://github.com/anilram30/estkit/releases/latest)

### **[Website & interactive benchmark ↗](https://anilram30.github.io/estkit)** · **[Estimator catalogue](docs/CATALOGUE.md)** · **[Report (PDF)](https://github.com/anilram30/estkit/releases/latest)** · **[References](docs/REFERENCES.md)**

</div>

---

**86 estimators · 12 families · 2 truth plants · 12 fault scenarios · ~9,000 benchmark runs · 0 runtime dependencies · float32-ready · Cortex-M7 verified**

estkit implements essentially every state estimator in the control and battery-management
literature — from the Kalman filter to moving-horizon estimation, particle filters, deterministic
observers and offline smoothers — against **one model-agnostic interface**, with **no dynamic
memory and no exceptions**, and then measures every one of them on the same missions, faults and
cells. The battery problem is one instance of the interface; a drone navigation model and an
airframe attitude model are others, and the same `Ekf<M>` estimates all three.

## Why it exists

A battery-management engineer is expected to know *which* estimator to use, *why*, and *at what
cost* on a microcontroller. estkit answers that empirically rather than by literature summary: it
implements all of them in one code base, runs them on one benchmark, and documents each in one
chapter of a 958-page report. Every algorithm is an original implementation made from its **primary
publication**, cited in the header of its source file and on its [catalogue page](docs/CATALOGUE.md).

## Use any filter on your own model — in twenty lines

A *model* is any class that provides a state transition, a measurement and the two noise
covariances. Jacobians, constraints and tunable parameters are optional and supplied numerically
when absent.

```cpp
#include "estkit/filters/ukf.hpp"
using namespace estkit;

struct MyModel {
    static constexpr int NX = 4, NU = 1, NY = 2;
    Vec<NX>    f(const Vec<NX>& x, const Vec<NU>& u) const;   // x_{k+1}
    Vec<NY>    h(const Vec<NX>& x, const Vec<NU>& u) const;   // y_k
    Mat<NX,NX> Q(const Vec<NX>&, const Vec<NU>&) const;
    Mat<NY,NY> R(const Vec<NX>&, const Vec<NU>&) const;
    // optional: F(x,u), H(x,u), B(x,u), constrain(x), NP/params()/set_params()
};

Ukf<MyModel> ukf{MyModel{}};
ukf.init(x0, P0);
for (...) { ukf.predict(u_prev); ukf.update(y, u); auto x = ukf.x(); }
```

The same four calls drive every estimator in the library, from a Luenberger observer to a
500-particle filter. See [`examples/`](examples) for a runnable drone (range-only) example, a
battery SOC example, and a Cortex-M7 build.

## The 86 estimators

Ten cell-level families plus pack-level and attitude estimators — the full table with the primary
reference and the benchmark result for each is in **[docs/CATALOGUE.md](docs/CATALOGUE.md)** and on
the **[website](https://anilram30.github.io/estkit/estimators.html)**.

| Family | Members |
|---|---|
| **Baselines** | Coulomb counting, OCV inversion, OCV-corrected Coulomb counting |
| **Classical & sigma-point Kalman** | LKF, EKF, IEKF, SO-EKF, UKF, CKF, CKF-5, CDKF, GHKF, SR-EKF, SR-UKF, U-D, EIF, SCKF, IPLF |
| **Adaptive Kalman** | Sage–Husa, innovation-based (IAE), variational Bayesian, strong-tracking, fading-memory, fuzzy, IMM, MMAE |
| **Robust & embedded Kalman** | Schmidt, H∞, Huber, maximum-correntropy, Student's-t, constrained, reduced-order, SVSF |
| **Deterministic observers** | Luenberger, steady-state KF, PI, sliding-mode, super-twisting, high-gain, extended-state, unknown-input, disturbance, adaptive, interval |
| **Joint / dual state–parameter (SOH)** | joint EKF/UKF, dual EKF/UKF, RLS-EKF, AWTLS capacity |
| **Particle / ensemble / Gaussian-sum** | bootstrap, fast, regularised, auxiliary, Rao–Blackwellised and unscented particle filters, EnKF, ETKF, GSF |
| **Data-driven hybrids** | direct neural regression, NN-EKF, ELM-RLS-EKF |
| **Moving-horizon estimation** | MHE (converged), MHE-RTI (real-time iteration) |
| **Smoothers (offline)** | extended and unscented RTS, fixed-lag, batch nonlinear least squares |
| **Pack-level & distributed** | decentralised EKF/UKF/reduced, bar-delta, federated, consensus, information-fusion |
| **Attitude (AHRS)** | complementary, Madgwick, Mahony (+ gated), multiplicative EKF, invariant EKF, TRIAD, QUEST |

## The benchmark

Two truth plants generate the data, both deliberately richer than the estimators' model:

- an **enhanced equivalent-circuit cell** — lumped thermal state, Arrhenius impedance, one-state and
  Preisach hysteresis, cycle/calendar ageing;
- an **electrochemical single-particle model** built from the published LG M50 electrode data
  (Chen et al. 2020) and validated against PyBaMM.

Four missions (eVTOL, fixed-wing, hover, ground-charge), twelve fault and operating scenarios
(initial-SOC error, current bias & gain, noise steps, outliers, cold/hot, ageing, parameter
mismatch, sensor dropout, Preisach hysteresis, and a combined worst case) and three noise seeds
produce the cell-level data set; a twelve-cell module, a 120-mission ageing study and a synthetic
instrumented flight produce the pack, state-of-health and attitude data sets — **~9,000 runs** in
[`results/final`](results).

**A few headlines** (steady-state SOC RMSE, % of capacity):

- The nominal problem is solved by any tuned closed-loop estimator; the field is separated only by faults.
- The plain **EKF** fails the combined-fault scenario (**12.44 %**) by covariance collapse, while the **UKF** (1.19 %), the adaptive filters and moving-horizon estimation stay under ~1 %.
- On the electrochemical plant the ranking **inverts**: every converged Kalman filter parks on a ~4 % bias with a clean innovation, while the **fading-memory EKF (0.36 % vs the EKF's 3.73 %)** and the fixed-gain observers lead.
- At module level, averaging inside the filter (**bar-delta**, **information fusion**) halves the per-cell error of twelve independent EKFs at a fraction of the cost.
- Single precision is enough for almost everything — but not universally: under the combined fault
  the sigma-point filters (**UKF**, **SR-UKF**, unscented RTS) lose covariance positive-definiteness in
  float32 on some toolchains (macOS Clang, MSVC) while holding on GCC 13 and Clang 18; the EKF-family
  and observer filters are unaffected. The best twelve run within 5× the cost of the EKF.

Explore it interactively: **[the benchmark explorer ↗](https://anilram30.github.io/estkit/benchmark.html)**.

## Cross-validated against reference libraries

The Kalman-type filters are cross-checked against **FilterPy**, the electrochemical plant against
**PyBaMM**, and the attitude filters against **ahrs** — each an independent reference implementation
run on the exported benchmark data, in CI:

| Reference | Quantity | Agreement |
|---|---|---|
| FilterPy 1.4 | EKF / UKF / RTS SOC & covariance | 1e-15 / 2.5e-12 / 7e-9 |
| PyBaMM 26 | SPM terminal voltage, 0.5–2C | 0.6–3.2 mV RMSE |
| ahrs 0.4 | Madgwick / Mahony attitude | 0.0014° |

These libraries are used only as references and are **not** part of estkit
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Embedded by construction

No heap allocation, no exceptions, fixed-size linear algebra with compile-time dimensions, a single
`Real` type that is `double` on a desktop and `float` on a microcontroller, warning-free under
`-Wall -Wextra -Wpedantic -Werror` on GCC 13 and Clang 18, builds and passes its tests on MSVC, and
it **cross-compiles and links for a Cortex-M7** (`arm-none-eabi-g++ -mcpu=cortex-m7 -mfpu=fpv5-d16
-fno-exceptions -fno-rtti`, float build) — all verified in [CI](.github/workflows/ci.yml).

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build                                      # unit tests + examples
./build/example_drone                                       # EKF/UKF/PF on a drone range model
./build/example_battery combined                            # SOC estimation on a hard scenario
./build/bench_cell --list                                   # every cell-level estimator
./build/bench_cell --plant ecm --profile evtol --scenario all --only EKF,UKF --out results/quick

scripts/run_all.sh 3                                        # the complete benchmark -> results/final
python3 scripts/catalogue.py                                # regenerate the catalogue & references
python3 scripts/site/build_site.py --out site               # regenerate the website
cd report && latexmk -pdf main.tex                          # build the 958-page report
```

Single-precision build: `cmake -S . -B build_f -DESTKIT_FLOAT=ON`.

## Layout

```
include/estkit/      the header-only library (core, filters, observers, parameter,
                     particle, learning, optimization, smoothers, battery, pack, attitude)
examples/            runnable examples (drone, battery, Cortex-M7)
benchmarks/          benchmark executables (bench_cell/pack/imu/soh) and estimator registries
tools/               ECM identification on the SPM plant, validation/model exporters
tests/               unit tests + the CI accuracy-regression reference
scripts/             run_all.sh, figure/table generators, validation, catalogue, site generator
results/final/       the recorded benchmark run (every table & figure is generated from it)
report/              LaTeX sources of the 958-page report and its figures
docs/                ESTIMATOR_API.md, CATALOGUE.md, REFERENCES.md, PROVENANCE.md
.github/workflows/   CI (build matrix, sanitizers, Cortex-M7, regression), validation, pages, release
```

## Licensing

- **Code** (`include/`, `benchmarks/`, `examples/`, `tests/`, `tools/`, `scripts/`) — **Apache License 2.0**, © 2026 Sreeram Anil ([LICENSE](LICENSE), [NOTICE](NOTICE)).
- **Report, figures and generated data** (`report/`, `results/`, the website) — **CC BY 4.0** ([report/LICENSE-CC-BY-4.0.txt](report/LICENSE-CC-BY-4.0.txt)).
- Every algorithm is implemented from the cited literature; no third-party source code is incorporated. Provenance of every component: [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Citing

If you use estkit, its benchmark or its report, please cite it — see [CITATION.cff](CITATION.cff).

## A note on tooling

AI assistance was used in preparing this project. All engineering decisions, algorithm
implementations, benchmark design and results are the author's, and every estimator is verified
against its primary reference and, where a reference implementation exists, cross-validated in CI.
