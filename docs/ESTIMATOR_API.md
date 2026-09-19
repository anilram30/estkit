<!-- Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0 -->
# estkit — estimator implementation contract

This document is the binding contract for every estimator in the library. Read it fully before
adding a filter/observer. The goal is a **publishable, model-agnostic, header-only C++17 library**
that also serves as a battery-management benchmark.

## 1. Layout

```
include/estkit/core/        linalg.hpp (Mat/Vec, Cholesky, QR, LU, DARE, Ackermann), rng.hpp, model.hpp (model concept + adapters)
include/estkit/filters/     Kalman-type filters (generic templates over a Model)
include/estkit/observers/   deterministic observers (generic templates)
include/estkit/particle/    particle / ensemble / Gaussian-sum methods (generic)
include/estkit/smoothers/   RTS, fixed-lag, batch NLS (generic, batch interface)
include/estkit/learning/    neural / data-driven methods
include/estkit/optimization/ moving-horizon estimation
include/estkit/battery/     cell model (EcmModel), plants, datasets, metrics, CellEstimator adapter, registry
include/estkit/pack/        pack-level (multi-cell) estimators
include/estkit/attitude/    IMU attitude filters (complementary, Madgwick, Mahony, MEKF, invariant EKF)
benchmarks/registry_<family>.cpp   ONE translation unit per family registers its estimators (see §5)
```

Rules: **header-only**, namespace `estkit`, no heap allocation inside estimators (fixed-size
`Mat<R,C>` / `Vec<N>` / plain arrays only — `std::vector` is allowed only in *batch* smoothers and
the host-side benchmark), no exceptions, no `std::function` inside the hot path, `Real` type
(`estkit::Real`, double by default, float with `-DESTKIT_REAL=float`) — never write `double`
literals that would break the float build (use `Real(0.5)`). Compiles with
`-Wall -Wextra -Wpedantic -Werror` (g++ 13 and clang 18). Every file starts with a header comment
containing the algorithm's exact literature references (authors, year, title, venue, volume/pages).

## 2. The model concept (core/model.hpp)

```cpp
struct MyModel {
  static constexpr int NX = ..., NU = ..., NY = ...;
  Vec<NX>    f(const Vec<NX>& x, const Vec<NU>& u) const;   // x_{k+1} = f(x_k,u_k)
  Vec<NY>    h(const Vec<NX>& x, const Vec<NU>& u) const;   // y_k = h(x_k,u_k)
  Mat<NX,NX> Q(const Vec<NX>& x, const Vec<NU>& u) const;   // process noise covariance
  Mat<NY,NY> R(const Vec<NX>& x, const Vec<NU>& u) const;   // measurement noise covariance
  // optional (use the free functions jac_F/jac_H/jac_B/constrain from model.hpp which fall back to numerics):
  Mat<NX,NX> F(x,u); Mat<NY,NX> H(x,u); Mat<NX,NU> B(x,u); Vec<NX> constrain(Vec<NX> x);
  // optional parameters for joint/dual estimation:
  static constexpr int NP; Vec<NP> params() const; void set_params(const Vec<NP>&);
};
```

The battery model `EcmModel` (battery/ecm_model.hpp) has NX=4 (`[z, v1, v2, h]`), NU=2
(`u = [i, T]`, current in A discharge-positive, temperature in K), NY=1 (terminal voltage), NP=2
(`[Q_Ah, R0]`). Index constants: `EcmModel::IZ, IV1, IV2, IH`, `EcmModel::UI, UT`. Extra
helpers: `m.ocv.ocv(z,T)`, `m.ocv.docv_dz(z,T)`, `m.R0(T)`, `m.a1(T)`, `m.a2(T)`, `m.p` (CellParams),
`m.dt`, `m.dh_di(T)`, `EcmModel::u_of(i,T)`, `m.x0(cfg)`, `m.P0(cfg)`.

**Never** hard-code battery specifics inside a generic filter. Access the model only through the
concept. If an algorithm needs a linear system (observers), obtain it from `jac_F`, `jac_H` at the
current estimate (gain scheduling) or at `init()` (fixed gain).

## 3. Generic filter interface (what `CellEstimator<Filter>` expects)

```cpp
template <class Model>
class MyFilter {
public:
  static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
  using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
  struct Options { /* all tuning knobs with sensible defaults */ } opts;   // public, settable before init()
  explicit MyFilter(const Model& m);
  void init(const X& x0, const Mat<NX,NX>& P0);      // P0 may be ignored by deterministic observers
  void predict(const U& u);                          // time update with u_{k-1}
  Y    predict_measurement(const U& u) const;        // h(x^-, u_k) BEFORE the update (used for voltage-residual metrics)
  void update(const Y& y, const U& u);               // measurement update with y_k, u_k
  const X& x() const;                                // current estimate
  const Mat<NX,NX>& P() const;                       // optional; omit if the method has no covariance
  Model& model(); const Model& model() const;
};
```

Timing convention (Plett 2004): at sample k the adapter calls `predict(u_{k-1})` (skipped at k=0),
then `predict_measurement(u_k)`, then `update(y_k, u_k)`.

Filters that must couple predict and update (e.g. MHE, particle filters) may do all work in
`update()` and keep `predict()` cheap, but they must still honour the interface.

Registration (in `benchmarks/registry_<family>.cpp`):

```cpp
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/filters/my_filter.hpp"
namespace estkit {
void register_kalman(EstimatorList& list) {
    add_cell_estimator<MyFilter<EcmModel>>(list, "MyFilter", "kalman");
    add_cell_estimator<MyFilter<EcmModel>>(list, "MyFilter-tuned", "kalman",
        [](MyFilter<EcmModel>& f, const EstimatorConfig& c) { f.opts.gamma = Real(2); (void)c; });
}
}
```

`add_cell_estimator` returns a reference; use `.with_capacity([](const F& f){ return f.x()[4]; })`
for estimators that estimate capacity (SOH), and `.with_bounds(lo, hi)` for interval observers.

Estimators that are not model-generic by nature (Coulomb counting, RLS identification, NN direct
regressors, pack methods) may implement `estkit::Estimator` directly (see battery/baselines.hpp).

Batch (offline) methods derive from `estkit::BatchEstimator` and implement
`run(const Real* i, const Real* v, const Real* T, int n, Real* soc_out, Real* v_pred_out)`.

## 4. Numerical hygiene

* Symmetrise covariances after updates (`P.symmetrize()`), use the Joseph form where relevant.
* Use `cholesky_safe`, `qr_r`, `chol_update`, `solve`, `solve_spd`, `inverse` from linalg.hpp.
* Guard divisions (innovation covariance) with a floor, never produce NaN; if something goes wrong,
  keep the previous estimate.
* Apply `constrain(model, x)` after the update when the estimate should stay physical (SOC in
  [-0.05, 1.05]) unless the algorithm is explicitly unconstrained.
* Keep everything O(NX^3) or better per step; no dynamic memory; no I/O.

## 5. Validation before you are done

Build: `cmake -S . -B build && cmake --build build -j2` (must be warning-free) and float build
`cmake -S . -B build_f -DESTKIT_FLOAT=ON && cmake --build build_f -j2`.
Run: `./build/bench_cell --plant ecm --profile evtol --scenario all --only NAME --out results/dev`
for each of your estimators and check the printed table. Acceptance for a *working* cell-level
SOC estimator on the ECM plant: `nominal` rmse_ss < 0.02 and `init_error` converges (conv >= 0,
no divergence). Robustness-oriented methods must additionally do what they claim on their target
scenario (e.g. Huber/Student-t on `outliers`, adaptive filters on `noise_step`, disturbance/
unknown-input observers on `current_bias`). Report the numbers you obtained in your final message.
Also run `./build/unit_tests`.

## 6. Documentation for the report

For each estimator also write `report/chapters/<family>/<name>.tex` following
`report/chapters/TEMPLATE.tex` (theory, every symbol defined, algorithm box, implementation notes,
complexity, references as `\cite{key}` using keys from `report/refs.bib`; add missing entries to
`report/refs_additions_<family>.bib` with complete bibliographic data).
