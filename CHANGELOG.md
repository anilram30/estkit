<!-- Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0 -->
# Changelog

## 1.0.0 — 2026-09-19

First public release.

- 70 cell-level estimators in ten families, 7 pack-level architectures, 9 attitude estimators,
  all against one model-agnostic interface (`f, h, Q, R`; optional Jacobians, constraints, parameters).
- Two truth plants (enhanced equivalent circuit with thermal, Arrhenius, one-state/Preisach hysteresis
  and ageing; electrochemical single-particle model from the LG M50 electrode data) and a
  single-particle-to-ECM identification tool.
- Benchmark: four missions, twelve fault scenarios, three seeds, 12-cell pack, 120-mission SOH study,
  synthetic instrumented flight; ~9 000 recorded runs in `results/final`.
- Cross-validation against FilterPy (1e-15 / 2.5e-12 / 7e-9), PyBaMM (0.6–3.2 mV) and ahrs (0.0014°).
- Report: 958 pages, one chapter per estimator, LaTeX sources included.
- Single-precision build (`-DESTKIT_FLOAT=ON`), warning-free with `-Wall -Wextra -Wpedantic -Werror`.
