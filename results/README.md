<!-- Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0 -->
# Benchmark data

`final/` is the recorded run that every table and figure of the report was generated from
(3 seeds, ECM and SPM plants, four missions, twelve scenarios, pack, attitude, SOH, float32 build).
It is regenerated from scratch by `scripts/run_all.sh 3`; the run is deterministic from the seeds.

| File | Content |
|---|---|
| `final/summary_ecm.csv`, `summary_spm.csv` | one row per estimator × profile × scenario × seed: RMSE, steady-state RMSE, MAE, max, final error, convergence time, divergence flag, voltage residual, ns/step, bytes, capacity error, bound violation |
| `final/float/summary_ecm.csv` | the same for the single-precision build (seed 1) |
| `final/summary_pack.csv`, `summary_imu.csv`, `summary_soh.csv`, `soh_flights.csv` | pack, attitude and multi-flight results |
| `final/ident_spm_ecm*.csv` | ECM identification on the SPM plant |
| `final/imu_nominal_seed1.csv` | the synthetic flight dataset |
| `final/profile_ts/` | mission time series used for the dataset figures |
| `final/log_*.txt` | console logs of the run |
| `model/` | model characteristic curves (OCV, Arrhenius, hysteresis, ageing) from `export_model_curves` |
| `final/ts/`, `final/validation/` (release asset `estkit_raw_timeseries.tar.gz`, 114 MB) | per-sample time series of every estimator (seed 1) and the exports used by the validation scripts |

Licence: CC BY 4.0 (see `../NOTICE`). Column definitions: report, Chapter "Benchmark methodology" and
Appendix "Build and run".
