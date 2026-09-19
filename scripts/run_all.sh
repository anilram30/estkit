#!/usr/bin/env bash
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
# Build (double + float) and run the complete benchmark suite into results/final.
# Usage: scripts/run_all.sh [SEEDS]   (default 3)
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${1:-3}
OUT=results/final
rm -rf "$OUT" build_final build_final_f
cmake -S . -B build_final -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build_final -j2
cmake -S . -B build_final_f -DCMAKE_BUILD_TYPE=Release -DESTKIT_FLOAT=ON > /dev/null
cmake --build build_final_f -j2
mkdir -p "$OUT"
echo "== unit tests"; ./build_final/unit_tests | tail -1; ./build_final_f/unit_tests | tail -1
echo "== ECM identification against the SPM plant"; ./build_final/ident_spm_ecm --r_scale 0.5 --k_scale 3 --R_ohm 0.005 --out "$OUT/ident_spm_ecm.csv" | tail -8
echo "== cell-level, ECM plant, eVTOL, all scenarios, $SEEDS seeds"
./build_final/bench_cell --plant ecm --profile evtol --scenario all --seeds "$SEEDS" --reps 2 --out "$OUT" > "$OUT/log_cell_ecm_evtol.txt"
echo "== cell-level, SPM plant, eVTOL, all scenarios, $SEEDS seeds"
./build_final/bench_cell --plant spm --profile evtol --scenario all --seeds "$SEEDS" --reps 2 --out "$OUT" > "$OUT/log_cell_spm_evtol.txt"
for PR in fixed_wing hover_hold ground_charge; do
  echo "== cell-level, ECM plant, $PR, all scenarios, 1 seed"
  ./build_final/bench_cell --plant ecm --profile "$PR" --scenario all --seeds 1 --reps 2 --out "$OUT" --timeseries 0 > "$OUT/log_cell_ecm_$PR.txt"
  # time series of the mission itself (for the dataset figures) — one estimator, nominal scenario
  ./build_final/bench_cell --plant ecm --profile "$PR" --scenario nominal --only CoulombCounting --seeds 1 --reps 1 --out "$OUT/profile_ts" --timeseries 1 > /dev/null
done
echo "== cell-level, float32 build, ECM plant, eVTOL"
./build_final_f/bench_cell --plant ecm --profile evtol --scenario all --seeds 1 --reps 2 --out "$OUT/float" --timeseries 0 > "$OUT/log_cell_float.txt"
echo "== pack-level"
./build_final/bench_pack --scenario all --seeds "$SEEDS" --reps 2 --out "$OUT" > "$OUT/log_pack.txt"
echo "== attitude (IMU)"
./build_final/bench_imu --scenario all --seeds "$SEEDS" --reps 2 --out "$OUT" > "$OUT/log_imu.txt"
./build_final/bench_imu --export-dataset "$OUT/imu_nominal_seed1.csv" > /dev/null
echo "== multi-flight SOH"
./build_final/bench_soh --flights 120 --accel 3 --out "$OUT" > "$OUT/log_soh.txt"
echo "== validation exports + cross-checks"
./build_final/export_validation "$OUT/validation" > /dev/null
./build_final/export_model_curves results/model > /dev/null
python3 scripts/validate_filterpy.py "$OUT/validation" report/tables | tail -4
python3 scripts/validate_pybamm.py "$OUT/validation" report/tables report/figures 2>/dev/null | tail -4
python3 scripts/validate_ahrs.py "$OUT/validation" report/tables | tail -3
echo "== done: $(date)"
