#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Benchmark regression check for CI: run a representative subset of the benchmark and compare
the accuracy metrics with the recorded reference values.

    python3 scripts/ci_regression.py --build build            # check
    python3 scripts/ci_regression.py --build build --record   # rewrite tests/regression_reference.json

Accuracy (steady-state RMSE, whole-run RMSE, divergence flag) must match the reference within a
tolerance that allows for compiler/libm differences; timing is reported but never checked.
"""
import os, sys, json, subprocess, argparse, tempfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, 'tests', 'regression_reference.json')
CELL = ['CoulombCounting', 'EKF', 'UKF', 'SR-UKF', 'IEKF', 'IAE-AKF', 'Fading-EKF', 'IMM', 'Huber-EKF', 'Schmidt-KF',
        'Luenberger', 'AdaptiveObserver', 'DOB', 'JointEKF', 'DualEKF', 'PF-Bootstrap', 'RBPF', 'NN-EKF', 'MHE-RTI', 'ERTS', 'URTS']
SCEN = ['nominal', 'init_error', 'current_bias', 'combined']
REL, ABS = 0.15, 0.0005     # 15 % relative or 0.05 percentage points absolute, whichever is larger


def run(build, out):
    exe = lambda n: os.path.join(build, n + ('.exe' if os.name == 'nt' else ''))
    for sc in SCEN:
        subprocess.run([exe('bench_cell'), '--plant', 'ecm', '--profile', 'evtol', '--scenario', sc, '--only', ','.join(CELL),
                        '--seeds', '1', '--reps', '1', '--out', out, '--timeseries', '0'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([exe('bench_cell'), '--plant', 'spm', '--profile', 'evtol', '--scenario', 'nominal', '--only', 'EKF,Fading-EKF,Luenberger',
                    '--seeds', '1', '--reps', '1', '--out', out, '--timeseries', '0'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([exe('bench_pack'), '--scenario', 'nominal', '--seeds', '1', '--reps', '1', '--out', out, '--timeseries', '0'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([exe('bench_imu'), '--scenario', 'nominal', '--seeds', '1', '--reps', '1', '--out', out, '--timeseries', '0'], check=True, stdout=subprocess.DEVNULL)
    rows = {}
    for f, key, cols in [('summary_ecm.csv', ('plant', 'estimator', 'scenario'), ('rmse', 'rmse_ss', 'diverged')),
                         ('summary_spm.csv', ('plant', 'estimator', 'scenario'), ('rmse', 'rmse_ss', 'diverged')),
                         ('summary_pack.csv', ('estimator', 'scenario'), ('rmse_cells', 'ss_rmse_cells', 'diverged')),
                         ('summary_imu.csv', ('estimator', 'scenario'), ('rms_angle_deg', 'ss_rms_angle_deg', 'diverged'))]:
        p = os.path.join(out, f)
        if not os.path.exists(p): continue
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            k = '|'.join(str(r[c]) for c in key)
            rows[k] = {c: float(r[c]) for c in cols}
            rows[k]['ns_per_step'] = float(r['ns_per_step'])
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--build', default='build'); ap.add_argument('--record', action='store_true'); a = ap.parse_args()
    with tempfile.TemporaryDirectory() as out:
        rows = run(a.build, out)
    if a.record:
        json.dump(rows, open(REF, 'w'), indent=1, sort_keys=True); print(f'recorded {len(rows)} reference rows'); return
    ref = json.load(open(REF))
    bad = []
    for k, r in ref.items():
        if k not in rows: bad.append(f'{k}: missing from run'); continue
        c = rows[k]
        for m in r:
            if m in ('ns_per_step',): continue
            if m == 'diverged':
                if int(c[m]) != int(r[m]): bad.append(f'{k}: diverged {c[m]} != {r[m]}')
                continue
            tol = max(REL * abs(r[m]), ABS if 'deg' not in m else 0.05)
            if abs(c[m] - r[m]) > tol: bad.append(f'{k}: {m} = {c[m]:.5f}, reference {r[m]:.5f} (tol {tol:.5f})')
    print(f'{len(ref)} reference rows checked, {len(bad)} deviations')
    for b in bad: print('  ', b)
    slow = sorted(rows.items(), key=lambda kv: -kv[1]['ns_per_step'])[:3]
    print('slowest per step:', ', '.join(f"{k.split('|')[-2]} {v['ns_per_step']:.0f} ns" for k, v in slow))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
