#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Cross-validation of the estkit Madgwick and Mahony attitude filters against the
Python `ahrs` library (Garcia, https://github.com/Mayitzin/ahrs), which implements
the published algorithms of Madgwick (2010/2011) and Mahony et al. (2008).

Conventions: estkit and ahrs both use Hamilton, scalar-first quaternions mapping
BODY -> NAV (NED).  The estkit dataset stores the accelerometer as SPECIFIC FORCE
(f = a - g, i.e. [0,0,-g] at rest), whereas ahrs expects the accelerometer to
point along +g at rest, so the accelerometer is negated before being fed to ahrs.
The estkit filters propagate the quaternion with the exact exponential map,
ahrs with a first-order Euler step of q_dot = 1/2 q (x) omega — the residual
difference is O(dt^2 |omega|^2) per step.

Usage: python3 scripts/validate_ahrs.py results/validation [report/tables]
"""
import sys, os, csv
import numpy as np


def load_csv(path):
    with open(path) as f:
        r = csv.reader(f); hdr = next(r); rows = [[float(x) for x in row] for row in r]
    return hdr, np.array(rows)


def qangle(a, b):
    """angle between two unit quaternions (deg), rows of arrays [w,x,y,z]"""
    d = np.abs(np.sum(a * b, axis=1)).clip(0, 1)
    return 2 * np.degrees(np.arccos(d))


def main():
    vdir = sys.argv[1] if len(sys.argv) > 1 else 'results/validation'
    tdir = sys.argv[2] if len(sys.argv) > 2 else None
    import ahrs
    from ahrs.filters import Madgwick, Mahony
    hdr, d = load_csv(os.path.join(vdir, 'val_ahrs.csv'))
    col = {h: i for i, h in enumerate(hdr)}
    gyr = d[:, [col['gx'], col['gy'], col['gz']]]
    acc = -d[:, [col['ax'], col['ay'], col['az']]]          # specific force -> gravity direction
    mag = d[:, [col['mx'], col['my'], col['mz']]]
    q_true = d[:, [col['qw_true'], col['qx_true'], col['qy_true'], col['qz_true']]]
    q_madg = d[:, [col['qw_madg'], col['qx_madg'], col['qy_madg'], col['qz_madg']]]
    q_mah = d[:, [col['qw_mah'], col['qx_mah'], col['qy_mah'], col['qz_mah']]]
    n = d.shape[0]
    q0 = q_true[0].copy()
    print('ahrs', ahrs.__version__, ';', n, 'samples at 100 Hz')
    # --- Madgwick (beta = 0.1) ---
    mw = Madgwick(frequency=100.0, beta=0.1)
    Q = np.zeros((n, 4)); q = q0.copy()
    for k in range(n):
        q = mw.updateMARG(q, gyr=gyr[k], acc=acc[k], mag=mag[k]); Q[k] = q
    dm = qangle(Q, q_madg)
    em_ours = qangle(q_madg, q_true); em_ahrs = qangle(Q, q_true)
    print('Madgwick: max angle(estkit, ahrs) = %.4f deg, mean %.4f deg; attitude RMS error vs truth: estkit %.3f deg, ahrs %.3f deg' % (dm.max(), dm.mean(), np.sqrt(np.mean(em_ours ** 2)), np.sqrt(np.mean(em_ahrs ** 2))))
    # --- Mahony (kp = 1, ki = 0.1) ---
    # ahrs.filters.Mahony hard-codes an ENU navigation frame (its magnetic reference is
    # b = [0, sqrt(hx^2+hy^2), hz], i.e. north along +y), whereas the estkit dataset is NED/FRD.
    # Both are related by the proper rotation P (180 deg about (1,1,0)/sqrt(2)):
    #   v_enu = P v_ned, v_flu = P v_frd, q_enu = qP (x) q_ned (x) qP^-1,  P = [[0,1,0],[1,0,0],[0,0,-1]]
    # In ENU/FLU the specific force itself is what ahrs expects (+g along body z at rest).
    P = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    qP = np.array([0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])

    def qmul(a, b):
        w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    def qconj(a): return np.array([a[0], -a[1], -a[2], -a[3]])
    gyr_e = gyr @ P.T; acc_e = -acc @ P.T; mag_e = mag @ P.T     # -acc restores the specific force (acc was negated above)
    mh = Mahony(frequency=100.0, k_P=1.0, k_I=0.1)
    Q2 = np.zeros((n, 4)); q = qmul(qmul(qP, q0), qconj(qP))
    for k in range(n):
        q = mh.updateMARG(q, gyr=gyr_e[k], acc=acc_e[k], mag=mag_e[k]); Q2[k] = qmul(qmul(qconj(qP), q), qP)
    dh = qangle(Q2, q_mah)
    eh_ours = qangle(q_mah, q_true); eh_ahrs = qangle(Q2, q_true)
    print('Mahony  : max angle(estkit, ahrs) = %.4f deg, mean %.4f deg; attitude RMS error vs truth: estkit %.3f deg, ahrs %.3f deg' % (dh.max(), dh.mean(), np.sqrt(np.mean(eh_ours ** 2)), np.sqrt(np.mean(eh_ahrs ** 2))))
    if tdir:
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, 'validation_ahrs.tex'), 'w') as f:
            f.write('% Copyright 2026 Sreeram Anil\n% SPDX-License-Identifier: CC-BY-4.0\n')
            f.write('\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\\caption{Cross-validation of the estkit Madgwick ($\\beta=0.1$) and Mahony ($k_P=1$, $k_I=0.1$) filters against the \\texttt{ahrs} %s Python library on the nominal flight dataset (%d samples, \\SI{100}{\\hertz}): angular distance between the two implementations\' quaternions, and attitude RMS error of each against the truth.}\\label{tab:val_ahrs}\n' % (ahrs.__version__, n))
            f.write('\\begin{tabular}{lcccc}\\toprule filter & max $\\angle(\\hat q_{\\mathrm{estkit}}, \\hat q_{\\mathrm{ahrs}})$ [deg] & mean [deg] & RMS err.\\ estkit [deg] & RMS err.\\ ahrs [deg] \\\\ \\midrule\n')
            f.write('Madgwick & %.4f & %.4f & %.3f & %.3f \\\\\n' % (dm.max(), dm.mean(), np.sqrt(np.mean(em_ours ** 2)), np.sqrt(np.mean(em_ahrs ** 2))))
            f.write('Mahony & %.4f & %.4f & %.3f & %.3f \\\\\n' % (dh.max(), dh.mean(), np.sqrt(np.mean(eh_ours ** 2)), np.sqrt(np.mean(eh_ahrs ** 2))))
            f.write('\\bottomrule\\end{tabular}\\end{table}\n')
        print('wrote', os.path.join(tdir, 'validation_ahrs.tex'))
    # CI gate: the two implementations must agree to well under a hundredth of a degree.
    worst = max(dm.max(), dh.max())
    print(f'worst angle(estkit, ahrs) = {worst:.4f} deg')
    if worst > 0.01:
        print('FAIL: exceeds 0.01 deg'); sys.exit(1)
    print('OK: estkit agrees with ahrs to better than 0.01 deg')


if __name__ == '__main__':
    main()
