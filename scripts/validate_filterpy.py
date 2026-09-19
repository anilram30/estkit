#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Cross-validation of the estkit EKF, UKF and RTS smoother against FilterPy.

FilterPy (Labbe, https://github.com/rlabbe/filterpy) is the reference open-source
Kalman-filter library used by the book "Kalman and Bayesian Filters in Python".
The very same discrete-time 2-RC battery model (same OCV lookup table, same
parameters, same noise covariances) is implemented here in NumPy and run through
FilterPy's ExtendedKalmanFilter, UnscentedKalmanFilter (Merwe scaled sigma points)
and rts_smoother on the dataset exported by tools/export_validation.  The
trajectories must agree to floating-point round-off.

Usage: python3 scripts/validate_filterpy.py results/validation [report/tables]
"""
import sys, os, csv
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, rts_smoother
import filterpy

R_GAS = 8.314462618
T_REF = 298.15


def load_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        rows = [[float(x) for x in row] for row in r]
    return hdr, np.array(rows)


class Model:
    """Exact NumPy replica of estkit::EcmModel (battery/ecm_model.hpp)."""

    def __init__(self, p, ocv_z, ocv_v):
        self.p = p
        self.zlo, self.zhi = ocv_z[0], ocv_z[-1]
        self.v = ocv_v
        self.n = len(ocv_v)
        dz = (self.zhi - self.zlo) / (self.n - 1)
        self.dz = dz
        self.dv = np.diff(ocv_v) / dz

    # entropic coefficient, identical to cell.hpp
    @staticmethod
    def dUdT(z):
        z = min(max(z, 0.0), 1.0)
        return 1e-4 * (-2.0 + 3.0 * z - 1.5 * z * z)

    @staticmethod
    def dUdT_dz(z):
        if z < 0 or z > 1:
            return 0.0
        return 1e-4 * (3.0 - 3.0 * z)

    def ocv(self, z, T):
        s = (z - self.zlo) / self.dz
        k = int(np.floor(s))
        k = min(max(k, 0), self.n - 2)
        zk = self.zlo + self.dz * k
        return self.v[k] + self.dv[k] * (z - zk) + (T - T_REF) * self.dUdT(z)

    def docv(self, z, T):
        s = (z - self.zlo) / self.dz
        k = int(np.floor(s))
        k = min(max(k, 0), self.n - 2)
        return self.dv[k] + (T - T_REF) * self.dUdT_dz(z)

    def arr(self, Ea, T):
        return np.exp(Ea / R_GAS * (1.0 / T - 1.0 / self.p['T_ref']))

    def R0(self, T): return self.p['R0'] * self.arr(self.p['Ea_R0'], T)
    def R1(self, T): return self.p['R1'] * self.arr(self.p['Ea_R1'], T)
    def R2(self, T): return self.p['R2'] * self.arr(self.p['Ea_R2'], T)
    def a1(self, T): return np.exp(-self.p['dt'] / (self.p['tau1'] * self.arr(self.p['Ea_tau'], T)))
    def a2(self, T): return np.exp(-self.p['dt'] / (self.p['tau2'] * self.arr(self.p['Ea_tau'], T)))
    def eta(self, i): return 1.0 if i >= 0 else self.p['eta_c']
    def cg(self, i): return -self.eta(i) * self.p['dt'] / (3600.0 * self.p['Q_Ah'])
    def ah(self, i): return np.exp(-abs(self.eta(i) * i * self.p['gamma'] * self.p['dt'] / (3600.0 * self.p['Q_Ah'])))

    def f(self, x, i, T):
        A1, A2, AH = self.a1(T), self.a2(T), self.ah(i)
        return np.array([x[0] + self.cg(i) * i,
                         A1 * x[1] + self.R1(T) * (1 - A1) * i,
                         A2 * x[2] + self.R2(T) * (1 - A2) * i,
                         AH * x[3] + (AH - 1) * np.sign(i)])

    def F(self, x, i, T):
        return np.diag([1.0, self.a1(T), self.a2(T), self.ah(i)])

    def B(self, x, i, T):
        dah = -np.sign(i) * self.eta(i) * self.p['gamma'] * self.p['dt'] / (3600.0 * self.p['Q_Ah']) * self.ah(i)
        return np.array([self.cg(i), self.R1(T) * (1 - self.a1(T)), self.R2(T) * (1 - self.a2(T)), dah * (x[3] + np.sign(i))])

    def h(self, x, i, T):
        return np.array([self.ocv(x[0], T) + self.p['M'] * x[3] - x[1] - x[2] - self.R0(T) * i])

    def H(self, x, i, T):
        return np.array([[self.docv(x[0], T), -1.0, -1.0, self.p['M']]])

    def Q(self, x, i, T):
        b = self.B(x, i, T)
        Qm = np.outer(b, b) * self.p['sigma_i'] ** 2
        for k in range(4):
            Qm[k, k] += self.p['qf%d' % k] ** 2
        return Qm

    def R(self, T):
        return np.array([[self.p['sigma_v'] ** 2 + (self.R0(T) * self.p['sigma_i']) ** 2]])


class BatteryEKF(ExtendedKalmanFilter):
    """FilterPy EKF with the nonlinear state propagation of the battery model."""

    def __init__(self, m):
        super().__init__(dim_x=4, dim_z=1)
        self.m = m
        self.u_prev = None

    def predict_x(self, u=0):
        i, T = u
        self.x = self.m.f(self.x, i, T)


def run_ekf(m, data, x0, P0):
    ekf = BatteryEKF(m)
    ekf.x = x0.copy(); ekf.P = P0.copy()
    n = data.shape[0]
    out = np.zeros((n, 9)); u_prev = None
    for k in range(n):
        i, v, T = data[k, 1], data[k, 2], data[k, 3]
        if u_prev is not None:
            ip, Tp = u_prev
            ekf.F = m.F(ekf.x, ip, Tp)
            ekf.Q = m.Q(ekf.x, ip, Tp)   # estkit evaluates Q at the PREDICTED state: emulate below
            xpred = m.f(ekf.x, ip, Tp)
            ekf.Q = m.Q(xpred, ip, Tp)
            ekf.predict(u=(ip, Tp))
            ekf.P = 0.5 * (ekf.P + ekf.P.T)
        ypred = m.h(ekf.x, i, T)[0]
        ekf.R = m.R(T)
        ekf.update(np.array([v]), HJacobian=lambda x, i=i, T=T: m.H(x, i, T), Hx=lambda x, i=i, T=T: m.h(x, i, T))
        ekf.P = 0.5 * (ekf.P + ekf.P.T)
        # estkit applies the state constraint (SOC in [-0.05, 1.05], h in [-1, 1])
        ekf.x[0] = min(max(ekf.x[0], -0.05), 1.05); ekf.x[3] = min(max(ekf.x[3], -1.0), 1.0)
        out[k, :4] = ekf.x; out[k, 4:8] = np.diag(ekf.P); out[k, 8] = ypred
        u_prev = (i, T)
    return out


def run_ukf(m, data, x0, P0, alpha=0.1, beta=2.0, kappa=0.0):
    pts = MerweScaledSigmaPoints(4, alpha=alpha, beta=beta, kappa=kappa)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=m.p['dt'], fx=lambda x, dt, u: m.f(x, u[0], u[1]), hx=lambda x, u: m.h(x, u[0], u[1]), points=pts)
    ukf.x = x0.copy(); ukf.P = P0.copy()
    n = data.shape[0]
    out = np.zeros((n, 9)); u_prev = None
    for k in range(n):
        i, v, T = data[k, 1], data[k, 2], data[k, 3]
        if u_prev is not None:
            ip, Tp = u_prev
            # estkit adds Q evaluated at the predicted mean; FilterPy adds self.Q inside predict():
            # evaluate the predicted mean first (cheap) to set Q consistently
            sig = pts.sigma_points(ukf.x, ukf.P)
            xm = sum(pts.Wm[s] * m.f(sig[s], ip, Tp) for s in range(sig.shape[0]))
            ukf.Q = m.Q(xm, ip, Tp)
            ukf.predict(u=(ip, Tp))
            ukf.P = 0.5 * (ukf.P + ukf.P.T)
        else:
            # FilterPy reuses the propagated sigma points in update(); at k = 0 nothing has been
            # propagated yet (sigmas_f is still zero), so draw them from (x0, P0) as estkit does
            ukf.sigmas_f = pts.sigma_points(ukf.x, ukf.P)
        # predicted measurement = sigma-mean of h over the propagated sigma points (Wan & van der Merwe form)
        ypred = sum(pts.Wm[s] * m.h(ukf.sigmas_f[s], i, T)[0] for s in range(ukf.sigmas_f.shape[0]))
        ukf.R = m.R(T)
        ukf.update(np.array([v]), u=(i, T))
        ukf.P = 0.5 * (ukf.P + ukf.P.T)
        ukf.x[0] = min(max(ukf.x[0], -0.05), 1.05); ukf.x[3] = min(max(ukf.x[3], -1.0), 1.0)
        out[k, :4] = ukf.x; out[k, 4:8] = np.diag(ukf.P); out[k, 8] = ypred
        u_prev = (i, T)
    return out


def run_ekf_rts(m, data, x0, P0):
    """Forward EKF storing posteriors, then FilterPy's rts_smoother with per-step F and Q.

    The battery model is affine in the state, x_{k+1} = F_k x_k + b_k with b_k = f(0, u_k), and F_k does not
    depend on x.  FilterPy's rts_smoother implements the purely linear recursion, so the known affine term is
    removed by the exact change of variables x~_k = x_k - c_k, c_{k+1} = F_k c_k + b_k, c_0 = 0 (the
    deterministic particular solution); x~ obeys x~_{k+1} = F_k x~_k + w_k and the backward pass is identical."""
    ekf = BatteryEKF(m); ekf.x = x0.copy(); ekf.P = P0.copy()
    n = data.shape[0]
    Xs, Ps, Fs, Qs = [], [], [], []
    u_prev = None
    for k in range(n):
        i, v, T = data[k, 1], data[k, 2], data[k, 3]
        if u_prev is not None:
            ip, Tp = u_prev
            ekf.F = m.F(ekf.x, ip, Tp); xpred = m.f(ekf.x, ip, Tp); ekf.Q = m.Q(xpred, ip, Tp)
            Fk, Qk = ekf.F.copy(), ekf.Q.copy()
            ekf.predict(u=(ip, Tp)); ekf.P = 0.5 * (ekf.P + ekf.P.T)
        else:
            Fk, Qk = np.eye(4), np.zeros((4, 4))
        ekf.R = m.R(T)
        ekf.update(np.array([v]), HJacobian=lambda x, i=i, T=T: m.H(x, i, T), Hx=lambda x, i=i, T=T: m.h(x, i, T))
        ekf.P = 0.5 * (ekf.P + ekf.P.T)
        ekf.x[0] = min(max(ekf.x[0], -0.05), 1.05); ekf.x[3] = min(max(ekf.x[3], -1.0), 1.0)
        Xs.append(ekf.x.copy()); Ps.append(ekf.P.copy()); Fs.append(Fk); Qs.append(Qk)
        u_prev = (i, T)
    # rts_smoother expects F_k, Q_k used to go from k to k+1: shift by one
    Fs = Fs[1:] + [Fs[-1]]; Qs = Qs[1:] + [Qs[-1]]
    # particular solution of the affine term
    c = np.zeros((n, 4))
    for k in range(n - 1):
        ip, Tp = data[k, 1], data[k, 3]
        c[k + 1] = Fs[k] @ c[k] + m.f(np.zeros(4), ip, Tp)
    Xt = np.array(Xs) - c
    xs, Ps_s, K, Pp = rts_smoother(Xt, np.array(Ps), np.array(Fs), np.array(Qs))
    return xs + c


def main():
    vdir = sys.argv[1] if len(sys.argv) > 1 else 'results/validation'
    tdir = sys.argv[2] if len(sys.argv) > 2 else None
    hdr, data = load_csv(os.path.join(vdir, 'val_cell_dataset.csv'))
    _, ocv = load_csv(os.path.join(vdir, 'val_ocv_table.csv'))
    phdr, prow = load_csv(os.path.join(vdir, 'val_params.csv'))
    p = dict(zip(phdr, prow[0]))
    m = Model(p, ocv[:, 0], ocv[:, 1])
    x0 = np.array([p['z0'], 0.0, 0.0, 0.0])
    P0 = np.diag([p['sigma_z0'] ** 2, 0.002 ** 2, 0.005 ** 2, 0.5 ** 2])
    n = data.shape[0]
    print(f'FilterPy {filterpy.__version__}, NumPy {np.__version__}; {n} samples')
    _, cekf = load_csv(os.path.join(vdir, 'val_ekf.csv'))
    _, cukf = load_csv(os.path.join(vdir, 'val_ukf.csv'))
    _, certs = load_csv(os.path.join(vdir, 'val_erts.csv'))
    results = []
    pe = run_ekf(m, data, x0, P0)
    dz = np.max(np.abs(pe[:, 0] - cekf[:, 0])); dP = np.max(np.abs(pe[:, 4] - cekf[:, 4])); dy = np.max(np.abs(pe[:, 8] - cekf[:, 8]))
    print(f'EKF : max |Δz| = {dz:.3e}, max |ΔP_zz| = {dP:.3e}, max |Δy_pred| = {dy:.3e} V; RMSE(SOC) estkit {np.sqrt(np.mean((cekf[:,0]-data[:,4])**2)):.5f} vs FilterPy {np.sqrt(np.mean((pe[:,0]-data[:,4])**2)):.5f}')
    results.append(('EKF', 'ExtendedKalmanFilter', dz, dP, dy))
    pu = run_ukf(m, data, x0, P0)
    dz = np.max(np.abs(pu[:, 0] - cukf[:, 0])); dP = np.max(np.abs(pu[:, 4] - cukf[:, 4])); dy = np.max(np.abs(pu[:, 8] - cukf[:, 8]))
    print(f'UKF : max |Δz| = {dz:.3e}, max |ΔP_zz| = {dP:.3e}, max |Δy_pred| = {dy:.3e} V; RMSE(SOC) estkit {np.sqrt(np.mean((cukf[:,0]-data[:,4])**2)):.5f} vs FilterPy {np.sqrt(np.mean((pu[:,0]-data[:,4])**2)):.5f}')
    results.append(('UKF', 'UnscentedKalmanFilter (Merwe sigma points)', dz, dP, dy))
    xs = run_ekf_rts(m, data, x0, P0)
    dz = np.max(np.abs(xs[:, 0] - certs[:, 0]))
    print(f'ERTS: max |Δz_smooth| = {dz:.3e}; RMSE(SOC) estkit {np.sqrt(np.mean((certs[:,0]-data[:,4])**2)):.5f} vs FilterPy {np.sqrt(np.mean((xs[:,0]-data[:,4])**2)):.5f}')
    results.append(('ERTS', '\\texttt{rts\\_smoother} on the EKF forward pass', dz, float('nan'), float('nan')))
    if tdir:
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, 'validation_filterpy.tex'), 'w') as f:
            f.write('% Copyright 2026 Sreeram Anil\n% SPDX-License-Identifier: CC-BY-4.0\n')
            f.write('\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\\caption{Cross-validation against FilterPy %s on the nominal eVTOL dataset (%d samples): maximum absolute differences between the estkit C++ trajectories and the FilterPy trajectories computed with an identical NumPy model.}\\label{tab:val_filterpy}\n' % (filterpy.__version__, n))
            f.write('\\begin{tabular}{llccc}\\toprule estkit & FilterPy reference & $\\max|\\Delta \\hat z|$ & $\\max|\\Delta P_{zz}|$ & $\\max|\\Delta \\hat y|$ [V] \\\\ \\midrule\n')
            for name, ref, dz, dP, dy in results:
                f.write('%s & %s & %.1e & %s & %s \\\\\n' % (name, ref, dz, ('%.1e' % dP) if dP == dP else '--', ('%.1e' % dy) if dy == dy else '--'))
            f.write('\\bottomrule\\end{tabular}\\end{table}\n')
        print('wrote', os.path.join(tdir, 'validation_filterpy.tex'))
    # CI gate: the reference agreement is deterministic, so enforce a tight tolerance.
    TOL = {'EKF': 1e-9, 'UKF': 1e-6, 'ERTS': 1e-6}
    worst = max(dz for _, _, dz, _, _ in results)
    fail = [name for name, _, dz, _, _ in results if dz > TOL.get(name, 1e-6)]
    print(f'worst |Δz| across EKF/UKF/ERTS = {worst:.2e}')
    if fail:
        print('FAIL: exceeds tolerance for', ', '.join(fail)); sys.exit(1)
    print('OK: estkit agrees with FilterPy to floating-point precision')


if __name__ == '__main__':
    main()
