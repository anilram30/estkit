#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Generate the LaTeX tables of the report from results/final.

Usage: python3 scripts/make_tables.py [results/final] [report/tables]
"""
import os, sys, re
import numpy as np
import pandas as pd

RES = sys.argv[1] if len(sys.argv) > 1 else 'results/final'
TAB = sys.argv[2] if len(sys.argv) > 2 else 'report/tables'
os.makedirs(TAB, exist_ok=True)

GROUP_ORDER = ['baselines', 'kalman', 'adaptive_kalman', 'robust_kalman', 'observers', 'soh', 'particle', 'learning', 'optimization', 'smoothers']
GROUP_LABEL = {'baselines': 'Baselines', 'kalman': 'Classical / sigma-point Kalman', 'adaptive_kalman': 'Adaptive Kalman', 'robust_kalman': 'Robust / embedded Kalman', 'observers': 'Deterministic observers',
               'soh': 'Joint / dual state-parameter (SOH)', 'particle': 'Particle / ensemble / Gaussian-sum', 'learning': 'Data-driven', 'optimization': 'Moving-horizon estimation', 'smoothers': 'Smoothers (offline)'}
SCEN_ORDER = ['nominal', 'init_error', 'current_bias', 'noise_step', 'outliers', 'cold', 'hot', 'aged_cell', 'param_mismatch', 'sensor_dropout', 'preisach', 'combined']


def safe(name):
    return re.sub(r'[^A-Za-z0-9]', '', name)


def tex(s):
    return str(s).replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')


def fmt(v, digits=2, pct=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    return f'{v * (100 if pct else 1):.{digits}f}'


TEX_HEADER = '% Copyright 2026 Sreeram Anil\n% SPDX-License-Identifier: CC-BY-4.0\n'


def write(name, content):
    with open(os.path.join(TAB, name), 'w') as f:
        f.write(TEX_HEADER + content)


def load(plant):
    p = os.path.join(RES, f'summary_{plant}.csv')
    return pd.read_csv(p) if os.path.exists(p) else None


ecm = load('ecm'); spm = load('spm')


def per_estimator_tables():
    ests = list(dict.fromkeys(ecm.estimator))
    ekf = ecm[(ecm.estimator == 'EKF') & (ecm.profile == 'evtol')].groupby('scenario').rmse_ss.mean()
    for est in ests:
        rows = []
        for plant, df in [('ecm', ecm), ('spm', spm)]:
            if df is None: continue
            d = df[(df.estimator == est) & (df.profile == 'evtol')]
            if len(d) == 0: continue
            g = d.groupby('scenario').agg(rmse=('rmse', 'mean'), rmse_sd=('rmse', 'std'), rmse_ss=('rmse_ss', 'mean'), maxe=('max_err', 'mean'), conv=('conv_time', 'mean'), dv=('diverged', 'max'), rmsev=('rmse_v', 'mean'), ns=('ns_per_step', 'median'), cap=('capacity_err_end', 'mean'), bv=('bound_violation', 'mean'), n=('seed', 'count'))
            for sc in SCEN_ORDER:
                if sc not in g.index: continue
                r = g.loc[sc]
                ekf_ref = ekf.get(sc, np.nan) if plant == 'ecm' else spm[(spm.estimator == 'EKF') & (spm.profile == 'evtol') & (spm.scenario == sc)].rmse_ss.mean() if spm is not None else np.nan
                rows.append((plant, sc, r, ekf_ref))
        L = []
        L.append('\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n')
        L.append('\\caption{%s: cell-level benchmark on the eVTOL mission (mean over seeds; RMSE and max error in \\%% SOC; $t_\\mathrm{conv}$: first time the error stays below 2\\,\\%% for 60\\,s, -- = never; EKF reference: steady-state RMSE of the plain EKF on the same data).}\\label{tab:cell_%s}\n' % (tex(est), safe(est)))
        L.append('\\begin{tabular}{llrrrrrcrr}\\toprule\nplant & scenario & RMSE & RMSE$_\\mathrm{ss}$ & max & $t_\\mathrm{conv}$ [s] & RMSE$_v$ [mV] & div. & EKF ref. & ns/step \\\\ \\midrule\n')
        last = None
        for plant, sc, r, ref in rows:
            if last is not None and plant != last: L.append('\\midrule\n')
            last = plant
            conv = '--' if (not np.isfinite(r.conv) or r.conv < 0) else f'{r.conv:.0f}'
            rmsev = '--' if not np.isfinite(r.rmsev) else f'{r.rmsev * 1e3:.2f}'
            extra = ''
            L.append('%s & %s & %s & %s & %s & %s & %s & %s & %s & %.0f \\\\\n' % (plant.upper(), tex(sc), fmt(r.rmse), fmt(r.rmse_ss), fmt(r.maxe), conv, rmsev, 'yes' if r['dv'] else 'no', fmt(ref), r.ns))
        L.append('\\bottomrule\\end{tabular}\\end{table}\n')
        # optional capacity / bound rows
        extra_rows = [(plant, sc, r) for plant, sc, r, _ in rows if np.isfinite(r.cap) or np.isfinite(r.bv)]
        if extra_rows:
            L.append('\\begin{table}[H]\\centering\\small\\caption{%s: additional outputs (capacity error at the end of the mission in Ah, averaged over the last 10\\,\\%% of the run; interval-bound violation rate).}\\label{tab:cellx_%s}\n' % (tex(est), safe(est)))
            L.append('\\begin{tabular}{llrr}\\toprule plant & scenario & $\\hat Q - Q$ [Ah] & bound violation [\\%] \\\\ \\midrule\n')
            for plant, sc, r in extra_rows:
                L.append('%s & %s & %s & %s \\\\\n' % (plant.upper(), tex(sc), ('%+.3f' % r.cap) if np.isfinite(r.cap) else '--', ('%.2f' % (100 * r.bv)) if np.isfinite(r.bv) else '--'))
            L.append('\\bottomrule\\end{tabular}\\end{table}\n')
        write(f'cell_{safe(est)}.tex', ''.join(L))
    print('per-estimator tables:', len(ests))


def overview_table(plant, df, label):
    if df is None: return
    d = df[df.profile == 'evtol']
    g = d.groupby(['group', 'estimator', 'scenario']).rmse_ss.mean().reset_index()
    scen = [c for c in SCEN_ORDER if c in set(g.scenario)]
    L = ['\\begingroup\\scriptsize\\setlength{\\tabcolsep}{2.6pt}\n\\begin{longtable}{l' + 'r' * len(scen) + '}\n']
    L.append('\\caption{Steady-state SOC RMSE [\\%%] for every estimator and scenario, %s plant, eVTOL mission (mean over seeds). Bold: best of the family in the column; `div\' marks divergence in at least one seed.}\\label{tab:overview_%s}\\\\\n' % (plant.upper(), label))
    L.append('\\toprule estimator & ' + ' & '.join('\\rotatebox{60}{%s}' % tex(c) for c in scen) + ' \\\\ \\midrule\\endfirsthead\n')
    L.append('\\toprule estimator & ' + ' & '.join('\\rotatebox{60}{%s}' % tex(c) for c in scen) + ' \\\\ \\midrule\\endhead\n')
    divs = d.groupby(['estimator', 'scenario']).diverged.max()
    for grp in GROUP_ORDER:
        gg = g[g.group == grp]
        if len(gg) == 0: continue
        L.append('\\multicolumn{%d}{l}{\\textit{%s}} \\\\\n' % (len(scen) + 1, GROUP_LABEL[grp]))
        ests = list(dict.fromkeys(gg.estimator))
        best = {c: gg[gg.scenario == c].rmse_ss.min() for c in scen}
        for e in ests:
            cells = []
            for c in scen:
                v = gg[(gg.estimator == e) & (gg.scenario == c)].rmse_ss
                if len(v) == 0: cells.append('--'); continue
                v = v.iloc[0]; s = f'{v * 100:.2f}'
                if divs.get((e, c), 0) > 0: s += '$^{\\mathrm{div}}$'
                if abs(v - best[c]) < 1e-12: s = '\\textbf{%s}' % s
                cells.append(s)
            L.append('\\quad %s & %s \\\\\n' % (tex(e), ' & '.join(cells)))
    L.append('\\bottomrule\\end{longtable}\\endgroup\n')
    write(f'overview_{label}.tex', ''.join(L))


def runtime_table():
    d = ecm[(ecm.profile == 'evtol') & (ecm.scenario == 'nominal')]
    g = d.groupby(['group', 'estimator']).agg(rmse=('rmse_ss', 'mean'), ns=('ns_per_step', 'median'), by=('bytes', 'first')).reset_index()
    ekf_ns = g[g.estimator == 'EKF'].ns.iloc[0]
    L = ['\\begin{longtable}{llrrrr}\n\\caption{Computational cost per sample on the benchmark machine (x86-64, single core, g++ 13 -O2, double precision; median over runs) with the estimator object size, relative cost to the EKF, and an order-of-magnitude projection to a Cortex-M7 class flight processor using the central scaling factor $\\kappa = 30$ of Chapter~\\ref{ch:metrics} (range 20--90; a measurement on the target supersedes this column). Batch smoothers report the per-sample cost of the whole record.}\\label{tab:runtime}\\\\\n']
    L.append('\\toprule estimator & family & ns/step & $\\times$EKF & bytes & est. Cortex-M7 [µs] \\\\ \\midrule\\endfirsthead\n\\toprule estimator & family & ns/step & $\\times$EKF & bytes & est. Cortex-M7 [µs] \\\\ \\midrule\\endhead\n')
    for grp in GROUP_ORDER:
        gg = g[g.group == grp].sort_values('ns')
        for _, r in gg.iterrows():
            L.append('%s & %s & %.0f & %.1f & %s & %.0f \\\\\n' % (tex(r.estimator), tex(grp), r.ns, r.ns / ekf_ns, ('%.1f\\,MB' % (r.by / 1e6)) if r.by > 1e6 else ('%d' % r.by), r.ns * 30 / 1e3))
    L.append('\\bottomrule\\end{longtable}\n')
    write('runtime.tex', ''.join(L))


def float_table():
    pf = os.path.join(RES, 'float', 'summary_ecm.csv')
    if not os.path.exists(pf): return
    f = pd.read_csv(pf); d = ecm[(ecm.profile == 'evtol') & (ecm.seed == 1)]
    m = f.merge(d, on=['estimator', 'scenario', 'group'], suffixes=('_f', '_d'))
    scen = ['nominal', 'init_error', 'aged_cell', 'combined']
    L = ['\\begin{longtable}{l' + 'rr' * len(scen) + '}\n\\caption{Single-precision (float32) build versus double precision: steady-state SOC RMSE [\\%] (seed 1, ECM plant, eVTOL). A large ratio indicates loss of numerical robustness in float32; `div\' marks divergence.}\\label{tab:float}\\\\\n']
    L.append('\\toprule estimator & ' + ' & '.join('\\multicolumn{2}{c}{%s}' % tex(c) for c in scen) + ' \\\\\n & ' + ' & '.join('f64 & f32' for _ in scen) + ' \\\\ \\midrule\\endfirsthead\n')
    L.append('\\toprule estimator & ' + ' & '.join('\\multicolumn{2}{c}{%s}' % tex(c) for c in scen) + ' \\\\ \\midrule\\endhead\n')
    for grp in GROUP_ORDER:
        mm = m[m.group == grp]
        for e in list(dict.fromkeys(mm.estimator)):
            cells = []
            for c in scen:
                r = mm[(mm.estimator == e) & (mm.scenario == c)]
                if len(r) == 0: cells += ['--', '--']; continue
                r = r.iloc[0]
                cells += [f'{r.rmse_ss_d * 100:.2f}', f'{r.rmse_ss_f * 100:.2f}' + ('$^{\\mathrm{div}}$' if r.diverged_f else '')]
            L.append('%s & %s \\\\\n' % (tex(e), ' & '.join(cells)))
    L.append('\\bottomrule\\end{longtable}\n')
    write('float_vs_double.tex', ''.join(L))


def profiles_table():
    d = ecm[ecm.scenario == 'nominal']
    g = d.groupby(['group', 'estimator', 'profile']).rmse_ss.mean().reset_index()
    profs = [p for p in ['evtol', 'fixed_wing', 'hover_hold', 'ground_charge'] if p in set(g.profile)]
    L = ['\\begin{longtable}{l' + 'r' * len(profs) + '}\n\\caption{Steady-state SOC RMSE [\\%] on the four mission profiles (nominal scenario, ECM plant).}\\label{tab:profiles}\\\\\n']
    L.append('\\toprule estimator & ' + ' & '.join(tex(p) for p in profs) + ' \\\\ \\midrule\\endfirsthead\n\\toprule estimator & ' + ' & '.join(tex(p) for p in profs) + ' \\\\ \\midrule\\endhead\n')
    for grp in GROUP_ORDER:
        gg = g[g.group == grp]
        for e in list(dict.fromkeys(gg.estimator)):
            cells = []
            for p in profs:
                v = gg[(gg.estimator == e) & (gg.profile == p)].rmse_ss
                cells.append(f'{v.iloc[0] * 100:.2f}' if len(v) else '--')
            L.append('%s & %s \\\\\n' % (tex(e), ' & '.join(cells)))
    L.append('\\bottomrule\\end{longtable}\n')
    write('profiles.tex', ''.join(L))


def pack_tables():
    p = os.path.join(RES, 'summary_pack.csv')
    if not os.path.exists(p): return
    s = pd.read_csv(p); ests = list(dict.fromkeys(s.estimator)); scen = list(dict.fromkeys(s.scenario))
    for e in ests:
        d = s[s.estimator == e].groupby('scenario').agg(rc=('rmse_cells', 'mean'), rmin=('rmse_min', 'mean'), rmean=('rmse_mean', 'mean'), rmax=('rmse_max', 'mean'), me=('max_cell_err', 'mean'), ss=('ss_rmse_cells', 'mean'), dv=('diverged', 'max'), ns=('ns_per_step', 'median'), by=('bytes', 'first'), msg=('messages', 'first'))
        L = ['\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\\caption{%s: pack-level benchmark (12-cell module, eVTOL mission, mean over seeds; all errors in \\%% SOC). RMSE$_\\mathrm{cells}$: per-cell RMSE over all cells; RMSE$_{\\min/\\mathrm{mean}/\\max}$: error of the weakest / mean / strongest-cell SOC; %d bytes of state, %d numbers exchanged per step.}\\label{tab:pack_%s}\n' % (tex(e), d.by.iloc[0], d.msg.iloc[0], safe(e))]
        L.append('\\begin{tabular}{lrrrrrrcr}\\toprule scenario & RMSE$_\\mathrm{cells}$ & RMSE$_{\\min}$ & RMSE$_\\mathrm{mean}$ & RMSE$_{\\max}$ & max cell err. & RMSE$_\\mathrm{ss}$ & div. & ns/step \\\\ \\midrule\n')
        for sc in scen:
            if sc not in d.index: continue
            r = d.loc[sc]
            L.append('%s & %s & %s & %s & %s & %s & %s & %s & %.0f \\\\\n' % (tex(sc), fmt(r.rc), fmt(r.rmin), fmt(r.rmean), fmt(r.rmax), fmt(r.me), fmt(r.ss), 'yes' if r['dv'] else 'no', r.ns))
        L.append('\\bottomrule\\end{tabular}\\end{table}\n')
        write(f'pack_{safe(e)}.tex', ''.join(L))
    g = s.groupby(['estimator', 'scenario']).rmse_cells.mean().reset_index()
    cost = s[s.scenario == 'nominal'].groupby('estimator').agg(ns=('ns_per_step', 'median'), by=('bytes', 'first'), msg=('messages', 'first'))
    L = ['\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\\caption{Pack-level overview: per-cell SOC RMSE [\\%] by scenario (mean over seeds), cost per pack step and communication load.}\\label{tab:pack_overview}\n']
    L.append('\\begin{tabular}{l' + 'r' * len(scen) + 'rrr}\\toprule estimator & ' + ' & '.join('\\rotatebox{60}{%s}' % tex(c) for c in scen) + ' & µs/step & kB & msgs \\\\ \\midrule\n')
    for e in ests:
        cells = [fmt(g[(g.estimator == e) & (g.scenario == c)].rmse_cells.mean()) for c in scen]
        L.append('%s & %s & %.1f & %.1f & %d \\\\\n' % (tex(e), ' & '.join(cells), cost.loc[e].ns / 1e3, cost.loc[e].by / 1e3, cost.loc[e].msg))
    L.append('\\bottomrule\\end{tabular}\\end{table}\n')
    write('pack_overview.tex', ''.join(L))


def imu_tables():
    p = os.path.join(RES, 'summary_imu.csv')
    if not os.path.exists(p): return
    s = pd.read_csv(p); ests = list(dict.fromkeys(s.estimator)); scen = list(dict.fromkeys(s.scenario))
    for e in ests:
        d = s[s.estimator == e].groupby('scenario').agg(a=('rms_angle_deg', 'mean'), r=('rms_roll_deg', 'mean'), pch=('rms_pitch_deg', 'mean'), y=('rms_yaw_deg', 'mean'), mx=('max_angle_deg', 'mean'), ss=('ss_rms_angle_deg', 'mean'), conv=('conv_time', 'mean'), b=('bias_rmse_dps', 'mean'), dv=('diverged', 'max'), ns=('ns_per_step', 'median'), by=('bytes', 'first'))
        L = ['\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\\caption{%s: attitude benchmark (mean over seeds; all angles in degrees; $t_\\mathrm{conv}$: first time the attitude error stays below 2$^\\circ$ for 30\\,s; bias RMSE in deg/s; %d bytes of state).}\\label{tab:imu_%s}\n' % (tex(e), d.by.iloc[0], safe(e))]
        L.append('\\begin{tabular}{lrrrrrrrrcr}\\toprule scenario & RMS & roll & pitch & yaw & max & RMS$_\\mathrm{ss}$ & $t_\\mathrm{conv}$ [s] & bias RMSE & lost & ns/step \\\\ \\midrule\n')
        for sc in scen:
            if sc not in d.index: continue
            r = d.loc[sc]
            conv = '--' if (not np.isfinite(r.conv) or r.conv < 0) else f'{r.conv:.0f}'
            L.append('%s & %.2f & %.2f & %.2f & %.2f & %.1f & %.2f & %s & %.2f & %s & %.0f \\\\\n' % (tex(sc), r.a, r.r, r.pch, r.y, r.mx, r.ss, conv, r.b, 'yes' if r['dv'] else 'no', r.ns))
        L.append('\\bottomrule\\end{tabular}\\end{table}\n')
        write(f'imu_{safe(e)}.tex', ''.join(L))
    g = s.groupby(['estimator', 'scenario']).rms_angle_deg.mean().reset_index()
    cost = s[s.scenario == 'nominal'].groupby('estimator').agg(ns=('ns_per_step', 'median'), by=('bytes', 'first'))
    L = ['\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\\caption{Attitude estimators: RMS attitude error [deg] by scenario (mean over seeds) and cost per 100\\,Hz sample.}\\label{tab:imu_overview}\n']
    L.append('\\begin{tabular}{l' + 'r' * len(scen) + 'rr}\\toprule estimator & ' + ' & '.join('\\rotatebox{60}{%s}' % tex(c) for c in scen) + ' & ns/step & bytes \\\\ \\midrule\n')
    for e in ests:
        cells = ['%.2f' % g[(g.estimator == e) & (g.scenario == c)].rms_angle_deg.mean() for c in scen]
        L.append('%s & %s & %.0f & %d \\\\\n' % (tex(e), ' & '.join(cells), cost.loc[e].ns, cost.loc[e].by))
    L.append('\\bottomrule\\end{tabular}\\end{table}\n')
    write('imu_overview.tex', ''.join(L))


def soh_table():
    p = os.path.join(RES, 'summary_soh.csv')
    if not os.path.exists(p): return
    s = pd.read_csv(p)
    L = ['\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\\caption{Multi-flight state-of-health benchmark: %d consecutive missions with accelerated ageing ($\\times$%.0f); capacity RMSE/MAE over flights 11--%d, final capacity error, mean per-flight SOC RMSE and cost.}\\label{tab:soh_overview}\n' % (s.flights.iloc[0], s.accel.iloc[0], s.flights.iloc[0])]
    L.append('\\begin{tabular}{llrrrrr}\\toprule estimator & family & RMSE $\\hat Q$ [mAh] & MAE $\\hat Q$ [mAh] & final [mAh] & SOC RMSE [\\%] & ns/step \\\\ \\midrule\n')
    for _, r in s.sort_values('capacity_rmse_Ah').iterrows():
        L.append('%s & %s & %.0f & %.0f & %+.0f & %.2f & %.0f \\\\\n' % (tex(r.estimator), tex(r.group), r.capacity_rmse_Ah * 1e3, r.capacity_mae_Ah * 1e3, r.final_capacity_err_Ah * 1e3, r.mean_soc_rmse * 100, r.ns_per_step))
    L.append('\\bottomrule\\end{tabular}\\end{table}\n')
    write('soh_overview.tex', ''.join(L))


def ranking_table():
    """Composite ranking: mean log RMSE_ss over all ECM scenarios (recursive estimators only)."""
    d = ecm[ecm.profile == 'evtol']
    g = d.groupby(['group', 'estimator', 'scenario']).rmse_ss.mean().reset_index()
    g['lr'] = np.log10(np.clip(g.rmse_ss, 1e-4, None))
    piv = g.groupby(['group', 'estimator']).agg(score=('lr', 'mean'), worst=('rmse_ss', 'max'), nominal=('rmse_ss', lambda v: v.iloc[list(g.loc[v.index].scenario).index('nominal')] if 'nominal' in list(g.loc[v.index].scenario) else np.nan)).reset_index()
    cost = d[d.scenario == 'nominal'].groupby('estimator').ns_per_step.median()
    piv['ns'] = piv.estimator.map(cost)
    piv = piv.sort_values('score')
    L = ['\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}\n\\begin{longtable}{rllrrrr}\n\\caption{Composite ranking of all cell-level estimators on the ECM plant: geometric-mean steady-state RMSE over the 12 scenarios (lower is better), worst-scenario RMSE, nominal RMSE and cost. Offline smoothers are listed for reference.}\\label{tab:ranking}\\\\\n']
    L.append('\\toprule rank & estimator & family & geo.-mean [\\%] & worst [\\%] & nominal [\\%] & ns/step \\\\ \\midrule\\endfirsthead\n\\toprule rank & estimator & family & geo.-mean [\\%] & worst [\\%] & nominal [\\%] & ns/step \\\\ \\midrule\\endhead\n')
    for k, (_, r) in enumerate(piv.iterrows()):
        L.append('%d & %s & %s & %.2f & %.2f & %.2f & %.0f \\\\\n' % (k + 1, tex(r.estimator), tex(r.group), 100 * 10 ** r.score, 100 * r.worst, 100 * r.nominal, r.ns))
    L.append('\\bottomrule\\end{longtable}\\endgroup\n')
    write('ranking.tex', ''.join(L))


if __name__ == '__main__':
    per_estimator_tables()
    overview_table('ecm', ecm, 'ecm'); overview_table('spm', spm, 'spm')
    runtime_table(); float_table(); profiles_table(); ranking_table()
    pack_tables(); imu_tables(); soh_table()
    print('tables written to', TAB)
