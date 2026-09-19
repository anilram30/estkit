#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Generate every figure of the report from results/final (and results/model).

Usage: python3 scripts/make_figures.py [results/final] [report/figures]
"""
import os, sys, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shrink_figures import shrink as _shrink   # 256-colour palette PNGs, ~3x smaller

RES = sys.argv[1] if len(sys.argv) > 1 else 'results/final'
FIG = sys.argv[2] if len(sys.argv) > 2 else 'report/figures'
MODEL = 'results/model'
os.makedirs(FIG, exist_ok=True)

# fixed categorical palette (validated, colour-blind safe adjacent pairs)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GRAY = '#6b6b68'; LIGHT = '#c9c8c2'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.2, 'figure.dpi': 130, 'savefig.dpi': 150, 'legend.frameon': False, 'axes.titleweight': 'bold', 'axes.titlesize': 9.5})

GROUP_ORDER = ['baselines', 'kalman', 'adaptive_kalman', 'robust_kalman', 'observers', 'soh', 'particle', 'learning', 'optimization', 'smoothers']
GROUP_LABEL = {'baselines': 'Baselines', 'kalman': 'Classical / sigma-point KF', 'adaptive_kalman': 'Adaptive KF', 'robust_kalman': 'Robust / embedded KF', 'observers': 'Deterministic observers',
               'soh': 'Joint / dual (SOH)', 'particle': 'Particle / ensemble', 'learning': 'Data-driven', 'optimization': 'Moving-horizon', 'smoothers': 'Smoothers (offline)'}
SCEN_ORDER = ['nominal', 'init_error', 'current_bias', 'noise_step', 'outliers', 'cold', 'hot', 'aged_cell', 'param_mismatch', 'sensor_dropout', 'preisach', 'combined']


def safe(name):
    return re.sub(r'[^A-Za-z0-9]', '', name)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight')
    plt.close(fig)
    _shrink(os.path.join(FIG, name))


# ----------------------------------------------------------------------------- cell-level per-estimator figures
def cell_estimator_figure(plant, est, scenario, title_suffix=''):
    path = os.path.join(RES, 'ts', f'{plant}__{est}__evtol__{scenario}__s1.csv')
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path)
    t = d.t / 60.0
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True, gridspec_kw={'height_ratios': [2.0, 1.4, 1.2]})
    ax = axs[0]
    ax.plot(t, d.soc_true, color=GRAY, lw=1.6, label='true SOC')
    ax.plot(t, d.soc_est, color=PAL[0], lw=1.1, label=f'{est} estimate')
    if d.soc_lower.notna().any() and d.soc_upper.notna().any():
        ax.fill_between(t, d.soc_lower, d.soc_upper, color=PAL[0], alpha=0.12, label='guaranteed interval')
    ax.set_ylabel('SOC [-]'); ax.set_ylim(-0.02, 1.05); ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.set_title(f'{est} — {plant.upper()} plant, eVTOL mission, scenario "{scenario}"{title_suffix}')
    ax2 = ax.twinx(); ax2.plot(t, d.i_meas, color=LIGHT, lw=0.6, zorder=0); ax2.set_ylabel('current [A]', color=GRAY); ax2.grid(False); ax2.tick_params(axis='y', colors=GRAY)
    ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color(LIGHT)
    ax = axs[1]
    err = (d.soc_est - d.soc_true) * 100
    ax.plot(t, err, color=PAL[1], lw=1.0, label='estimation error')
    if d.soc_std.notna().any() and (d.soc_std > 0).any():
        s = d.soc_std * 100 * 2
        ax.fill_between(t, -s, s, color=PAL[1], alpha=0.15, label='±2σ (filter)')
    lim = max(2.0, min(40.0, np.nanpercentile(np.abs(err), 99.5) * 1.3))
    ax.set_ylim(-lim, lim); ax.set_ylabel('SOC error [%]'); ax.axhline(0, color=GRAY, lw=0.5); ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax = axs[2]
    if d.v_pred.notna().any():
        res = (d.v_meas - d.v_pred) * 1e3
        ax.plot(t, res, color=PAL[2], lw=0.8, label='innovation v_meas − v_pred')
        lim = max(5.0, min(300.0, np.nanpercentile(np.abs(res), 99.5) * 1.3))
        ax.set_ylim(-lim, lim)
    else:
        ax.text(0.5, 0.5, 'no voltage prediction for this estimator', ha='center', va='center', transform=ax.transAxes, color=GRAY)
    ax.set_ylabel('residual [mV]'); ax.set_xlabel('time [min]'); ax.axhline(0, color=GRAY, lw=0.5); ax.legend(loc='upper right', fontsize=8)
    savefig(fig, f'cell{"" if plant == "ecm" else "spm"}_{safe(est)}_{scenario}.png')
    return True


def all_cell_figures():
    summ = pd.read_csv(os.path.join(RES, 'summary_ecm.csv'))
    ests = list(dict.fromkeys(summ.estimator))
    n = 0
    for est in ests:
        for sc in ['nominal', 'init_error', 'current_bias', 'combined']:
            n += cell_estimator_figure('ecm', est, sc)
        for sc in ['nominal', 'cold']:
            n += cell_estimator_figure('spm', est, sc)
    print(f'cell figures: {n}')


# ----------------------------------------------------------------------------- overview figures
def load_summary(plant):
    p = os.path.join(RES, f'summary_{plant}.csv')
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p)
    return s


def overview_heatmap(plant):
    s = load_summary(plant)
    if s is None:
        return
    s = s[s.profile == 'evtol']
    g = s.groupby(['group', 'estimator', 'scenario']).rmse_ss.mean().reset_index()
    order = [e for grp in GROUP_ORDER for e in dict.fromkeys(g[g.group == grp].estimator)]
    scen = [c for c in SCEN_ORDER if c in set(g.scenario)]
    M = np.full((len(order), len(scen)), np.nan)
    for i, e in enumerate(order):
        for j, c in enumerate(scen):
            v = g[(g.estimator == e) & (g.scenario == c)].rmse_ss
            if len(v): M[i, j] = v.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(8.5, 0.24 * len(order) + 1.6))
    cmap = plt.get_cmap('Blues')
    im = ax.imshow(np.log10(np.clip(M, 0.02, 40)), cmap=cmap, aspect='auto', vmin=np.log10(0.02), vmax=np.log10(40))
    ax.set_xticks(range(len(scen))); ax.set_xticklabels(scen, rotation=45, ha='right')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    ax.grid(False)
    for i in range(len(order)):
        for j in range(len(scen)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}' if M[i, j] < 10 else f'{M[i, j]:.0f}', ha='center', va='center', fontsize=5.5, color='white' if M[i, j] > 1.5 else '#0b0b0b')
    # group separators
    y = 0
    for grp in GROUP_ORDER:
        k = sum(1 for e in order if e in set(g[g.group == grp].estimator))
        if k: y += k; ax.axhline(y - 0.5, color='white', lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label('steady-state SOC RMSE [%] (log colour scale)')
    cb.set_ticks(np.log10([0.02, 0.05, 0.1, 0.3, 1, 3, 10, 40])); cb.set_ticklabels(['0.02', '0.05', '0.1', '0.3', '1', '3', '10', '40'])
    ax.set_title(f'Steady-state SOC RMSE [%] — {plant.upper()} plant, eVTOL mission (mean over seeds)')
    savefig(fig, f'overview_heatmap_{plant}.png')


def overview_runtime():
    s = load_summary('ecm'); s = s[(s.profile == 'evtol') & (s.scenario == 'nominal')]
    g = s.groupby(['group', 'estimator']).agg(rmse=('rmse_ss', 'mean'), ns=('ns_per_step', 'median'), bytes=('bytes', 'first')).reset_index()
    g = g[~g.group.isin(['smoothers'])]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for k, grp in enumerate(GROUP_ORDER):
        gg = g[g.group == grp]
        if len(gg) == 0: continue
        ax.scatter(gg.ns / 1e3, gg.rmse * 100, s=28, color=PAL[k % 8], marker='o' if k < 8 else 's', label=GROUP_LABEL[grp], edgecolor='white', linewidth=0.5, zorder=3)
        for _, r in gg.iterrows():
            ax.annotate(r.estimator, (r.ns / 1e3, r.rmse * 100), fontsize=5.5, xytext=(3, 2), textcoords='offset points', color='#333')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('execution time per step [µs]  (x86-64, single core, -O2)'); ax.set_ylabel('steady-state SOC RMSE, nominal scenario [%]')
    ax.set_title('Accuracy versus computational cost (ECM plant, nominal eVTOL mission)'); ax.legend(fontsize=7, loc='upper right', ncol=2)
    savefig(fig, 'overview_runtime_vs_rmse.png')


def overview_bars(plant, scenarios=('nominal', 'init_error', 'current_bias', 'aged_cell', 'outliers', 'combined')):
    s = load_summary(plant)
    if s is None: return
    s = s[s.profile == 'evtol']
    g = s.groupby(['group', 'estimator', 'scenario']).rmse_ss.mean().reset_index()
    for grp in GROUP_ORDER:
        gg = g[g.group == grp]
        if len(gg) == 0: continue
        ests = list(dict.fromkeys(gg.estimator))
        scen = [c for c in scenarios if c in set(gg.scenario)]
        fig, ax = plt.subplots(figsize=(8.2, 3.6))
        w = 0.8 / len(scen)
        for j, c in enumerate(scen):
            vals = [gg[(gg.estimator == e) & (gg.scenario == c)].rmse_ss.mean() * 100 for e in ests]
            ax.bar(np.arange(len(ests)) + (j - len(scen) / 2 + 0.5) * w, vals, width=w * 0.92, color=PAL[j % 8], label=c, zorder=3)
        ekf = g[(g.estimator == 'EKF') & (g.scenario == 'nominal')].rmse_ss.mean() * 100
        ax.axhline(ekf, color=GRAY, lw=0.8, ls='--'); ax.text(len(ests) - 0.5, ekf, ' EKF nominal', color=GRAY, fontsize=7, va='bottom', ha='right')
        ax.set_yscale('log'); ax.set_xticks(range(len(ests))); ax.set_xticklabels(ests, rotation=30, ha='right', fontsize=7.5)
        ax.set_ylabel('steady-state SOC RMSE [%]'); ax.set_title(f'{GROUP_LABEL[grp]} — {plant.upper()} plant, eVTOL mission'); ax.legend(fontsize=7, ncol=len(scen))
        savefig(fig, f'family_{grp}_{plant}.png')


def overview_float():
    pf = os.path.join(RES, 'float', 'summary_ecm.csv')
    if not os.path.exists(pf): return
    f = pd.read_csv(pf); d = load_summary('ecm')
    d = d[(d.profile == 'evtol') & (d.seed == 1)]
    m = f.merge(d, on=['estimator', 'scenario'], suffixes=('_f', '_d'))
    m = m[m.scenario.isin(['nominal', 'init_error', 'combined'])]
    piv = m.pivot_table(index='estimator', columns='scenario', values=['rmse_ss_f', 'rmse_ss_d'])
    ests = [e for e in dict.fromkeys(d.estimator) if e in piv.index]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(ests))
    for j, sc in enumerate(['nominal', 'init_error', 'combined']):
        ratio = [piv.loc[e, ('rmse_ss_f', sc)] / max(piv.loc[e, ('rmse_ss_d', sc)], 1e-6) for e in ests]
        ax.bar(x + (j - 1) * 0.27, ratio, width=0.25, color=PAL[j], label=sc, zorder=3)
    ax.axhline(1, color=GRAY, lw=0.8); ax.set_yscale('log'); ax.set_ylim(0.3, 30)
    ax.set_xticks(x); ax.set_xticklabels(ests, rotation=90, fontsize=6); ax.set_ylabel('RMSE(float32) / RMSE(float64)')
    ax.set_title('Single-precision build: steady-state SOC RMSE relative to the double-precision build (ECM plant)'); ax.legend(fontsize=7)
    savefig(fig, 'overview_float_vs_double.png')


# ----------------------------------------------------------------------------- datasets & model
def dataset_figures():
    for prof in ['evtol', 'fixed_wing', 'hover_hold', 'ground_charge']:
        p = os.path.join(RES, 'ts', f'ecm__CoulombCounting__{prof}__nominal__s1.csv')
        if not os.path.exists(p): p = os.path.join(RES, 'profile_ts', 'ts', f'ecm__CoulombCounting__{prof}__nominal__s1.csv')
        if not os.path.exists(p): continue
        d = pd.read_csv(p); t = d.t / 60
        fig, axs = plt.subplots(4, 1, figsize=(7.2, 6.4), sharex=True)
        axs[0].plot(t, d.i_meas, color=PAL[0], lw=0.8); axs[0].set_ylabel('current [A]')
        axs[1].plot(t, d.v_true, color=PAL[1], lw=0.9); axs[1].set_ylabel('voltage [V]')
        axs[2].plot(t, d.soc_true, color=PAL[2]); axs[2].set_ylabel('SOC [-]')
        axs[3].plot(t, d.T_true - 273.15, color=PAL[3]); axs[3].set_ylabel('cell temp. [°C]'); axs[3].set_xlabel('time [min]')
        axs[0].set_title(f'Mission profile "{prof}" — ECM plant, nominal sensors (seed 1)')
        savefig(fig, f'dataset_{prof}.png')
    # OCV
    o = pd.read_csv(os.path.join(MODEL, 'ocv.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.0))
    axs[0].plot(o.z, o.ocv, color=PAL[0]); axs[0].set_xlabel('SOC z'); axs[0].set_ylabel('OCV [V]'); axs[0].set_title('Open-circuit voltage (Chen 2020 electrodes)')
    axs[1].plot(o.z, o.docv_dz, color=PAL[1]); axs[1].set_xlabel('SOC z'); axs[1].set_ylabel('dOCV/dz [V]'); axs[1].set_title('OCV slope (observability gain)')
    axs[2].plot(o.x_n, o.U_n, color=PAL[2], label='graphite U_n(x)'); axs[2].plot(o.y_p, o.U_p, color=PAL[3], label='NMC811 U_p(y)'); axs[2].set_xlabel('stoichiometry'); axs[2].set_ylabel('potential vs Li [V]'); axs[2].legend(fontsize=7); axs[2].set_title('Electrode potentials')
    fig.tight_layout(); savefig(fig, 'model_ocv.png')
    # Arrhenius
    a = pd.read_csv(os.path.join(MODEL, 'arrhenius.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.0))
    axs[0].plot(a.T_C, a.R0 * 1e3, color=PAL[0], label='R0'); axs[0].plot(a.T_C, a.R1 * 1e3, color=PAL[1], label='R1'); axs[0].plot(a.T_C, a.R2 * 1e3, color=PAL[2], label='R2')
    axs[0].set_yscale('log'); axs[0].set_xlabel('temperature [°C]'); axs[0].set_ylabel('resistance [mΩ]'); axs[0].legend(fontsize=7); axs[0].set_title('Arrhenius impedance scaling (nominal ECM)')
    axs[1].plot(a.T_C, a.tau1, color=PAL[0], label='τ1'); axs[1].plot(a.T_C, a.tau2, color=PAL[1], label='τ2'); axs[1].set_yscale('log'); axs[1].set_xlabel('temperature [°C]'); axs[1].set_ylabel('time constant [s]'); axs[1].legend(fontsize=7); axs[1].set_title('RC time constants')
    fig.tight_layout(); savefig(fig, 'model_arrhenius.png')
    # hysteresis
    pr = pd.read_csv(os.path.join(MODEL, 'preisach.csv')); h1 = pd.read_csv(os.path.join(MODEL, 'hysteresis_onestate.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.0))
    for k, (br, c) in enumerate([('major_down', PAL[0]), ('major_up', PAL[1]), ('minor_down', PAL[2]), ('minor_up', PAL[3]), ('minor_down2', PAL[4])]):
        q = pr[pr.branch == br]; axs[0].plot(q.z, q.v_hyst * 1e3, color=c, label=br.replace('_', ' '))
    axs[0].set_xlabel('SOC z'); axs[0].set_ylabel('hysteresis voltage [mV]'); axs[0].set_title('Preisach operator (M_p = 15 mV): major and minor loops'); axs[0].legend(fontsize=6.5)
    for br, c in [('discharge', PAL[0]), ('charge', PAL[1])]:
        q = h1[h1.branch == br]; axs[1].plot(q.z, q.v_hyst * 1e3, color=c, label=br)
    axs[1].set_xlabel('SOC z'); axs[1].set_ylabel('M h [mV]'); axs[1].set_title('One-state hysteresis (Plett), 1C cycle'); axs[1].legend(fontsize=7)
    fig.tight_layout(); savefig(fig, 'model_hysteresis.png')
    # ageing
    ag = pd.read_csv(os.path.join(MODEL, 'aging.csv')); ac = pd.read_csv(os.path.join(MODEL, 'aging_calendar.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.0))
    for k, T in enumerate([10, 25, 45]):
        q = ag[ag.T_C == T]; axs[0].plot(q.efc, q.Q_loss * 100, color=PAL[k], label=f'{T} °C')
        q = ac[ac.T_C == T]; axs[1].plot(q.days, q.Q_loss * 100, color=PAL[k], label=f'{T} °C')
    axs[0].set_xlabel('equivalent full cycles'); axs[0].set_ylabel('capacity loss [%]'); axs[0].set_title('Cycle ageing (Arrhenius × Ah^0.55)'); axs[0].legend(fontsize=7)
    axs[1].set_xlabel('storage time [days]'); axs[1].set_ylabel('capacity loss [%]'); axs[1].set_title('Calendar ageing (√t law)'); axs[1].legend(fontsize=7)
    fig.tight_layout(); savefig(fig, 'model_aging.png')
    # SPM identification residual
    rp = os.path.join(RES, 'ident_spm_ecm_record.csv')
    if os.path.exists(rp):
        r = pd.read_csv(rp); t = r.t / 60
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw={'height_ratios': [1, 2, 1.2]})
        axs[0].plot(t, r.i, color=PAL[0], lw=0.7); axs[0].set_ylabel('current [A]')
        axs[1].plot(t, r.v_spm, color=GRAY, lw=1.4, label='SPM plant'); axs[1].plot(t, r.v_ecm, color=PAL[1], lw=0.8, ls='--', label='fitted 2-RC ECM'); axs[1].set_ylabel('voltage [V]'); axs[1].legend(fontsize=7)
        axs[2].plot(t, (r.v_spm - r.v_ecm) * 1e3, color=PAL[2], lw=0.7); axs[2].set_ylabel('residual [mV]'); axs[2].set_xlabel('time [min]')
        axs[0].set_title('ECM identification on the SPM plant (pulse train + fixed-wing mission, 25 °C)')
        savefig(fig, 'model_spm_fit.png')
    sp = pd.read_csv(os.path.join(MODEL, 'spm_pulse.csv'))
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    axs[0].plot(sp.t / 60, sp.v, color=PAL[0]); axs[0].set_ylabel('voltage [V]'); axs[0].set_title('SPM plant: 1C for 10 min, rest, 4C pulse — surface vs bulk stoichiometry')
    axs[1].plot(sp.t / 60, sp.soc, color=GRAY, label='bulk SOC'); axs[1].plot(sp.t / 60, (sp.x_surf - 0.0434) / (0.9014 - 0.0434), color=PAL[1], label='graphite surface (normalised)'); axs[1].plot(sp.t / 60, 1 - (sp.y_surf - 0.27) / (0.8426 - 0.27), color=PAL[2], label='NMC surface (normalised)')
    axs[1].set_ylabel('stoichiometry [-]'); axs[1].set_xlabel('time [min]'); axs[1].legend(fontsize=7)
    savefig(fig, 'model_spm_pulse.png')
    # IMU dataset
    ip = os.path.join(RES, 'imu_nominal_seed1.csv')
    if os.path.exists(ip):
        d = pd.read_csv(ip)
        # euler from quaternion
        w, x, y, z = d.qw, d.qx, d.qy, d.qz
        roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))); pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))); yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        fig, axs = plt.subplots(4, 1, figsize=(7.2, 7.0), sharex=True)
        tt = d.t / 60
        axs[0].plot(tt, roll, color=PAL[0], label='roll'); axs[0].plot(tt, pitch, color=PAL[1], label='pitch'); axs[0].plot(tt, yaw, color=PAL[2], label='yaw'); axs[0].set_ylabel('attitude [deg]'); axs[0].legend(fontsize=7, ncol=3); axs[0].set_title('Synthetic flight for the attitude benchmark (nominal, seed 1)')
        axs[1].plot(tt, np.degrees(d.gx), color=PAL[0], lw=0.5); axs[1].plot(tt, np.degrees(d.gy), color=PAL[1], lw=0.5); axs[1].plot(tt, np.degrees(d.gz), color=PAL[2], lw=0.5); axs[1].set_ylabel('gyro [deg/s]')
        axs[2].plot(tt, d.ax, color=PAL[0], lw=0.5); axs[2].plot(tt, d.ay, color=PAL[1], lw=0.5); axs[2].plot(tt, d.az, color=PAL[2], lw=0.5); axs[2].set_ylabel('specific force [m/s²]')
        axs[3].plot(tt, d.mx, color=PAL[0], lw=0.5); axs[3].plot(tt, d.my, color=PAL[1], lw=0.5); axs[3].plot(tt, d.mz, color=PAL[2], lw=0.5); axs[3].set_ylabel('magnetometer [µT]'); axs[3].set_xlabel('time [min]')
        savefig(fig, 'dataset_imu.png')
    # pack dataset
    pp = os.path.join(RES, 'ts', 'pack__Decentralized-EKF__evtol__nominal__s1.csv')
    if os.path.exists(pp):
        d = pd.read_csv(pp)
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        for c in range(12):
            ax.plot(d.t / 60, d[f'soc_true_{c}'], color=PAL[c % 8], lw=0.7, alpha=0.8)
        ax.plot(d.t / 60, d.soc_true_min, color='k', lw=1.4, ls='--', label='weakest cell'); ax.plot(d.t / 60, d.soc_true_max, color='k', lw=1.4, ls=':', label='strongest cell')
        ax.set_xlabel('time [min]'); ax.set_ylabel('cell SOC [-]'); ax.set_title('12-cell series module: true cell SOCs (nominal spread, seed 1)'); ax.legend(fontsize=7)
        savefig(fig, 'dataset_pack.png')


# ----------------------------------------------------------------------------- pack / imu / soh
def pack_figures():
    p = os.path.join(RES, 'summary_pack.csv')
    if not os.path.exists(p): return
    s = pd.read_csv(p)
    ests = list(dict.fromkeys(s.estimator))
    for est in ests:
        for sc in ['nominal', 'weak_cell', 'large_imbalance']:
            path = os.path.join(RES, 'ts', f'pack__{est}__evtol__{sc}__s1.csv')
            if not os.path.exists(path): continue
            d = pd.read_csv(path); t = d.t / 60
            fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={'height_ratios': [2, 1.3]})
            axs[0].plot(t, d.soc_true_min, color=GRAY, lw=1.5, label='true weakest cell'); axs[0].plot(t, d.soc_true_max, color=LIGHT, lw=1.5, label='true strongest cell')
            axs[0].plot(t, d.soc_est_min, color=PAL[0], lw=1.0, label=f'{est}: est. weakest'); axs[0].plot(t, d.soc_est_max, color=PAL[1], lw=1.0, label=f'{est}: est. strongest')
            axs[0].set_ylabel('SOC [-]'); axs[0].legend(fontsize=7, ncol=2); axs[0].set_title(f'{est} — pack scenario "{sc}"')
            for c in range(12):
                axs[1].plot(t, (d[f'soc_est_{c}'] - d[f'soc_true_{c}']) * 100, color=PAL[c % 8], lw=0.6, alpha=0.85)
            axs[1].axhline(0, color=GRAY, lw=0.5); axs[1].set_ylabel('per-cell SOC error [%]'); axs[1].set_xlabel('time [min]')
            lim = max(2, min(30, np.nanpercentile(np.abs(np.concatenate([(d[f'soc_est_{c}'] - d[f'soc_true_{c}']).values * 100 for c in range(12)])), 99.5) * 1.3)); axs[1].set_ylim(-lim, lim)
            savefig(fig, f'pack_{safe(est)}_{sc}.png')
    g = s.groupby(['estimator', 'scenario']).agg(rmse=('rmse_cells', 'mean'), rmin=('rmse_min', 'mean'), ns=('ns_per_step', 'median'), by=('bytes', 'first')).reset_index()
    scen = list(dict.fromkeys(s.scenario))
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={'width_ratios': [2.2, 1]})
    w = 0.8 / len(scen)
    for j, c in enumerate(scen):
        vals = [g[(g.estimator == e) & (g.scenario == c)].rmse.mean() * 100 for e in ests]
        axs[0].bar(np.arange(len(ests)) + (j - len(scen) / 2 + 0.5) * w, vals, width=w * 0.9, color=PAL[j % 8], label=c, zorder=3)
    axs[0].set_xticks(range(len(ests))); axs[0].set_xticklabels(ests, rotation=25, ha='right', fontsize=7.5); axs[0].set_ylabel('per-cell SOC RMSE [%]'); axs[0].legend(fontsize=6.5, ncol=2); axs[0].set_title('Pack-level accuracy by scenario')
    gn = g[g.scenario == 'nominal']
    axs[1].scatter(gn.ns / 1e3, gn.by / 1e3, color=PAL[0], s=30, zorder=3)
    for _, r in gn.iterrows(): axs[1].annotate(r.estimator, (r.ns / 1e3, r.by / 1e3), fontsize=6, xytext=(3, 2), textcoords='offset points')
    axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel('µs per pack step'); axs[1].set_ylabel('state memory [kB]'); axs[1].set_title('Cost')
    fig.tight_layout(); savefig(fig, 'pack_overview.png')


def imu_figures():
    p = os.path.join(RES, 'summary_imu.csv')
    if not os.path.exists(p): return
    s = pd.read_csv(p); ests = list(dict.fromkeys(s.estimator))
    for est in ests:
        for sc in ['nominal', 'init_error', 'gyro_bias']:
            path = os.path.join(RES, 'ts', f'imu__{est}__{sc}__s1.csv')
            if not os.path.exists(path): continue
            d = pd.read_csv(path); t = d.t / 60
            fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True, gridspec_kw={'height_ratios': [2, 1.3, 1]})
            for k, (a, lab) in enumerate([('roll', 'roll'), ('pitch', 'pitch'), ('yaw', 'yaw')]):
                axs[0].plot(t, d[f'{a}_true'], color=GRAY, lw=1.3, alpha=0.7)
                axs[0].plot(t, d[f'{a}_est'], color=PAL[k], lw=0.9, label=f'{lab} (est.)')
            axs[0].set_ylabel('attitude [deg]'); axs[0].legend(fontsize=7, ncol=3); axs[0].set_title(f'{est} — attitude scenario "{sc}" (grey: truth)')
            axs[1].plot(t, d.angle_err_deg, color=PAL[1], lw=0.8); axs[1].set_ylabel('attitude error [deg]'); axs[1].set_ylim(0, max(2, min(60, np.nanpercentile(d.angle_err_deg, 99.5) * 1.3)))
            axs[2].plot(t, d.bias_true_x, color=GRAY, lw=1.2, label='true gyro bias x'); axs[2].plot(t, d.bias_est_x, color=PAL[2], lw=0.9, label='estimated'); axs[2].set_ylabel('bias [deg/s]'); axs[2].set_xlabel('time [min]'); axs[2].legend(fontsize=7)
            savefig(fig, f'imu_{safe(est)}_{sc}.png')
    g = s.groupby(['estimator', 'scenario']).rms_angle_deg.mean().reset_index()
    scen = list(dict.fromkeys(s.scenario))
    fig, ax = plt.subplots(figsize=(8.5, 3.6)); w = 0.8 / len(scen)
    for j, c in enumerate(scen):
        vals = [g[(g.estimator == e) & (g.scenario == c)].rms_angle_deg.mean() for e in ests]
        ax.bar(np.arange(len(ests)) + (j - len(scen) / 2 + 0.5) * w, vals, width=w * 0.9, color=PAL[j % 8], label=c, zorder=3)
    ax.set_yscale('log'); ax.set_xticks(range(len(ests))); ax.set_xticklabels(ests, rotation=25, ha='right', fontsize=7.5); ax.set_ylabel('attitude RMS error [deg]'); ax.legend(fontsize=6.5, ncol=3); ax.set_title('Attitude estimators by scenario (mean over seeds)')
    savefig(fig, 'imu_overview.png')


def soh_figures():
    p = os.path.join(RES, 'soh_flights.csv')
    if not os.path.exists(p): return
    d = pd.read_csv(p); ests = list(dict.fromkeys(d.estimator))
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
    q = d[d.estimator == ests[0]]
    axs[0].plot(q.flight, q.Q_true, color='k', lw=1.8, label='true capacity')
    for k, e in enumerate(ests):
        q = d[d.estimator == e]; axs[0].plot(q.flight, q.Q_est, color=PAL[k % 8], lw=1.0, label=e); axs[1].plot(q.flight, q.Q_err * 1000, color=PAL[k % 8], lw=1.0)
    axs[0].set_ylabel('capacity [Ah]'); axs[0].legend(fontsize=7, ncol=3); axs[0].set_title('Capacity tracking over consecutive missions (accelerated ageing ×3, one cycle per day)')
    axs[1].axhline(0, color=GRAY, lw=0.5); axs[1].set_ylabel('capacity error [mAh]'); axs[1].set_xlabel('flight number'); axs[1].set_ylim(-400, 400)
    savefig(fig, 'soh_capacity_tracking.png')
    for k, e in enumerate(ests):
        q = d[d.estimator == e]
        fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
        axs[0].plot(q.flight, q.Q_true, color='k', lw=1.6, label='true'); axs[0].plot(q.flight, q.Q_est, color=PAL[0], lw=1.0, label=f'{e}'); axs[0].set_ylabel('capacity [Ah]'); axs[0].legend(fontsize=7); axs[0].set_title(f'{e}: capacity and per-flight SOC accuracy over 120 missions')
        axs[1].plot(q.flight, q.soc_rmse_flight * 100, color=PAL[1], lw=1.0); axs[1].set_ylabel('flight SOC RMSE [%]'); axs[1].set_xlabel('flight number')
        savefig(fig, f'soh_{safe(e)}.png')


if __name__ == '__main__':
    dataset_figures()
    all_cell_figures()
    for pl in ['ecm', 'spm']:
        overview_heatmap(pl); overview_bars(pl)
    overview_runtime(); overview_float()
    pack_figures(); imu_figures(); soh_figures()
    print('figures written to', FIG)
