#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Figures for the synthesis chapters (Part XIV) of the report, generated from results/final.

Usage: python3 scripts/make_results_figures.py [results/final] [report/figures]

Produces
  res_spm_traces.png           SOC error traces of selected estimators on the SPM plant, nominal mission
  res_ecm_combined_traces.png  SOC error traces of selected estimators on the ECM plant, `combined` scenario
  res_ranking_bars.png         geometric-mean and worst-case steady-state RMSE of every online estimator (ECM)
  res_ecm_vs_spm.png           nominal steady-state RMSE on the ECM plant against the SPM plant, per estimator
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shrink_figures import shrink as _shrink   # 256-colour palette PNGs, ~3x smaller

RES = sys.argv[1] if len(sys.argv) > 1 else 'results/final'
FIG = sys.argv[2] if len(sys.argv) > 2 else 'report/figures'
os.makedirs(FIG, exist_ok=True)

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GRAY = '#6b6b68'; LIGHT = '#c9c8c2'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 1.2, 'figure.dpi': 130, 'savefig.dpi': 150,
                     'legend.frameon': False, 'axes.titleweight': 'bold', 'axes.titlesize': 9.5})
GROUP_ORDER = ['baselines', 'kalman', 'adaptive_kalman', 'robust_kalman', 'observers', 'soh', 'particle', 'learning', 'optimization', 'smoothers']
GROUP_LABEL = {'baselines': 'Baselines', 'kalman': 'Classical / sigma-point KF', 'adaptive_kalman': 'Adaptive KF',
               'robust_kalman': 'Robust / embedded KF', 'observers': 'Deterministic observers', 'soh': 'Joint / dual (SOH)',
               'particle': 'Particle / ensemble', 'learning': 'Data-driven', 'optimization': 'Moving-horizon', 'smoothers': 'Smoothers (offline)'}
GROUP_COLOR = {g: PAL[i % len(PAL)] for i, g in enumerate(GROUP_ORDER)}
GROUP_COLOR['optimization'] = '#0b5394'; GROUP_COLOR['smoothers'] = GRAY


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight'); plt.close(fig)
    _shrink(os.path.join(FIG, name))


def traces(plant, scenario, ests, name, title, ylim):
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    first = None
    for k, est in enumerate(ests):
        p = os.path.join(RES, 'ts', f'{plant}__{est}__evtol__{scenario}__s1.csv')
        if not os.path.exists(p):
            print('missing', p); continue
        d = pd.read_csv(p)
        if first is None:
            first = d
        ax.plot(d.t / 60, (d.soc_est - d.soc_true) * 100, color=PAL[k % len(PAL)] if k < len(PAL) else GRAY,
                lw=1.1 if k < len(PAL) else 0.9, ls='-' if k < len(PAL) else '--', label=est)
    if first is not None:
        ax2 = ax.twinx(); ax2.plot(first.t / 60, first.i_meas, color=LIGHT, lw=0.6, zorder=0); ax2.grid(False)
        ax2.set_ylabel('current [A]', color=GRAY); ax2.tick_params(axis='y', colors=GRAY)
        ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color(LIGHT)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlabel('time [min]'); ax.set_ylabel('SOC error [% of capacity]'); ax.set_ylim(*ylim)
    ax.set_title(title); ax.legend(loc='lower left', ncol=4, fontsize=7.5)
    ax.set_zorder(ax2.get_zorder() + 1) if first is not None else None
    ax.patch.set_visible(False)
    savefig(fig, name)


def ranking_bars(name):
    d = pd.read_csv(os.path.join(RES, 'summary_ecm.csv'))
    e = d[d.profile == 'evtol']
    g = e.groupby(['estimator', 'scenario'])['rmse_ss'].mean().unstack() * 100
    fam = e.groupby('estimator')['group'].first()
    geo = np.exp(np.log(g.clip(lower=1e-3)).mean(axis=1)); worst = g.max(axis=1)
    t = pd.DataFrame({'geo': geo, 'worst': worst, 'fam': fam}).sort_values('geo', ascending=False)
    fig, ax = plt.subplots(figsize=(7.4, 11.5))
    y = np.arange(len(t))
    ax.barh(y, t.geo, color=[GROUP_COLOR[f] for f in t.fam], height=0.7)
    ax.plot(t.worst, y, 'k|', ms=6, mew=1.2, label='worst scenario')
    ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=7.5); ax.set_xscale('log')
    ax.set_xlim(0.1, 40); ax.set_xlabel('steady-state SOC RMSE [% of capacity]: bar = geometric mean over 12 scenarios, tick = worst scenario')
    ax.set_title('Composite ranking of the 70 cell-level estimators, ECM plant')
    from matplotlib.patches import Patch
    handles = [Patch(color=GROUP_COLOR[g_], label=GROUP_LABEL[g_]) for g_ in GROUP_ORDER]
    handles.append(plt.Line2D([], [], color='k', marker='|', ls='', label='worst scenario'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.045), fontsize=7.5, ncol=3)
    ax.set_ylim(-0.7, len(t) - 0.3)
    savefig(fig, name)


def ecm_vs_spm(name):
    d = pd.read_csv(os.path.join(RES, 'summary_ecm.csv')); s = pd.read_csv(os.path.join(RES, 'summary_spm.csv'))
    e = d[(d.profile == 'evtol') & (d.scenario == 'nominal')].groupby('estimator')['rmse_ss'].mean() * 100
    q = s[s.scenario == 'nominal'].groupby('estimator')['rmse_ss'].mean() * 100
    fam = d.groupby('estimator')['group'].first()
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for g_ in GROUP_ORDER:
        idx = [i for i in e.index if fam[i] == g_ and i in q.index]
        ax.scatter(e[idx], q[idx], s=26, color=GROUP_COLOR[g_], label=GROUP_LABEL[g_], zorder=3)
        for i in idx:
            if e[i] < 0.075 and q[i] > 2.4:      # the dense cluster of EKF-type filters is labelled once
                continue
            ax.annotate(i, (e[i], q[i]), fontsize=5.8, xytext=(3, 2), textcoords='offset points', color='#333')
    ax.annotate('cluster: EKF and its algebraic equivalents, sigma-point, cubature,\n$\\mathbf{R}$-adapting, M-estimator and constrained filters, smoothers, ELM-RLS-EKF, AWTLS\n(ECM 0.02-0.07 %, SPM 2.6-4.7 %)',
                (0.05, 3.6), xytext=(0.013, 1.35), fontsize=6.5, color='#333',
                arrowprops=dict(arrowstyle='->', color='#333', lw=0.6))
    ax.set_xscale('log'); ax.set_yscale('log')
    lim = [0.012, 6]
    ax.plot(lim, lim, color=GRAY, lw=0.7, ls='--'); ax.set_xlim(*lim); ax.set_ylim(0.2, 8)
    ax.set_xlabel('nominal steady-state RMSE, ECM plant [%]'); ax.set_ylabel('nominal steady-state RMSE, SPM plant [%]')
    ax.set_title('The two plants rank the estimators differently')
    ax.legend(fontsize=7.5, loc='lower right', ncol=2)
    savefig(fig, name)


if __name__ == '__main__':
    traces('spm', 'nominal', ['EKF', 'UKF', 'MHE', 'Fading-EKF', 'Luenberger', 'Schmidt-KF', 'DualEKF', 'CoulombCounting'],
           'res_spm_traces.png', 'Electrochemical (SPM) plant, nominal eVTOL mission: SOC error of eight estimators', (-6, 3))
    traces('ecm', 'combined', ['EKF', 'UKF', 'IMM', 'MHE-RTI', 'AdaptiveObserver', 'Fading-EKF', 'DualEKF', 'Luenberger'],
           'res_ecm_combined_traces.png', 'ECM plant, `combined` scenario: SOC error of eight estimators', (-32, 6))
    ranking_bars('res_ranking_bars.png')
    ecm_vs_spm('res_ecm_vs_spm.png')
    print('done')
