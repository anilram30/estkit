#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Cross-validation of the estkit single-particle-model plant against PyBaMM.

PyBaMM (Sulzer et al. 2021) is the reference open-source battery-modelling
framework.  Its `lithium_ion.SPM` with the built-in "Chen2020" parameter set
(LG M50, Chen et al. 2020) is solved for constant-current discharges at 25 degC
(isothermal, no contact resistance) and compared with the estkit C++ SPM
(plant_spm.hpp, r_scale = k_scale = 1, R_ohm = 0) exported by tools/export_validation.

Both models share the same electrode OCP functions, exchange-current densities,
diffusivities, geometry and initial stoichiometries; differences come only from
the numerical discretisation (estkit: 20 finite-volume shells, backward Euler,
0.1 s; PyBaMM: default finite-volume mesh and the CasADi/IDAKLU DAE solver).

Usage: python3 scripts/validate_pybamm.py results/validation [report/tables] [report/figures]
"""
import sys, os, csv
import numpy as np


def load_csv(path):
    with open(path) as f:
        r = csv.reader(f); hdr = next(r); rows = [[float(x) for x in row] for row in r]
    return hdr, np.array(rows)


def main():
    vdir = sys.argv[1] if len(sys.argv) > 1 else 'results/validation'
    tdir = sys.argv[2] if len(sys.argv) > 2 else None
    fdir = sys.argv[3] if len(sys.argv) > 3 else None
    import pybamm
    print('PyBaMM', pybamm.__version__)
    param = pybamm.ParameterValues('Chen2020')
    results = []
    curves = {}
    for crate in [0.5, 1.0, 2.0]:
        _, ours = load_csv(os.path.join(vdir, 'val_spm_%.1fC.csv' % crate))
        model = pybamm.lithium_ion.SPM()
        p = param.copy()
        p['Current function [A]'] = crate * 5.0
        p['Ambient temperature [K]'] = 298.15
        p['Initial temperature [K]'] = 298.15
        try:
            p['Contact resistance [Ohm]'] = 0.0
        except Exception:
            pass
        sim = pybamm.Simulation(model, parameter_values=p)
        t_end = ours[-1, 0]
        t_eval = np.arange(0.0, t_end + 1.0, 1.0)
        sol = sim.solve(t_eval)
        t_pb = sol['Time [s]'].entries
        v_pb = sol['Terminal voltage [V]'].entries if 'Terminal voltage [V]' in sol.all_models[0].variables else sol['Voltage [V]'].entries
        # compare on the common time span
        t_max = min(t_pb[-1], ours[-1, 0])
        tt = np.arange(0.0, t_max, 1.0)
        v_ours = np.interp(tt, ours[:, 0], ours[:, 1])
        v_ref = np.interp(tt, t_pb, v_pb)
        diff = v_ours - v_ref
        rmse = np.sqrt(np.mean(diff ** 2)) * 1e3; maxd = np.max(np.abs(diff)) * 1e3
        print('%.1fC: %d s compared, RMSE %.2f mV, max |diff| %.2f mV (PyBaMM end %.0f s, estkit end %.0f s)' % (crate, len(tt), rmse, maxd, t_pb[-1], ours[-1, 0]))
        results.append((crate, len(tt), rmse, maxd, t_pb[-1], ours[-1, 0]))
        curves[crate] = (tt, v_ours, v_ref)
    if tdir:
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, 'validation_pybamm.tex'), 'w') as f:
            f.write('% Copyright 2026 Sreeram Anil\n% SPDX-License-Identifier: CC-BY-4.0\n')
            f.write('\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\\caption{Cross-validation of the estkit SPM plant (LG M50 parameters, isothermal \\SI{25}{\\celsius}, no contact resistance) against PyBaMM %s \\texttt{lithium\\_ion.SPM} with the \\texttt{Chen2020} parameter set: terminal-voltage differences over constant-current discharges from 100\\,\\%% SOC.}\\label{tab:val_pybamm}\n' % pybamm.__version__)
            f.write('\\begin{tabular}{ccccc}\\toprule C-rate & compared span [s] & RMSE [mV] & max $|\\Delta v|$ [mV] & time to cut-off PyBaMM / estkit [s] \\\\ \\midrule\n')
            for crate, n, rmse, maxd, te1, te2 in results:
                f.write('%.1f & %d & %.2f & %.2f & %.0f / %.0f \\\\\n' % (crate, n, rmse, maxd, te1, te2))
            f.write('\\bottomrule\\end{tabular}\\end{table}\n')
        print('wrote', os.path.join(tdir, 'validation_pybamm.tex'))
    if fdir:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        os.makedirs(fdir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        for crate, (tt, vo, vr) in curves.items():
            axs[0].plot(tt / 60, vr, '-', lw=2.2, alpha=0.45, label='PyBaMM SPM %.1fC' % crate)
            axs[0].plot(tt / 60, vo, '--', lw=1.2, label='estkit SPM %.1fC' % crate)
            axs[1].plot(tt / 60, (vo - vr) * 1e3, lw=1.0, label='%.1fC' % crate)
        axs[0].set_ylabel('terminal voltage [V]'); axs[0].legend(ncol=3, fontsize=8); axs[0].grid(alpha=0.3)
        axs[1].set_ylabel('estkit − PyBaMM [mV]'); axs[1].set_xlabel('time [min]'); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, 'validation_pybamm.png'), dpi=150)
        print('wrote', os.path.join(fdir, 'validation_pybamm.png'))


if __name__ == '__main__':
    main()
