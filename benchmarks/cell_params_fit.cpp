// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// Parameters of the 2-RC ECM identified against the electrochemical SPM plant
// (power-cell design: Chen 2020 electrode data with particle radii x0.5, reaction
// rate constants x3, lumped ohmic resistance 5 mOhm) with tools/ident_spm_ecm
// (Levenberg-Marquardt, dynamic identification profile: HPPC-style pulse train +
// fixed-wing mission, isothermal 25 degC).  Result recorded in results/ident_spm_ecm.csv:
//   R0 = 9.52 mOhm, R1 = 2.08 mOhm (tau1 = 15.1 s), R2 = 2.55 mOhm (tau2 = 555.8 s),
//   residual voltage RMSE 4.0 mV at 25 degC (Butler-Volmer kinetics and SOC-dependent
//   diffusion impedance are not representable by a constant-parameter 2-RC circuit).
// Repeating the fit at -10, 10, 25 and 45 degC and regressing ln(p) on 1/T gives the
// Arrhenius activation energies Ea_R0 = 20.2 kJ/mol, Ea_R1 = 17.1 kJ/mol,
// Ea_R2 = 42.1 kJ/mol; the time constants are practically temperature independent
// (Ea_tau ~ 0): the temperature dependence of the diffusion impedance is absorbed by R2.
#include "estkit/battery/cell.hpp"
namespace estkit {
CellParams CellParams::ecm_fitted_to_spm() {
    CellParams p = CellParams::nominal();
    p.R0 = Real(0.00952095);
    p.R1 = Real(0.00207697); p.tau1 = Real(15.1272);
    p.R2 = Real(0.00254659); p.tau2 = Real(555.79);
    p.Ea_R0 = Real(20185); p.Ea_R1 = Real(17142); p.Ea_R2 = Real(42140); p.Ea_tau = Real(0);
    p.M = Real(0);            // the SPM has no hysteresis
    return p;
}
}
