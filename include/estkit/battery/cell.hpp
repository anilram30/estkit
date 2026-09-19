// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/battery/cell.hpp — cell parameters, open-circuit voltage (OCV) and its
//  temperature/SOC derivatives, Arrhenius temperature dependence, Preisach
//  hysteresis operator and the semi-empirical ageing law.
//
//  Chemistry: NMC811 / graphite, 5 Ah 21700 class (LG M50 electrode data).
//
//  Sources (see report bibliography for full citations)
//    * Electrode open-circuit potentials U_n(x), U_p(y): Chen et al. (2020),
//      J. Electrochem. Soc. 167, 080534 (fitted functions as distributed in the
//      PyBaMM "Chen2020" parameter set).
//    * Equivalent-circuit "ESC" model with one-state hysteresis: Plett (2004,
//      J. Power Sources 134, Parts 1-3) and Plett (2015) BMS Vol. I, Ch. 2.
//    * Lumped thermal model: Lin et al. (2014), J. Power Sources 257, 1-11;
//      heat generation: Bernardi, Pawlikowski & Newman (1985), JES 132, 5-12.
//    * Arrhenius temperature dependence of impedance: Plett (2015) Vol. I §2.10.
//    * Preisach hysteresis: Mayergoyz (1991) "Mathematical Models of Hysteresis";
//      battery application: Baronti et al. (2014), IEEE Trans. Magn. 50, 7300704.
//    * Ageing: structural form of Wang et al. (2011), J. Power Sources 196,
//      3942-3948 (Arrhenius x power law in Ah-throughput) plus calendar term of
//      the type used by Schmalstieg et al. (2014), J. Power Sources 257, 325-334.
//      Coefficients are representative values for an NMC cell (not fitted data).
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include "../core/linalg.hpp"

namespace estkit {

constexpr Real kFaraday = Real(96485.33212);   // C/mol
constexpr Real kRgas    = Real(8.314462618);   // J/(mol K)
constexpr Real kT0      = Real(273.15);        // K
constexpr Real kTref    = Real(298.15);        // K (25 degC)

// -----------------------------------------------------------------------------
//  Electrode open-circuit potentials (Chen et al. 2020, LG M50)
// -----------------------------------------------------------------------------
struct Chen2020 {
    // graphite anode potential vs Li/Li+ as a function of stoichiometry x
    static Real U_n(Real x) {
        return Real(1.9793) * std::exp(Real(-39.3631) * x) + Real(0.2482)
             - Real(0.0909) * std::tanh(Real(29.8538) * (x - Real(0.1234)))
             - Real(0.04478) * std::tanh(Real(14.9159) * (x - Real(0.2769)))
             - Real(0.0205) * std::tanh(Real(30.4444) * (x - Real(0.6103)));
    }
    static Real dU_n(Real x) {
        auto sech2 = [](Real u) { const Real t = std::tanh(u); return Real(1) - t * t; };
        return Real(-39.3631) * Real(1.9793) * std::exp(Real(-39.3631) * x)
             - Real(0.0909) * Real(29.8538) * sech2(Real(29.8538) * (x - Real(0.1234)))
             - Real(0.04478) * Real(14.9159) * sech2(Real(14.9159) * (x - Real(0.2769)))
             - Real(0.0205) * Real(30.4444) * sech2(Real(30.4444) * (x - Real(0.6103)));
    }
    // NMC811 cathode potential vs Li/Li+ as a function of stoichiometry y
    static Real U_p(Real y) {
        return Real(-0.8090) * y + Real(4.4875)
             - Real(0.0428) * std::tanh(Real(18.5138) * (y - Real(0.5542)))
             - Real(17.7326) * std::tanh(Real(15.7890) * (y - Real(0.3117)))
             + Real(17.5842) * std::tanh(Real(15.9308) * (y - Real(0.3120)));
    }
    static Real dU_p(Real y) {
        auto sech2 = [](Real u) { const Real t = std::tanh(u); return Real(1) - t * t; };
        return Real(-0.8090)
             - Real(0.0428) * Real(18.5138) * sech2(Real(18.5138) * (y - Real(0.5542)))
             - Real(17.7326) * Real(15.7890) * sech2(Real(15.7890) * (y - Real(0.3117)))
             + Real(17.5842) * Real(15.9308) * sech2(Real(15.9308) * (y - Real(0.3120)));
    }
    // Electrode geometry / loading (Chen 2020, Table; PyBaMM Chen2020)
    static constexpr Real A_elec  = Real(0.1027);     // m^2 electrode area
    static constexpr Real L_n     = Real(85.2e-6);    // m
    static constexpr Real L_p     = Real(75.6e-6);    // m
    static constexpr Real eps_n   = Real(0.75);       // active-material volume fraction
    static constexpr Real eps_p   = Real(0.665);
    static constexpr Real R_n     = Real(5.86e-6);    // m particle radius
    static constexpr Real R_p     = Real(5.22e-6);
    static constexpr Real cmax_n  = Real(33133);      // mol/m^3
    static constexpr Real cmax_p  = Real(63104);
    static constexpr Real D_n     = Real(3.3e-14);    // m^2/s solid diffusivity at 25 degC
    static constexpr Real D_p     = Real(4.0e-15);
    static constexpr Real k_n     = Real(6.48e-7);    // A m^-2 (m^3/mol)^1.5 reaction-rate constant
    static constexpr Real k_p     = Real(3.42e-6);
    static constexpr Real Ea_k_n  = Real(35000);      // J/mol
    static constexpr Real Ea_k_p  = Real(17800);
    static constexpr Real Ea_D_n  = Real(30000);      // representative (not in Chen 2020)
    static constexpr Real Ea_D_p  = Real(30000);
    static constexpr Real c_e     = Real(1000);       // mol/m^3 electrolyte concentration
    // stoichiometries at 100 % SOC (PyBaMM initial concentrations / c_max)
    static constexpr Real x100    = Real(0.9014);
    static constexpr Real y100    = Real(0.2700);
    static constexpr Real Q_nom_Ah = Real(5.0);       // nominal capacity used to define SOC

    // charge per unit stoichiometry change [As]:  F * eps * L * A * c_max
    static constexpr Real q_n() { return kFaraday * eps_n * L_n * A_elec * cmax_n; }
    static constexpr Real q_p() { return kFaraday * eps_p * L_p * A_elec * cmax_p; }
    // stoichiometry as a function of SOC z (lithium balance, Q_nom Ah between z=0 and 1)
    static constexpr Real dx() { return Q_nom_Ah * Real(3600) / q_n(); }
    static constexpr Real dy() { return Q_nom_Ah * Real(3600) / q_p(); }
    static Real x_of_z(Real z) { return x100 - (Real(1) - z) * dx(); }
    static Real y_of_z(Real z) { return y100 + (Real(1) - z) * dy(); }
    // full-cell equilibrium voltage and slope at reference temperature
    static Real ocv(Real z) { return U_p(y_of_z(z)) - U_n(x_of_z(z)); }
    static Real docv_dz(Real z) { return -dy() * dU_p(y_of_z(z)) - dx() * dU_n(x_of_z(z)); }
};

// Entropic coefficient dU/dT [V/K] as a function of SOC. Representative smooth
// curve of the magnitude reported for NMC/graphite cells (-0.05 ... -0.25 mV/K,
// cf. Forgez et al. 2010 for LFP; Plett 2015 Vol. I §2.9 for the modelling form).
inline Real entropic_dUdT(Real z) {
    z = clampr(z, Real(0), Real(1));
    return Real(1e-4) * (Real(-2.0) + Real(3.0) * z - Real(1.5) * z * z);
}
inline Real entropic_dUdT_dz(Real z) {   // d/dz of the entropic coefficient
    if (z < Real(0) || z > Real(1)) return Real(0);
    return Real(1e-4) * (Real(3.0) - Real(3.0) * z);
}

// -----------------------------------------------------------------------------
//  OCV lookup table (what a BMS would store in flash): uniform grid with linear
//  interpolation and linear extrapolation outside [z_lo, z_hi].
// -----------------------------------------------------------------------------
template <int NPTS = 241>
struct OcvTable {
    Real z_lo = Real(-0.1), z_hi = Real(1.1);
    Real v[NPTS];
    Real dv[NPTS];  // slope of each segment (dv[NPTS-1] unused)
    OcvTable() { build(); }
    void build() {
        for (int k = 0; k < NPTS; ++k) v[k] = Chen2020::ocv(z_lo + (z_hi - z_lo) * Real(k) / Real(NPTS - 1));
        const Real dz = (z_hi - z_lo) / Real(NPTS - 1);
        for (int k = 0; k < NPTS - 1; ++k) dv[k] = (v[k + 1] - v[k]) / dz;
        dv[NPTS - 1] = dv[NPTS - 2];
    }
    Real ocv(Real z, Real T = kTref) const {
        const Real dz = (z_hi - z_lo) / Real(NPTS - 1);
        Real s = (z - z_lo) / dz; int k = int(std::floor(s));
        k = clampr(k, 0, NPTS - 2);
        const Real zk = z_lo + dz * Real(k);
        return v[k] + dv[k] * (z - zk) + (T - kTref) * entropic_dUdT(z);
    }
    Real docv_dz(Real z, Real T = kTref) const {
        const Real dz = (z_hi - z_lo) / Real(NPTS - 1);
        Real s = (z - z_lo) / dz; int k = int(std::floor(s));
        k = clampr(k, 0, NPTS - 2);
        return dv[k] + (T - kTref) * entropic_dUdT_dz(z);
    }
};

// -----------------------------------------------------------------------------
//  Equivalent-circuit parameters (2-RC "ESC" model) at the reference temperature
//  plus thermal and ageing constants.  Current sign convention: DISCHARGE > 0.
// -----------------------------------------------------------------------------
struct CellParams {
    // --- electrical (25 degC) ---
    Real Q_Ah   = Real(5.0);       // total capacity [Ah]
    Real eta_c  = Real(0.995);     // coulombic efficiency on charge (=1 on discharge)
    Real R0     = Real(0.0120);    // ohm  series resistance
    Real R1     = Real(0.0060);    // ohm  fast RC branch
    Real tau1   = Real(5.0);       // s
    Real R2     = Real(0.0080);    // ohm  slow RC branch
    Real tau2   = Real(120.0);     // s
    Real M      = Real(0.005);     // V    one-state hysteresis magnitude
    Real gamma  = Real(50.0);      // -    hysteresis rate constant
    // --- Arrhenius activation energies [J/mol] (Plett 2015 Vol. I, §2.10) ---
    Real Ea_R0  = Real(20000);
    Real Ea_R1  = Real(25000);
    Real Ea_R2  = Real(30000);
    Real Ea_tau = Real(25000);     // time constants shrink with temperature (faster diffusion)
    Real T_ref  = kTref;
    // --- thermal (lumped, Lin et al. 2014) ---
    Real C_th   = Real(70.0);      // J/K   (m = 70 g, c_p = 1000 J/kg/K)
    Real R_th   = Real(3.0);       // K/W   cell-to-ambient thermal resistance
    // --- ageing (Wang 2011 structure): dQ_loss = B exp(-Ea/RT) z Ah^(z-1) dAh ---
    Real age_B      = Real(454.0);     // [Ah^-z] chosen so that 20 % loss after 1000 EFC (10,000 Ah throughput) at 25 degC
    Real age_Ea     = Real(31700);     // J/mol (Wang 2011)
    Real age_z      = Real(0.55);      // power-law exponent (Wang 2011)
    Real age_cal_A  = Real(36.4);      // calendar term Q_loss_cal = A exp(-Ea/RT) sqrt(t_days): 5 % after 2 years at 25 degC
    Real age_cal_Ea = Real(24500);     // J/mol
    Real age_R_gain = Real(2.5);       // R0 growth factor: R0 = R0_ref (1 + age_R_gain * Q_loss_frac)

    // temperature-scaled parameters
    Real arr(Real Ea, Real T) const { return safe_exp(Ea / kRgas * (Real(1) / T - Real(1) / T_ref)); }
    Real R0_T(Real T) const { return R0 * arr(Ea_R0, T); }
    Real R1_T(Real T) const { return R1 * arr(Ea_R1, T); }
    Real R2_T(Real T) const { return R2 * arr(Ea_R2, T); }
    Real tau1_T(Real T) const { return tau1 * arr(Ea_tau, T); }
    Real tau2_T(Real T) const { return tau2 * arr(Ea_tau, T); }

    // Representative high-power cell (default). The OCV comes from Chen 2020.
    static CellParams nominal() { return CellParams(); }
    // Parameters identified against the SPM plant with tools/ident_spm_ecm
    // (dynamic-profile least squares, see report Ch. "Electrochemical plant").
    static CellParams ecm_fitted_to_spm();
};

// -----------------------------------------------------------------------------
//  Discrete Preisach hysteresis operator  (Mayergoyz 1991, Ch. 1)
//  Input u = SOC in [0,1]; output = hysteresis voltage in [-Mp, +Mp].
//  Relays gamma_{ab} with thresholds a >= b on an L x L grid; weight function
//  mu(a,b) concentrated near the diagonal (narrow minor loops).
// -----------------------------------------------------------------------------
template <int L = 32>
class Preisach {
public:
    static constexpr int NREL = L * (L - 1) / 2;   // relays with alpha > beta (strict)
    void init(Real Mp, Real width = Real(0.15)) {
        Mp_ = Mp; int n = 0; Real wsum = 0;
        for (int ia = 1; ia < L; ++ia) for (int ib = 0; ib < ia; ++ib) {
            const Real a = Real(ia) / Real(L - 1), b = Real(ib) / Real(L - 1);
            const Real w = std::exp(-sq((a - b) / width));
            alpha_[n] = a; beta_[n] = b; mu_[n] = w; wsum += w; ++n;
        }
        for (int k = 0; k < NREL; ++k) mu_[k] *= Mp_ / wsum;
        for (int k = 0; k < NREL; ++k) r_[k] = Real(0);   // demagnetised (neutral) state
        u_prev_ = Real(0.5);
    }
    // update with input u, return output voltage
    Real update(Real u) {
        u = clampr(u, Real(0), Real(1));
        Real y = 0;
        for (int k = 0; k < NREL; ++k) {
            if (u >= alpha_[k]) r_[k] = Real(1);
            else if (u <= beta_[k]) r_[k] = Real(-1);
            y += mu_[k] * r_[k];
        }
        u_prev_ = u;
        return y;
    }
    Real output() const { Real y = 0; for (int k = 0; k < NREL; ++k) y += mu_[k] * r_[k]; return y; }
    // set relays consistent with a monotone approach to u from below (charged state) / above
    void set_saturated(Real u, bool from_below) {
        for (int k = 0; k < NREL; ++k) {
            if (from_below) r_[k] = (u >= alpha_[k]) ? Real(1) : Real(-1);
            else            r_[k] = (u <= beta_[k]) ? Real(-1) : Real(1);
        }
    }
private:
    Real alpha_[NREL], beta_[NREL], mu_[NREL], r_[NREL];
    Real Mp_ = 0, u_prev_ = 0;
};

}  // namespace estkit
