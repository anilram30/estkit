// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/battery/plant_spm.hpp — electrochemical Single Particle Model (SPM) plant
//
//  Governing equations (Doyle, Fuller & Newman 1993 reduced to one representative
//  particle per electrode; see Marquis et al. 2019, JES 166, A3693 for the
//  asymptotic derivation, and Chen et al. 2020, JES 167, 080534 for parameters)
//
//   solid diffusion, electrode k in {n,p}:   dc_k/dt = (1/r^2) d/dr ( r^2 D_k dc_k/dr )
//        dc_k/dr|_{r=0} = 0,   -D_k dc_k/dr|_{r=R_k} = J_k,    J_k = i_k / (F a_k L_k A)
//        a_k = 3 eps_k / R_k (specific interfacial area), i_n = +I, i_p = -I
//   Butler-Volmer (alpha = 1/2):
//        eta_k = (2 R T / F) asinh( F J_k / (2 j0_k) )
//        j0_k  = k_k(T) sqrt(c_e) sqrt(c_k,surf) sqrt(c_k,max - c_k,surf)
//   terminal voltage:
//        V = U_p(y_surf) + eta_p - U_n(x_surf) - eta_n - R_ohm(T) I
//   SOC (bulk definition):  z = 1 - (Ah discharged)/Q_nom  ==  (x_avg - x_0)/(x_100 - x_0)
//
//  Numerics: finite volumes in r with N shells per particle, backward Euler in
//  time (unconditionally stable), tridiagonal solve by the Thomas algorithm.
//  Surface concentration reconstructed from the flux boundary condition.
// =============================================================================
#pragma once
#include "cell.hpp"
#include "plant_ecm.hpp"

namespace estkit {

template <int NR = 20>
class SpmPlant final : public PlantBase {
public:
    // lumped ohmic resistance (electrolyte + contact) not captured by the SPM
    Real R_ohm_ref = Real(0.010);   // ohm at 25 degC
    Real Ea_R_ohm  = Real(20000);   // J/mol
    Real C_th = Real(70.0), R_th = Real(3.0);
    bool enable_thermal = true;
    // "Power-cell" design scaling relative to the LG M50 energy cell (Chen 2020):
    //   particle radii x r_scale (smaller particles -> larger interfacial area a = 3 eps / R
    //   and shorter diffusion time R^2/D), reaction-rate constants x k_scale.
    //   r_scale = k_scale = 1 reproduces the published M50 parameters exactly.
    Real r_scale = Real(1), k_scale = Real(1);
    Real R_n() const { return Chen2020::R_n * r_scale; }
    Real R_p() const { return Chen2020::R_p * r_scale; }

    void reset(Real z0, Real T0, Real T_amb) override {
        T_ = T0; T_amb_ = T_amb; Ah_ = Real(0); z0_ = z0;
        const Real cn = Chen2020::x_of_z(z0) * Chen2020::cmax_n;
        const Real cp = Chen2020::y_of_z(z0) * Chen2020::cmax_p;
        for (int j = 0; j < NR; ++j) { cn_[j] = cn; cp_[j] = cp; }
        setup_geometry();
        I_last_ = Real(0);
        eta_n_ = eta_p_ = Real(0);
    }
    void set_ambient(Real T_amb) override { T_amb_ = T_amb; }
    Real capacity_Ah() const override { return Chen2020::Q_nom_Ah; }
    Real temperature() const override { return T_; }
    Real ah_throughput() const override { return Ah_; }
    // bulk SOC from the average negative-particle concentration (lithium balance)
    Real soc() const override {
        const Real xavg = average(cn_) / Chen2020::cmax_n;
        return Real(1) - (Chen2020::x100 - xavg) / Chen2020::dx();
    }
    Real ocv_now() const override { return Chen2020::ocv(soc()) + (T_ - kTref) * entropic_dUdT(soc()); }
    Real x_surf() const { return surf_conc(cn_, Jn_last_, D_n(T_)) / Chen2020::cmax_n; }
    Real y_surf() const { return surf_conc(cp_, Jp_last_, D_p(T_)) / Chen2020::cmax_p; }

    Real voltage(Real I) const override {
        const Real Jn = flux_n(I), Jp = flux_p(I);
        const Real xs = clampr(surf_conc(cn_, Jn, D_n(T_)) / Chen2020::cmax_n, Real(1e-6), Real(1) - Real(1e-6));
        const Real ys = clampr(surf_conc(cp_, Jp, D_p(T_)) / Chen2020::cmax_p, Real(1e-6), Real(1) - Real(1e-6));
        const Real Un = Chen2020::U_n(xs) + (T_ - kTref) * Real(0.5) * entropic_dUdT(soc()) * Real(-1);   // split entropic term
        const Real Up = Chen2020::U_p(ys) + (T_ - kTref) * Real(0.5) * entropic_dUdT(soc());
        const Real eta_n = overpotential(Jn, j0(Chen2020::k_n, Chen2020::Ea_k_n, xs, Chen2020::cmax_n));
        const Real eta_p = overpotential(Jp, j0(Chen2020::k_p, Chen2020::Ea_k_p, ys, Chen2020::cmax_p));
        return Up + eta_p - Un - eta_n - R_ohm(T_) * I;
    }

    void step(Real I, Real dt) override {
        const Real v = voltage(I);
        const Real q_irr = I * (ocv_now() - v);
        const Real q_rev = -I * T_ * entropic_dUdT(soc());
        const Real Jn = flux_n(I), Jp = flux_p(I);
        diffuse(cn_, D_n(T_), R_n(), Jn, dt);
        diffuse(cp_, D_p(T_), R_p(), Jp, dt);
        Jn_last_ = Jn; Jp_last_ = Jp; I_last_ = I;
        Ah_ += std::fabs(I) * dt / Real(3600);
        if (enable_thermal) T_ += dt / C_th * (q_irr + q_rev - (T_ - T_amb_) / R_th);
    }

private:
    // --- geometry (finite-volume shells, cell centres at (j+1/2) dr) ---
    void setup_geometry() {
        for (int k = 0; k < 2; ++k) {
            const Real R = (k == 0) ? R_n() : R_p();
            const Real dr = R / Real(NR);
            for (int j = 0; j < NR; ++j) {
                const Real ri = Real(j) * dr, ro = Real(j + 1) * dr;
                vol_[k][j] = (ro * ro * ro - ri * ri * ri) / Real(3);   // times 4 pi (cancels)
                face_[k][j] = ro * ro;                                    // outer face area / 4 pi
            }
            dr_[k] = dr;
        }
    }
    static Real arr(Real Ea, Real T) { return safe_exp(Ea / kRgas * (Real(1) / kTref - Real(1) / T)); }
    Real D_n(Real T) const { return Chen2020::D_n * arr(Chen2020::Ea_D_n, T); }
    Real D_p(Real T) const { return Chen2020::D_p * arr(Chen2020::Ea_D_p, T); }
    Real R_ohm(Real T) const { return R_ohm_ref * safe_exp(Ea_R_ohm / kRgas * (Real(1) / T - Real(1) / kTref)); }
    // molar surface fluxes [mol m^-2 s^-1], positive = lithium leaving the particle
    Real flux_n(Real I) const { const Real a = Real(3) * Chen2020::eps_n / R_n(); return I / (kFaraday * a * Chen2020::L_n * Chen2020::A_elec); }
    Real flux_p(Real I) const { const Real a = Real(3) * Chen2020::eps_p / R_p(); return -I / (kFaraday * a * Chen2020::L_p * Chen2020::A_elec); }
    Real j0(Real k, Real Ea, Real xs, Real cmax) const {
        const Real cs = xs * cmax;
        return k_scale * k * arr(Ea, T_) * std::sqrt(Chen2020::c_e) * std::sqrt(cs) * std::sqrt(cmax - cs);
    }
    Real overpotential(Real J, Real j0v) const { return (Real(2) * kRgas * T_ / kFaraday) * std::asinh(kFaraday * J / (Real(2) * j0v)); }
    Real average(const Real (&c)[NR]) const {
        const int k = (&c == &cn_) ? 0 : 1; Real s = 0, vs = 0;
        for (int j = 0; j < NR; ++j) { s += c[j] * vol_[k][j]; vs += vol_[k][j]; }
        return s / vs;
    }
    Real surf_conc(const Real (&c)[NR], Real J, Real D) const {
        const int k = (&c == &cn_) ? 0 : 1;
        return c[NR - 1] - J * dr_[k] / (Real(2) * D);   // linear reconstruction using the flux BC
    }
    // backward-Euler finite-volume step of the radial diffusion equation
    void diffuse(Real (&c)[NR], Real D, Real R, Real J, Real dt) {
        const int k = (&c == &cn_) ? 0 : 1;
        const Real dr = dr_[k];
        Real a[NR], b[NR], cc[NR], rhs[NR];
        for (int j = 0; j < NR; ++j) {
            const Real lo = (j > 0) ? dt * D * face_[k][j - 1] / (dr * vol_[k][j]) : Real(0);
            const Real hi = (j < NR - 1) ? dt * D * face_[k][j] / (dr * vol_[k][j]) : Real(0);
            a[j] = -lo; b[j] = Real(1) + lo + hi; cc[j] = -hi; rhs[j] = c[j];
        }
        rhs[NR - 1] -= dt * J * R * R / vol_[k][NR - 1];    // surface flux (leaving particle)
        // Thomas algorithm
        for (int j = 1; j < NR; ++j) { const Real m = a[j] / b[j - 1]; b[j] -= m * cc[j - 1]; rhs[j] -= m * rhs[j - 1]; }
        c[NR - 1] = rhs[NR - 1] / b[NR - 1];
        for (int j = NR - 2; j >= 0; --j) c[j] = (rhs[j] - cc[j] * c[j + 1]) / b[j];
    }

    Real cn_[NR], cp_[NR];
    Real vol_[2][NR], face_[2][NR], dr_[2];
    Real T_ = kTref, T_amb_ = kTref, Ah_ = 0, z0_ = 1, I_last_ = 0, Jn_last_ = 0, Jp_last_ = 0, eta_n_ = 0, eta_p_ = 0;
};

}  // namespace estkit
