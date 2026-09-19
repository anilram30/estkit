// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  ident_spm_ecm — identify 2-RC equivalent-circuit parameters from the
//  electrochemical SPM plant (dynamic-profile least squares).
//
//  Method (Plett 2015 Vol. I Ch. 2 "ECM parameter identification"; Hu, Li & Peng
//  2012): simulate the SPM (LG M50 parameters, Chen et al. 2020) isothermally at
//  25 degC on an identification profile (HPPC-style pulse train at several SOC
//  levels followed by the fixed-wing mission), then minimise
//        J(theta) = sum_k ( v_spm[k] - v_ecm(theta)[k] )^2,   theta = [R0, R1, tau1, R2, tau2]
//  with the ECM driven open-loop by the same current and the same Coulomb-counted
//  SOC, using Levenberg-Marquardt (Nocedal & Wright 2006, Ch. 10) on
//  log-parameters (positivity), numeric Jacobian.  The OCV is shared by
//  construction (both models use the Chen 2020 electrode potentials).
//
//  Usage: ident_spm_ecm [--out params.csv]
// =============================================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstdlib>
#include "estkit/battery/plant_spm.hpp"
#include "estkit/battery/plant_ecm.hpp"
#include "estkit/battery/datasets.hpp"

using namespace estkit;

struct Ident {
    std::vector<Real> t, i, v_spm, z;
    Real dt = Real(0.1);
};

static Real g_r_scale = 1, g_k_scale = 1, g_R_ohm = Real(0.010);
static Ident make_profile(Real T) {
    Ident d;
    SpmPlant<20> spm; spm.enable_thermal = false; spm.r_scale = g_r_scale; spm.k_scale = g_k_scale; spm.R_ohm_ref = g_R_ohm; spm.reset(Real(0.95), T, T);
    Rng rng(42);
    // (a) pulse train: at each SOC level 10 s 2C discharge, 40 s rest, 10 s 1C charge, 40 s rest, then 1C discharge for 5 % SOC
    std::vector<Real> icmd;
    auto push = [&](Real cur, Real dur) { for (int k = 0; k < int(dur / d.dt); ++k) icmd.push_back(cur); };
    push(0, 60);
    for (int lvl = 0; lvl < 8; ++lvl) {
        push(10, 10); push(0, 40); push(-5, 10); push(0, 40); push(20, 5); push(0, 40);
        push(5, 180);   // 5 % SOC step down
    }
    push(0, 120);
    // (b) fixed-wing mission with gusts (starts ~0.55 SOC after the pulse train)
    MissionProfile mp = mission_fixed_wing();
    std::vector<Real> im; std::vector<int> seg;
    generate_current(mp, d.dt, Chen2020::Q_nom_Ah, rng, im, seg);
    for (Real x : im) icmd.push_back(x);
    push(0, 300);
    for (size_t k = 0; k < icmd.size(); ++k) {
        const Real cur = icmd[k];
        d.t.push_back(Real(k) * d.dt); d.i.push_back(cur); d.z.push_back(spm.soc()); d.v_spm.push_back(spm.voltage(cur));
        spm.step(cur, d.dt);
        if (spm.soc() < Real(0.08)) break;
    }
    return d;
}

// simulate the ECM voltage for theta (open loop, SOC from Coulomb counting of the same current)
static void ecm_voltage(const Ident& d, const Real* th, std::vector<Real>& v, Real T = kTref) {
    CellParams p = CellParams::nominal();
    p.R0 = th[0]; p.R1 = th[1]; p.tau1 = th[2]; p.R2 = th[3]; p.tau2 = th[4]; p.M = Real(0);
    p.T_ref = T;   // parameters are fitted AT this temperature (no Arrhenius scaling inside the fit)
    EcmPlant e(p); e.enable_aging = false; e.enable_thermal = false; e.reset(Real(0.95), T, T);
    v.resize(d.i.size());
    for (size_t k = 0; k < d.i.size(); ++k) { v[k] = e.voltage(d.i[k]); e.step(d.i[k], d.dt); }
}

struct FitResult { Real th[5]; Real rmse_mV; std::vector<Real> v; };
static FitResult fit_at(const Ident& d, Real T, bool verbose) {
    // Levenberg-Marquardt on log-parameters
    const int NP = 5;
    Real th[NP] = { Real(0.02), Real(0.01), Real(10), Real(0.01), Real(300) };
    Real lth[NP]; for (int j = 0; j < NP; ++j) lth[j] = std::log(th[j]);
    std::vector<Real> v, vp, vm; ecm_voltage(d, th, v, T);
    auto cost = [&](const std::vector<Real>& vv) { double s = 0; for (size_t k = 0; k < vv.size(); ++k) s += sq(double(vv[k] - d.v_spm[k])); return s; };
    double J = cost(v); double lambda = 1e-3;
    std::printf("iter  0  rmse = %.3f mV\n", 1e3 * std::sqrt(J / d.i.size()));
    for (int it = 1; it <= 40; ++it) {
        // Jacobian (n x NP) by central differences in log-space
        const size_t n = d.i.size();
        std::vector<std::vector<Real>> Jc(NP);
        for (int j = 0; j < NP; ++j) {
            Real tp[NP], tm[NP]; const Real h = Real(1e-3);
            for (int q = 0; q < NP; ++q) { tp[q] = std::exp(lth[q] + (q == j ? h : 0)); tm[q] = std::exp(lth[q] - (q == j ? h : 0)); }
            ecm_voltage(d, tp, vp, T); ecm_voltage(d, tm, vm, T);
            Jc[j].resize(n); for (size_t k = 0; k < n; ++k) Jc[j][k] = (vp[k] - vm[k]) / (2 * h);
        }
        Mat<5, 5> JtJ; Vec<5> Jtr;
        for (int a = 0; a < NP; ++a) { for (int b = 0; b < NP; ++b) { double s = 0; for (size_t k = 0; k < n; ++k) s += double(Jc[a][k]) * double(Jc[b][k]); JtJ(a, b) = Real(s); }
            double s = 0; for (size_t k = 0; k < n; ++k) s += double(Jc[a][k]) * double(d.v_spm[k] - v[k]); Jtr[a] = Real(s); }
        bool accepted = false;
        for (int tries = 0; tries < 12 && !accepted; ++tries) {
            Mat<5, 5> A = JtJ; for (int q = 0; q < NP; ++q) A(q, q) *= (Real(1) + Real(lambda));
            Vec<5> step = solve(A, Jtr);
            Real lnew[NP], tnew[NP]; for (int q = 0; q < NP; ++q) { lnew[q] = lth[q] + step[q]; tnew[q] = std::exp(lnew[q]); }
            std::vector<Real> vn; ecm_voltage(d, tnew, vn, T); const double Jn = cost(vn);
            if (Jn < J) { J = Jn; v = vn; for (int q = 0; q < NP; ++q) { lth[q] = lnew[q]; th[q] = tnew[q]; } lambda = std::max(lambda / 3, 1e-9); accepted = true; }
            else lambda *= 4;
        }
        if (verbose) std::printf("iter %2d  rmse = %.3f mV   R0=%.4f R1=%.4f tau1=%.1f R2=%.4f tau2=%.1f  lambda=%.1e\n", it, 1e3 * std::sqrt(J / d.i.size()), th[0], th[1], th[2], th[3], th[4], lambda);
        if (!accepted) break;
    }
    FitResult r; for (int q = 0; q < NP; ++q) r.th[q] = th[q]; r.rmse_mV = Real(1e3 * std::sqrt(J / d.i.size())); r.v = v; return r;
}

int main(int argc, char** argv) {
    std::string out = "results/ident_spm_ecm.csv";
    for (int a = 1; a < argc; ++a) { std::string k = argv[a]; if (k == "--out" && a + 1 < argc) out = argv[++a]; else if (k == "--r_scale" && a + 1 < argc) g_r_scale = Real(std::atof(argv[++a])); else if (k == "--k_scale" && a + 1 < argc) g_k_scale = Real(std::atof(argv[++a])); else if (k == "--R_ohm" && a + 1 < argc) g_R_ohm = Real(std::atof(argv[++a])); }
    // --- reference temperature fit (25 degC), verbose ---
    Ident d25 = make_profile(kTref);
    std::printf("identification record: %zu samples (%.0f s)\n", d25.i.size(), d25.t.back());
    FitResult r25 = fit_at(d25, kTref, true);
    const Real* th = r25.th;
    std::printf("\nFitted 2-RC ECM to SPM (25 degC):\n  R0 = %.5f ohm\n  R1 = %.5f ohm, tau1 = %.2f s (C1 = %.0f F)\n  R2 = %.5f ohm, tau2 = %.2f s (C2 = %.0f F)\n  overall voltage RMSE = %.3f mV\n",
                th[0], th[1], th[2], th[2] / th[1], th[3], th[4], th[4] / th[3], r25.rmse_mV);
    // --- fits at other temperatures -> Arrhenius activation energies ---
    const Real Ts[4] = { kT0 - Real(10), kT0 + Real(10), kT0 + Real(25), kT0 + Real(45) };
    Real lnR0[4], lnR1[4], lnR2[4], lnt1[4], lnt2[4], invT[4];
    for (int q = 0; q < 4; ++q) {
        Ident dq = make_profile(Ts[q]); FitResult rq = fit_at(dq, Ts[q], false);
        lnR0[q] = std::log(rq.th[0]); lnR1[q] = std::log(rq.th[1]); lnt1[q] = std::log(rq.th[2]); lnR2[q] = std::log(rq.th[3]); lnt2[q] = std::log(rq.th[4]);
        invT[q] = Real(1) / Ts[q];
        std::printf("T = %5.1f degC: R0=%.5f R1=%.5f tau1=%.1f R2=%.5f tau2=%.1f  rmse=%.2f mV\n", Ts[q] - kT0, rq.th[0], rq.th[1], rq.th[2], rq.th[3], rq.th[4], rq.rmse_mV);
    }
    // ln(p(T)) = ln(p_ref) + Ea/R (1/T - 1/Tref)  ->  slope of ln p vs 1/T = Ea/R
    auto slope = [&](const Real* y) { Real mx = 0, my = 0; for (int q = 0; q < 4; ++q) { mx += invT[q]; my += y[q]; } mx /= 4; my /= 4; Real sxy = 0, sxx = 0; for (int q = 0; q < 4; ++q) { sxy += (invT[q] - mx) * (y[q] - my); sxx += sq(invT[q] - mx); } return sxy / sxx; };
    const Real EaR0 = slope(lnR0) * kRgas, EaR1 = slope(lnR1) * kRgas, EaR2 = slope(lnR2) * kRgas, Eat1 = slope(lnt1) * kRgas, Eat2 = slope(lnt2) * kRgas;
    std::printf("Arrhenius activation energies [J/mol]: R0 %.0f, R1 %.0f, R2 %.0f, tau1 %.0f, tau2 %.0f\n", EaR0, EaR1, EaR2, Eat1, Eat2);
    FILE* f = std::fopen(out.c_str(), "w");
    if (f) { std::fprintf(f, "R0,R1,tau1,R2,tau2,rmse_mV,Ea_R0,Ea_R1,Ea_R2,Ea_tau1,Ea_tau2\n%.6g,%.6g,%.6g,%.6g,%.6g,%.4f,%.0f,%.0f,%.0f,%.0f,%.0f\n", th[0], th[1], th[2], th[3], th[4], r25.rmse_mV, EaR0, EaR1, EaR2, Eat1, Eat2); std::fclose(f); }
    // dump the identification record for plotting
    FILE* g = std::fopen((out.substr(0, out.size() - 4) + "_record.csv").c_str(), "w");
    if (g) { std::fprintf(g, "t,i,v_spm,v_ecm,soc\n"); for (size_t k = 0; k < d25.i.size(); k += 10) std::fprintf(g, "%.1f,%.4f,%.5f,%.5f,%.5f\n", double(d25.t[k]), double(d25.i[k]), double(d25.v_spm[k]), double(r25.v[k]), double(d25.z[k])); std::fclose(g); }
    return 0;
}
