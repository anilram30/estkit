// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  export_model_curves — dumps model characteristics for the report figures:
//  OCV/slope/entropic coefficient, Arrhenius impedance scaling, Preisach loops,
//  ageing curves, and the SPM/ECM identification residual.
//  Usage: export_model_curves <out_dir>
// =============================================================================
#include <cstdio>
#include <string>
#include <filesystem>
#include "estkit/battery/plant_ecm.hpp"
#include "estkit/battery/plant_spm.hpp"

using namespace estkit;

int main(int argc, char** argv) {
    std::string out = argc > 1 ? argv[1] : "results/model";
    { std::error_code ec; std::filesystem::create_directories(out, ec); }
    // --- OCV, slope, entropic coefficient, electrode potentials ---
    {
        FILE* f = std::fopen((out + "/ocv.csv").c_str(), "w");
        std::fprintf(f, "z,ocv,docv_dz,dUdT,x_n,U_n,y_p,U_p\n");
        for (int k = 0; k <= 1000; ++k) {
            const Real z = Real(k) / 1000;
            std::fprintf(f, "%.4f,%.6f,%.6f,%.8f,%.5f,%.6f,%.5f,%.6f\n", double(z), double(Chen2020::ocv(z)), double(Chen2020::docv_dz(z)), double(entropic_dUdT(z)),
                         double(Chen2020::x_of_z(z)), double(Chen2020::U_n(Chen2020::x_of_z(z))), double(Chen2020::y_of_z(z)), double(Chen2020::U_p(Chen2020::y_of_z(z))));
        }
        std::fclose(f);
    }
    // --- Arrhenius scaling of the ECM impedance ---
    {
        CellParams p = CellParams::nominal(), q = CellParams::ecm_fitted_to_spm();
        FILE* f = std::fopen((out + "/arrhenius.csv").c_str(), "w");
        std::fprintf(f, "T_C,R0,R1,R2,tau1,tau2,R0_spmfit,R1_spmfit,R2_spmfit\n");
        for (int T = -20; T <= 60; T += 2) {
            const Real TK = kT0 + Real(T);
            std::fprintf(f, "%d,%.6f,%.6f,%.6f,%.3f,%.3f,%.6f,%.6f,%.6f\n", T, double(p.R0_T(TK)), double(p.R1_T(TK)), double(p.R2_T(TK)), double(p.tau1_T(TK)), double(p.tau2_T(TK)), double(q.R0_T(TK)), double(q.R1_T(TK)), double(q.R2_T(TK)));
        }
        std::fclose(f);
    }
    // --- Preisach major loop and a minor loop; one-state hysteresis for comparison ---
    {
        Preisach<32> pr; pr.init(Real(0.015)); pr.set_saturated(Real(1), true);
        FILE* f = std::fopen((out + "/preisach.csv").c_str(), "w");
        std::fprintf(f, "z,v_hyst,branch\n");
        for (int k = 1000; k >= 0; --k) std::fprintf(f, "%.4f,%.6f,major_down\n", k / 1000.0, double(pr.update(Real(k) / 1000)));
        for (int k = 0; k <= 1000; ++k) std::fprintf(f, "%.4f,%.6f,major_up\n", k / 1000.0, double(pr.update(Real(k) / 1000)));
        for (int k = 1000; k >= 500; --k) std::fprintf(f, "%.4f,%.6f,minor_down\n", k / 1000.0, double(pr.update(Real(k) / 1000)));
        for (int k = 500; k <= 800; ++k) std::fprintf(f, "%.4f,%.6f,minor_up\n", k / 1000.0, double(pr.update(Real(k) / 1000)));
        for (int k = 800; k >= 300; --k) std::fprintf(f, "%.4f,%.6f,minor_down2\n", k / 1000.0, double(pr.update(Real(k) / 1000)));
        std::fclose(f);
        // one-state hysteresis under a 1C discharge / charge cycle
        EcmPlant e; e.enable_aging = false; e.enable_thermal = false; e.reset(Real(1), kTref, kTref);
        f = std::fopen((out + "/hysteresis_onestate.csv").c_str(), "w"); std::fprintf(f, "z,v_hyst,branch\n");
        for (int k = 0; k < 36000; ++k) { e.step(Real(5), Real(0.1)); if (k % 100 == 0) std::fprintf(f, "%.4f,%.6f,discharge\n", double(e.soc()), double(e.hysteresis_voltage())); }
        for (int k = 0; k < 36000; ++k) { e.step(Real(-5), Real(0.1)); if (k % 100 == 0) std::fprintf(f, "%.4f,%.6f,charge\n", double(e.soc()), double(e.hysteresis_voltage())); }
        std::fclose(f);
    }
    // --- ageing: 1C full cycles at three temperatures ---
    {
        FILE* f = std::fopen((out + "/aging.csv").c_str(), "w");
        std::fprintf(f, "T_C,efc,Q_loss,R0_growth\n");
        for (int T : {10, 25, 45}) {
            EcmPlant e; e.enable_thermal = false; e.reset(Real(1), kT0 + Real(T), kT0 + Real(T));
            for (int c = 1; c <= 1000; ++c) {
                for (int k = 0; k < 3600; ++k) e.step(Real(5), Real(1));
                for (int k = 0; k < 3600; ++k) e.step(Real(-5), Real(1));
                if (c % 20 == 0) std::fprintf(f, "%d,%d,%.5f,%.5f\n", T, c, double(e.capacity_loss()), double(1 + e.p.age_R_gain * e.capacity_loss()));
            }
        }
        std::fclose(f);
        // calendar ageing
        f = std::fopen((out + "/aging_calendar.csv").c_str(), "w"); std::fprintf(f, "T_C,days,Q_loss\n");
        for (int T : {10, 25, 45}) { EcmPlant e; e.reset(Real(1), kT0 + Real(T), kT0 + Real(T)); for (int d = 0; d <= 730; d += 10) { std::fprintf(f, "%d,%d,%.5f\n", T, d, double(e.capacity_loss())); e.age_calendar(Real(10)); } }
        std::fclose(f);
    }
    // --- SPM: 1C discharge, surface vs bulk stoichiometry (power-cell variant) ---
    {
        SpmPlant<20> s; s.r_scale = Real(0.5); s.k_scale = Real(3); s.R_ohm_ref = Real(0.005); s.reset(Real(1), kTref, kTref);
        FILE* f = std::fopen((out + "/spm_pulse.csv").c_str(), "w"); std::fprintf(f, "t,i,v,soc,x_surf,y_surf,T\n");
        for (int k = 0; k < 24000; ++k) {
            const Real t = k * Real(0.1); Real I = (t < 60) ? 0 : (t < 660 ? 5 : (t < 900 ? 0 : (t < 960 ? 20 : 0)));
            if (k % 10 == 0) std::fprintf(f, "%.1f,%.2f,%.6f,%.6f,%.6f,%.6f,%.3f\n", double(t), double(I), double(s.voltage(I)), double(s.soc()), double(s.x_surf()), double(s.y_surf()), double(s.temperature() - kT0));
            s.step(I, Real(0.1));
        }
        std::fclose(f);
    }
    std::printf("model curves written to %s\n", out.c_str());
    return 0;
}
