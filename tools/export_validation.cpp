// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  export_validation — dumps datasets and C++ estimator trajectories so that the
//  Python scripts in scripts/validate_*.py can cross-check the library against
//  FilterPy (EKF / UKF / RTS), PyBaMM (SPM plant) and AHRS (Madgwick / Mahony).
//  Usage: export_validation <out_dir>
// =============================================================================
#include <cstdio>
#include <string>
#include <vector>
#include <filesystem>
#include "estkit/battery/datasets.hpp"
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/filters/ekf.hpp"
#include "estkit/filters/ukf.hpp"
#include "estkit/smoothers/rts.hpp"
#include "estkit/attitude/imu_dataset.hpp"
#include "estkit/attitude/madgwick.hpp"
#include "estkit/attitude/mahony.hpp"

using namespace estkit;

int main(int argc, char** argv) {
    std::string out = argc > 1 ? argv[1] : "results/validation";
    { std::error_code ec; std::filesystem::create_directories(out, ec); }
    // ------------------------------------------------------------------ cell / FilterPy
    Scenario sc; const MissionProfile mp = mission_evtol();
    Dataset d = make_dataset("ecm", mp, sc, 1);
    {
        FILE* f = std::fopen((out + "/val_cell_dataset.csv").c_str(), "w");
        std::fprintf(f, "t,i_meas,v_meas,T_meas,soc_true,v_true\n");
        for (int k = 0; k < d.n; ++k) std::fprintf(f, "%.6f,%.17g,%.17g,%.17g,%.17g,%.17g\n", double(d.t[k]), double(d.i_meas[k]), double(d.v_meas[k]), double(d.T_meas[k]), double(d.soc_true[k]), double(d.v_true[k]));
        std::fclose(f);
        EcmModel m(d.est_cfg);
        f = std::fopen((out + "/val_ocv_table.csv").c_str(), "w");
        std::fprintf(f, "z,ocv\n");
        const int N = 241; for (int k = 0; k < N; ++k) std::fprintf(f, "%.17g,%.17g\n", double(m.ocv.z_lo + (m.ocv.z_hi - m.ocv.z_lo) * k / (N - 1)), double(m.ocv.v[k]));
        std::fclose(f);
        const CellParams& p = m.p;
        f = std::fopen((out + "/val_params.csv").c_str(), "w");
        std::fprintf(f, "Q_Ah,eta_c,R0,R1,tau1,R2,tau2,M,gamma,Ea_R0,Ea_R1,Ea_R2,Ea_tau,T_ref,dt,z0,sigma_z0,sigma_v,sigma_i,qf0,qf1,qf2,qf3\n");
        std::fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     double(p.Q_Ah), double(p.eta_c), double(p.R0), double(p.R1), double(p.tau1), double(p.R2), double(p.tau2), double(p.M), double(p.gamma), double(p.Ea_R0), double(p.Ea_R1), double(p.Ea_R2), double(p.Ea_tau), double(p.T_ref),
                     double(d.est_cfg.dt), double(d.est_cfg.z0), double(d.est_cfg.sigma_z0), double(d.est_cfg.sigma_v), double(d.est_cfg.sigma_i), double(m.q_floor[0]), double(m.q_floor[1]), double(m.q_floor[2]), double(m.q_floor[3]));
        std::fclose(f);
        // EKF and UKF trajectories (posterior state and covariance diagonal)
        CellEstimator<Ekf<EcmModel>> ekf("EKF", "k"); ekf.reset(d.est_cfg);
        // for the FilterPy comparison the UKF reuses the propagated sigma points (Wan & van der Merwe form)
        CellEstimator<Ukf<EcmModel>> ukf("UKF", "k", [](Ukf<EcmModel>& f, const EstimatorConfig&) { f.opts.redraw_sigma_points = false; }); ukf.reset(d.est_cfg);
        FILE* fe = std::fopen((out + "/val_ekf.csv").c_str(), "w"); FILE* fu = std::fopen((out + "/val_ukf.csv").c_str(), "w");
        std::fprintf(fe, "z,v1,v2,h,P00,P11,P22,P33,ypred\n"); std::fprintf(fu, "z,v1,v2,h,P00,P11,P22,P33,ypred\n");
        for (int k = 0; k < d.n; ++k) {
            ekf.step(d.i_meas[k], d.v_meas[k], d.T_meas[k]); ukf.step(d.i_meas[k], d.v_meas[k], d.T_meas[k]);
            const auto& xe = ekf.filter().x(); const auto& Pe = ekf.filter().P(); const auto& xu = ukf.filter().x(); const auto& Pu = ukf.filter().P();
            std::fprintf(fe, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", double(xe[0]), double(xe[1]), double(xe[2]), double(xe[3]), double(Pe(0, 0)), double(Pe(1, 1)), double(Pe(2, 2)), double(Pe(3, 3)), double(ekf.voltage_pred()));
            std::fprintf(fu, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", double(xu[0]), double(xu[1]), double(xu[2]), double(xu[3]), double(Pu(0, 0)), double(Pu(1, 1)), double(Pu(2, 2)), double(Pu(3, 3)), double(ukf.voltage_pred()));
        }
        std::fclose(fe); std::fclose(fu);
        // RTS smoother
        EstimatorList all = make_all_estimators();
        for (auto& e : all) if (std::string(e->name()) == "ERTS") {
            auto* b = dynamic_cast<BatchEstimator*>(e.get()); b->reset(d.est_cfg);
            std::vector<Real> soc(d.n), vp(d.n); b->run(d.i_meas.data(), d.v_meas.data(), d.T_meas.data(), d.n, soc.data(), vp.data());
            FILE* fr = std::fopen((out + "/val_erts.csv").c_str(), "w"); std::fprintf(fr, "z_smooth\n");
            for (int k = 0; k < d.n; ++k) std::fprintf(fr, "%.17g\n", double(soc[k]));
            std::fclose(fr);
        }
    }
    // ------------------------------------------------------------------ SPM / PyBaMM (LG M50, published parameters, isothermal 25 degC)
    {
        for (double crate : {0.5, 1.0, 2.0}) {
            SpmPlant<20> spm; spm.enable_thermal = false; spm.R_ohm_ref = Real(0); spm.r_scale = Real(1); spm.k_scale = Real(1); spm.reset(Real(1), kTref, kTref);
            char name[64]; std::snprintf(name, sizeof(name), "/val_spm_%.1fC.csv", crate);
            FILE* f = std::fopen((out + name).c_str(), "w"); std::fprintf(f, "t,v,soc,x_surf,y_surf\n");
            const Real I = Real(crate) * Chen2020::Q_nom_Ah, dt = Real(0.1);
            for (int k = 0; k < int(3600 * 1.05 / crate / dt); ++k) {
                const Real v = spm.voltage(I);
                if (v < Real(2.5)) break;
                if (k % 10 == 0) std::fprintf(f, "%.2f,%.8f,%.8f,%.8f,%.8f\n", double(k * dt), double(v), double(spm.soc()), double(spm.x_surf()), double(spm.y_surf()));
                spm.step(I, dt);
            }
            std::fclose(f);
        }
    }
    // ------------------------------------------------------------------ attitude / AHRS
    {
        ImuDataset im = make_imu_dataset(imu_scenarios()[0], 1);
        AttitudeConfig cfg; cfg.dt = im.dt; cfg.q0 = im.q0_est; cfg.mag_ref_nav = im.mag_ref_nav; cfg.g = im.g;
        MadgwickFilter mw; mw.opts.beta = Real(0.1); mw.opts.zeta = Real(0); mw.opts.gate = false; mw.reset(cfg);
        MahonyFilter mh; mh.opts.kp = Real(1.0); mh.opts.ki = Real(0.1); mh.opts.mag_ref_measured = true; mh.opts.gate = false; mh.reset(cfg);
        FILE* f = std::fopen((out + "/val_ahrs.csv").c_str(), "w");
        std::fprintf(f, "t,gx,gy,gz,ax,ay,az,mx,my,mz,qw_true,qx_true,qy_true,qz_true,qw_madg,qx_madg,qy_madg,qz_madg,qw_mah,qx_mah,qy_mah,qz_mah\n");
        for (int k = 0; k < im.n; ++k) {
            mw.step(im.gyro[k], im.accel[k], im.mag[k]); mh.step(im.gyro[k], im.accel[k], im.mag[k]);
            const Quat a = mw.quaternion(), b = mh.quaternion(), q = im.q_true[k];
            std::fprintf(f, "%.4f,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n", double(im.t[k]),
                         double(im.gyro[k][0]), double(im.gyro[k][1]), double(im.gyro[k][2]), double(im.accel[k][0]), double(im.accel[k][1]), double(im.accel[k][2]), double(im.mag[k][0]), double(im.mag[k][1]), double(im.mag[k][2]),
                         double(q.w), double(q.x), double(q.y), double(q.z), double(a.w), double(a.x), double(a.y), double(a.z), double(b.w), double(b.x), double(b.y), double(b.z));
        }
        std::fclose(f);
        std::printf("attitude: q0_est = %.6f %.6f %.6f %.6f, mag_ref_nav = %.4f %.4f %.4f\n", double(im.q0_est.w), double(im.q0_est.x), double(im.q0_est.y), double(im.q0_est.z), double(im.mag_ref_nav[0]), double(im.mag_ref_nav[1]), double(im.mag_ref_nav[2]));
    }
    std::printf("validation exports written to %s\n", out.c_str());
    return 0;
}
