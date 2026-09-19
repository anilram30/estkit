// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  bench_imu — attitude (AHRS) estimator benchmark on a synthetic flight
//  Usage: bench_imu [--scenario NAME|all] [--only NAME] [--seeds K] [--out DIR] [--reps R] [--list] [--export-dataset FILE]
//  Output: <out>/summary_imu.csv ; <out>/ts/imu__<est>__<scenario>__s1.csv
// =============================================================================
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include "estkit/attitude/attitude_estimator.hpp"

using namespace estkit;
static std::vector<std::string> split(const std::string& s, char sep) { std::vector<std::string> o; std::stringstream ss(s); std::string it; while (std::getline(ss, it, sep)) if (!it.empty()) o.push_back(it); return o; }
static void mkdirp(const std::string& p) { std::error_code ec; std::filesystem::create_directories(p, ec); }
static bool file_empty(const std::string& p) { std::error_code ec; return !std::filesystem::exists(p, ec) || std::filesystem::file_size(p, ec) == 0; }

int main(int argc, char** argv) {
    std::string scenario = "all", only, out = "results", export_ds; int seeds = 1, reps = 2; bool list_only = false, timeseries = true;
    for (int a = 1; a < argc; ++a) {
        std::string k = argv[a]; auto val = [&]() { return std::string(a + 1 < argc ? argv[++a] : ""); };
        if (k == "--scenario") scenario = val(); else if (k == "--only") only = val(); else if (k == "--seeds") seeds = std::stoi(val());
        else if (k == "--out") out = val(); else if (k == "--reps") reps = std::stoi(val()); else if (k == "--timeseries") timeseries = std::stoi(val()) != 0;
        else if (k == "--list") list_only = true; else if (k == "--export-dataset") export_ds = val();
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    AttitudeList ests = make_all_attitude_estimators();
    if (list_only) { for (auto& e : ests) std::printf("%-28s %s\n", e->name(), e->group()); return 0; }
    if (!export_ds.empty()) {   // dump the nominal dataset for cross-validation with external libraries
        ImuDataset d = make_imu_dataset(imu_scenarios()[0], 1);
        std::ofstream f(export_ds);
        f << "t,gx,gy,gz,ax,ay,az,mx,my,mz,qw,qx,qy,qz,bgx,bgy,bgz\n";
        for (int k = 0; k < d.n; ++k) f << d.t[k] << ',' << d.gyro[k][0] << ',' << d.gyro[k][1] << ',' << d.gyro[k][2] << ',' << d.accel[k][0] << ',' << d.accel[k][1] << ',' << d.accel[k][2] << ','
                                        << d.mag[k][0] << ',' << d.mag[k][1] << ',' << d.mag[k][2] << ',' << d.q_true[k].w << ',' << d.q_true[k].x << ',' << d.q_true[k].y << ',' << d.q_true[k].z << ','
                                        << d.gyro_bias_true[k][0] << ',' << d.gyro_bias_true[k][1] << ',' << d.gyro_bias_true[k][2] << '\n';
        std::printf("exported %d samples to %s (mag_ref_nav = %.3f, %.3f, %.3f uT)\n", d.n, export_ds.c_str(), d.mag_ref_nav[0], d.mag_ref_nav[1], d.mag_ref_nav[2]);
        return 0;
    }
    std::set<std::string> only_set; for (auto& s : split(only, ',')) only_set.insert(s);
    std::vector<ImuScenario> scs; for (auto& s : imu_scenarios()) if (scenario == "all" || s.name == scenario) scs.push_back(s);
    mkdirp(out); if (timeseries) mkdirp(out + "/ts");
    const std::string sp = out + "/summary_imu.csv"; const bool hdr = file_empty(sp);
    std::ofstream summary(sp, std::ios::app);
    if (hdr) summary << "estimator,group,scenario,seed,rms_angle_deg,rms_roll_deg,rms_pitch_deg,rms_yaw_deg,max_angle_deg,ss_rms_angle_deg,conv_time,bias_rmse_dps,diverged,ns_per_step,bytes\n";
    for (const auto& sc : scs) for (int seed = 1; seed <= seeds; ++seed) {
        const ImuDataset d = make_imu_dataset(sc, uint64_t(seed));
        AttitudeConfig cfg; cfg.dt = d.dt; cfg.q0 = d.q0_est; cfg.mag_ref_nav = d.mag_ref_nav; cfg.g = d.g; cfg.gyro_noise = sc.gyro_noise; cfg.accel_noise = sc.accel_noise + sc.vibration; cfg.mag_noise = sc.mag_noise; cfg.gyro_bias_rw = sc.gyro_bias_rw; cfg.seed = seed;
        for (auto& e : ests) {
            if (!only_set.empty() && !only_set.count(e->name())) continue;
            std::vector<Quat> qe(d.n); std::vector<Vec<3>> be(d.n);
            double best_ns = 1e300;
            for (int rep = 0; rep < reps; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                e->reset(cfg);
                for (int k = 0; k < d.n; ++k) { e->step(d.gyro[k], d.accel[k], d.mag[k]); qe[k] = e->quaternion(); if (rep == 0) be[k] = e->gyro_bias(); }
                const auto t1 = std::chrono::steady_clock::now();
                best_ns = std::min(best_ns, std::chrono::duration<double, std::nano>(t1 - t0).count() / d.n);
            }
            double se = 0, se_ss = 0, sr = 0, spp = 0, sy = 0, maxa = 0, sb = 0; int nss = 0; bool div = false; double conv = -1; int run = 0; const int win = int(30 / d.dt);
            std::vector<double> ang(d.n);
            for (int k = 0; k < d.n; ++k) {
                const double a = Quat::angle_between(d.q_true[k], qe[k]) * 180 / kPi; ang[k] = a;
                sb += (be[k] - d.gyro_bias_true[k]).norm() * (be[k] - d.gyro_bias_true[k]).norm();
                if (!std::isfinite(a)) { div = true; continue; }
                Real r1, p1, y1, r2, p2, y2; d.q_true[k].to_euler(r1, p1, y1); qe[k].to_euler(r2, p2, y2);
                auto wrap = [](double x) { while (x > kPi) { x -= 2 * kPi; } while (x < -kPi) { x += 2 * kPi; } return x; };
                const double er = wrap(r2 - r1) * 180 / kPi, ep = wrap(p2 - p1) * 180 / kPi, ey = wrap(y2 - y1) * 180 / kPi;
                se += a * a; sr += er * er; spp += ep * ep; sy += ey * ey; maxa = std::max(maxa, a);
                if (k >= d.n / 2) { se_ss += a * a; ++nss; if (a > 45) div = true; }
                if (a < 2.0) { ++run; if (run >= win && conv < 0) conv = (k - win + 1) * d.dt; } else run = 0;
            }
            summary << e->name() << ',' << e->group() << ',' << sc.name << ',' << seed << ',' << std::sqrt(se / d.n) << ',' << std::sqrt(sr / d.n) << ',' << std::sqrt(spp / d.n) << ',' << std::sqrt(sy / d.n) << ','
                    << maxa << ',' << std::sqrt(se_ss / std::max(nss, 1)) << ',' << conv << ',' << std::sqrt(sb / d.n) * 180 / kPi << ',' << (div ? 1 : 0) << ',' << best_ns << ',' << e->state_bytes() << '\n';
            summary.flush();
            std::printf("[imu|%-15s|s%d] %-22s rms=%6.3f deg (roll %.3f pitch %.3f yaw %.3f) ss=%.3f max=%.2f conv=%.1fs div=%d %7.0f ns/step\n", sc.name.c_str(), seed, e->name(), std::sqrt(se / d.n), std::sqrt(sr / d.n), std::sqrt(spp / d.n), std::sqrt(sy / d.n), std::sqrt(se_ss / std::max(nss, 1)), maxa, conv, div ? 1 : 0, best_ns);
            if (timeseries && seed == 1) {
                std::ofstream ts(out + "/ts/imu__" + e->name() + "__" + sc.name + "__s1.csv");
                ts << "t,roll_true,pitch_true,yaw_true,roll_est,pitch_est,yaw_est,angle_err_deg,bias_est_x,bias_true_x\n";
                for (int k = 0; k < d.n; k += 10) { Real r1, p1, y1, r2, p2, y2; d.q_true[k].to_euler(r1, p1, y1); qe[k].to_euler(r2, p2, y2);
                    ts << d.t[k] << ',' << r1 * 180 / kPi << ',' << p1 * 180 / kPi << ',' << y1 * 180 / kPi << ',' << r2 * 180 / kPi << ',' << p2 * 180 / kPi << ',' << y2 * 180 / kPi << ',' << ang[k] << ',' << be[k][0] * 180 / kPi << ',' << d.gyro_bias_true[k][0] * 180 / kPi << '\n'; }
            }
        }
    }
    return 0;
}
