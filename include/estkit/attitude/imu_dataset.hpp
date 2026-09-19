// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/attitude/imu_dataset.hpp — synthetic 9-axis IMU data for a small
//  aircraft / drone flight, with truth attitude, for benchmarking attitude
//  estimators (aerospace GNC heritage of the state-estimation toolbox).
//
//  Frames: navigation = NED (x north, y east, z down), body = FRD (x forward,
//  y right, z down).  q maps body -> nav.  Gravity g_nav = [0, 0, +9.81].
//  Accelerometer measures specific force  f_b = R^T (a_nav - g_nav)   (at rest: R^T [0,0,-g])
//  Gyroscope measures body rate           w_b = omega + b_g + n_g      (bias random walk)
//  Magnetometer measures                  m_b = R^T m_nav + b_m + n_m,  m_nav = B [cos I, 0, sin I]
//  (magnetic inclination I = 65 deg, B = 48 uT: central Europe; a residual hard-iron
//  offset of ~0.4 uT (0.8 % of the field) represents a calibrated compass).
//  Truth attitude follows a scripted mission (take-off, climb, coordinated
//  turns, cruise with turbulence, descent); body rates come from the Euler-rate
//  kinematics and the linear acceleration from the flight-path velocity.
// =============================================================================
#pragma once
#include <string>
#include <vector>
#include "../core/quaternion.hpp"
#include "../core/rng.hpp"

namespace estkit {

struct ImuScenario {
    std::string name = "nominal";
    Real gyro_noise = Real(0.1 * kPi / 180);     // rad/s per sample (100 Hz)
    Real gyro_bias0 = Real(0.5 * kPi / 180);     // rad/s initial bias magnitude per axis
    Real gyro_bias_rw = Real(0.002 * kPi / 180); // rad/s per sqrt(sample) random walk
    Real accel_noise = Real(0.05);               // m/s^2
    Real accel_bias = Real(0.02);                // m/s^2
    Real mag_noise = Real(0.5);                  // uT
    Real init_error_deg = Real(0);               // estimator initial attitude error (all axes)
    Real mag_disturb_uT = Real(0);               // magnetic disturbance magnitude applied for 20 s at t = 300 s
    Real vibration = Real(0);                    // extra accelerometer vibration noise (m/s^2)
    Real turbulence = Real(1);                   // scale of the attitude oscillations
};

inline std::vector<ImuScenario> imu_scenarios() {
    std::vector<ImuScenario> s; ImuScenario a;
    a.name = "nominal"; s.push_back(a);
    a = ImuScenario(); a.name = "init_error"; a.init_error_deg = Real(30); s.push_back(a);
    a = ImuScenario(); a.name = "gyro_bias"; a.gyro_bias0 = Real(3.0 * kPi / 180); s.push_back(a);
    a = ImuScenario(); a.name = "mag_disturbance"; a.mag_disturb_uT = Real(25); s.push_back(a);
    a = ImuScenario(); a.name = "vibration"; a.vibration = Real(1.0); s.push_back(a);
    a = ImuScenario(); a.name = "high_dynamics"; a.turbulence = Real(3); s.push_back(a);
    return s;
}

struct ImuDataset {
    std::string scenario; uint64_t seed = 0; Real dt = Real(0.01); int n = 0;
    std::vector<Real> t;
    std::vector<Vec<3>> gyro, accel, mag;        // measured
    std::vector<Vec<3>> gyro_bias_true;
    std::vector<Quat> q_true;
    Vec<3> mag_ref_nav;                          // known reference field (uT)
    Real g = Real(9.80665);
    Quat q0_est;                                 // initial attitude handed to estimators
};

// Three independent draws in a defined order (function-argument evaluation order is
// unspecified in C++, which would make the dataset compiler-dependent).
inline Vec<3> noise3(Rng& rng, Real s) { Vec<3> n; n[2] = s * rng.normal(); n[1] = s * rng.normal(); n[0] = s * rng.normal(); return n; }   // z,y,x: the order that reproduces the recorded dataset
inline ImuDataset make_imu_dataset(const ImuScenario& sc, uint64_t seed, Real duration_s = Real(900), Real dt = Real(0.01)) {
    ImuDataset d; d.scenario = sc.name; d.seed = seed; d.dt = dt; d.n = int(duration_s / dt);
    Rng rng(seed * 7919 + 101);
    const Real incl = Real(65 * kPi / 180), B = Real(48);
    d.mag_ref_nav = vec3(B * std::cos(incl), Real(0), B * std::sin(incl));
    const Vec<3> g_nav = vec3(0, 0, d.g);
    // --- scripted Euler-angle mission ---
    auto script = [&](Real t, Real& roll, Real& pitch, Real& yaw, Real& V) {
        const Real d2r = Real(kPi / 180);
        Real r = 0, p = 0, y = 0; V = Real(45);
        if (t < 60) { p = 0; V = Real(20 + t * 0.4); }                                            // taxi / take-off roll
        else if (t < 180) { p = 10; V = Real(44); }                                                // climb
        else if (t < 240) { r = 25; p = 4; y = (t - 180) * Real(5.2); }                            // right turn (coordinated)
        else if (t < 480) { p = 2; y = 312; }                                                      // cruise
        else if (t < 540) { r = -25; p = 3; y = 312 - (t - 480) * Real(5.2); }                     // left turn
        else if (t < 780) { p = -4; y = 0; V = Real(40); }                                         // descent
        else if (t < 840) { r = 20; p = 1; y = (t - 780) * Real(4.0); }                            // base turn
        else { p = -3; y = 240; V = Real(35); }                                                    // final
        // smooth transitions + turbulence
        const Real turb = sc.turbulence;
        r += turb * (Real(3) * std::sin(Real(2 * kPi * 0.5) * t) + Real(1.5) * std::sin(Real(2 * kPi * 0.13) * t + 1));
        p += turb * (Real(1.5) * std::sin(Real(2 * kPi * 0.3) * t + 2) + Real(0.7) * std::sin(Real(2 * kPi * 0.07) * t));
        y += turb * (Real(1.0) * std::sin(Real(2 * kPi * 0.05) * t + 3));
        roll = r * d2r; pitch = p * d2r; yaw = y * d2r;
    };
    // low-pass the script (2 s time constant) so that rates are finite; then derive rates numerically
    Real rf = 0, pf = 0, yf = 0, Vf = 20; bool first = true;
    const Real alpha = dt / Real(2.0);
    std::vector<Real> R(d.n), P(d.n), Y(d.n), Vs(d.n);
    for (int k = 0; k < d.n; ++k) {
        Real r, p, y, V; script(k * dt, r, p, y, V);
        if (first) { rf = r; pf = p; yf = y; Vf = V; first = false; }
        // unwrap yaw before filtering
        while (y - yf > kPi) y -= 2 * kPi;
        while (y - yf < -kPi) y += 2 * kPi;
        rf += alpha * (r - rf); pf += alpha * (p - pf); yf += alpha * (y - yf); Vf += alpha * (V - Vf);
        R[k] = rf; P[k] = pf; Y[k] = yf; Vs[k] = Vf;
    }
    d.t.resize(d.n); d.gyro.resize(d.n); d.accel.resize(d.n); d.mag.resize(d.n); d.gyro_bias_true.resize(d.n); d.q_true.resize(d.n);
    Vec<3> bg = vec3(sc.gyro_bias0, -sc.gyro_bias0 * Real(0.6), sc.gyro_bias0 * Real(0.4));
    const Vec<3> ba = vec3(sc.accel_bias, -sc.accel_bias, sc.accel_bias * Real(0.5));
    const Vec<3> bm = vec3(Real(0.3), Real(-0.15), Real(0.25));   // residual hard-iron after calibration (uT)
    Vec<3> v_prev;
    for (int k = 0; k < d.n; ++k) {
        const Real t = k * dt; d.t[k] = t;
        const Quat q = Quat::from_euler(R[k], P[k], Y[k]); d.q_true[k] = q;
        // Euler rates by central differences
        const int kp = std::min(k + 1, d.n - 1), km = std::max(k - 1, 0);
        const Real rd = (R[kp] - R[km]) / ((kp - km) * dt), pd = (P[kp] - P[km]) / ((kp - km) * dt), yd = (Y[kp] - Y[km]) / ((kp - km) * dt);
        // body rates from ZYX Euler-rate kinematics
        const Real sr = std::sin(R[k]), cr = std::cos(R[k]), sp = std::sin(P[k]), cp = std::cos(P[k]);
        const Vec<3> omega = vec3(rd - yd * sp, pd * cr + yd * cp * sr, yd * cp * cr - pd * sr);
        // flight-path velocity (nav) and its derivative -> linear acceleration
        const Real gamma = P[k] - Real(3 * kPi / 180);
        const Vec<3> v_nav = vec3(Vs[k] * std::cos(Y[k]) * std::cos(gamma), Vs[k] * std::sin(Y[k]) * std::cos(gamma), -Vs[k] * std::sin(gamma));
        Vec<3> a_nav; if (k > 0) a_nav = (v_nav - v_prev) / dt; v_prev = v_nav;
        // sensors
        bg = bg + noise3(rng, sc.gyro_bias_rw);
        d.gyro_bias_true[k] = bg;
        d.gyro[k] = omega + bg + noise3(rng, sc.gyro_noise);
        const Real an = sc.accel_noise + sc.vibration;
        d.accel[k] = q.rotate_inv(a_nav - g_nav) + ba + noise3(rng, an);
        Vec<3> m_nav = d.mag_ref_nav;
        if (sc.mag_disturb_uT > 0 && t >= 300 && t < 320) m_nav = m_nav + vec3(Real(0), sc.mag_disturb_uT, Real(0));
        d.mag[k] = q.rotate_inv(m_nav) + bm + noise3(rng, sc.mag_noise);
    }
    // initial estimate handed to the estimators
    const Real e = sc.init_error_deg * Real(kPi / 180);
    d.q0_est = (d.q_true[0] * Quat::from_euler(e, e, e)).normalized();
    return d;
}

}  // namespace estkit
