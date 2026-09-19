// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/attitude/mahony.hpp — explicit nonlinear complementary filter on
//  SO(3) with gyro-bias estimation ("Mahony filter")
//
//  References
//    * Mahony, R., Hamel, T. & Pflimlin, J.-M. (2008). Nonlinear complementary
//      filters on the special orthogonal group. IEEE Transactions on Automatic
//      Control 53(5), 1203-1218.   (explicit complementary filter, Eq. (32)-(33);
//      Lyapunov / passivity analysis, Theorem 5.1)
//    * Hamel, T. & Mahony, R. (2006). Attitude estimation on SO(3) based on
//      direct inertial measurements. Proc. IEEE ICRA, 2170-2175.
//    * Euston, M. et al. (2008). A complementary filter for attitude estimation
//      of a fixed-wing UAV. Proc. IEEE/RSJ IROS, 340-345.
//
//  Algorithm (Mahony et al. 2008, Eq. (32)-(33)), here written for the BODY ->
//  NAV Hamilton quaternion q with R = R(q):
//
//      v_hat_i = R^T v_0i                            predicted body direction
//      omega_mes = sum_i k_i ( v_i x v_hat_i )                          (32a)
//      b_dot   = -k_I omega_mes                                          (33)
//      R_dot   = R [ omega_y - b + k_P omega_mes ]_x                     (32)
//
//  with v_0i the known NAV reference directions (normalised gravity reference
//  d_a = [0,0,-1] for the specific force and the normalised cfg.mag_ref_nav for
//  the magnetometer) and v_i the corresponding normalised measurements.
//
//  Why the signs are these.  Write the true attitude as q = q_hat (x) exp(delta)
//  with delta a body-frame rotation vector.  Then v_i = R^T v_0i =
//  (I - [delta]_x) v_hat_i + O(delta^2), so
//      omega_mes = sum_i k_i (v_i x v_hat_i) = [ sum_i k_i (I - v_hat_i v_hat_i^T) ] delta
//                = M delta,      M = M^T >= 0,
//  and applying the body-frame correction q_hat <- q_hat (x) exp(k_P omega_mes dt)
//  gives delta_dot = -k_P M delta - b_tilde, b_tilde = b - b_hat, while (33)
//  gives b_tilde_dot = +k_I M delta: the linearisation is the double integrator
//  s^2 + k_P m s + k_I m = 0, asymptotically stable for k_P, k_I > 0.  M is
//  positive definite as soon as two non-collinear directions are measured,
//  which is the observability condition of the paper (Theorem 5.1: almost
//  global asymptotic stability, the unstable set being a measure-zero set of
//  180-degree errors).
//
//  Discrete implementation (dt = 1/100 s): the bias integrator is applied
//  first, then q is propagated with the exact exponential map of the corrected
//  rate, q <- q (x) exp(omega_c dt), which preserves ||q|| = 1 by construction.
//
//  Cross-validation: with k_acc = k_mag = 1, gate = false and
//  mag_ref_measured = true this is the algorithm implemented by the Python
//  `ahrs` library (ahrs.filters.Mahony.updateMARG) applied to (-accel, mag).
// =============================================================================
#pragma once
#include "attitude_estimator.hpp"
#include "attitude_util.hpp"

namespace estkit {

class MahonyFilter : public AttitudeEstimator {
public:
    struct Options {
        Real kp = Real(1.0);            // proportional gain [1/s]
        Real ki = Real(0.2);            // integral (bias) gain [1/s^2]
        Real k_acc = Real(1.0);         // weight of the accelerometer direction
        Real k_mag = Real(1.0);         // weight of the magnetometer direction
        bool mag_yaw_only = false;      // project the magnetic error onto the vertical
        bool mag_ref_measured = false;  // Madgwick-style reference from the measured field
        Real bias_max = Real(10 * kPi / 180);   // anti-windup clamp [rad/s]
        // Optional magnitude gating (not in the 2008 paper; keep false for
        // cross-validation against reference implementations).
        bool gate = false;
        Real gate_tau = Real(0.5);
        Real accel_extra = Real(0.5), accel_dead = Real(0.3), accel_k = Real(15);   // m/s^2
        Real accel_rate_v = Real(45);                            // m/s manoeuvre bound
        Real accel_rate_dead = Real(0.02);                       // rad/s dead-zone
        Real accel_reject = Real(1.0);                           // m/s^2, editing threshold
        Real mag_extra = Real(1.5), mag_dead = Real(2.0), mag_k = Real(10);         // uT
        Real mag_reject = Real(3.0);                             // uT, editing threshold
    } opts;

    const char* name() const override { return name_; }
    void set_name(const char* n) { name_ = n; }   // registered instance name (e.g. "Mahony-Gated")
    const char* group() const override { return "attitude"; }

    void reset(const AttitudeConfig& cfg) override {
        q_ = cfg.q0.normalized();
        b_ = Vec<3>();
        dt_ = cfg.dt;
        g_ = cfg.g;
        mref_ = att::unit(cfg.mag_ref_nav);
        mref_norm_ = cfg.mag_ref_nav.norm();
        sigma_a_ = cfg.accel_noise;
        sigma_m_ = cfg.mag_noise;
        agate_.reset();
        mgate_.reset();
    }

    void step(const Vec<3>& gyro, const Vec<3>& accel, const Vec<3>& mag) override {
        Vec<3> omega_mes;
        const Real an = accel.norm(), mn = mag.norm();
        const Vec<3> wc0 = gyro - b_;

        // --- accelerometer: NAV reference d_a = [0,0,-1] (NED specific force) --
        Vec<3> va_hat = q_.rotate_inv(vec3(Real(0), Real(0), Real(-1)));
        if (an > Real(1e-6)) {
            Real wa = Real(1);
            if (opts.gate) {
                const Real e = agate_.excess(an, g_, dt_, opts.gate_tau, opts.accel_dead);
                const Real sx = std::sqrt(sq(opts.accel_k * e) + sq(att::manoeuvre_sigma(wc0, opts.accel_rate_v, opts.accel_rate_dead)));
                wa = (sx > opts.accel_reject) ? Real(0)
                                              : att::obs_weight(sq(sigma_a_) + sq(opts.accel_extra), sq(sx));
            }
            if (wa > Real(0)) omega_mes = omega_mes + cross(Vec<3>(accel / an), va_hat) * (opts.k_acc * wa);
        }

        // --- magnetometer -----------------------------------------------------
        if (mn > Real(1e-6)) {
            Real wm = Real(1);
            if (opts.gate) {
                const Real e = mgate_.excess(mn, mref_norm_, dt_, opts.gate_tau, opts.mag_dead);
                const Real sx = opts.mag_k * e;
                wm = (sx > opts.mag_reject) ? Real(0)
                                            : att::obs_weight(sq(sigma_m_) + sq(opts.mag_extra), sq(sx));
            }
            if (wm > Real(0)) {
                const Vec<3> vm = mag / mn;
                Vec<3> dm = mref_;
                if (opts.mag_ref_measured) {
                    const Vec<3> h = q_.rotate(vm);
                    dm = att::unit(vec3(std::sqrt(h[0] * h[0] + h[1] * h[1]), Real(0), h[2]));
                }
                const Vec<3> vm_hat = q_.rotate_inv(dm);
                Vec<3> em = cross(vm, vm_hat);
                if (opts.mag_yaw_only) {
                    // keep only the component about the body-frame vertical, so
                    // that magnetic errors cannot corrupt roll and pitch
                    const Vec<3> up = -va_hat;                    // = R^T [0,0,1]
                    em = up * dot(em, up);
                }
                omega_mes = omega_mes + em * (opts.k_mag * wm);
            }
        }

        // --- (33) bias integrator, then (32) attitude propagation -------------
        b_ = b_ - omega_mes * (opts.ki * dt_);
        for (int i = 0; i < 3; ++i) b_[i] = clampr(b_[i], -opts.bias_max, opts.bias_max);
        const Vec<3> omega_c = gyro - b_ + omega_mes * opts.kp;
        q_ = q_.integrate(omega_c, dt_);
    }

    Quat quaternion() const override { return q_; }
    Vec<3> gyro_bias() const override { return b_; }
    size_t state_bytes() const override { return sizeof(*this); }

private:
    const char* name_ = "Mahony";
    Quat q_;
    Vec<3> b_, mref_;
    Real dt_ = Real(0.01), g_ = Real(9.80665), mref_norm_ = Real(48);
    Real sigma_a_ = Real(0.05), sigma_m_ = Real(0.5);
    att::NormGate agate_, mgate_;
};

}  // namespace estkit
