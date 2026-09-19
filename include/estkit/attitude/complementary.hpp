// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/attitude/complementary.hpp — quaternion complementary filter
//  (gyro high-pass + accelerometer/magnetometer low-pass, fixed gains)
//
//  References
//    * Higgins, W. T. (1975). A comparison of complementary and Kalman
//      filtering. IEEE Transactions on Aerospace and Electronic Systems
//      AES-11(3), 321-325.                    (the complementary/Kalman duality)
//    * Euston, M., Coote, P., Mahony, R., Kim, J. & Hamel, T. (2008). A
//      complementary filter for attitude estimation of a fixed-wing UAV.
//      Proc. IEEE/RSJ International Conference on Intelligent Robots and
//      Systems (IROS), 340-345.       (centripetal / specific-force correction)
//    * Titterton, D. H. & Weston, J. L. (2004). Strapdown Inertial Navigation
//      Technology, 2nd ed., IET, Ch. 11.          (tilt / heading construction)
//    * Markley, F. L. & Crassidis, J. L. (2014). Fundamentals of Spacecraft
//      Attitude Determination and Control. Springer, Ch. 5.
//
//  Idea.  The gyroscope is accurate at high frequency and drifts at low
//  frequency; the accelerometer/magnetometer attitude is noisy and biased by
//  manoeuvres at high frequency but drift-free at low frequency.  With G(s) the
//  low-pass  1/(tau s + 1)  the complementary pair  G(s) + (1-G(s)) = 1  splits
//  the spectrum without any covariance bookkeeping (Higgins 1975 shows this is
//  the steady-state Kalman filter of a two-state model, which is why the single
//  knob tau is enough).
//
//  Implementation (per sample, dt = 1/100 s)
//    1. gyro branch (high-pass):    q <- q (x) exp(omega dt)
//    2. specific-force correction (optional, Euston et al. 2008):
//         with body velocity v_b ~ [V,0,0] and V_dot ~ 0,
//         f_b = v_dot_b + omega x v_b - R^T g_nav
//         => gravity estimate  a_c = f_b - omega x [V,0,0]^T
//                                  = [f_x, f_y - omega_z V, f_z + omega_y V]^T
//    3. tilt blend (low-pass, gain gamma_a = dt/(tau_a+dt)):
//         (phi_a, theta_a) from a_c; q_meas = q_euler(phi_a, theta_a, psi_hat);
//         delta = log(q^* (x) q_meas);   q <- q (x) exp(gamma_a delta)
//    4. heading blend (low-pass, gain gamma_m = dt/(tau_m+dt)) about the NAV
//       vertical so that it cannot disturb roll/pitch:
//         psi_m = atan2(-y_h, x_h) + D;  e = wrap(psi_m - psi_hat)
//         q <- exp_z(gamma_m e) (x) q
//    5. both blends are de-rated by a disturbance weight
//         w = s_base^2/(s_base^2 + s_disturbance^2)
//       built from the low-pass filtered magnitude anomaly (|f| vs g, |m| vs
//       |m_ref|), so that a sustained unmodelled specific force or a magnetic
//       disturbance reduces the gain of the corresponding branch; beyond a
//       rejection threshold the branch is switched off entirely.
//
//  The filter has NO bias state on purpose: it is the memory-light baseline
//  against which the bias-estimating filters (Mahony, MEKF, IEKF) are measured.
//  A constant gyro bias b therefore shows up as a steady-state attitude error
//  of magnitude ~ |b| tau (see the chapter).
// =============================================================================
#pragma once
#include "attitude_estimator.hpp"
#include "attitude_util.hpp"

namespace estkit {

class ComplementaryFilter : public AttitudeEstimator {
public:
    struct Options {
        Real tau_acc = Real(1.0);        // s, tilt cross-over time constant (1-2 s)
        Real tau_mag = Real(2.0);        // s, heading cross-over time constant
        // --- specific-force (centripetal) compensation, Euston et al. 2008 ----
        bool centripetal = true;         // subtract omega x v_b from the accelerometer
        Real airspeed = Real(45);        // m/s, nominal true airspeed of the platform
        // --- disturbance de-rating of the two correction branches -------------
        //  w = s_base^2/(s_base^2 + s_extra^2): the scalar Kalman gain ratio,
        //  which multiplies the blending gain gamma of that branch.
        bool gate = true;
        Real gate_tau = Real(0.5);       // s, low-pass on |f| and |m| before gating
        Real accel_extra = Real(1.0);    // m/s^2, residual specific-force model error
        Real accel_dead = Real(2.0);     // m/s^2 tolerated ||f|-g| (the centripetal
                                         // correction itself is only good to ~1 m/s^2)
        Real accel_k = Real(5);          // extra sigma per m/s^2 of excess
        Real accel_reject = Real(6);     // m/s^2: above this the tilt branch is disabled
        Real accel_rate_v = Real(0);     // m/s, manoeuvre bound (0: the centripetal
                                         // correction already removes the mean)
        Real accel_rate_dead = Real(0.02);  // rad/s, rate uncertainty dead-zone
        Real mag_extra = Real(1.5);      // uT, hard/soft-iron allowance
        Real mag_dead = Real(2.0);       // uT tolerated ||m|-|m_ref||
        Real mag_k = Real(10);           // extra sigma per uT of excess
        Real mag_reject = Real(3.0);     // uT: above this the heading branch is disabled
    } opts;

    const char* name() const override { return "Complementary"; }
    const char* group() const override { return "attitude"; }

    void reset(const AttitudeConfig& cfg) override {
        q_ = cfg.q0.normalized();
        dt_ = cfg.dt;
        g_ = cfg.g;
        mref_ = cfg.mag_ref_nav;
        mref_norm_ = mref_.norm();
        decl_ = std::atan2(mref_[1], mref_[0]);
        sigma_a_ = cfg.accel_noise;
        sigma_m_ = cfg.mag_noise;
        agate_.reset();
        mgate_.reset();
    }

    void step(const Vec<3>& gyro, const Vec<3>& accel, const Vec<3>& mag) override {
        // --- 1. gyro branch ---------------------------------------------------
        q_ = q_.integrate(gyro, dt_);

        // --- 2. specific-force correction ------------------------------------
        Vec<3> f = accel;
        if (opts.centripetal) f = f - cross(gyro, vec3(opts.airspeed, Real(0), Real(0)));

        // --- 3. tilt blend ----------------------------------------------------
        const Real fn = f.norm();
        if (fn > Real(1e-3)) {
            Real wa = Real(1);
            if (opts.gate) {
                const Real e = agate_.excess(fn, g_, dt_, opts.gate_tau, opts.accel_dead);
                const Real sx = std::sqrt(sq(opts.accel_k * e) + sq(att::manoeuvre_sigma(gyro, opts.accel_rate_v, opts.accel_rate_dead)));
                wa = (sx > opts.accel_reject) ? Real(0)
                                              : att::obs_weight(sq(sigma_a_) + sq(opts.accel_extra), sq(sx));
            }
            if (wa > Real(0)) {
                Real ra = 0, pa = 0;
                att::tilt_from_accel(f, ra, pa);
                Real r = 0, p = 0, y = 0;
                q_.to_euler(r, p, y);
                const Quat q_meas = Quat::from_euler(ra, pa, y);
                const Vec<3> dv = (q_.conj() * q_meas).log();     // body-frame rotation vector
                const Real gamma = wa * dt_ / (opts.tau_acc + dt_);
                q_ = (q_ * Quat::exp(dv * gamma)).normalized();
            }
        }

        // --- 4. heading blend -------------------------------------------------
        const Real mn = mag.norm();
        if (mn > Real(1e-3)) {
            Real wm = Real(1);
            if (opts.gate) {
                const Real e = mgate_.excess(mn, mref_norm_, dt_, opts.gate_tau, opts.mag_dead);
                const Real sx = opts.mag_k * e;
                wm = (sx > opts.mag_reject) ? Real(0)
                                            : att::obs_weight(sq(sigma_m_) + sq(opts.mag_extra), sq(sx));
            }
            if (wm > Real(0)) {
                Real r = 0, p = 0, y = 0;
                q_.to_euler(r, p, y);
                const Real psi_m = att::heading_from_mag(mag, r, p) + decl_;
                const Real e = att::wrap_pi(psi_m - y);
                const Real u = wm * dt_ / (opts.tau_mag + dt_) * e;
                const Quat qz(std::cos(u / 2), Real(0), Real(0), std::sin(u / 2));
                q_ = (qz * q_).normalized();                       // rotation about the NAV vertical
            }
        }
    }

    Quat quaternion() const override { return q_; }
    size_t state_bytes() const override { return sizeof(*this); }

private:
    Quat q_;
    Real dt_ = Real(0.01), g_ = Real(9.80665), mref_norm_ = Real(48), decl_ = Real(0);
    Real sigma_a_ = Real(0.05), sigma_m_ = Real(0.5);
    Vec<3> mref_;
    att::NormGate agate_, mgate_;
};

}  // namespace estkit
