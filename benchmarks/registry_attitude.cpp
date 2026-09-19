// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  Attitude / IMU (AHRS) estimator family — registered here
//  (see include/estkit/attitude/)
//
//  Six estimators covering the whole spectrum of attitude estimation:
//    * Complementary  — fixed-gain quaternion complementary filter with the
//      fixed-wing centripetal correction (Higgins 1975; Euston et al. 2008);
//    * Madgwick       — gradient-descent MARG filter (Madgwick et al. 2011);
//    * Mahony         — explicit nonlinear complementary filter on SO(3) with
//      bias estimation (Mahony, Hamel & Pflimlin 2008, Eq. (32)-(33));
//    * MEKF           — multiplicative EKF, 6-state error (Lefferts et al. 1982;
//      Markley 2003);
//    * InvariantEKF   — right-invariant EKF (Barrau & Bonnabel 2017);
//    * TRIAD / QUEST  — memory-less single-frame solutions of Wahba's problem
//      (Black 1964; Shuster & Oh 1981).
//
//  All tuning is expressed through each filter's public Options so that the
//  headers stay free of benchmark-specific numbers.  The values below are the
//  literature-recommended ones; the per-chapter sensitivity studies report what
//  happens away from them.
// =============================================================================
#include "estkit/attitude/attitude_estimator.hpp"
#include "estkit/attitude/complementary.hpp"
#include "estkit/attitude/madgwick.hpp"
#include "estkit/attitude/mahony.hpp"
#include "estkit/attitude/mekf.hpp"
#include "estkit/attitude/iekf_attitude.hpp"
#include "estkit/attitude/triad_quest.hpp"

namespace estkit {

void register_attitude(AttitudeList& list) {
    // --- 1. Fixed-gain quaternion complementary filter -----------------------
    //  tau_acc = 1 s / tau_mag = 2 s are the classical cross-over time constants
    //  (Higgins 1975) and the values the assignment prescribes: fast enough that
    //  a 0.5 deg/s uncompensated gyro bias only costs ~0.5 deg of tilt, slow
    //  enough that the 0.5 Hz turbulence in the accelerometer is attenuated by
    //  two decades.  The centripetal correction (Euston et al. 2008) uses the
    //  nominal cruise airspeed of the simulated aircraft (45 m/s); without it
    //  the filter tracks the accelerometer into a 23 deg roll error in every
    //  coordinated turn (measured: 18.5 deg RMS without, 6.9 deg with).
    {
        auto f = std::make_unique<ComplementaryFilter>();
        f->opts.tau_acc = Real(1.0);
        f->opts.tau_mag = Real(2.0);
        f->opts.centripetal = true;
        f->opts.airspeed = Real(45);
        f->opts.gate = true;
        list.push_back(std::move(f));
    }

    // --- 2. Madgwick MARG gradient-descent filter ----------------------------
    //  beta = 0.1 rad/s is the benchmark value requested by the assignment and
    //  sits at the top of Madgwick's recommended range (beta = sqrt(3/4)
    //  omega_beta, i.e. omega_beta ~ 6.6 deg/s of assumed gyro error).
    //  zeta = 0 keeps the registered filter bit-compatible with the published
    //  algorithm and with ahrs.filters.Madgwick; the zeta sensitivity is reported
    //  in the chapter.
    {
        auto f = std::make_unique<MadgwickFilter>();
        f->opts.beta = Real(0.1);
        f->opts.zeta = Real(0);      // published default; see the chapter for zeta > 0
        f->opts.gate = false;        // published algorithm: no manoeuvre rejection
        list.push_back(std::move(f));
    }
    //  Practical variant: bias-drift compensation (zeta) + manoeuvre gate (Section 3.4 of the report;
    //  the gate must be paired with a bias state, see the chapter).
    {
        auto f = std::make_unique<MadgwickFilter>();
        f->opts.beta = Real(0.1);
        f->opts.zeta = Real(0.005);
        f->opts.gate = true;
        f->set_name("Madgwick-Gated");
        list.push_back(std::move(f));
    }

    // --- 3. Mahony explicit complementary filter -----------------------------
    //  k_P = 1 /s, k_I = 0.1 /s^2: the linearised loop s^2 + k_P m s + k_I m
    //  (m ~ 1 with two unit direction measurements) has omega_n = 0.32 rad/s and
    //  damping 1.6 — an over-damped 3 s attitude loop and a ~20 s bias loop.
    //  k_I is at the bottom of the assignment's 0.1-0.3 band because the
    //  specific-force disturbance of the turns drives the bias integrator: at
    //  k_I = 0.2 the measured bias RMSE doubles (1.24 -> 2.09 deg/s).
    //  Equal weights on the two directions, no gating: the algorithm exactly as
    //  published.
    {
        auto f = std::make_unique<MahonyFilter>();
        f->opts.kp = Real(1.0);
        f->opts.ki = Real(0.1);
        f->opts.k_acc = Real(1.0);
        f->opts.k_mag = Real(1.0);
        f->opts.mag_ref_measured = false;   // known inertial reference, as in the paper
        f->opts.gate = false;               // published algorithm: Eq. (32)-(33) only
        list.push_back(std::move(f));
    }
    //  Practical variant with the manoeuvre gate (accelerometer weight reduced when |f| deviates from g).
    {
        auto f = std::make_unique<MahonyFilter>();
        f->opts.kp = Real(1.0);
        f->opts.ki = Real(0.1);
        f->opts.k_acc = Real(1.0);
        f->opts.k_mag = Real(1.0);
        f->opts.mag_ref_measured = false;
        f->opts.gate = true;
        f->set_name("Mahony-Gated");
        list.push_back(std::move(f));
    }

    // --- 4. Multiplicative EKF ------------------------------------------------
    //  The process noise comes from the config (gyro_noise, gyro_bias_rw); only
    //  the *measurement* model needs engineering judgement, because the
    //  dominant error of both vector observations is model error, not sensor
    //  noise: accel_extra = 0.5 m/s^2 (~3 deg of tilt) covers the residual
    //  linear acceleration of ordinary flight, mag_extra = 1.5 uT covers the
    //  residual hard-iron offset of the simulated (calibrated) compass
    //  (|b_hard| = 0.42 uT) with margin.  The
    //  magnitude gate plus the rate-driven manoeuvre bound handle the two large
    //  disturbances — the specific force of a coordinated turn (which a
    //  magnitude check alone cannot see) and the 25 uT magnetic disturbance.
    //  Beyond 1 m/s^2 / 3 uT of modelled disturbance the observation is edited
    //  out rather than merely de-weighted, because a manoeuvre lasts thousands
    //  of samples and a finite R would still integrate its bias.
    {
        auto f = std::make_unique<Mekf>();
        f->opts.p0_att_deg = Real(30);
        f->opts.p0_bias_dps = Real(3);
        f->opts.accel_extra = Real(0.5);
        f->opts.mag_extra = Real(1.5);
        f->opts.gate = true;
        f->opts.accel_rate_v = Real(45);       // nominal airspeed: |omega x v_b| bound
        f->opts.accel_rate_dead = Real(0.02);  // rad/s of rate uncertainty
        f->opts.accel_reject = Real(1.0);      // m/s^2 -> edit the accelerometer out
        f->opts.centripetal = false;           // standard MEKF measurement model
        list.push_back(std::move(f));
    }

    // --- 5. Right-invariant EKF ----------------------------------------------
    //  Same noise model and the same measurement-covariance policy as the MEKF,
    //  so that the benchmark isolates the effect of the error parametrisation.
    {
        auto f = std::make_unique<InvariantEkf>();
        f->opts.p0_att_deg = Real(30);
        f->opts.p0_bias_dps = Real(3);
        f->opts.accel_extra = Real(0.5);
        f->opts.mag_extra = Real(1.5);
        f->opts.gate = true;
        f->opts.accel_rate_v = Real(45);
        f->opts.accel_rate_dead = Real(0.02);
        f->opts.accel_reject = Real(1.0);
        f->opts.centripetal = false;
        list.push_back(std::move(f));
    }

    // --- 6. TRIAD and QUEST (memory-less baselines) ---------------------------
    //  TRIAD trusts the accelerometer completely; QUEST weighs the two
    //  observations by the inverse variance of their direction errors, derived
    //  from the configured sensor noise (so the vibration scenario
    //  automatically shifts the weight onto the magnetometer).
    {
        auto f = std::make_unique<TriadQuest>("TRIAD", TriadQuest::Method::Triad);
        f->opts.primary = 0;
        list.push_back(std::move(f));
    }
    {
        auto f = std::make_unique<TriadQuest>("QUEST", TriadQuest::Method::Quest);
        list.push_back(std::move(f));
    }
}

}  // namespace estkit
