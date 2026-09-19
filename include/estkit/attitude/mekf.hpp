// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/attitude/mekf.hpp — multiplicative extended Kalman filter (MEKF)
//  with a 6-state error [delta-theta, delta-b]
//
//  References
//    * Lefferts, E. J., Markley, F. L. & Shuster, M. D. (1982). Kalman
//      filtering for spacecraft attitude estimation. Journal of Guidance,
//      Control, and Dynamics 5(5), 417-429.        (the reference formulation)
//    * Markley, F. L. (2003). Attitude error representations for Kalman
//      filtering. Journal of Guidance, Control, and Dynamics 26(2), 311-317.
//      (the multiplicative error and the reset step)
//    * Crassidis, J. L., Markley, F. L. & Cheng, Y. (2007). Survey of nonlinear
//      attitude estimation methods. Journal of Guidance, Control, and Dynamics
//      30(1), 12-28.
//    * Farrenkopf, R. L. (1978). Analytic steady-state accuracy solutions for
//      two common spacecraft attitude estimators. Journal of Guidance and
//      Control 1(4), 282-284.            (the gyro noise/bias process model)
//    * Bucy, R. S. & Joseph, P. D. (1968). Filtering for Stochastic Processes
//      with Applications to Guidance. Wiley.        (Joseph covariance update)
//
//  State.  A reference quaternion q_hat (BODY -> NAV) and a gyro bias b_hat
//  carry the estimate; the *filter* state is the 6-vector
//      x = [ delta_theta ; delta_b ],   q = q_hat (x) exp(delta_theta),
//                                       b = b_hat + delta_b,
//  which is kept identically zero between updates (the "multiplicative" trick:
//  the three-parameter error never has to represent a large rotation, so the
//  quaternion norm constraint never makes P singular -- Markley 2003, Sec. II).
//
//  Propagation.  With omega_hat = omega_y - b_hat and the gyro model
//  omega_y = omega + b + n_v, b_dot = n_u,
//      q_hat <- q_hat (x) exp(omega_hat dt)                               (1)
//      d/dt delta_theta = -[omega_hat]_x delta_theta - delta_b - n_v      (2)
//      d/dt delta_b     = n_u                                             (3)
//  so F = [[-[omega_hat]_x, -I],[0, 0]] and (Lefferts et al. 1982, Eq. (2.9);
//  Markley & Crassidis 2014, Eq. (6.60)) with a = omega_hat dt, th = |a|,
//      Phi_11 = exp(-[a]_x) = I - (sin th/th)[a]_x + ((1-cos th)/th^2)[a]_x^2
//      Phi_12 = -dt ( I - ((1-cos th)/th^2)[a]_x + ((th-sin th)/th^3)[a]_x^2 )
//      Phi_22 = I                                                          (4)
//  and the van Loan / Farrenkopf discrete process noise
//      Q_11 = (sigma_v^2 dt + sigma_u^2 dt^3/3) I
//      Q_12 = Q_21 = -(sigma_u^2 dt^2/2) I,   Q_22 = sigma_u^2 dt I        (5)
//
//  Measurements.  Any body-frame observation of a known NAV vector r,
//  y = R^T r + noise, linearises about the reference attitude as
//      y ~ R_hat^T r + [R_hat^T r]_x delta_theta,  i.e.
//      H = [ [R_hat^T r]_x   0_{3x3} ]                                     (6)
//  (from R(q_hat (x) exp(d))^T = (I - [d]_x) R_hat^T).  Two such blocks are
//  stacked: r_a = [0,0,-g] for the specific force (NED convention: a level
//  accelerometer reads [0,0,-g]) and r_m = cfg.mag_ref_nav for the
//  magnetometer.
//
//  Update and reset (Markley 2003, Sec. V).  Joseph form for P, then
//      q_hat <- q_hat (x) exp(delta_theta_plus),  b_hat <- b_hat + delta_b_plus,
//      delta_theta <- 0, delta_b <- 0                                      (7)
//  The first-order reset leaves P unchanged; the second-order correction
//  (I - [delta_theta/2]_x) P_11 (.)^T is O(|delta_theta|^2) and is omitted, as
//  in all standard implementations.
//
//  Measurement-noise model.  On an aircraft the dominant error of the two
//  vector observations is not the sensor noise but the *model* error: the
//  specific force contains the manoeuvre acceleration (a coordinated turn at
//  bank phi hides the whole g tan(phi) lateral component) and the magnetic
//  field contains hard-iron and local disturbances.  R is therefore
//      R_a = (sigma_a^2 + sigma_a,extra^2 + (k_a e_a)^2 + (V_ref |omega_yz|)^2) I,
//      R_m = (sigma_m^2 + sigma_m,extra^2 + (k_m e_m)^2) I,                (8)
//  where e_a, e_m are the low-pass filtered deviations of |f| from g and of
//  |m| from |m_ref| and V_ref |omega_yz| = |omega x v_b| is the manoeuvre bound
//  of attitude_util.hpp (a magnitude check alone cannot see a coordinated turn,
//  where |f| = g/cos(phi) but the direction of f is wrong by the full bank
//  angle).  This is the standard "adaptive measurement covariance" of practical
//  AHRS design; set opts.gate = false to recover the textbook constant-R filter.
// =============================================================================
#pragma once
#include "attitude_estimator.hpp"
#include "attitude_util.hpp"

namespace estkit {

class Mekf : public AttitudeEstimator {
public:
    struct Options {
        Real p0_att_deg = Real(30);      // initial 1-sigma attitude uncertainty
        Real p0_bias_dps = Real(3);      // initial 1-sigma gyro-bias uncertainty
        Real q_scale = Real(1);          // multiplicative tuning of the process noise
        Real accel_extra = Real(0.5);    // m/s^2, unmodelled specific force
        Real mag_extra = Real(1.5);      // uT, hard/soft-iron and field-model error
        bool use_mag = true;
        bool joseph = true;
        bool exact_phi = true;           // closed-form Phi (false: 2nd-order series)
        // adaptive measurement covariance, Eq. (8)
        bool gate = true;
        Real gate_tau = Real(0.5);       // s
        Real accel_dead = Real(0.3);     // m/s^2
        Real accel_k = Real(15);         // extra sigma per m/s^2 of excess
        Real accel_reject = Real(1.0);   // m/s^2: above this disturbance sigma the
                                         // accelerometer update is SKIPPED (editing)
        Real accel_rate_v = Real(45);    // m/s, nominal airspeed for the manoeuvre bound (0 = off)
        Real accel_rate_dead = Real(0.02);  // rad/s, rate uncertainty dead-zone of that bound
        bool centripetal = false;        // Euston et al. (2008) mean correction f <- f - omega x v_b
        Real airspeed = Real(45);        // m/s, nominal true airspeed used by that correction
        Real mag_dead = Real(2.0);       // uT
        Real mag_k = Real(10);           // extra sigma per uT of excess
        Real mag_reject = Real(3.0);     // uT: above this the magnetometer is skipped
        Real bias_max = Real(15 * kPi / 180);
    } opts;

    const char* name() const override { return "MEKF"; }
    const char* group() const override { return "attitude"; }

    void reset(const AttitudeConfig& cfg) override {
        q_ = cfg.q0.normalized();
        b_ = Vec<3>();
        dt_ = cfg.dt;
        g_ = cfg.g;
        mref_ = cfg.mag_ref_nav;
        mref_norm_ = mref_.norm();
        sigma_a_ = cfg.accel_noise;
        sigma_m_ = cfg.mag_noise;
        // continuous-time densities from the per-sample sigmas of the dataset:
        // an angle-random-walk of sigma_d rad/s per sample integrates to
        // sigma_d dt of angle per step, i.e. sigma_v = sigma_d sqrt(dt);
        // a bias increment of sigma_rw per sample is sigma_u = sigma_rw/sqrt(dt).
        sigma_v_ = cfg.gyro_noise * std::sqrt(dt_);
        sigma_u_ = cfg.gyro_bias_rw / std::sqrt(dt_);
        P_ = Mat<6, 6>();
        const Real pa = sq(opts.p0_att_deg * Real(kPi / 180)), pb = sq(opts.p0_bias_dps * Real(kPi / 180));
        for (int i = 0; i < 3; ++i) { P_(i, i) = pa; P_(i + 3, i + 3) = pb; }
        agate_.reset();
        mgate_.reset();
    }

    void step(const Vec<3>& gyro, const Vec<3>& accel, const Vec<3>& mag) override {
        const Vec<3> w = gyro - b_;      // bias-compensated rate (also drives the gate)
        predict(w);
        // Optional centripetal (mean) correction of the specific force; with a
        // bias-compensated rate it is far more accurate than in a filter with no
        // bias state, because the residual is |db| V rather than |b| V.
        const Vec<3> f = opts.centripetal ? Vec<3>(accel - cross(w, vec3(opts.airspeed, Real(0), Real(0)))) : accel;
        update(f, mag, w);
    }

    Quat quaternion() const override { return q_; }
    Vec<3> gyro_bias() const override { return b_; }
    size_t state_bytes() const override { return sizeof(*this); }
    const Mat<6, 6>& P() const { return P_; }

private:
    // ---- (1)-(5) time update -------------------------------------------------
    void predict(const Vec<3>& w) {
        q_ = q_.integrate(w, dt_);

        Mat<3, 3> Phi11, Phi12;
        transition_blocks(w, dt_, opts.exact_phi, Phi11, Phi12);
        Mat<6, 6> Phi = Mat<6, 6>::identity();
        Phi.set_block<3, 3>(0, 0, Phi11);
        Phi.set_block<3, 3>(0, 3, Phi12);

        const Real sv2 = sq(sigma_v_) * opts.q_scale, su2 = sq(sigma_u_) * opts.q_scale;
        const Real q11 = sv2 * dt_ + su2 * dt_ * dt_ * dt_ / Real(3);
        const Real q12 = -su2 * dt_ * dt_ / Real(2);
        const Real q22 = su2 * dt_;
        Mat<6, 6> Q;
        for (int i = 0; i < 3; ++i) { Q(i, i) = q11; Q(i, i + 3) = q12; Q(i + 3, i) = q12; Q(i + 3, i + 3) = q22; }

        P_ = Phi * P_ * Phi.t() + Q;
        P_.symmetrize();
    }

    // ---- (6)-(8) measurement update -----------------------------------------
    void update(const Vec<3>& accel, const Vec<3>& mag, const Vec<3>& w) {
        const Real an = accel.norm(), mn = mag.norm();
        const Vec<3> ra = vec3(Real(0), Real(0), -g_);
        const Vec<3> ha = q_.rotate_inv(ra), hm = q_.rotate_inv(mref_);

        //  Disturbance variance of each observation, Eq. (8).  A manoeuvre or a
        //  magnetic anomaly is a SYSTEMATIC error that lasts for thousands of
        //  samples, not white noise: merely inflating R still lets the filter
        //  integrate it, so beyond a threshold the observation is edited out
        //  altogether (Gelb 1974, Sec. 4.5; Tapley et al. 2004, Sec. 4.15).
        Real sxa = Real(0), sxm = Real(0);
        if (opts.gate) {
            sxa = std::sqrt(sq(opts.accel_k * agate_.excess(an, g_, dt_, opts.gate_tau, opts.accel_dead))
                            + sq(att::manoeuvre_sigma(w, opts.accel_rate_v, opts.accel_rate_dead)));
            sxm = opts.mag_k * mgate_.excess(mn, mref_norm_, dt_, opts.gate_tau, opts.mag_dead);
        }
        Real sa2 = (sxa > opts.accel_reject) ? Real(1e12) : sq(sigma_a_) + sq(opts.accel_extra) + sq(sxa);
        Real sm2 = (sxm > opts.mag_reject) ? Real(1e12) : sq(sigma_m_) + sq(opts.mag_extra) + sq(sxm);
        if (an < Real(1e-6)) sa2 = Real(1e12);
        if (!opts.use_mag || mn < Real(1e-6)) sm2 = Real(1e12);

        Mat<6, 6> H;
        H.set_block<3, 3>(0, 0, skew(ha));
        H.set_block<3, 3>(3, 0, skew(hm));
        Vec<6> z;
        for (int i = 0; i < 3; ++i) { z[i] = accel[i] - ha[i]; z[i + 3] = mag[i] - hm[i]; }
        Mat<6, 6> R;
        for (int i = 0; i < 3; ++i) { R(i, i) = sa2; R(i + 3, i + 3) = sm2; }

        const Mat<6, 6> S = H * P_ * H.t() + R;
        const Mat<6, 6> K = P_ * H.t() * inverse(S);
        if (!K.is_finite()) return;
        const Vec<6> dx = K * z;
        if (!dx.is_finite()) return;

        if (opts.joseph) {
            const Mat<6, 6> IKH = Mat<6, 6>::identity() - K * H;
            P_ = IKH * P_ * IKH.t() + K * R * K.t();
        } else {
            P_ = (Mat<6, 6>::identity() - K * H) * P_;
        }
        P_.symmetrize();

        // (7) multiplicative reset
        const Vec<3> dth = vec3(dx[0], dx[1], dx[2]);
        q_ = (q_ * Quat::exp(dth)).normalized();
        for (int i = 0; i < 3; ++i) b_[i] = clampr(b_[i] + dx[i + 3], -opts.bias_max, opts.bias_max);
    }

    // Phi_11 = exp(-[a]_x), Phi_12 = -int_0^dt exp(-[w]_x tau) dtau, a = w dt
    static void transition_blocks(const Vec<3>& w, Real dt, bool exact, Mat<3, 3>& Phi11, Mat<3, 3>& Phi12) {
        const Vec<3> a = w * dt;
        const Real th = a.norm();
        const Mat<3, 3> A = skew(a), A2 = A * A, I = Mat<3, 3>::identity();
        Real c1 = Real(1), c2 = Real(0.5), d1 = Real(0.5), d2 = Real(1) / Real(6);
        if (exact && th > Real(1e-6)) {
            c1 = std::sin(th) / th;
            c2 = (Real(1) - std::cos(th)) / (th * th);
            d1 = c2;
            d2 = (th - std::sin(th)) / (th * th * th);
        }
        Phi11 = I - A * c1 + A2 * c2;
        Phi12 = (I - A * d1 + A2 * d2) * (-dt);
    }

    Quat q_;
    Vec<3> b_, mref_;
    Mat<6, 6> P_;
    Real dt_ = Real(0.01), g_ = Real(9.80665), mref_norm_ = Real(48);
    Real sigma_a_ = Real(0.05), sigma_m_ = Real(0.5), sigma_v_ = Real(1e-4), sigma_u_ = Real(1e-4);
    att::NormGate agate_, mgate_;
};

}  // namespace estkit
