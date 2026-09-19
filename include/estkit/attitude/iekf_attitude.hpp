// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/attitude/iekf_attitude.hpp — right-invariant extended Kalman filter
//  (RIEKF) on SO(3) x R^3 for attitude and gyro bias
//
//  References
//    * Barrau, A. & Bonnabel, S. (2017). The invariant extended Kalman filter
//      as a stable observer. IEEE Transactions on Automatic Control 62(4),
//      1797-1812.        (group-affine systems, log-linear error, Theorem 1-2;
//                         the "imperfect IEKF" with unknown biases, Sec. V)
//    * Barrau, A. & Bonnabel, S. (2018). Invariant Kalman filtering. Annual
//      Review of Control, Robotics, and Autonomous Systems 1, 237-257.
//    * Bonnabel, S. (2007). Left-invariant extended Kalman filter and attitude
//      estimation. Proc. 46th IEEE Conference on Decision and Control, 1027-1032.
//    * Markley, F. L. (2003). Attitude error representations for Kalman
//      filtering. JGCD 26(2), 311-317.       (for the MEKF it is compared with)
//
//  Why an invariant error.  The attitude kinematics R_dot = R [omega]_x are
//  equivariant under LEFT translations R -> R_0 R (a change of the navigation
//  frame), so the natural error is the RIGHT-invariant one
//      eta = R_hat R^T                                                    (1)
//  With the same input applied to the true and the estimated system,
//      eta_dot = R_hat [omega]_x R^T - R_hat [omega]_x R^T = 0,           (2)
//  i.e. the invariant error is *exactly conserved* -- no linearisation, no
//  dependence on the trajectory.  This is the group-affine / log-linear
//  property of Barrau & Bonnabel (2017, Theorem 1): the error propagation
//  matrix of the attitude block is the identity, whatever the angular rate and
//  whatever the size of the error.  Compare the MEKF, whose error obeys
//  d/dt delta_theta = -[omega_hat]_x delta_theta: correct only to first order
//  and driven by the (possibly wrong) estimated rate.
//
//  With a gyro bias the system is no longer group-affine (the bias does not
//  live in the group); this is the "imperfect IEKF" of Barrau & Bonnabel
//  (2017, Sec. V).  With omega_hat = omega_y - b_hat, beta = b - b_hat and
//  omega_y = omega + b + n_v,
//      eta_dot = R_hat [omega_hat - omega]_x R^T = [R_hat (beta + n_v)]_x eta
//  so writing eta = exp([xi]_x) with xi the NAV-frame attitude error,
//      xi_dot   = R_hat (beta + n_v)                                      (3)
//      beta_dot = n_u                                                     (4)
//  The state-transition matrix is therefore
//      Phi = [[ I , R_hat dt ],[ 0 , I ]]                                 (5)
//  exactly (F is nilpotent), and the discrete process noise is
//      Q_11 = (sigma_v^2 dt + sigma_u^2 dt^3/3) I,
//      Q_12 = (sigma_u^2 dt^2/2) R_hat,  Q_22 = sigma_u^2 dt I            (6)
//  (the rotation drops out of Q_11 because R_hat R_hat^T = I).
//
//  Right-invariant observations.  Gravity and the magnetic field are measured
//  as y = R^T r + n with r a KNOWN navigation vector: these are exactly the
//  "right-invariant" observations y = chi^{-1} b of Barrau & Bonnabel.  The
//  invariant innovation is formed in the NAV frame,
//      z = R_hat y - r = eta r - r + R_hat n = -[r]_x xi + R_hat n        (7)
//  so that
//      H = [ -[r]_x , 0_{3x3} ],   cov(R_hat n) = R_hat (sigma^2 I) R_hat^T
//                                              = sigma^2 I                (8)
//  H does not depend on the state AT ALL -- this is the second half of the
//  invariance payoff, and it is what removes the "false observability" that
//  makes an ordinary EKF inconsistent after a large heading error.
//
//  Update and reset.  Since eta = R_hat R^T = exp([xi]_x), the true attitude is
//  R = exp(-[xi]_x) R_hat, so the correction is a LEFT multiplication
//      q_hat <- exp(-xi_plus) (x) q_hat,   b_hat <- b_hat + beta_plus     (9)
//  with (xi_plus, beta_plus) = K z and the Joseph covariance update.
//
//  Measurement-noise model: identical to the MEKF (see mekf.hpp, Eq. (8)) --
//  constant allowances for the unmodelled specific force and the hard-iron
//  offset, a low-pass magnitude gate, and the rate-driven manoeuvre bound
//  V_ref |omega_yz| -- so that the benchmark difference between the two filters
//  is purely the error parametrisation.
//
//  Cost: identical to the MEKF (one 6x6 inverse per step); the only difference
//  is which frame the attitude error lives in and what Phi and H look like.
// =============================================================================
#pragma once
#include "attitude_estimator.hpp"
#include "attitude_util.hpp"

namespace estkit {

class InvariantEkf : public AttitudeEstimator {
public:
    struct Options {
        Real p0_att_deg = Real(30);
        Real p0_bias_dps = Real(3);
        Real q_scale = Real(1);
        Real accel_extra = Real(0.5);    // m/s^2, unmodelled specific force
        Real mag_extra = Real(1.5);      // uT, hard/soft-iron and field-model error
        bool use_mag = true;
        bool joseph = true;
        bool gate = true;
        Real gate_tau = Real(0.5);
        Real accel_dead = Real(0.3);
        Real accel_k = Real(15);
        Real accel_reject = Real(1.0);   // m/s^2: above this disturbance sigma the
                                         // accelerometer update is SKIPPED (editing)
        Real accel_rate_v = Real(45);    // m/s, nominal airspeed for the manoeuvre bound (0 = off)
        Real accel_rate_dead = Real(0.02);  // rad/s, rate uncertainty dead-zone of that bound
        bool centripetal = false;        // Euston et al. (2008) mean correction f <- f - omega x v_b
        Real airspeed = Real(45);        // m/s, nominal true airspeed used by that correction
        Real mag_dead = Real(2.0);
        Real mag_k = Real(10);
        Real mag_reject = Real(3.0);     // uT: above this the magnetometer is skipped
        Real bias_max = Real(15 * kPi / 180);
    } opts;

    const char* name() const override { return "InvariantEKF"; }
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
    // ---- (5),(6) time update ------------------------------------------------
    void predict(const Vec<3>& w) {
        q_ = q_.integrate(w, dt_);
        const Mat<3, 3> Rh = q_.R();

        Mat<6, 6> Phi = Mat<6, 6>::identity();
        Phi.set_block<3, 3>(0, 3, Mat<3, 3>(Rh * dt_));

        const Real sv2 = sq(sigma_v_) * opts.q_scale, su2 = sq(sigma_u_) * opts.q_scale;
        const Real q11 = sv2 * dt_ + su2 * dt_ * dt_ * dt_ / Real(3);
        const Real q22 = su2 * dt_;
        const Mat<3, 3> Q12 = Rh * (su2 * dt_ * dt_ / Real(2));
        Mat<6, 6> Q;
        for (int i = 0; i < 3; ++i) { Q(i, i) = q11; Q(i + 3, i + 3) = q22; }
        Q.set_block<3, 3>(0, 3, Q12);
        Q.set_block<3, 3>(3, 0, Mat<3, 3>(Q12.t()));

        P_ = Phi * P_ * Phi.t() + Q;
        P_.symmetrize();
    }

    // ---- (7)-(9) measurement update -----------------------------------------
    void update(const Vec<3>& accel, const Vec<3>& mag, const Vec<3>& w) {
        const Real an = accel.norm(), mn = mag.norm();
        const Vec<3> ra = vec3(Real(0), Real(0), -g_);
        const Mat<3, 3> Rh = q_.R();

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

        // invariant innovation in the NAV frame, Eq. (7)
        const Vec<3> za = Rh * accel - ra, zm = Rh * mag - mref_;
        Mat<6, 6> H;
        H.set_block<3, 3>(0, 0, Mat<3, 3>(-skew(ra)));
        H.set_block<3, 3>(3, 0, Mat<3, 3>(-skew(mref_)));
        Vec<6> z;
        for (int i = 0; i < 3; ++i) { z[i] = za[i]; z[i + 3] = zm[i]; }
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

        // (9) left (navigation-frame) correction
        const Vec<3> xi = vec3(dx[0], dx[1], dx[2]);
        q_ = (Quat::exp(Vec<3>(-xi)) * q_).normalized();
        for (int i = 0; i < 3; ++i) b_[i] = clampr(b_[i] + dx[i + 3], -opts.bias_max, opts.bias_max);
    }

    Quat q_;
    Vec<3> b_, mref_;
    Mat<6, 6> P_;
    Real dt_ = Real(0.01), g_ = Real(9.80665), mref_norm_ = Real(48);
    Real sigma_a_ = Real(0.05), sigma_m_ = Real(0.5), sigma_v_ = Real(1e-4), sigma_u_ = Real(1e-4);
    att::NormGate agate_, mgate_;
};

}  // namespace estkit
