// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/attitude/madgwick.hpp — Madgwick's gradient-descent MARG filter
//
//  References
//    * Madgwick, S. O. H., Harrison, A. J. L. & Vaidyanathan, R. (2011).
//      Estimation of IMU and MARG orientation using a gradient descent
//      algorithm. Proc. IEEE International Conference on Rehabilitation
//      Robotics (ICORR), 1-7.
//    * Madgwick, S. O. H. (2010). An efficient orientation filter for inertial
//      and inertial/magnetic sensor arrays. Internal report, Department of
//      Mechanical Engineering, University of Bristol.  (full derivation, the
//      magnetic-distortion compensation of Sec. 3.3 and the gyroscope bias
//      drift compensation of Sec. 3.4)
//
//  Algorithm.  The orientation is a unit quaternion q (BODY -> NAV, Hamilton,
//  scalar first).  Two estimates of q_dot are fused:
//
//    (i)  q_dot_omega = 1/2 q (x) [0, omega_c],   omega_c = omega_y - b   (7)
//    (ii) a single normalised gradient-descent step on the objective that a set
//         of known NAV reference directions d_i must map onto the measured BODY
//         directions s_i,
//              f_i(q) = R(q)^T d_i - s_i in R^3,   J_i = d f_i / d q in R^{3x4}
//              grad F = sum_i J_i^T f_i,     q_dot_eps = grad F / ||grad F||
//
//  combined with the fixed gain beta (Madgwick eq. (42)-(44)):
//
//              q_dot = q_dot_omega - beta q_dot_eps                        (A)
//              q_{k+1} = normalise( q_k + q_dot dt )                       (B)
//
//  Magnetic-distortion compensation (Sec. 3.3): rather than a tabulated field,
//  the NAV reference of the magnetometer is recomputed each step from the
//  measurement itself,
//              h = R(q) m_hat,   d_m = [ sqrt(h_x^2+h_y^2), 0, h_z ]       (C)
//  which forces the reference into the vertical plane containing north and
//  therefore makes any error in the *declination* of the measured field
//  affect the heading only, never roll and pitch.
//
//  Gyroscope bias drift compensation (Sec. 3.4): the normalised gradient is a
//  quaternion rate, so the equivalent body rate of the orientation error is
//              omega_eps = 2 q^* (x) q_dot_eps                             (D)
//              b <- b + zeta omega_eps dt,   omega_c = omega_y - b         (E)
//  an integral action whose gain zeta has the units of an angular acceleration
//  (Madgwick recommends zeta = sqrt(3/4) times the gyroscope bias drift rate).
//
//  FRAME MAPPING to this benchmark.  Madgwick's published equations are written
//  with a gravity reference d_a = [0,0,1] and an accelerometer that reads
//  +1 g on the vertical axis at rest (the "reaction force" convention of a
//  z-up earth frame).  This dataset is NED/FRD with the specific-force
//  convention f = a - g, so a level accelerometer reads [0,0,-g]: the NAV
//  reference of the *normalised accelerometer* is
//              d_a = [0, 0, -1]                                            (F)
//  Since f(q) and J(q) are both linear in d, flipping the sign of d flips the
//  sign of both, and grad F = J^T f is unchanged -- i.e. this implementation
//  is bit-for-bit Madgwick's algorithm applied to the negated accelerometer.
//  The quaternion convention is unchanged: Madgwick's reference implementation
//  propagates q_dot = 1/2 q (x) omega with omega in the sensor frame and
//  evaluates R(q)^T d, which is exactly the BODY -> NAV Hamilton quaternion
//  used here.  The magnetic part (C) transfers verbatim, because it uses the
//  measured h_z whatever its sign (h_z > 0 here: the field dips downwards in
//  the northern hemisphere and NED z points down).
//
//  Cross-validation: with opts.zeta = 0 this is the algorithm implemented by
//  the Python `ahrs` library (ahrs.filters.Madgwick.updateMARG) applied to
//  (-accel, mag, gyro).
// =============================================================================
#pragma once
#include "attitude_estimator.hpp"
#include "attitude_util.hpp"

namespace estkit {

class MadgwickFilter : public AttitudeEstimator {
public:
    struct Options {
        Real beta = Real(0.1);      // gradient-descent gain [rad/s]; beta = sqrt(3/4) * omega_beta
        Real zeta = Real(0);        // bias-drift integral gain [rad/s^2]; 0 = published default
        bool use_mag = true;        // false -> IMU-only (heading unobservable)
        Real bias_max = Real(10 * kPi / 180);   // anti-windup clamp on the bias state
        // Optional magnitude gating of the two objectives (NOT part of the
        // published algorithm; keep false for cross-validation).
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
    void set_name(const char* n) { name_ = n; }   // registered instance name (e.g. "Madgwick-Gated")
    const char* group() const override { return "attitude"; }

    void reset(const AttitudeConfig& cfg) override {
        q_ = cfg.q0.normalized();
        b_ = Vec<3>();
        dt_ = cfg.dt;
        g_ = cfg.g;
        mref_norm_ = cfg.mag_ref_nav.norm();
        sigma_a_ = cfg.accel_noise;
        sigma_m_ = cfg.mag_noise;
        agate_.reset();
        mgate_.reset();
    }

    void step(const Vec<3>& gyro, const Vec<3>& accel, const Vec<3>& mag) override {
        const Real an = accel.norm(), mn = mag.norm();
        const Vec<3> wc0 = gyro - b_;
        Real grad[4] = {Real(0), Real(0), Real(0), Real(0)};
        bool have_grad = false;

        if (an > Real(1e-6)) {
            Real wa = Real(1);
            if (opts.gate) {
                const Real e = agate_.excess(an, g_, dt_, opts.gate_tau, opts.accel_dead);
                const Real sx = std::sqrt(sq(opts.accel_k * e) + sq(att::manoeuvre_sigma(wc0, opts.accel_rate_v, opts.accel_rate_dead)));
                wa = (sx > opts.accel_reject) ? Real(0)
                                              : att::obs_weight(sq(sigma_a_) + sq(opts.accel_extra), sq(sx));
            }
            if (wa > Real(0)) {
                // (F): NAV reference of the normalised specific force in NED
                accum(q_, vec3(Real(0), Real(0), Real(-1)), Vec<3>(accel / an), wa, grad);
                have_grad = true;
            }
        }
        if (opts.use_mag && mn > Real(1e-6)) {
            Real wm = Real(1);
            if (opts.gate) {
                const Real e = mgate_.excess(mn, mref_norm_, dt_, opts.gate_tau, opts.mag_dead);
                const Real sx = opts.mag_k * e;
                wm = (sx > opts.mag_reject) ? Real(0)
                                            : att::obs_weight(sq(sigma_m_) + sq(opts.mag_extra), sq(sx));
            }
            if (wm > Real(0)) {
                const Vec<3> mh = mag / mn;
                const Vec<3> h = q_.rotate(mh);                                   // (C)
                const Real bx = std::sqrt(h[0] * h[0] + h[1] * h[1]);
                accum(q_, vec3(bx, Real(0), h[2]), mh, wm, grad);
                have_grad = true;
            }
        }

        Vec<3> w = gyro - b_;
        Quat qdot_eps(Real(0), Real(0), Real(0), Real(0));
        if (have_grad) {
            const Real gn = std::sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2] + grad[3] * grad[3]);
            if (gn > Real(1e-12)) {
                qdot_eps = Quat(grad[0] / gn, grad[1] / gn, grad[2] / gn, grad[3] / gn);
                if (opts.zeta > Real(0)) {                                        // (D),(E)
                    const Quat we = att::qscale(q_.conj() * qdot_eps, Real(2));
                    b_ = b_ + we.vec() * (opts.zeta * dt_);
                    for (int i = 0; i < 3; ++i) b_[i] = clampr(b_[i], -opts.bias_max, opts.bias_max);
                    w = gyro - b_;
                }
            }
        }

        // (A),(B)
        const Quat qw = att::qscale(q_ * Quat(Real(0), w[0], w[1], w[2]), Real(0.5));
        const Quat qdot = att::qadd(qw, att::qscale(qdot_eps, -opts.beta));
        q_ = att::qadd(q_, att::qscale(qdot, dt_)).normalized();
    }

    Quat quaternion() const override { return q_; }
    Vec<3> gyro_bias() const override { return b_; }
    size_t state_bytes() const override { return sizeof(*this); }

private:
    const char* name_ = "Madgwick";
    // grad += wgt * J^T f  for  f(q) = R(q)^T d - s,  J = df/dq in R^{3x4}
    static void accum(const Quat& q, const Vec<3>& d, const Vec<3>& s, Real wgt, Real (&grad)[4]) {
        const Real w = q.w, x = q.x, y = q.y, z = q.z;
        const Real dx = d[0], dy = d[1], dz = d[2];
        const Real two = Real(2), four = Real(4);
        const Real f[3] = {
            dx * (Real(1) - two * y * y - two * z * z) + dy * (two * x * y + two * w * z) + dz * (two * x * z - two * w * y) - s[0],
            dx * (two * x * y - two * w * z) + dy * (Real(1) - two * x * x - two * z * z) + dz * (two * y * z + two * w * x) - s[1],
            dx * (two * x * z + two * w * y) + dy * (two * y * z - two * w * x) + dz * (Real(1) - two * x * x - two * y * y) - s[2]};
        const Real J[3][4] = {
            {two * (dy * z - dz * y), two * (dy * y + dz * z), two * (dy * x - dz * w) - four * dx * y, two * (dy * w + dz * x) - four * dx * z},
            {two * (dz * x - dx * z), two * (dx * y + dz * w) - four * dy * x, two * (dx * x + dz * z), two * (dz * y - dx * w) - four * dy * z},
            {two * (dx * y - dy * x), two * (dx * z - dy * w) - four * dz * x, two * (dx * w + dy * z) - four * dz * y, two * (dx * x + dy * y)}};
        for (int k = 0; k < 4; ++k)
            for (int i = 0; i < 3; ++i) grad[k] += wgt * J[i][k] * f[i];
    }

    Quat q_;
    Vec<3> b_;
    Real dt_ = Real(0.01), g_ = Real(9.80665), mref_norm_ = Real(48);
    Real sigma_a_ = Real(0.05), sigma_m_ = Real(0.5);
    att::NormGate agate_, mgate_;
};

}  // namespace estkit
