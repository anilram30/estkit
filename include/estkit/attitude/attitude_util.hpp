// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/attitude/attitude_util.hpp — small shared helpers for the attitude
//  (AHRS) estimator family.  Nothing here is an estimator; these are the frame
//  conversions, the vector-magnitude disturbance detector and the rotation
//  matrix -> quaternion conversion that several of the filters need.
//
//  Frame conventions of the benchmark dataset (imu_dataset.hpp)
//    * navigation frame: NED (x north, y east, z down), gravity g_nav=[0,0,+g];
//    * body frame: FRD (x forward, y right, z down);
//    * q maps BODY -> NAV: v_nav = R(q) v_body (Hamilton, scalar first);
//    * accelerometer measures specific force  f_b = R^T (a_nav - g_nav),
//      so at rest f_b = R^T [0,0,-g]:  the *nav reference* of the normalised
//      accelerometer is  d_a = [0,0,-1]  (i.e. "up" in NED);
//    * magnetometer measures m_b = R^T m_nav + b_hard + n, with the reference
//      m_nav = B [cos I, 0, sin I] handed to the estimators in cfg.mag_ref_nav.
//
//  References for the conventions
//    * Kuipers, J. B. (1999). Quaternions and Rotation Sequences. Princeton
//      University Press, Ch. 7.
//    * Markley, F. L. & Crassidis, J. L. (2014). Fundamentals of Spacecraft
//      Attitude Determination and Control. Springer, Ch. 2 and Ch. 5.
//    * Shepperd, S. W. (1978). Quaternion from rotation matrix. Journal of
//      Guidance and Control 1(3), 223-224.        (numerically safe R -> q)
//    * Titterton, D. H. & Weston, J. L. (2004). Strapdown Inertial Navigation
//      Technology, 2nd ed., IET, Ch. 3 and Ch. 11.   (tilt / heading formulas)
// =============================================================================
#pragma once
#include <cmath>
#include "../core/quaternion.hpp"

namespace estkit {
namespace att {

// ---- small vector helpers ---------------------------------------------------
inline Vec<3> unit(const Vec<3>& v) { const Real n = v.norm(); return (n > Real(1e-12)) ? Vec<3>(v / n) : v; }
inline Real wrap_pi(Real a) { while (a > kPi) a -= Real(2) * kPi; while (a < -kPi) a += Real(2) * kPi; return a; }

// ---- first-order low-pass filter (time constant tau seconds) ----------------
struct Lpf1 {
    Real y = Real(0);
    bool started = false;
    void reset() { y = Real(0); started = false; }
    Real step(Real u, Real dt, Real tau) {
        if (!started) { y = u; started = true; return y; }
        const Real a = (tau > Real(0)) ? dt / (tau + dt) : Real(1);
        y += a * (u - y);
        return y;
    }
};

// ---- magnitude-based disturbance detector -----------------------------------
//  A vector observation whose magnitude is known a priori (|f| = g at rest,
//  |m| = B for an undisturbed magnetic field) can be checked against that
//  magnitude: a sustained deviation means the *model* of the measurement is
//  wrong (manoeuvre specific force, magnetic disturbance) rather than the
//  sensor being noisy.  The magnitude is low-pass filtered first so that
//  zero-mean vibration, which does not bias the attitude, does not trip the
//  detector (it inflates the norm only by sqrt(|v|^2+3 sigma^2)-|v|).
//  Returns the *excess* deviation beyond a dead-zone, in the units of the
//  measurement; the filters turn it into an extra standard deviation or into a
//  gain de-rating factor.
struct NormGate {
    Lpf1 lp;
    void reset() { lp.reset(); }
    Real excess(Real y_norm, Real ref_norm, Real dt, Real tau, Real dead) {
        const Real f = lp.step(y_norm, dt, tau);
        const Real e = std::fabs(f - ref_norm) - dead;
        return (e > Real(0)) ? e : Real(0);
    }
};

// ---- manoeuvre bound on the unmodelled specific force -----------------------
//  A magnitude check alone CANNOT detect a coordinated turn: there
//  |f| = g/cos(phi) differs from g by only 10 % at 25 deg of bank while the
//  *direction* of f is wrong by the full bank angle.  What does detect it is
//  the angular rate: with body velocity v_b the unmodelled part of the specific
//  force is v_dot_b + omega x v_b, and for a fixed-wing aircraft flying along
//  its x axis, |omega x v_b| = V sqrt(omega_y^2 + omega_z^2).  Bounding it with
//  the nominal airspeed V_ref of the airframe (a design constant, not a
//  measurement) gives a rate-driven standard deviation that inflates R exactly
//  when the aircraft manoeuvres.  Euston et al. (2008) use the same expression
//  with a measured airspeed to *correct* the mean; here it is used
//  conservatively, to widen the covariance, and needs no extra sensor.
//  omega must be bias-compensated, and a dead-zone `rate_dead` covering the
//  residual rate uncertainty (bias + noise) must be subtracted, otherwise an
//  uncompensated gyro bias would permanently disable the accelerometer and
//  dead-lock a filter whose bias estimator needs that accelerometer.
inline Real manoeuvre_sigma(const Vec<3>& omega, Real v_ref, Real rate_dead = Real(0.02)) {
    const Real r = std::sqrt(omega[1] * omega[1] + omega[2] * omega[2]) - rate_dead;
    return (r > Real(0)) ? v_ref * r : Real(0);
}

// Reliability weight of a vector observation whose variance is base_var and
// whose model error has variance extra_var: the scalar Kalman gain ratio
// base/(base+extra), used by the fixed-gain filters that have no covariance.
inline Real obs_weight(Real base_var, Real extra_var) {
    const Real d = base_var + extra_var;
    return (d > Real(0)) ? base_var / d : Real(0);
}

// ---- roll / pitch from the specific force (NED/FRD) -------------------------
//  d_b = -f/|f| is the body-frame "down" direction, i.e. the third ROW of R:
//  d_b = [-sin(theta), cos(theta) sin(phi), cos(theta) cos(phi)]^T.
inline void tilt_from_accel(const Vec<3>& f, Real& roll, Real& pitch) {
    const Vec<3> d = unit(Vec<3>(-f));
    pitch = std::asin(clampr(-d[0], Real(-1), Real(1)));
    roll = std::atan2(d[1], d[2]);
}

// ---- tilt-compensated magnetic heading --------------------------------------
//  m_level = R_y(theta) R_x(phi) m_b = R_z(psi)^T m_nav; with m_nav in the
//  north-down plane (declination D) this gives psi = atan2(-y_h, x_h) + D.
inline Real heading_from_mag(const Vec<3>& m, Real roll, Real pitch) {
    const Real sr = std::sin(roll), cr = std::cos(roll), sp = std::sin(pitch), cp = std::cos(pitch);
    const Real xh = m[0] * cp + m[1] * sp * sr + m[2] * sp * cr;
    const Real yh = m[1] * cr - m[2] * sr;
    return std::atan2(-yh, xh);
}

// ---- rotation matrix -> quaternion (Shepperd 1978) --------------------------
//  Picks the largest of {tr R, R00, R11, R22} so that the divisor is never
//  smaller than 1/2, which bounds the relative error for any rotation.
inline Quat quat_from_R(const Mat<3, 3>& R) {
    const Real tr = R.trace();
    Real m[4] = {tr, R(0, 0), R(1, 1), R(2, 2)};
    int i = 0;
    for (int k = 1; k < 4; ++k) if (m[k] > m[i]) i = k;
    Quat q;
    if (i == 0) {
        const Real r = std::sqrt(std::fmax(Real(1) + tr, Real(1e-12))), s = Real(0.5) / r;
        q.w = Real(0.5) * r; q.x = (R(2, 1) - R(1, 2)) * s; q.y = (R(0, 2) - R(2, 0)) * s; q.z = (R(1, 0) - R(0, 1)) * s;
    } else if (i == 1) {
        const Real r = std::sqrt(std::fmax(Real(1) + R(0, 0) - R(1, 1) - R(2, 2), Real(1e-12))), s = Real(0.5) / r;
        q.x = Real(0.5) * r; q.w = (R(2, 1) - R(1, 2)) * s; q.y = (R(0, 1) + R(1, 0)) * s; q.z = (R(0, 2) + R(2, 0)) * s;
    } else if (i == 2) {
        const Real r = std::sqrt(std::fmax(Real(1) - R(0, 0) + R(1, 1) - R(2, 2), Real(1e-12))), s = Real(0.5) / r;
        q.y = Real(0.5) * r; q.w = (R(0, 2) - R(2, 0)) * s; q.x = (R(0, 1) + R(1, 0)) * s; q.z = (R(1, 2) + R(2, 1)) * s;
    } else {
        const Real r = std::sqrt(std::fmax(Real(1) - R(0, 0) - R(1, 1) + R(2, 2), Real(1e-12))), s = Real(0.5) / r;
        q.z = Real(0.5) * r; q.w = (R(1, 0) - R(0, 1)) * s; q.x = (R(0, 2) + R(2, 0)) * s; q.y = (R(1, 2) + R(2, 1)) * s;
    }
    return q.normalized();
}

// ---- non-unit quaternion arithmetic (needed by the gradient-descent filter) --
inline Quat qadd(const Quat& a, const Quat& b) { return Quat(a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z); }
inline Quat qscale(const Quat& a, Real s) { return Quat(a.w * s, a.x * s, a.y * s, a.z * s); }

}  // namespace att
}  // namespace estkit
