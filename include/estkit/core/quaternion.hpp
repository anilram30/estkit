// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/core/quaternion.hpp — unit quaternions and SO(3) helpers for attitude
//  estimation.  Conventions (aerospace / JPL-free "Hamilton" convention):
//    * q = [w, x, y, z], scalar first, Hamilton product q ⊗ p.
//    * q represents the rotation from the BODY frame to the NAVIGATION frame:
//      v_nav = R(q) v_body,  R(q) = I + 2 w [v]x + 2 [v]x^2.
//    * Euler angles: ZYX (yaw psi, pitch theta, roll phi), aerospace sequence
//      (Kuipers 1999, Ch. 7; Markley & Crassidis 2014, Ch. 2).
//    * Kinematics: q_dot = 1/2 q ⊗ [0, omega_body].  Discrete propagation by the
//      exact exponential map for a constant rate over dt.
// =============================================================================
#pragma once
#include <cmath>
#include "linalg.hpp"

namespace estkit {

struct Quat {
    Real w = 1, x = 0, y = 0, z = 0;
    Quat() = default;
    Quat(Real w_, Real x_, Real y_, Real z_) : w(w_), x(x_), y(y_), z(z_) {}
    static Quat identity() { return Quat(); }
    Real norm() const { return std::sqrt(w * w + x * x + y * y + z * z); }
    Quat normalized() const { const Real n = norm(); return (n > Real(0)) ? Quat(w / n, x / n, y / n, z / n) : Quat(); }
    Quat conj() const { return Quat(w, -x, -y, -z); }
    Vec<3> vec() const { Vec<3> v; v[0] = x; v[1] = y; v[2] = z; return v; }
    // Hamilton product
    Quat operator*(const Quat& p) const {
        return Quat(w * p.w - x * p.x - y * p.y - z * p.z,
                    w * p.x + x * p.w + y * p.z - z * p.y,
                    w * p.y - x * p.z + y * p.w + z * p.x,
                    w * p.z + x * p.y - y * p.x + z * p.w);
    }
    // rotation matrix body -> nav
    Mat<3, 3> R() const {
        Mat<3, 3> m;
        m(0, 0) = 1 - 2 * (y * y + z * z); m(0, 1) = 2 * (x * y - w * z);     m(0, 2) = 2 * (x * z + w * y);
        m(1, 0) = 2 * (x * y + w * z);     m(1, 1) = 1 - 2 * (x * x + z * z); m(1, 2) = 2 * (y * z - w * x);
        m(2, 0) = 2 * (x * z - w * y);     m(2, 1) = 2 * (y * z + w * x);     m(2, 2) = 1 - 2 * (x * x + y * y);
        return m;
    }
    Vec<3> rotate(const Vec<3>& v) const { return R() * v; }            // body -> nav
    Vec<3> rotate_inv(const Vec<3>& v) const { return R().t() * v; }    // nav -> body
    // exponential map: rotation vector (rad) -> quaternion
    static Quat exp(const Vec<3>& rv) {
        const Real a = rv.norm();
        if (a < Real(1e-9)) return Quat(1, rv[0] / 2, rv[1] / 2, rv[2] / 2).normalized();
        const Real s = std::sin(a / 2) / a;
        return Quat(std::cos(a / 2), rv[0] * s, rv[1] * s, rv[2] * s);
    }
    // logarithm: quaternion -> rotation vector (rad)
    Vec<3> log() const {
        Quat q = normalized(); if (q.w < 0) { q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z; }
        const Real vn = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        Vec<3> rv; if (vn < Real(1e-9)) { rv[0] = 2 * q.x; rv[1] = 2 * q.y; rv[2] = 2 * q.z; return rv; }
        const Real a = 2 * std::atan2(vn, q.w);
        rv[0] = q.x / vn * a; rv[1] = q.y / vn * a; rv[2] = q.z / vn * a; return rv;
    }
    // integrate body rate omega (rad/s) over dt: q_{k+1} = q_k ⊗ exp(omega dt)
    Quat integrate(const Vec<3>& omega, Real dt) const { return ((*this) * Quat::exp(omega * dt)).normalized(); }
    // ZYX Euler angles (rad): roll phi (x), pitch theta (y), yaw psi (z)
    static Quat from_euler(Real roll, Real pitch, Real yaw) {
        const Real cr = std::cos(roll / 2), sr = std::sin(roll / 2), cp = std::cos(pitch / 2), sp = std::sin(pitch / 2), cy = std::cos(yaw / 2), sy = std::sin(yaw / 2);
        return Quat(cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy);
    }
    void to_euler(Real& roll, Real& pitch, Real& yaw) const {
        roll = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
        const Real sp = clampr(2 * (w * y - z * x), Real(-1), Real(1));
        pitch = std::asin(sp);
        yaw = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    }
    // angular distance between two attitudes (rad)
    static Real angle_between(const Quat& a, const Quat& b) { return (a.conj() * b).log().norm(); }
};

inline Mat<3, 3> skew(const Vec<3>& v) { Mat<3, 3> m; m(0, 1) = -v[2]; m(0, 2) = v[1]; m(1, 0) = v[2]; m(1, 2) = -v[0]; m(2, 0) = -v[1]; m(2, 1) = v[0]; return m; }
inline Vec<3> cross(const Vec<3>& a, const Vec<3>& b) { Vec<3> c; c[0] = a[1] * b[2] - a[2] * b[1]; c[1] = a[2] * b[0] - a[0] * b[2]; c[2] = a[0] * b[1] - a[1] * b[0]; return c; }
inline Vec<3> vec3(Real a, Real b, Real c) { Vec<3> v; v[0] = a; v[1] = b; v[2] = c; return v; }

}  // namespace estkit
