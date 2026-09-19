// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/attitude/triad_quest.hpp — single-frame (memory-less) attitude
//  determination from two vector observations: TRIAD and QUEST
//
//  References
//    * Black, H. D. (1964). A passive system for determining the attitude of a
//      satellite. AIAA Journal 2(7), 1350-1351.                      (TRIAD)
//    * Wahba, G. (1965). A least squares estimate of satellite attitude.
//      SIAM Review 7(3), 409.                                (the loss function)
//    * Shuster, M. D. & Oh, S. D. (1981). Three-axis attitude determination
//      from vector observations. Journal of Guidance and Control 4(1), 70-77.
//                                                     (QUEST, Eq. (63)-(72))
//    * Markley, F. L. & Mortari, D. (2000). Quaternion attitude estimation
//      using vector observations. Journal of the Astronautical Sciences
//      48(2-3), 359-380.               (survey; Davenport's q-method and QUEST)
//    * Keat, J. (1977). Analysis of least-squares attitude determination
//      routine DOAID. Computer Sciences Corp. report CSC/TM-77/6034.
//                                                  (Davenport's K matrix)
//
//  Wahba's problem.  Given m body-frame unit observations b_i of known NAV-frame
//  unit references r_i with weights a_i > 0, find the attitude matrix
//  A = R^T (NAV -> BODY) minimising
//      L(A) = 1/2 sum_i a_i || b_i - A r_i ||^2 = sum_i a_i - g(A),
//      g(A) = sum_i a_i b_i^T A r_i = sum_i a_i r_i^T R b_i.              (1)
//  Substituting the Hamilton quaternion q = [w, v] with
//  R(q) = (w^2-|v|^2) I + 2 v v^T + 2 w [v]_x turns (1) into the quadratic form
//      g(q) = q^T K q,   K = [[ sigma , Z^T ],[ Z , S - sigma I ]],       (2)
//      B = sum_i a_i b_i r_i^T,  sigma = tr B,  S = B + B^T,
//      Z = sum_i a_i ( b_i x r_i ),
//  so the optimal q is the eigenvector of K for its largest eigenvalue
//  lambda_max (Davenport's q-method).  NOTE that (2) is written directly for
//  our BODY -> NAV quaternion; no transposition is needed afterwards.
//
//  QUEST (Shuster & Oh 1981).  lambda_max is obtained from the characteristic
//  polynomial of K, which for a 4x4 of this structure reduces to
//      lambda^4 - (a+b) lambda^2 - c lambda + (a b + c sigma - d) = 0,    (3)
//      a = sigma^2 - kappa,  b = sigma^2 + Z^T Z,
//      c = Delta + Z^T S Z,  d = Z^T S^2 Z,
//      kappa = tr(adj S),  Delta = det S,
//  solved by Newton's method from lambda_0 = sum_i a_i (exact when the
//  observations are consistent, so one or two iterations suffice).  With
//      alpha = lambda^2 - sigma^2 + kappa,  beta = lambda - sigma,
//      gamma = (lambda + sigma) alpha - Delta,
//      X = ( alpha I + beta S + S^2 ) Z = adj((lambda+sigma) I - S) Z,    (4)
//  the eigenvector is q = [gamma, X] / sqrt(gamma^2 + |X|^2).  (4) follows
//  from the lower block row of K q = lambda q, v = ((lambda+sigma)I - S)^{-1} Z w,
//  taking w = det((lambda+sigma)I - S) = gamma.
//
//  TRIAD (Black 1964).  With exactly two observations an exact (unweighted)
//  solution exists: build the orthonormal triads
//      t1 = b1, t2 = (b1 x b2)/|b1 x b2|, t3 = t1 x t2   (body)
//      s1 = r1, s2 = (r1 x r2)/|r1 x r2|, s3 = s1 x s2   (nav)
//  and set R = sum_i s_i t_i^T, so that R b1 = r1 exactly and the second
//  observation only fixes the remaining rotation about b1.  TRIAD therefore
//  trusts one sensor completely; QUEST is its weighted least-squares
//  generalisation (Shuster & Oh 1981, Sec. IV, show the two coincide for two
//  observations in the limit a1/a2 -> infinity).
//
//  Both are used here WITHOUT a gyroscope, as the memory-less baselines of the
//  attitude family: they have no state, no tuning, and no way of rejecting a
//  specific-force disturbance, so they show exactly how much of the AHRS
//  problem is filtering rather than algebra.
// =============================================================================
#pragma once
#include "attitude_estimator.hpp"
#include "attitude_util.hpp"

namespace estkit {

class TriadQuest : public AttitudeEstimator {
public:
    enum class Method { Triad, Quest };
    struct Options {
        Method method = Method::Quest;
        int primary = 0;               // TRIAD: 0 = accelerometer exact, 1 = magnetometer exact
        Real w_acc = Real(-1);         // <0: derive the weight from the configured noise
        Real w_mag = Real(-1);
        int newton_iters = 8;          // Newton iterations on the characteristic equation (3)
    } opts;

    explicit TriadQuest(const char* nm = "QUEST", Method m = Method::Quest) : name_(nm) { opts.method = m; }

    const char* name() const override { return name_; }
    const char* group() const override { return "attitude"; }

    void reset(const AttitudeConfig& cfg) override {
        q_ = cfg.q0.normalized();
        rm_ = att::unit(cfg.mag_ref_nav);
        // Weight each observation by the inverse variance of its DIRECTION
        // error: a vector measurement of magnitude |v| with per-axis noise
        // sigma has a direction error of sigma/|v| radians (Shuster & Oh 1981,
        // Sec. II: a_i = sigma_tot^2/sigma_i^2 with sum_i a_i = 1).
        Real wa = opts.w_acc, wm = opts.w_mag;
        if (wa < Real(0)) wa = Real(1) / sq(std::fmax(cfg.accel_noise, Real(1e-6)) / cfg.g);
        if (wm < Real(0)) wm = Real(1) / sq(std::fmax(cfg.mag_noise, Real(1e-6)) / std::fmax(cfg.mag_ref_nav.norm(), Real(1e-6)));
        const Real s = wa + wm;
        a_acc_ = wa / s;
        a_mag_ = wm / s;
    }

    void step(const Vec<3>& gyro, const Vec<3>& accel, const Vec<3>& mag) override {
        (void)gyro;                                   // single-frame: no gyroscope
        const Real an = accel.norm(), mn = mag.norm();
        if (!(an > Real(1e-6)) || !(mn > Real(1e-6))) return;
        const Vec<3> b1 = accel / an, b2 = mag / mn;
        const Vec<3> r1 = vec3(Real(0), Real(0), Real(-1));   // NED reference of the specific force
        const Vec<3> r2 = rm_;
        const Quat q = (opts.method == Method::Triad)
                           ? triad(b1, r1, b2, r2, opts.primary)
                           : quest(b1, r1, a_acc_, b2, r2, a_mag_, opts.newton_iters, q_);
        if (q.norm() > Real(1e-9)) q_ = q.normalized();
    }

    Quat quaternion() const override { return q_; }
    size_t state_bytes() const override { return sizeof(*this); }

    // ---- TRIAD (Black 1964) -------------------------------------------------
    static Quat triad(const Vec<3>& b1, const Vec<3>& r1, const Vec<3>& b2, const Vec<3>& r2, int primary) {
        const Vec<3>& p_b = (primary == 0) ? b1 : b2;
        const Vec<3>& p_r = (primary == 0) ? r1 : r2;
        const Vec<3>& s_b = (primary == 0) ? b2 : b1;
        const Vec<3>& s_r = (primary == 0) ? r2 : r1;
        const Vec<3> t2 = att::unit(cross(p_b, s_b)), u2 = att::unit(cross(p_r, s_r));
        const Vec<3> t3 = cross(p_b, t2), u3 = cross(p_r, u2);
        // R = sum_i u_i t_i^T maps BODY -> NAV
        const Mat<3, 3> R = outer(p_r, p_b) + outer(u2, t2) + outer(u3, t3);
        return att::quat_from_R(R);
    }

    // ---- QUEST (Shuster & Oh 1981) ------------------------------------------
    static Quat quest(const Vec<3>& b1, const Vec<3>& r1, Real a1,
                      const Vec<3>& b2, const Vec<3>& r2, Real a2, int iters, const Quat& fallback) {
        const Mat<3, 3> B = outer(b1, r1) * a1 + outer(b2, r2) * a2;         // (2)
        const Real sig = B.trace();
        const Mat<3, 3> S = B + B.t();
        const Vec<3> Z = cross(b1, r1) * a1 + cross(b2, r2) * a2;
        const Mat<3, 3> S2 = S * S;
        const Real kappa = Real(0.5) * (sq(S.trace()) - S2.trace());          // tr(adj S)
        const Real Delta = det3(S);
        const Real aa = sq(sig) - kappa;
        const Real bb = sq(sig) + dot(Z, Z);
        const Real cc = Delta + dot(Z, S * Z);
        const Real dd = dot(Z, S2 * Z);
        const Real con = aa * bb + cc * sig - dd;
        Real lam = a1 + a2;                                                   // (3), Newton
        for (int k = 0; k < iters; ++k) {
            const Real l2 = lam * lam;
            const Real f = l2 * l2 - (aa + bb) * l2 - cc * lam + con;
            const Real fp = Real(4) * l2 * lam - Real(2) * (aa + bb) * lam - cc;
            if (!(std::fabs(fp) > Real(1e-12))) break;
            const Real dl = f / fp;
            lam -= dl;
            if (std::fabs(dl) < Real(1e-12)) break;
        }
        const Real alpha = lam * lam - sq(sig) + kappa;                       // (4)
        const Real beta = lam - sig;
        const Real gamma = (lam + sig) * alpha - Delta;
        const Vec<3> X = (Mat<3, 3>::identity() * alpha + S * beta + S2) * Z;
        Quat q(gamma, X[0], X[1], X[2]);
        const Real n = q.norm();
        if (std::isfinite(n) && n > Real(1e-9)) return q.normalized();
        // Degenerate closed form (adj((lambda+sigma)I-S) Z = 0 and gamma = 0):
        // two steps of inverse iteration on Davenport's K, seeded with TRIAD.
        Mat<4, 4> K;
        K(0, 0) = sig;
        for (int i = 0; i < 3; ++i) { K(0, i + 1) = Z[i]; K(i + 1, 0) = Z[i]; }
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) K(i + 1, j + 1) = S(i, j) - (i == j ? sig : Real(0));
        const Real shift = lam + Real(1e-6) * (Real(1) + std::fabs(lam));
        for (int i = 0; i < 4; ++i) K(i, i) -= shift;
        Vec<4> x;
        x[0] = fallback.w; x[1] = fallback.x; x[2] = fallback.y; x[3] = fallback.z;
        for (int it = 0; it < 2; ++it) {
            const Vec<4> y = solve(K, x);
            if (!y.is_finite() || !(y.norm() > Real(1e-300))) return fallback;
            x = y / y.norm();
        }
        return Quat(x[0], x[1], x[2], x[3]).normalized();
    }

private:
    static Real det3(const Mat<3, 3>& m) {
        return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
             - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
             + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    }

    const char* name_;
    Quat q_;
    Vec<3> rm_;
    Real a_acc_ = Real(0.8), a_mag_ = Real(0.2);
};

}  // namespace estkit
