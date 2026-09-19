// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/battery/ecm_model_reduced.hpp — REDUCED-ORDER estimator model for the
//  2-RC equivalent circuit: the slow RC branch is removed from the state vector
//  and reconstructed quasi-statically (open loop) from the current.
//
//  References
//    * Simon, D. (2006). Optimal State Estimation: Kalman, H-infinity, and
//      Nonlinear Approaches. Wiley, Ch. 10 (reduced-order Kalman filtering;
//      the neglected dynamics are accounted for as an additional noise source).
//    * Plett, G. L. (2016). Battery Management Systems, Volume II:
//      Equivalent-Circuit Methods. Artech House, §3 (model order selection and
//      reduction for the ESC model).
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs - Part 2/3. J. Power Sources 134,
//      262-292.                                     (the full 2-RC "ESC" model)
//    * Chen, C. H. et al. (2020). Development of experimental techniques for
//      parameterization of multi-scale lithium-ion battery models.
//      J. Electrochem. Soc. 167, 080534.                     (OCV / chemistry)
//
//  Model ------------------------------------------------------------------
//  State  x = [ z, v1, h ]^T   (n_x = 3 instead of 4)
//  Input  u = [ i, T ]^T,  output y = v.
//
//     z+  = z  - eta(i) dt/(3600 Q) i
//     v1+ = a1 v1 + R1(T)(1-a1) i,        a1 = exp(-dt/tau1(T))
//     h+  = ah h + (ah - 1) sgn(i),       ah = exp(-|eta(i) i gamma dt/(3600 Q)|)
//     y   = OCV(z,T) + M h - v1 - v2_qs - R0(T) i
//
//  The slow branch is NOT a state: it is a deterministic functional of the input
//  history, obtained by passing R2(T) i through its own first-order low pass
//
//     v2_qs+ = a2 v2_qs + R2(T)(1-a2) i,  a2 = exp(-dt/tau2(T)),                (*)
//
//  which is exactly the quasi-static (input-driven) part of the true v2 dynamics
//  with the feedback path to the estimator removed.  Because tau2 ~ 120 s is much
//  longer than the correlation time of the SOC information in the measurement,
//  (*) is essentially uncorrectable from a single terminal-voltage channel, so
//  dropping it from the state costs little accuracy but removes one row/column
//  from every covariance operation.
//
//  Following Simon (2006, Ch. 10), the reconstruction error of the neglected
//  branch is accounted for as an extra, uncorrelated measurement-noise term:
//      R = sigma_v^2 + (R0 sigma_i)^2 + sigma_v2^2
//  with sigma_v2 the assumed standard deviation of  v2_true - v2_qs  (it is zero
//  when the parameters are exact and grows with the R2/tau2 mismatch).
//
//  IMPORTANT ---------------------------------------------------------------
//  v2_qs is an internal, non-estimated state carried inside the model object and
//  advanced by f().  The model must therefore be used with a filter that
//  evaluates f() exactly once per sampling period and supplies analytic
//  Jacobians (the EKF family; F, H and B below are analytic so the numeric
//  fall-backs in core/model.hpp are never used).  Sigma-point or particle
//  filters, which evaluate f() many times per step, would corrupt (*).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "cell.hpp"
#include "ecm_model.hpp"

namespace estkit {

struct EcmModelReduced {
    static constexpr int NX = 3, NU = 2, NY = 1, NP = 2;
    static constexpr int IZ = 0, IV1 = 1, IH = 2;   // state indices (no v2)
    static constexpr int UI = 0, UT = 1;            // input indices
    CellParams p;
    Real dt = Real(0.1);
    Real sigma_i = Real(0.05), sigma_v = Real(0.002);
    Real sigma_v2 = Real(0.002);                    // assumed error of the quasi-static slow branch [V]
    Real q_floor[NX] = {Real(1e-5), Real(1e-4), Real(1e-3)};
    OcvTable<241> ocv;
    mutable Real v2_ = Real(0);                     // quasi-static slow-branch voltage (not estimated)

    EcmModelReduced() = default;
    explicit EcmModelReduced(const EstimatorConfig& c)
        : p(c.params), dt(c.dt), sigma_i(c.sigma_i), sigma_v(c.sigma_v) {}

    // --- temperature-scheduled coefficients (identical to EcmModel) ---
    Real R0(Real T) const { return p.R0_T(T); }
    Real a1(Real T) const { return std::exp(-dt / p.tau1_T(T)); }
    Real a2(Real T) const { return std::exp(-dt / p.tau2_T(T)); }
    Real eta(Real i) const { return i >= Real(0) ? Real(1) : p.eta_c; }
    Real coulomb_gain(Real i) const { return -eta(i) * dt / (Real(3600) * p.Q_Ah); }
    Real ah(Real i) const { return std::exp(-std::fabs(eta(i) * i * p.gamma * dt / (Real(3600) * p.Q_Ah))); }
    Real v2() const { return v2_; }
    void set_v2(Real v) const { v2_ = v; }

    // --- generic model interface ---
    Vec<NX> f(const Vec<NX>& x, const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        const Real A1 = a1(T), A2 = a2(T), AH = ah(i);
        Vec<NX> xn;
        xn[IZ]  = x[IZ] + coulomb_gain(i) * i;
        xn[IV1] = A1 * x[IV1] + p.R1_T(T) * (Real(1) - A1) * i;
        xn[IH]  = AH * x[IH] + (AH - Real(1)) * sgn(i);
        v2_ = A2 * v2_ + p.R2_T(T) * (Real(1) - A2) * i;      // (*) quasi-static slow branch
        return xn;
    }
    Mat<NX, NX> F(const Vec<NX>& /*x*/, const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        Mat<NX, NX> Fm;
        Fm(IZ, IZ) = Real(1); Fm(IV1, IV1) = a1(T); Fm(IH, IH) = ah(i);
        return Fm;
    }
    Mat<NX, NU> B(const Vec<NX>& x, const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        Mat<NX, NU> Bm;
        Bm(IZ, UI)  = coulomb_gain(i);
        Bm(IV1, UI) = p.R1_T(T) * (Real(1) - a1(T));
        const Real dah_di = -sgn(i) * eta(i) * p.gamma * dt / (Real(3600) * p.Q_Ah) * ah(i);
        Bm(IH, UI)  = dah_di * (x[IH] + sgn(i));
        return Bm;
    }
    Vec<NY> h(const Vec<NX>& x, const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        Vec<NY> y; y[0] = ocv.ocv(x[IZ], T) + p.M * x[IH] - x[IV1] - v2_ - R0(T) * i;
        return y;
    }
    Mat<NY, NX> H(const Vec<NX>& x, const Vec<NU>& u) const {
        Mat<NY, NX> Hm;
        Hm(0, IZ) = ocv.docv_dz(x[IZ], u[UT]); Hm(0, IV1) = Real(-1); Hm(0, IH) = p.M;
        return Hm;
    }
    Real dh_di(Real T) const { return -R0(T); }
    Mat<NX, NX> Q(const Vec<NX>& x, const Vec<NU>& u) const {
        const Vec<NX> b = B(x, u).col(UI);
        Mat<NX, NX> Qm = outer(b, b) * sq(sigma_i);
        for (int k = 0; k < NX; ++k) Qm(k, k) += sq(q_floor[k]);
        return Qm;
    }
    // Measurement noise + the reduction penalty (Simon 2006, Ch. 10)
    Mat<NY, NY> R(const Vec<NX>& /*x*/, const Vec<NU>& u) const {
        Mat<NY, NY> Rm;
        Rm(0, 0) = sq(sigma_v) + sq(R0(u[UT]) * sigma_i) + sq(sigma_v2);
        return Rm;
    }
    Vec<NX> constrain(Vec<NX> x) const {
        x[IZ] = clampr(x[IZ], Real(-0.05), Real(1.05));
        x[IH] = clampr(x[IH], Real(-1), Real(1));
        return x;
    }
    // --- tunable parameters theta = [Q_Ah, R0] ---
    Vec<NP> params() const { Vec<NP> th; th[0] = p.Q_Ah; th[1] = p.R0; return th; }
    void set_params(const Vec<NP>& th) { p.Q_Ah = std::max(th[0], Real(0.5)); p.R0 = std::max(th[1], Real(1e-4)); }

    // --- initial condition helpers ---
    static Vec<NU> u_of(Real i, Real T) { Vec<NU> u; u[UI] = i; u[UT] = T; return u; }
    Vec<NX> x0(const EstimatorConfig& c) const { Vec<NX> x; x[IZ] = c.z0; return x; }
    Mat<NX, NX> P0(const EstimatorConfig& c) const {
        Mat<NX, NX> P;
        P(IZ, IZ) = sq(c.sigma_z0); P(IV1, IV1) = sq(Real(0.002)); P(IH, IH) = sq(Real(0.5));
        return P;
    }
};

}  // namespace estkit
