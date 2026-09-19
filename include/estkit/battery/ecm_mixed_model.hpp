// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/battery/ecm_mixed_model.hpp — the 2-RC equivalent-circuit cell model
//  written in the CONDITIONALLY LINEAR ("mixed linear/nonlinear") form required
//  by the marginalised (Rao-Blackwellised) particle filter, estkit/particle/rbpf.hpp.
//
//  References
//    * Schon, Thomas, Gustafsson, Fredrik & Nordlund, Per-Johan (2005).
//      Marginalized particle filters for mixed linear/nonlinear state-space
//      models. IEEE Transactions on Signal Processing 53(7), 2279-2289.
//    * Plett, Gregory L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs — Part 2/3. J. Power Sources 134,
//      262-276 / 277-292.                                   (the ESC cell model)
//
//  ---------------------------------------------------------------------------
//  The mixed-model concept (what rbpf.hpp requires of its template argument)
//  ---------------------------------------------------------------------------
//    static constexpr int NXN;   // nonlinear states  x_n  (sampled by the PF)
//    static constexpr int NXL;   // linear states     x_l  (Kalman filtered)
//    static constexpr int NX = NXN + NXL;   // full state, ordered [x_n ; x_l]
//    static constexpr int NU, NY;
//
//    x_n(k+1) = f_n(x_n(k), u(k)) + w_n(k),            w_n ~ N(0, Q_n(x_n,u))
//    x_l(k+1) = A_l(u(k)) x_l(k) + b_l(u(k)) + w_l(k), w_l ~ N(0, Q_l(u))
//    y(k)     = h_n(x_n(k),u(k)) + C_l(x_n(k),u(k)) x_l(k) + v(k),  v ~ N(0,R(u))
//
//    Vec<NXN>     f_n(const Vec<NXN>&, const Vec<NU>&) const;
//    Mat<NXN,NXN> Q_n(const Vec<NXN>&, const Vec<NU>&) const;
//    Vec<NXN>     constrain_n(Vec<NXN>) const;            // optional projection
//    Mat<NXL,NXL> A_l(const Vec<NU>&) const;
//    Vec<NXL>     b_l(const Vec<NU>&) const;
//    Mat<NXL,NXL> Q_l(const Vec<NU>&) const;
//    Vec<NY>      h_n(const Vec<NXN>&, const Vec<NU>&) const;
//    Mat<NY,NXL>  C_l(const Vec<NXN>&, const Vec<NU>&) const;
//    Mat<NY,NY>   R(const Vec<NU>&) const;
//
//  The two structural requirements of Schon et al. that make the marginalisation
//  exact are (i) the nonlinear dynamics must NOT depend on the linear states
//  (there is no A_n x_l term) and (ii) the measurement must be affine in x_l.
//  Both hold for the ESC battery model:
//
//     x_n = [ z , h ]^T      SOC and the one-state hysteresis level
//     x_l = [ v1, v2 ]^T     the two RC-branch (diffusion) voltages
//
//     z(k+1) = z(k) - eta(i) dt i /(3600 Q)                   (no v1, v2)
//     h(k+1) = a_h h(k) + (a_h - 1) sgn(i)                    (no v1, v2)
//     v1(k+1) = a1 v1(k) + R1 (1-a1) i                        (linear, given u)
//     v2(k+1) = a2 v2(k) + R2 (1-a2) i                        (linear, given u)
//     y      = OCV(z,T) + M h - R0(T) i  +  [-1  -1] [v1 v2]^T
//              \______________________/     \_____________/
//                     h_n(x_n,u)              C_l x_l
//
//  so the RC voltages — the states that are genuinely linear-Gaussian — can be
//  integrated out analytically and only the 2-dimensional (z,h) sub-space is
//  explored by particles (Rao-Blackwellisation).
//
//  Process noise.  The physical driver of the process noise is the current-sensor
//  error, which enters every state through B(:,i); its exact covariance therefore
//  couples the nonlinear and the linear block, Q_nl = b_n b_l^T sigma_i^2.  Schon
//  et al. §III-B treat that cross-term (it adds a gain from the realised
//  nonlinear transition into the linear mean).  Here Q_nl is dropped:
//  |b_z| sigma_i = 2.8e-7 and |b_h| sigma_i < 1e-5 per step are two to three
//  orders of magnitude below the diagonal floors of the RC states, so the
//  neglected correlation is numerically irrelevant (see the RBPF chapter).
//
//  State ORDER.  The full state exposed to the benchmark is [z, h, v1, v2] —
//  note that this differs from EcmModel's [z, v1, v2, h].  Only index IZ = 0 is
//  used by the cell adapter, and it agrees in both orderings.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "cell.hpp"
#include "ecm_model.hpp"

namespace estkit {

struct EcmMixedModel {
    static constexpr int NXN = 2, NXL = 2;
    static constexpr int NX = NXN + NXL, NU = 2, NY = 1;
    static constexpr int IZ = 0, IH = 1, IV1 = 2, IV2 = 3;   // full-state indices
    static constexpr int UI = 0, UT = 1;                     // input indices

    CellParams p;
    Real dt = Real(0.1);
    Real sigma_i = Real(0.05), sigma_v = Real(0.002);
    // std-dev floors of the process noise, in the [z, h, v1, v2] order
    Real q_floor[NX] = {Real(1e-5), Real(1e-3), Real(1e-4), Real(1e-4)};
    OcvTable<241> ocv;

    EcmMixedModel() = default;
    explicit EcmMixedModel(const EstimatorConfig& c) : p(c.params), dt(c.dt), sigma_i(c.sigma_i), sigma_v(c.sigma_v) {}

    // --- temperature/current-scheduled coefficients (identical to EcmModel) ---
    Real R0(Real T) const { return p.R0_T(T); }
    Real a1(Real T) const { return std::exp(-dt / p.tau1_T(T)); }
    Real a2(Real T) const { return std::exp(-dt / p.tau2_T(T)); }
    Real eta(Real i) const { return i >= Real(0) ? Real(1) : p.eta_c; }
    Real coulomb_gain(Real i) const { return -eta(i) * dt / (Real(3600) * p.Q_Ah); }
    Real ah(Real i) const { return std::exp(-std::fabs(eta(i) * i * p.gamma * dt / (Real(3600) * p.Q_Ah))); }

    // --- nonlinear block  x_n = [z, h] ---------------------------------------
    Vec<NXN> f_n(const Vec<NXN>& xn, const Vec<NU>& u) const {
        const Real i = u[UI];
        const Real AH = ah(i);
        Vec<NXN> xp;
        xp[0] = xn[0] + coulomb_gain(i) * i;
        xp[1] = AH * xn[1] + (AH - Real(1)) * sgn(i);
        return xp;
    }
    Mat<NXN, NXN> Q_n(const Vec<NXN>& xn, const Vec<NU>& u) const {
        const Real i = u[UI];
        Vec<NXN> b;
        b[0] = coulomb_gain(i);
        const Real dah_di = -sgn(i) * eta(i) * p.gamma * dt / (Real(3600) * p.Q_Ah) * ah(i);
        b[1] = dah_di * (xn[1] + sgn(i));
        Mat<NXN, NXN> Qm = outer(b, b) * sq(sigma_i);
        Qm(0, 0) += sq(q_floor[0]);
        Qm(1, 1) += sq(q_floor[1]);
        return Qm;
    }
    Vec<NXN> constrain_n(Vec<NXN> xn) const {
        xn[0] = clampr(xn[0], Real(-0.05), Real(1.05));
        xn[1] = clampr(xn[1], Real(-1), Real(1));
        return xn;
    }

    // --- linear block  x_l = [v1, v2] ----------------------------------------
    Mat<NXL, NXL> A_l(const Vec<NU>& u) const {
        Mat<NXL, NXL> A; A(0, 0) = a1(u[UT]); A(1, 1) = a2(u[UT]); return A;
    }
    Vec<NXL> b_l(const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        Vec<NXL> b;
        b[0] = p.R1_T(T) * (Real(1) - a1(T)) * i;
        b[1] = p.R2_T(T) * (Real(1) - a2(T)) * i;
        return b;
    }
    Mat<NXL, NXL> Q_l(const Vec<NU>& u) const {
        const Real T = u[UT];
        Vec<NXL> b;
        b[0] = p.R1_T(T) * (Real(1) - a1(T));
        b[1] = p.R2_T(T) * (Real(1) - a2(T));
        Mat<NXL, NXL> Qm = outer(b, b) * sq(sigma_i);
        Qm(0, 0) += sq(q_floor[2]);
        Qm(1, 1) += sq(q_floor[3]);
        return Qm;
    }

    // --- measurement  y = h_n(x_n,u) + C_l x_l + v ---------------------------
    Vec<NY> h_n(const Vec<NXN>& xn, const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        Vec<NY> y; y[0] = ocv.ocv(xn[0], T) + p.M * xn[1] - R0(T) * i;
        return y;
    }
    Mat<NY, NXL> C_l(const Vec<NXN>& /*xn*/, const Vec<NU>& /*u*/) const {
        Mat<NY, NXL> C; C(0, 0) = Real(-1); C(0, 1) = Real(-1); return C;
    }
    Mat<NY, NY> R(const Vec<NU>& u) const {
        Mat<NY, NY> Rm; Rm(0, 0) = sq(sigma_v) + sq(R0(u[UT]) * sigma_i); return Rm;
    }

    // --- benchmark glue (same contract as EcmModel) ---------------------------
    static Vec<NU> u_of(Real i, Real T) { Vec<NU> u; u[UI] = i; u[UT] = T; return u; }
    Vec<NX> x0(const EstimatorConfig& c) const { Vec<NX> x; x[IZ] = c.z0; return x; }
    Mat<NX, NX> P0(const EstimatorConfig& c) const {
        Mat<NX, NX> P;
        P(IZ, IZ) = sq(c.sigma_z0); P(IH, IH) = sq(Real(0.5));
        P(IV1, IV1) = sq(Real(0.002)); P(IV2, IV2) = sq(Real(0.005));
        return P;
    }
};

}  // namespace estkit
