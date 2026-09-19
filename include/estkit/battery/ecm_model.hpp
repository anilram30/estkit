// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/battery/ecm_model.hpp — the estimator-side discrete-time 2-RC
//  equivalent-circuit model ("ESC" model, Plett 2004 / Plett 2015 Vol. I Ch. 2)
//  written against the generic estkit model concept (core/model.hpp).
//
//  State  x = [ z, v1, v2, h ]^T
//     z   state of charge (-),  v1, v2 RC-branch voltages (V), h in [-1,1] hysteresis
//  Input  u = [ i, T ]^T   current (A, discharge positive), cell temperature (K)
//  Output y = v = OCV(z,T) + M h - v1 - v2 - R0(T) i
//
//  Discrete dynamics (sample time dt, ZOH on the current):
//     z+  = z  - eta(i) dt/(3600 Q) i
//     v1+ = a1 v1 + R1(T) (1 - a1) i,    a1 = exp(-dt/tau1(T))
//     v2+ = a2 v2 + R2(T) (1 - a2) i,    a2 = exp(-dt/tau2(T))
//     h+  = ah h + (ah - 1) sgn(i),      ah = exp(-|eta(i) i gamma dt/(3600 Q)|)
//  Tunable parameters (for joint/dual estimation): theta = [Q_Ah, R0]
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "cell.hpp"

namespace estkit {

// What the benchmark tells an estimator at reset().
struct EstimatorConfig {
    CellParams params;          // ECM parameters the estimator is allowed to use (may be mismatched)
    Real dt = Real(0.1);        // sample time [s]
    Real z0 = Real(0.9);        // initial SOC guess
    Real sigma_z0 = Real(0.1);  // assumed initial SOC standard deviation
    Real sigma_v = Real(0.002); // assumed voltage-sensor noise std [V]
    Real sigma_i = Real(0.05);  // assumed current-sensor noise std [A]
    Real sigma_T = Real(0.2);   // assumed temperature-sensor noise std [K]
    Real T0 = kTref;            // initial temperature [K]
    uint64_t seed = 1;          // seed for stochastic estimators
};

struct EcmModel {
    static constexpr int NX = 4, NU = 2, NY = 1, NP = 2;
    static constexpr int IZ = 0, IV1 = 1, IV2 = 2, IH = 3;   // state indices
    static constexpr int UI = 0, UT = 1;                     // input indices
    CellParams p;
    Real dt = Real(0.1);
    Real sigma_i = Real(0.05), sigma_v = Real(0.002);
    Real q_floor[NX] = {Real(1e-5), Real(1e-4), Real(1e-4), Real(1e-3)};   // std-dev floors of the process noise
    OcvTable<241> ocv;

    EcmModel() = default;
    explicit EcmModel(const EstimatorConfig& c) : p(c.params), dt(c.dt), sigma_i(c.sigma_i), sigma_v(c.sigma_v) {}

    // --- temperature-scheduled coefficients ---
    Real R0(Real T) const { return p.R0_T(T); }
    Real a1(Real T) const { return std::exp(-dt / p.tau1_T(T)); }
    Real a2(Real T) const { return std::exp(-dt / p.tau2_T(T)); }
    Real eta(Real i) const { return i >= Real(0) ? Real(1) : p.eta_c; }
    Real coulomb_gain(Real i) const { return -eta(i) * dt / (Real(3600) * p.Q_Ah); }   // dz/di per step
    Real ah(Real i) const { return std::exp(-std::fabs(eta(i) * i * p.gamma * dt / (Real(3600) * p.Q_Ah))); }

    // --- generic model interface ---
    Vec<NX> f(const Vec<NX>& x, const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        const Real A1 = a1(T), A2 = a2(T), AH = ah(i);
        Vec<NX> xn;
        xn[IZ]  = x[IZ] + coulomb_gain(i) * i;
        xn[IV1] = A1 * x[IV1] + p.R1_T(T) * (Real(1) - A1) * i;
        xn[IV2] = A2 * x[IV2] + p.R2_T(T) * (Real(1) - A2) * i;
        xn[IH]  = AH * x[IH] + (AH - Real(1)) * sgn(i);
        return xn;
    }
    Mat<NX, NX> F(const Vec<NX>& /*x*/, const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        Mat<NX, NX> Fm;
        Fm(IZ, IZ) = Real(1); Fm(IV1, IV1) = a1(T); Fm(IV2, IV2) = a2(T); Fm(IH, IH) = ah(i);
        return Fm;
    }
    // input Jacobian df/du (column 0: current, column 1: temperature ~ 0)
    Mat<NX, NU> B(const Vec<NX>& x, const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        Mat<NX, NU> Bm;
        Bm(IZ, UI)  = coulomb_gain(i);
        Bm(IV1, UI) = p.R1_T(T) * (Real(1) - a1(T));
        Bm(IV2, UI) = p.R2_T(T) * (Real(1) - a2(T));
        // d/di [ah h + (ah-1) sgn(i)] = dah/di (h + sgn(i)),  dah/di = -sgn(i) eta gamma dt/(3600 Q) ah
        const Real dah_di = -sgn(i) * eta(i) * p.gamma * dt / (Real(3600) * p.Q_Ah) * ah(i);
        Bm(IH, UI)  = dah_di * (x[IH] + sgn(i));
        return Bm;
    }
    Vec<NY> h(const Vec<NX>& x, const Vec<NU>& u) const {
        const Real i = u[UI], T = u[UT];
        Vec<NY> y; y[0] = ocv.ocv(x[IZ], T) + p.M * x[IH] - x[IV1] - x[IV2] - R0(T) * i;
        return y;
    }
    Mat<NY, NX> H(const Vec<NX>& x, const Vec<NU>& u) const {
        Mat<NY, NX> Hm;
        Hm(0, IZ) = ocv.docv_dz(x[IZ], u[UT]); Hm(0, IV1) = Real(-1); Hm(0, IV2) = Real(-1); Hm(0, IH) = p.M;
        return Hm;
    }
    Real dh_di(Real T) const { return -R0(T); }   // feed-through term D
    // process noise: current-sensor noise mapped through B plus diagonal floor
    Mat<NX, NX> Q(const Vec<NX>& x, const Vec<NU>& u) const {
        const Vec<NX> b = B(x, u).col(UI);
        Mat<NX, NX> Qm = outer(b, b) * sq(sigma_i);
        for (int k = 0; k < NX; ++k) Qm(k, k) += sq(q_floor[k]);
        return Qm;
    }
    Mat<NY, NY> R(const Vec<NX>& /*x*/, const Vec<NU>& u) const {
        Mat<NY, NY> Rm; Rm(0, 0) = sq(sigma_v) + sq(R0(u[UT]) * sigma_i); return Rm;
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
        Mat<NX, NX> P; P(IZ, IZ) = sq(c.sigma_z0); P(IV1, IV1) = sq(Real(0.002)); P(IV2, IV2) = sq(Real(0.005)); P(IH, IH) = sq(Real(0.5));
        return P;
    }
};

// -----------------------------------------------------------------------------
//  Benchmark interface for cell-level SOC estimators (recursive, one sample at a time).
//  Timing convention (Plett 2004 Part 3):  step(i_k, v_k, T_k) performs
//     time update   x_k^-  = f(x_{k-1}^+, u_{k-1})   (skipped at k = 0)
//     meas. update  x_k^+  from  v_k with  y_k^- = h(x_k^-, u_k)
// -----------------------------------------------------------------------------
class Estimator {
public:
    virtual ~Estimator() = default;
    virtual const char* name() const = 0;
    virtual const char* group() const = 0;
    virtual void reset(const EstimatorConfig& cfg) = 0;
    virtual void step(Real i, Real v, Real T) = 0;
    virtual Real soc() const = 0;
    virtual Real soc_std() const { return Real(-1); }          // -1 = not available
    virtual Real capacity_Ah() const { return kNaN; }            // SOH-capable estimators override
    virtual Real voltage_pred() const { return kNaN; }           // predicted terminal voltage y_k^- (prior)
    virtual size_t state_bytes() const = 0;                      // memory footprint of the estimator object
    virtual bool is_batch() const { return false; }              // offline (smoother) methods
    virtual Real soc_lower() const { return kNaN; }              // interval / set-membership observers
    virtual Real soc_upper() const { return kNaN; }
};

// Offline / batch estimators (smoothers, full-information NLS): given the full
// record they return the SOC trajectory.
class BatchEstimator : public Estimator {
public:
    bool is_batch() const override { return true; }
    // process the whole record; soc_out / v_pred_out must be filled with n entries
    virtual void run(const Real* i, const Real* v, const Real* T, int n, Real* soc_out, Real* v_pred_out) = 0;
    void step(Real, Real, Real) override {}
    Real soc() const override { return kNaN; }
};

}  // namespace estkit
