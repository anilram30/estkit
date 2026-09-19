// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/eso.hpp — extended state observer (ESO) of active disturbance
//  rejection control, with Gao's bandwidth parameterisation.
//
//  References
//    * Han, J. (2009). From PID to active disturbance rejection control. IEEE
//      Transactions on Industrial Electronics 56(3), 900-906.
//    * Gao, Z. (2003). Scaling and bandwidth-parameterization based controller
//      tuning. Proc. American Control Conference, vol. 6, 4989-4996.
//    * Zheng, Q., Gao, L. Q. & Gao, Z. (2007). On stability analysis of active
//      disturbance rejection control for nonlinear time-varying plants with
//      unknown dynamics. Proc. 46th IEEE Conference on Decision and Control, 3501-3506.
//    * Luenberger, D. G. (1971). An introduction to observers. IEEE TAC 16(6), 596-602.
//
//  Idea.  Everything that is not in the model -- unmodelled dynamics, parameter
//  drift, an offset on the integrated input -- is lumped into one extra state,
//  the "total disturbance" d, appended to the channel that cannot correct itself:
//
//      x_{k+1} = f(x_k, u_k) + g d_k ,   d_{k+1} = d_k ,   y_k = h(x_k, u_k).   (1)
//
//  g in R^{NX} defaults to the unit vector e_{j*} with
//  j* = argmax_j ||(I - A)^{-1} e_j||_1 (obs::slowest_state), i.e. the state with
//  the largest DC gain from a constant disturbance.  For the cell model
//  A = diag(1, a1, a2, ah) and j* is the SOC integrator, so d is exactly a
//  lumped SOC-rate error: current-sensor bias divided by the capacity, coulombic
//  inefficiency, and capacity fade all appear there.
//
//  Observer (current-estimator form)
//      x^-_k = f(x^+_{k-1}, u_{k-1}) + g d^+_{k-1} ,   d^-_k = d^+_{k-1}
//      r_k   = y_k - h(x^-_k, u_k)
//      x^+_k = x^-_k + L_x r_k ,     d^+_k = d^-_k + l_d r_k                    (2)
//
//  Bandwidth parameterisation (Gao 2003).  Instead of tuning NX+1 gains, ALL
//  observer poles are placed at one point,
//      z_i = exp(-omega_o dt) ,  i = 1..NX+1 ,                                  (3)
//  leaving a single knob omega_o (the observer bandwidth in rad/s).  The gain is
//  then obtained from Ackermann's formula applied to the augmented pair
//      Abar = [[A, g],[0, 1]] ,   Cbar = [C, 0] ,                               (4)
//  and converted to the current-estimator form by L = Abar^{-1} Lbar.  Placing
//  all poles at a single location is also what keeps the design numerically
//  sound on this plant: the cell model has three eigenvalues within 2 % of each
//  other, and a *spread* pole set would demand gains of order 10^2-10^3 to pull
//  them apart (see the Luenberger chapter).  The correction of the non-augmented
//  states is exactly the Luenberger correction implied by the same placement.
//
//  Convergence.  For the augmented linear time-varying pair, (2)-(4) give
//      eta_k = (I - Lbar Cbar) Abar eta_{k-1} + noise ,
//  with eta = [xhat - x ; dhat - d], so the estimation error converges
//  geometrically at the rate exp(-omega_o dt) provided the augmented pair is
//  observable, which holds iff (A, C) is observable and
//      rank [ A - I   g ; C   0 ] = NX + 1 .                                    (5)
//  If d is not constant but has a bounded rate |d_dot| <= M, the error is
//  ultimately bounded by O(M / omega_o) (Zheng, Gao & Gao 2007).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "luenberger.hpp"

namespace estkit {

template <class Model>
class ExtendedStateObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NA = NX + 1;
    static_assert(NY == 1, "ExtendedStateObserver uses Ackermann pole placement (NY == 1)");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        // z-plane location of all NX+1 observer poles: pole_o = exp(-omega_o dt).
        // Default 0.998 = exp(-0.02 * 0.1): omega_o = 0.02 rad/s at dt = 0.1 s.
        Real pole_o = Real(0.998);
        Real g[NX];               // disturbance direction; all zero => auto (slowest state)
        Real g_scale = Real(1);   // magnitude of the disturbance input (units of d)
        Real d_max = Real(1e30);  // anti-windup bound on the disturbance state
        Real q_d = Real(1e-6);    // Riccati fallback: random-walk std of d
        Real q_scale = Real(1), r_scale = Real(1);
        Real regain_tol = Real(1e-3);
        int  riccati_iters_init = 30000, riccati_iters_refine = 40;
        bool riccati_exact = true;         // exact DARE (doubling) instead of the truncated recursion
        // Certified decay rate: a design is accepted only if rho <= 1 - rho_margin
        // over the whole scheduling family (see obs::gain_is_robust).
        Real rho_margin = Real(1e-4);
        Real gain_smooth = Real(1);        // exponential blending of a re-designed gain (1 = replace)
        Real robust_lo = Real(0.8), robust_hi = Real(1.25);  // scheduling-uncertainty range for the gain check
        Real max_gain = Real(1e4);
        bool fallback_to_riccati = true;
        bool apply_constraints = true;

        Options() { for (int i = 0; i < NX; ++i) g[i] = Real(0); }
    } opts;

    explicit ExtendedStateObserver(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; d_ = Real(0); have_design_ = false; riccati_init_ = false; La_ = Vec<NA>();
        Lx_ = X(); ld_ = Real(0); residual_ = Y();
        Pa_ = Mat<NA, NA>();
        Pa_.template set_block<NX, NX>(0, 0, P0);
        Pa_(NX, NX) = sq(Real(1e-3));
    }

    void predict(const U& u) { x_ = m_.f(x_, u) + g_ * d_; }
    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NX, NX> A = jac_F(m_, x_, u);
        const Mat<NY, NX> C = jac_H(m_, x_, u);
        schedule_gain(A, C, u);
        const Y r = y - m_.h(x_, u);
        residual_ = r;
        const X dx = Lx_ * r[0];
        const Real dd = ld_ * r[0];
        if (dx.is_finite() && std::isfinite(dd)) { x_ = x_ + dx; d_ = clampr(d_ + dd, -opts.d_max, opts.d_max); }
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    Real disturbance() const { return d_; }       // estimated total disturbance (per step, in g-units)
    const X& gain() const { return Lx_; }
    Real disturbance_gain() const { return ld_; }
    const Y& residual() const { return residual_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void build_g(const Mat<NX, NX>& A) {
        Real user = Real(0);
        for (int i = 0; i < NX; ++i) user += std::fabs(opts.g[i]);
        g_ = X();
        if (user > Real(0)) { for (int i = 0; i < NX; ++i) g_[i] = opts.g[i] * opts.g_scale; }
        else                { g_[obs::slowest_state<NX>(A)] = opts.g_scale; }
    }

    void schedule_gain(const Mat<NX, NX>& A, const Mat<NY, NX>& C, const U& u) {
        if (have_design_ && obs::lin_change<NX, NY>(A_ref_, C_ref_, A, C) < opts.regain_tol) return;
        A_ref_ = A; C_ref_ = C;
        build_g(A);
        Mat<NA, NA> Aa = Mat<NA, NA>::identity();
        Aa.template set_block<NX, NX>(0, 0, A);
        for (int i = 0; i < NX; ++i) Aa(i, NX) = g_[i];
        Mat<NY, NA> Ca;
        Ca.template set_block<NY, NX>(0, 0, C);
        Vec<NA> L;
        Real p[NA];
        for (int i = 0; i < NA; ++i) p[i] = opts.pole_o;
        bool ok = obs::place_gain<NA>(Aa, Ca, p, L);
        if (ok && L.norm_inf() > opts.max_gain) ok = false;
        // Certify the bandwidth-parameterised assignment before using it.
        const Real rho_max = Real(1) - opts.rho_margin;
        if (ok && !obs::gain_is_robust<NA, NY>(Aa, Ca, L, opts.robust_lo, opts.robust_hi, rho_max)) ok = false;
        if (!ok && opts.fallback_to_riccati) {
            Mat<NA, NA> Qa;
            Qa.template set_block<NX, NX>(0, 0, m_.Q(x_, u) * opts.q_scale);
            Qa(NX, NX) = sq(opts.q_d);
            const Mat<NY, NY> Ra = m_.R(x_, u) * opts.r_scale;
            if (!riccati_init_) { Pa_ = obs::dare_design<NA, NY>(Aa, Ca, Qa, Ra, opts.riccati_exact, opts.riccati_iters_init); riccati_init_ = true; }
            const Vec<NA> Lr = obs::riccati_refine<NA, NY>(Aa, Ca, Qa, Ra, Pa_, opts.riccati_iters_refine).col(0);
            if (Lr.is_finite() && (obs::gain_is_robust<NA, NY>(Aa, Ca, Lr, opts.robust_lo, opts.robust_hi, rho_max) || !have_design_)) { L = Lr; ok = true; }
        }
        if (ok) {
            obs::blend_gain<NA>(La_, L, opts.gain_smooth, have_design_);
            for (int i = 0; i < NX; ++i) Lx_[i] = La_[i];
            ld_ = La_[NX];
            have_design_ = true;
        }
    }

    Model m_;
    X x_, g_, Lx_;
    Y residual_;
    Real d_ = Real(0), ld_ = Real(0);
    Mat<NX, NX> A_ref_;
    Mat<NY, NX> C_ref_;
    Vec<NA> La_;
    Mat<NA, NA> Pa_;
    bool have_design_ = false, riccati_init_ = false;
};

}  // namespace estkit
