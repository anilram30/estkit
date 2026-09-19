// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/smo.hpp — first-order sliding-mode observer (SMO) with a
//  boundary layer, and the adaptive-switching-gain variant.
//
//  References
//    * Utkin, V. I. (1992). Sliding Modes in Control and Optimization. Springer,
//      Berlin. (variable-structure observers, boundary-layer regularisation)
//    * Slotine, J.-J. E., Hedrick, J. K. & Misawa, E. A. (1987). On sliding
//      observers for nonlinear systems. J. Dynamic Systems, Measurement, and
//      Control 109(3), 245-252.
//    * Kim, I.-S. (2006). The novel state of charge estimation method for lithium
//      battery using sliding mode observer. Journal of Power Sources 163(1), 584-590.
//    * Kim, I.-S. (2008). Nonlinear state of charge estimator for hybrid electric
//      vehicle battery. IEEE Transactions on Power Electronics 23(4), 2027-2034.
//    * Chen, X., Shen, W., Cao, Z. & Kapoor, A. (2014). A novel approach for state
//      of charge estimation based on adaptive switching gain sliding mode observer
//      in electric vehicles. Journal of Power Sources 246, 667-678.
//    * Ioannou, P. A. & Kokotovic, P. V. (1984). Instability analysis and
//      improvement of robustness of adaptive control. Automatica 20(5), 583-594.
//      (sigma-modification used to keep the adapted gain bounded)
//
//  Algorithm (one sampling period, current-estimator form, NY = 1)
//      x^-_k = f(x^+_{k-1}, u_{k-1})
//      r_k   = y_k - h(x^-_k, u_k)                    (sliding variable s = r)
//      x^+_k = x^-_k + L r_k + kappa_k K_s sat(r_k / phi)
//
//  L is the linear (Luenberger) gain obtained by pole placement or by the LQE
//  Riccati equation; K_s in R^{NX} is the switching-gain distribution vector;
//  phi > 0 is the boundary-layer half-width that replaces sgn(.) by sat(./phi)
//  and so removes chattering at the price of a |r| <= O(phi) ultimate bound
//  (Utkin 1992, Sec. 8.3; Slotine et al. 1987).
//
//  Reaching condition.  With V = s^2/2 and s = r = -C e + delta (delta the lumped
//  model error bounded by |delta| <= D), a single step outside the boundary layer
//  changes s by
//      Delta s = -(C L) r - (C K_s) sgn(r) + O(||A - I||, D),
//  so V decreases as long as
//      (C K_s) > D + |C (A - I) e| ,                                       (R1)
//  i.e. the projection of the switching gain on the output direction must
//  dominate the lumped uncertainty.  Once |r| <= phi the trajectory stays in the
//  boundary layer and the estimate converges to an O(phi) neighbourhood.
//
//  Choice of K_s.  If Options::ks is left at zero the switching gain is derived
//  from the linear gain, but applied to ONE state only:
//      K_s = ks_scale phi L_{j*} e_{j*} ,  j* = argmax_j ||(I-A)^{-1} e_j||_1 ,
//  the state with the largest DC gain from a disturbance (SOC for the cell).
//  Spreading the discontinuous term over all states is unsafe with a
//  gain-scheduled L: the entries of L that belong to weakly observable modes
//  change sign with the operating point (the hysteresis entry of the cell model
//  does so between hover and cruise), and a sgn() term with a flipping sign is
//  positive feedback.  With this rule the switching term contributes an
//  additional gain ks_scale L_{j*} inside the boundary layer and saturates at
//  ks_scale phi L_{j*} outside it -- a bounded-influence correction.  |K_s| is
//  additionally capped at ks_max, because L itself is scheduled: at an operating
//  point where the design briefly asks for a large L_{j*} the *saturated* part of
//  the injection would otherwise become a large constant drift on the estimate.
//
//  Adaptive switching gain (Chen et al. 2014).  D is rarely known.  The gain
//  multiplier kappa is adapted online from a low-pass filtered residual
//  |r|_f (time constant adapt_tau), which isolates the systematic part of the
//  residual from the measurement noise:
//      kappa_{k+1} = kappa_k + gamma (|r_k|_f - phi)            if |r_k|_f >  phi
//      kappa_{k+1} = kappa_k - sigma (kappa_k - kappa_min)      if |r_k|_f <= phi
//  and projected onto [kappa_min, kappa_max].  The first line raises the gain
//  until the sliding surface is reached (so (R1) is eventually satisfied without
//  knowing D); the second is the sigma-modification that lets the gain decay
//  again once the surface is reached, which limits chattering and keeps kappa
//  bounded (Ioannou & Kokotovic 1984).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "luenberger.hpp"

namespace estkit {

template <class Model>
class SlidingModeObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static_assert(NY == 1, "SlidingModeObserver assumes a scalar sliding variable (NY == 1)");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    using Design = obs::Design;

    struct Options {
        // ---- linear part ----
        Design design = Design::PolePlacement;
        Real pole[NX];
        Real q_scale = Real(1), r_scale = Real(1), q_extra[NX];
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
        // ---- switching part ----
        Real phi = Real(0.005);       // boundary-layer half-width [output units]
        Real ks[NX];                  // switching-gain vector; all zero => auto (see below)
        Real ks_max = Real(1e30);     // absolute cap on |K_s| (state units per step)
        Real ks_scale = Real(2);      // auto rule: K_s = ks_scale * phi * L_{j*} e_{j*}, so
                                      // that at |r| = phi the switching term is ks_scale
                                      // times the linear term on the dominant state and it
                                      // saturates beyond (bounded influence)
        // ---- adaptation (Chen et al. 2014) ----
        bool adaptive_gain = false;
        Real adapt_rate = Real(2);        // gamma, per step and per unit of |r|_f - phi
        Real adapt_leak = Real(1e-3);     // sigma, per-step relaxation toward kappa_min
        Real adapt_tau = Real(200);       // steps; low-pass on |r| used by the adaptation
        Real kappa_min = Real(0.5), kappa_max = Real(8), kappa0 = Real(1);
        bool apply_constraints = true;

        Options() {
            for (int i = 0; i < NX; ++i) { pole[i] = Real(0.9975); q_extra[i] = Real(0); ks[i] = Real(0); }
        }
    } opts;

    explicit SlidingModeObserver(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0; have_design_ = false; riccati_init_ = false; L_ = X(); Ks_ = X();
        kappa_ = opts.kappa0; rf_ = Real(0); residual_ = Y();
    }

    void predict(const U& u) { x_ = m_.f(x_, u); }
    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NX, NX> A = jac_F(m_, x_, u);
        const Mat<NY, NX> C = jac_H(m_, x_, u);
        schedule_gain(A, C, u);
        const Y r = y - m_.h(x_, u);
        residual_ = r;
        const Real s = r[0];
        const Real phi = (opts.phi > Real(0)) ? opts.phi : Real(0);
        const Real sw = sat(s, phi);                       // sat(s/phi) in [-1, 1]
        if (opts.adaptive_gain) adapt(std::fabs(s), phi);
        X dx;
        for (int i = 0; i < NX; ++i) dx[i] = L_[i] * s + kappa_ * Ks_[i] * sw;
        if (dx.is_finite()) x_ = x_ + dx;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const X& gain() const { return L_; }
    const X& switching_gain() const { return Ks_; }
    Real kappa() const { return kappa_; }               // adapted switching-gain multiplier
    const Y& residual() const { return residual_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    // Adaptation on a low-pass filtered |r|: the instantaneous residual is
    // dominated by the measurement noise, and driving the gain up on individual
    // noise excursions would raise kappa (and hence the effective gain inside the
    // boundary layer) until the loop chatters.  The filter makes the dead zone
    // act on the *systematic* part of the residual, which is what the switching
    // term has to dominate.
    void adapt(Real absr, Real phi) {
        const Real a = std::exp(-Real(1) / std::max(opts.adapt_tau, Real(1)));
        rf_ = a * rf_ + (Real(1) - a) * absr;
        if (rf_ > phi) kappa_ += opts.adapt_rate * (rf_ - phi);
        else           kappa_ -= opts.adapt_leak * (kappa_ - opts.kappa_min);
        kappa_ = clampr(kappa_, opts.kappa_min, opts.kappa_max);
    }

    void schedule_gain(const Mat<NX, NX>& A, const Mat<NY, NX>& C, const U& u) {
        if (have_design_ && obs::lin_change<NX, NY>(A_ref_, C_ref_, A, C) < opts.regain_tol) return;
        A_ref_ = A; C_ref_ = C;
        X L;
        bool ok = false;
        if (opts.design == Design::PolePlacement) {
            ok = obs::place_gain<NX>(A, C, opts.pole, L);
            if (ok && L.norm_inf() > opts.max_gain) ok = false;
        }
        // Certify the placement over the whole scheduling family before using it.
        const Real rho_max = Real(1) - opts.rho_margin;
        if (ok && !obs::gain_is_robust<NX, NY>(A, C, L, opts.robust_lo, opts.robust_hi, rho_max)) ok = false;
        // The LQE fallback is available at *every* re-design: the linear part of
        // a sliding-mode observer must stay certified, otherwise the switching
        // term is injected on top of an unstable equivalent dynamics.
        if (opts.design == Design::Riccati || (!ok && opts.fallback_to_riccati)) {
            Mat<NX, NX> Qd = m_.Q(x_, u) * opts.q_scale;
            for (int i = 0; i < NX; ++i) Qd(i, i) += sq(opts.q_extra[i]);
            const Mat<NY, NY> Rd = m_.R(x_, u) * opts.r_scale;
            if (!riccati_init_) { P_ = obs::dare_design<NX, NY>(A, C, Qd, Rd, opts.riccati_exact, opts.riccati_iters_init); riccati_init_ = true; }
            const X Lr = obs::riccati_refine<NX, NY>(A, C, Qd, Rd, P_, opts.riccati_iters_refine).col(0);
            if (Lr.is_finite() && (obs::gain_is_robust<NX, NY>(A, C, Lr, opts.robust_lo, opts.robust_hi, rho_max) || !have_design_)) { L = Lr; ok = true; }
        }
        if (!ok) return;
        obs::blend_gain<NX>(L_, L, opts.gain_smooth, have_design_);
        Real user = Real(0);
        for (int i = 0; i < NX; ++i) user += std::fabs(opts.ks[i]);
        Ks_ = X();
        if (user > Real(0)) {
            for (int i = 0; i < NX; ++i) Ks_[i] = opts.ks[i];
        } else {
            // Auto rule.  The switching term is applied to ONE state -- the one
            // with the largest DC gain from a disturbance (SOC for the cell) --
            // with magnitude ks_scale phi L_{j*}.  Restricting it to a single
            // component matters: the remaining entries of a gain-scheduled L
            // change sign with the operating point (the hysteresis entry does so
            // between hover and cruise), and a switching term whose sign flips
            // turns the discontinuous injection into positive feedback.
            const int j = obs::slowest_state<NX>(A);
            if (L_[j] * C(0, j) > Real(0))
                Ks_[j] = clampr(opts.ks_scale * opts.phi * L_[j], -opts.ks_max, opts.ks_max);
        }
        have_design_ = true;
    }

    Model m_;
    X x_, L_, Ks_;
    Y residual_;
    Real kappa_ = Real(1), rf_ = Real(0);
    Mat<NX, NX> A_ref_, P_;
    Mat<NY, NX> C_ref_;
    bool have_design_ = false, riccati_init_ = false;
};

}  // namespace estkit
