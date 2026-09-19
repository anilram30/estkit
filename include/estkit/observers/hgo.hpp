// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/hgo.hpp — high-gain observer (HGO) with anti-peaking
//  saturation of the correction.
//
//  References
//    * Khalil, H. K. (2002). Nonlinear Systems, 3rd ed., Prentice Hall,
//      Section 14.5 (high-gain observers, the peaking phenomenon).
//    * Khalil, H. K. & Praly, L. (2014). High-gain observers in nonlinear feedback
//      control. International Journal of Robust and Nonlinear Control 24(6), 993-1015.
//    * Esfandiari, F. & Khalil, H. K. (1992). Output feedback stabilization of
//      fully linearizable systems. International Journal of Control 56(5), 1007-1037.
//      (saturation as the standard remedy for peaking)
//    * Ackermann, J. (1972); Ogata, K. (2010), Sec. 10-6 (pole placement).
//
//  Idea.  In the observable canonical coordinates xi = T x of a single-output
//  system, where T is built from the observability matrix
//      O = [C ; C A ; ... ; C A^{n-1}] ,
//  the observer gain is scaled by powers of a single small parameter eps,
//      L_i = alpha_i / eps^i ,  i = 1..n ,                                     (1)
//  with alpha_i the coefficients of a Hurwitz polynomial
//  s^n + alpha_1 s^{n-1} + ... + alpha_n.  The error dynamics then become
//  n-times faster as eps -> 0 and the observer dominates any bounded model
//  uncertainty; the price is the *peaking phenomenon*: the transient reaches an
//  amplitude O(1/eps^{n-1}) before it decays (Khalil 2002, Sec. 14.5).
//
//  Equivalent pole assignment.  Scaling the gain as in (1) is exactly equivalent
//  to placing the continuous-time observer poles at -lambda_i / eps, where the
//  lambda_i are the roots of the nominal polynomial.  Discretising with the
//  sample time dt gives the z-plane poles
//      p_i = exp(-lambda_i mu / eps) ,   mu = omega_0 dt ,                     (2)
//  which is what this implementation uses: (2) is placed with the shifted-and-
//  scaled Ackermann formula of obs::place_gain in the *original* coordinates,
//  avoiding the explicit canonical transformation.  That matters here: for the
//  2-RC cell model sampled at 10 Hz, det(O) ~ 1e-17, so forming T from O and
//  inverting it would be numerically meaningless, while Ackermann's formula
//  (which only needs O^{-1} e_n) stays usable.  mu = omega_0 dt is the nominal
//  (eps = 1) bandwidth per sample.
//
//  Choice of the lambda_i.  The default is the classical *binomial* observer,
//  lambda_i = 1 for all i, i.e. the nominal polynomial (s+1)^n and
//  alpha_i = binom(n, i), which is the standard textbook choice (Khalil 2002,
//  Ex. 14.7).  It is also the only numerically usable one on this plant: three of
//  the four open-loop modes lie within 2 % of each other, and any *spread*
//  lambda_i demands that those nearly coincident modes be pulled apart, which
//  costs gains of order 10^2-10^3 (see the Luenberger chapter).
//
//  Anti-peaking saturation.  Following Esfandiari & Khalil (1992), the correction
//  is saturated outside a compact set known to contain the trajectory.  Here the
//  per-step state correction is limited componentwise,
//      x^+ = x^- + sat( L r ; c_i ) ,   c_i = kappa sqrt(P0_ii) ,              (3)
//  where P0 is the initial covariance handed to init() (the natural state scale
//  supplied by the application) and kappa = Options::corr_sigmas.  Model::constrain
//  is applied afterwards, which is the second standard projection.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "luenberger.hpp"

namespace estkit {

template <class Model>
class HighGainObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static_assert(NY == 1, "HighGainObserver uses Ackermann pole placement (NY == 1)");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real eps = Real(0.2);          // high-gain parameter (smaller = faster, more peaking)
        Real mu = Real(1e-3);          // nominal per-sample bandwidth omega_0 dt
                                       // (omega_0 = 1/100 s^-1 at dt = 0.1 s, so the
                                       //  closed-loop time constant is eps * 100 s)
        Real lambda[NX];               // nominal (eps = 1) root locations (binomial: all 1)
        Real corr_sigmas = Real(0.5);  // anti-peaking limit, in units of sqrt(P0_ii) per step
        Real regain_tol = Real(1e-3);
        Real gain_smooth = Real(1);        // exponential blending of a re-designed gain (1 = replace)
        Real robust_lo = Real(0.8), robust_hi = Real(1.25);  // scheduling-uncertainty range for the gain check
        Real max_gain = Real(1e6);
        bool fallback_to_riccati = true;
        int  riccati_iters_init = 30000, riccati_iters_refine = 40;
        bool riccati_exact = true;         // exact DARE (doubling) instead of the truncated recursion
        // Certified decay rate: a design is accepted only if rho <= 1 - rho_margin
        // over the whole scheduling family (see obs::gain_is_robust).
        Real rho_margin = Real(1e-4);
        Real q_scale = Real(1), r_scale = Real(1);
        Real q_extra[NX];              // extra process-noise std per state (start-up LQE design)
        bool apply_constraints = true;

        Options() { for (int i = 0; i < NX; ++i) { lambda[i] = Real(1); q_extra[i] = Real(0); } }
    } opts;

    explicit HighGainObserver(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0; have_design_ = false; riccati_init_ = false; L_ = X(); residual_ = Y();
        for (int i = 0; i < NX; ++i) {
            const Real p = P0(i, i);
            scale_[i] = (p > Real(0)) ? std::sqrt(p) : Real(1);
        }
    }

    void predict(const U& u) { x_ = m_.f(x_, u); }
    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NX, NX> A = jac_F(m_, x_, u);
        const Mat<NY, NX> C = jac_H(m_, x_, u);
        schedule_gain(A, C, u);
        const Y r = y - m_.h(x_, u);
        residual_ = r;
        X dx = L_ * r[0];
        if (!dx.is_finite()) return;
        if (opts.corr_sigmas > Real(0)) {                      // anti-peaking saturation (3)
            for (int i = 0; i < NX; ++i) {
                const Real lim = opts.corr_sigmas * scale_[i];
                dx[i] = clampr(dx[i], -lim, lim);
            }
        }
        x_ = x_ + dx;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const X& gain() const { return L_; }
    const Y& residual() const { return residual_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void schedule_gain(const Mat<NX, NX>& A, const Mat<NY, NX>& C, const U& u) {
        if (have_design_ && obs::lin_change<NX, NY>(A_ref_, C_ref_, A, C) < opts.regain_tol) return;
        A_ref_ = A; C_ref_ = C;
        const Real e = (opts.eps > Real(1e-6)) ? opts.eps : Real(1e-6);
        Real p[NX];
        for (int i = 0; i < NX; ++i) p[i] = std::exp(-opts.lambda[i] * opts.mu / e);
        X L;
        bool ok = obs::place_gain<NX>(A, C, p, L);
        if (ok && L.norm_inf() > opts.max_gain) ok = false;
        const Real rho_max = Real(1) - opts.rho_margin;
        if (ok && !obs::gain_is_robust<NX, NY>(A, C, L, opts.robust_lo, opts.robust_hi, rho_max)) ok = false;
        if (!ok && opts.fallback_to_riccati) {
            Mat<NX, NX> Qd = m_.Q(x_, u) * opts.q_scale;
            for (int i = 0; i < NX; ++i) Qd(i, i) += sq(opts.q_extra[i]);
            const Mat<NY, NY> Rd = m_.R(x_, u) * opts.r_scale;
            if (!riccati_init_) { P_ = obs::dare_design<NX, NY>(A, C, Qd, Rd, opts.riccati_exact, opts.riccati_iters_init); riccati_init_ = true; }
            const X Lr = obs::riccati_refine<NX, NY>(A, C, Qd, Rd, P_, opts.riccati_iters_refine).col(0);
            if (Lr.is_finite() && (obs::gain_is_robust<NX, NY>(A, C, Lr, opts.robust_lo, opts.robust_hi, rho_max) || !have_design_)) { L = Lr; ok = true; }
        }
        if (ok) { obs::blend_gain<NX>(L_, L, opts.gain_smooth, have_design_); have_design_ = true; }
    }

    Model m_;
    X x_, L_;
    Y residual_;
    Real scale_[NX] = {};
    Mat<NX, NX> A_ref_, P_;
    Mat<NY, NX> C_ref_;
    bool have_design_ = false, riccati_init_ = false;
};

}  // namespace estkit
