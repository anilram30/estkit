// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/adaptive_observer.hpp — joint state and parameter adaptive
//  observer (Kreisselmeier / Zhang form).
//
//  References
//    * Kreisselmeier, G. (1977). Adaptive observers with exponential rate of
//      convergence. IEEE Transactions on Automatic Control 22(1), 2-8.
//    * Zhang, Q. (2002). Adaptive observer for multiple-input-multiple-output
//      (MIMO) linear time-varying systems. IEEE Transactions on Automatic Control
//      47(3), 525-529.
//    * Zhang, Q. & Clavel, A. (2001). Adaptive observer with exponential
//      forgetting factor for linear time varying systems. Proc. 40th IEEE
//      Conference on Decision and Control, 3886-3891.
//    * Ioannou, P. A. & Sun, J. (1996). Robust Adaptive Control. Prentice Hall.
//      (normalised gradient law, sigma-modification, persistency of excitation)
//
//  Problem.  The model depends on an unknown constant parameter vector theta
//  (Model::params(); for the cell theta = [Q_Ah, R0]).  A Luenberger observer
//  designed for a wrong theta leaves a bias that no output injection can remove,
//  so theta must be estimated together with the state.
//
//  Zhang's construction.  Write the linearised error model with the parameter
//  sensitivities
//      Psi_k = df/dtheta ,   Phi_k = dh/dtheta                                 (1)
//  (obtained here by central finite differences through Model::set_params, so no
//  analytic derivative is required).  Introduce the *filtered sensitivity*
//  (regressor) matrix Upsilon in R^{NX x NP} propagated through the observer
//  dynamics; in current-estimator form
//      Upsilon^-_k = A Upsilon^+_{k-1} + Psi_{k-1} ,                           (2a)
//      Omega_k     = C Upsilon^-_k + Phi_k ,          (output sensitivity)     (2b)
//      Upsilon^+_k = Upsilon^-_k - L Omega_k ,                                 (2c)
//  which is exactly Zhang's Upsilon_{k+1} = (A - L C) Upsilon_k + Psi - L Phi
//  split across the time and measurement updates.  The key property is that, to
//  first order, the output error depends on the parameter error through Omega,
//      r_k = y_k - h(xhat^-_k, u_k) ~ -C e^-_k - Omega_k (thetahat - theta) ,  (3)
//  so Omega is the correct regressor for a gradient/least-squares update:
//      Delta_k     = Gamma Omega_k^T r_k / (1 + ||Omega_k||^2 / nu) ,          (4)
//      thetahat_k  = thetahat_{k-1} + Delta_k ,
//      xhat^+_k    = xhat^-_k + L r_k + Upsilon^-_k Delta_k .                  (5)
//  The last term of (5) is Zhang's state/parameter coupling: it corrects the state
//  for the instantaneous change of the parameter estimate and is what makes the
//  joint error dynamics triangular and therefore exponentially stable whenever
//  Omega is persistently exciting (Zhang 2002, Thm. 1).
//
//  Scaling.  The physical parameters of a cell differ by three orders of
//  magnitude (Q_Ah ~ 5 Ah, R0 ~ 0.012 Ohm), so the adaptation is carried out on
//  the *relative* deviation phi defined by theta = theta_0 (1 + phi), phi
//  projected onto [phi_min, phi_max].  All sensitivities, gains and bounds are
//  then dimensionless and one scalar Gamma works for every parameter.
//  A sigma-modification (leak) keeps phi bounded when excitation is poor.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "luenberger.hpp"

namespace estkit {

template <class Model>
class AdaptiveObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static_assert(has_params<Model>::value, "AdaptiveObserver requires Model::NP, params(), set_params()");
    static constexpr int NP = Model::NP;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    using Design = obs::Design;

    struct Options {
        // ---- state observer ----
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
        // ---- parameter adaptation ----
        Real gamma[NP];              // per-parameter adaptation gain on the relative deviation
        Real nu = Real(1);           // normalisation constant in (4)
        Real fd_step = Real(1e-4);   // relative finite-difference step for Psi, Phi
        Real phi_min = Real(-0.45), phi_max = Real(0.45);   // projection bounds on theta/theta_0 - 1
        Real leak = Real(0);         // sigma-modification (per-step pull of phi toward 0)
        Real ups_max = Real(1e4);    // clamp on the filtered sensitivity (anti-windup)
        bool state_coupling = true;  // include the Upsilon Delta term of (5)
        bool apply_constraints = true;

        Options() {
            for (int i = 0; i < NX; ++i) { pole[i] = Real(0.9975); q_extra[i] = Real(0); }
            for (int i = 0; i < NP; ++i) gamma[i] = Real(1);
        }
    } opts;

    explicit AdaptiveObserver(const Model& m) : m_(m) { theta0_ = m_.params(); }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0; have_design_ = false; riccati_init_ = false; L_ = X(); residual_ = Y();
        theta0_ = m_.params();
        phi_ = Vec<NP>();
        Ups_ = Mat<NX, NP>();
        Psi_ = Mat<NX, NP>();
    }

    void predict(const U& u) {
        Psi_ = sens_f(x_, u);                    // df/dphi at the posterior estimate
        const Mat<NX, NX> A = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        Ups_ = A * Ups_ + Psi_;                  // (2a)
        clamp_ups();
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NX, NX> A = jac_F(m_, x_, u);
        const Mat<NY, NX> C = jac_H(m_, x_, u);
        schedule_gain(A, C, u);
        const Y r = y - m_.h(x_, u);
        residual_ = r;
        const Mat<NY, NP> Phi = sens_h(x_, u);
        const Mat<NY, NP> Om = C * Ups_ + Phi;           // (2b)
        // normalised gradient update (4)
        Real n2 = Real(0);
        for (int k = 0; k < NY * NP; ++k) n2 += Om.d[k] * Om.d[k];
        const Real den = Real(1) + n2 / std::max(opts.nu, Real(1e-12));
        Vec<NP> del;
        const Vec<NP> g = Om.t() * r;
        for (int i = 0; i < NP; ++i) del[i] = opts.gamma[i] * g[i] / den - opts.leak * phi_[i];
        if (del.is_finite()) {
            for (int i = 0; i < NP; ++i) {
                const Real pn = clampr(phi_[i] + del[i], opts.phi_min, opts.phi_max);
                del[i] = pn - phi_[i];        // projected increment (used in (5) too)
                phi_[i] = pn;
            }
            apply_params();
        } else {
            del = Vec<NP>();
        }
        // state update (5)
        X dx = L_ * r[0];
        if (opts.state_coupling) dx = dx + Ups_ * del;
        if (dx.is_finite()) x_ = x_ + dx;
        Ups_ = Ups_ - L_ * Om;                           // (2c)
        clamp_ups();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    Vec<NP> params() const { return m_.params(); }
    const Vec<NP>& relative_deviation() const { return phi_; }
    const Mat<NX, NP>& sensitivity() const { return Ups_; }
    const X& gain() const { return L_; }
    const Y& residual() const { return residual_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void apply_params() {
        Vec<NP> th;
        for (int i = 0; i < NP; ++i) th[i] = theta0_[i] * (Real(1) + phi_[i]);
        m_.set_params(th);
    }
    void clamp_ups() {
        for (int k = 0; k < NX * NP; ++k) {
            if (!std::isfinite(Ups_.d[k])) Ups_.d[k] = Real(0);
            Ups_.d[k] = clampr(Ups_.d[k], -opts.ups_max, opts.ups_max);
        }
    }
    // central differences of f and h w.r.t. the relative parameter deviation phi.
    // The model is perturbed in place (set_params is cheap) and restored, which
    // avoids copying the model (and its OCV table) NP times per step.
    Mat<NX, NP> sens_f(const X& x, const U& u) {
        Mat<NX, NP> J;
        const Vec<NP> th = m_.params();
        for (int i = 0; i < NP; ++i) {
            const Real d = opts.fd_step * std::fabs(theta0_[i]);
            if (!(d > Real(0))) continue;
            Vec<NP> tp = th, tm = th;
            tp[i] += d; tm[i] -= d;
            m_.set_params(tp); const X fp = m_.f(x, u);
            m_.set_params(tm); const X fm = m_.f(x, u);
            m_.set_params(th);
            // dtheta_i = theta0_i * dphi_i  ->  df/dphi_i = theta0_i * df/dtheta_i
            for (int j = 0; j < NX; ++j) J(j, i) = theta0_[i] * (fp[j] - fm[j]) / (Real(2) * d);
        }
        return J;
    }
    Mat<NY, NP> sens_h(const X& x, const U& u) {
        Mat<NY, NP> J;
        const Vec<NP> th = m_.params();
        for (int i = 0; i < NP; ++i) {
            const Real d = opts.fd_step * std::fabs(theta0_[i]);
            if (!(d > Real(0))) continue;
            Vec<NP> tp = th, tm = th;
            tp[i] += d; tm[i] -= d;
            m_.set_params(tp); const Y hp = m_.h(x, u);
            m_.set_params(tm); const Y hm = m_.h(x, u);
            m_.set_params(th);
            for (int j = 0; j < NY; ++j) J(j, i) = theta0_[i] * (hp[j] - hm[j]) / (Real(2) * d);
        }
        return J;
    }

    void schedule_gain(const Mat<NX, NX>& A, const Mat<NY, NX>& C, const U& u) {
        if (have_design_ && obs::lin_change<NX, NY>(A_ref_, C_ref_, A, C) < opts.regain_tol) return;
        A_ref_ = A; C_ref_ = C;
        X L;
        bool ok = false;
        if (opts.design == Design::PolePlacement) {
            if constexpr (NY == 1) {
                ok = obs::place_gain<NX>(A, C, opts.pole, L);
                if (ok && L.norm_inf() > opts.max_gain) ok = false;
            }
        }
        // Certify before use: an uncertified state gain makes the *parameter*
        // update diverge too, because the sensitivity filter Upsilon is driven
        // by the same closed-loop matrix (I - L C) A.
        const Real rho_max = Real(1) - opts.rho_margin;
        if (ok && !obs::gain_is_robust<NX, NY>(A, C, L, opts.robust_lo, opts.robust_hi, rho_max)) ok = false;
        if (opts.design == Design::Riccati || (!ok && opts.fallback_to_riccati)) {
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
    Vec<NP> theta0_, phi_;
    Mat<NX, NP> Ups_, Psi_;
    Mat<NX, NX> A_ref_, P_;
    Mat<NY, NX> C_ref_;
    bool have_design_ = false, riccati_init_ = false;
};

}  // namespace estkit
