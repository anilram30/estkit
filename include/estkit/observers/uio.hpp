// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/uio.hpp — unknown-input observer (UIO) with exact decoupling
//  of one input-channel disturbance.
//
//  References
//    * Chen, J., Patton, R. J. & Zhang, H.-Y. (1996). Design of unknown input
//      observers and robust fault detection filters. International Journal of
//      Control 63(1), 85-105.
//    * Chen, J. & Patton, R. J. (1999). Robust Model-Based Fault Diagnosis for
//      Dynamic Systems. Kluwer, Boston, Chapter 3.
//    * Darouach, M., Zasadzinski, M. & Xu, S. J. (1994). Full-order observers for
//      linear systems with unknown inputs. IEEE Transactions on Automatic Control
//      39(3), 606-609.
//    * Hou, M. & Muller, P. C. (1992). Design of observers for linear systems with
//      unknown inputs. IEEE Transactions on Automatic Control 37(6), 871-875.
//    * Zhao, S., Duncan, S. R. & Howey, D. A. (2017). Observability analysis and
//      state estimation of lithium-ion batteries in the presence of sensor biases.
//      IEEE Transactions on Control Systems Technology 25(1), 326-333.
//
//  Problem.  The plant carries an unknown, arbitrarily time-varying input d on
//  one channel of u (for a battery: the current-sensor bias, u_true = u - d e_j):
//
//      x_{k+1} = f(x_k, u_k) + E d_k ,   E = -df/du_j ,   y_k = h(x_k, u_k).    (1)
//
//  Design (Chen, Patton & Zhang 1996).  Choose
//      H = E (C E)^+ ,      T = I - H C ,                                       (2)
//      F = T A - K1 C ,     K = K1 + F H ,                                      (3)
//  and run
//      z_{k+1} = F z_k + T B u_k + K y_k ,     xhat_k = z_k + H y_k .           (4)
//  Then, because T E = E - E (CE)^+ (C E) = 0 whenever rank(C E) = rank(E),
//      e_{k+1} = x_{k+1} - xhat_{k+1} = F e_k                                   (5)
//  *exactly and independently of d*.  Existence therefore requires
//      (i)  rank(C E) = rank(E)   and   (ii) (T A, C) detectable.               (6)
//  Condition (ii) is equivalent to (A, E, C) having no invariant zero on or
//  outside the unit circle; the invariant zeros are *fixed modes* of F and cannot
//  be moved by K1 -- for the cell model one such mode sits at |z| = 0.99973
//  (tau ~ 375 s), which is exactly the price paid for the exact decoupling.
//
//  Nonlinear realisation.  The model here is nonlinear, so (4) is used with the
//  linearisation (A, C, E) at the current estimate and the substitutions
//      T B u_k   ->  T ( f(xhat_k, u_k) - A xhat_k )          (affine part of f)
//      y_k       ->  ytilde_k = C xhat_k + ( y_k - h(xhat_k, u_k) )
//  which reduce to (4) exactly when f and h are affine.  The approximation is
//  first order in the estimation error: the neglected terms are the second-order
//  remainders of f and h, i.e. O(||e||^2) through the curvature of the OCV.
//
//  Direct feed-through.  Classical UIO theory assumes y = C x, but a current-
//  sensor bias also enters the terminal voltage through the ohmic drop,
//  y = h(x, u - d e_j) ~ C x - D_j d with D_j = dh/du_j = -R0.  That path is NOT
//  decoupled and leaves a residual sensitivity (K + H) D_j d in (5); for the cell
//  it is proportional to R0 (12 mOhm) and is quantified in the chapter.
//
//  Disturbance reconstruction.  From (1), y_{k+1} - h(prediction) = C E d_k + v,
//  so dhat = r / (C E), low-pass filtered with time constant d_tau_steps.
//
//  Conditioning of the decoupling, and graceful degradation.  Condition (6i) is a
//  rank test, but the *numerical* quality of the design is governed by
//      ||H||_inf = ||E||_inf / |C E| ,
//  which is the factor by which a persistent OUTPUT model error delta shows up in
//  the state estimate (xhat = z + H ytilde feeds the raw measurement into the
//  state along E).  Exact rejection of the unknown input is therefore bought with
//  a 1/|C E| sensitivity to every unmodelled effect.  For the 2-RC cell the two
//  dominant terms of C E = (dOCV/dz) E_z - E_v1 - E_v2 + M E_h partially cancel,
//  and how much they cancel depends on the parameter set: with the nominal ECM
//  parameters ||H|| = 2.1, but with the parameters fitted to the single-particle
//  plant (whose R1 is an order of magnitude smaller) ||H|| = 15-17, and the
//  structured ECM-vs-SPM voltage error is then amplified sevenfold.
//  The design is consequently relaxed to a partial decoupling
//      H = theta E (C E)^+ ,                                                     (7)
//  for which T E = (1 - theta) E: the unknown input is attenuated by (1 - theta)
//  rather than removed and the remainder is rejected by the K1 feedback like any
//  other disturbance.  theta = 1 recovers the exact Chen-Patton-Zhang observer;
//  theta = 0 is an ordinary LQE observer, so the estimator degrades continuously
//  to "no decoupling" instead of failing.  Two independent factors set theta,
//      theta = min(1, h_max |C E| / ||E||_inf) * min(1, d_max / |dhat|_filtered),
//  an a-priori conditioning budget and an a-posteriori admissibility test.  The
//  second is the important one: dhat = r/(C E) is the unknown input that the
//  residual implies, so if its filtered magnitude exceeds the physical bound
//  d_max the residual demonstrably is NOT an input bias, hypothesis (1) is false,
//  and the observer must stop paying for a decoupling it cannot use.  On the ECM
//  plant |dhat| settles at 0.25-0.4 A against d_max = 5 A and the gate stays at
//  one; on the SPM plant the structured ECM-vs-SPM voltage error implies 30-70 A
//  and the gate closes to ~0.1.
//
//  Re-basing.  z estimates T x = x - H C x, so its meaning changes with H.  Every
//  re-design therefore shifts z by (H_old - H_new) ytilde, which keeps
//  xhat = z + H ytilde continuous; without it a change of the decoupling moves
//  the estimate by several volts times the change of H.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "luenberger.hpp"
#include <limits>

namespace estkit {

template <class Model>
class UnknownInputObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static_assert(NY == 1, "UnknownInputObserver is implemented for a scalar output");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    using Design = obs::Design;

    struct Options {
        int  u_index = 0;              // input channel carrying the unknown bias
        Design design = Design::Riccati;   // K1 design on (TA, C)
        Real pole[NX];                 // used when design == PolePlacement
        Real q_scale = Real(1), r_scale = Real(1), q_extra[NX];
        Real regain_tol = Real(1e-3);
        int  riccati_iters_init = 30000, riccati_iters_refine = 40;
        bool riccati_exact = true;         // exact DARE (doubling) instead of the truncated recursion
        // Certified decay rate: a design is accepted only if rho <= 1 - rho_margin
        // over the whole scheduling family (see obs::gain_is_robust).
        Real rho_margin = Real(1e-4);
        Real rank_tol = Real(1e-12);   // relative floor on |C E|
        Real d_tau_steps = Real(600);  // low-pass on the reconstructed disturbance
        Real d_max = std::numeric_limits<Real>::infinity();  // physical bound on the unknown input
        Real d_mon_steps = Real(6000);  // horizon of the admissibility monitor [samples]
        Real gate_tol = Real(0.05);     // change of the admissibility gate that forces a re-design
        // Largest measurement amplification ||H||_inf the decoupling may use.
        // H = E (C E)^+ injects the raw output into the state along E, so a
        // persistent *output* model error delta appears in the estimate as
        // ||H|| delta: exact decoupling of the unknown input is bought with a
        // 1/|C E| sensitivity to everything the model gets wrong.  When the
        // matching condition is satisfied but ill-conditioned the design is
        // therefore relaxed to a *partial* decoupling H = theta E (C E)^+ with
        //      theta = min(1, h_max |C E| / ||E||_inf) ,
        // for which T E = (1 - theta) E: the unknown input is attenuated by
        // (1 - theta) instead of removed, and the remainder is left to the K1
        // feedback.  theta = 0 degrades continuously into an ordinary LQE
        // observer, so the estimator never does worse than not decoupling at all.
        Real h_max = Real(1e30);
        Real robust_lo = Real(0.8), robust_hi = Real(1.25);  // scheduling-uncertainty range
        Real max_gain = Real(1e4);
        bool apply_constraints = true;

        Options() { for (int i = 0; i < NX; ++i) { pole[i] = Real(0.9975); q_extra[i] = Real(0); } }
    } opts;

    explicit UnknownInputObserver(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0; z_ = X(); ytil_ = Y(); residual_ = Y();
        have_design_ = false; riccati_init_ = false; first_ = true; force_design_ = false;
        d_ = Real(0); ce_ = Real(0); theta_ = Real(0); dmag_ = Real(0); gate_ = Real(1); decoupled_ = false;
    }

    void predict(const U& u) {
        const Mat<NX, NX> A = jac_F(m_, x_, u);
        const X xm = m_.f(x_, u);
        if (have_design_) {
            const X zn = F_ * z_ + T_ * (xm - A * x_) + K_ * ytil_;
            if (zn.is_finite()) z_ = zn;
        }
        x_ = xm;
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NX, NX> A = jac_F(m_, x_, u);
        const Mat<NY, NX> C = jac_H(m_, x_, u);
        const Y r = y - m_.h(x_, u);
        residual_ = r;
        const Y ytil = C * x_ + r;
        schedule_design(A, C, u, ytil);
        ytil_ = ytil;
        if (first_) { z_ = x_ - H_ * ytil_; first_ = false; }
        const X xn = z_ + H_ * ytil_;
        if (xn.is_finite()) x_ = xn;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
        // Disturbance reconstruction (first-order, low-pass filtered).  The raw
        // estimate r/(C E) inherits the 1/|C E| amplification, so it is clamped
        // to the physically admissible range before and after filtering; without
        // the clamp a 0.5 V structured model error reads as a 70 A "bias".
        if (std::fabs(ce_) > Real(0)) {
            const Real a = std::exp(-Real(1) / std::max(opts.d_tau_steps, Real(1)));
            const Real raw = r[0] / ce_;
            d_ = clampr(a * d_ + (Real(1) - a) * clampr(raw, -opts.d_max, opts.d_max), -opts.d_max, opts.d_max);
            // Admissibility monitor: |dhat| filtered on a much longer horizon.
            // This is the evidence used to decide whether the matching
            // assumption "the residual is an input bias" still holds.
            const Real b = std::exp(-Real(1) / std::max(opts.d_mon_steps, Real(1)));
            dmag_ = b * dmag_ + (Real(1) - b) * std::fabs(raw);
            if (std::fabs(admissibility() - gate_) > opts.gate_tol) force_design_ = true;
        }
    }

    // Fraction of the exact decoupling that the reconstructed disturbance still
    // justifies.  dhat = r/(C E) is the unknown input implied by the residual;
    // if its magnitude exceeds the physical bound d_max, the residual cannot be
    // an input bias, the matching assumption (1) is wrong, and the decoupling is
    // backed off in proportion.  On the ECM plant dhat settles at 0.25-0.4 A
    // against d_max = 5 A and the gate stays at 1 (exact decoupling); on the SPM
    // plant the structured ECM-vs-SPM voltage error implies 30-70 A and the gate
    // closes to ~0.1, leaving an ordinary LQE observer.
    Real admissibility() const {
        if (!(opts.d_max < std::numeric_limits<Real>::infinity()) || !(dmag_ > opts.d_max)) return Real(1);
        return opts.d_max / dmag_;
    }

    const X& x() const { return x_; }
    Real disturbance() const { return d_; }      // estimated bias on input channel u_index
    bool decoupled() const { return decoupled_; }  // rank(CE) == rank(E) satisfied
    Real decoupling() const { return theta_; }     // theta in [0,1]: 1 = exact decoupling
    Real gate() const { return gate_; }            // disturbance-admissibility factor in (0,1]
    const X& gain() const { return K1_; }
    const Y& residual() const { return residual_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void schedule_design(const Mat<NX, NX>& A, const Mat<NY, NX>& C, const U& u, const Y& ytil) {
        if (have_design_ && !force_design_ && obs::lin_change<NX, NY>(A_ref_, C_ref_, A, C) < opts.regain_tol) return;
        force_design_ = false;
        A_ref_ = A; C_ref_ = C;
        // E = -df/du_j : the state increment produced by a positive bias on the
        // measured input (u_true = u_meas - d e_j).
        const X E = jac_B(m_, x_, u).col(opts.u_index) * Real(-1);
        const Real ce = (C * E)[0];
        const Real scale = (Real(1) + C.norm_inf()) * (Real(1) + E.norm_inf());
        decoupled_ = (std::fabs(ce) > opts.rank_tol * scale) && (E.norm_inf() > Real(0));
        ce_ = decoupled_ ? ce : Real(0);
        X H;
        theta_ = Real(0);
        gate_ = admissibility();
        if (decoupled_) {
            // Partial decoupling.  theta = 1 (the exact Chen-Patton-Zhang design)
            // unless either the conditioning budget h_max or the disturbance
            // admissibility gate says the decoupling is not justified.
            const Real hinf = E.norm_inf() / std::fabs(ce);
            theta_ = gate_ * ((hinf > opts.h_max) ? opts.h_max / hinf : Real(1));
            H = E * (theta_ / ce);                  // H = theta E (C E)^+
        }
        const Mat<NX, NX> T = Mat<NX, NX>::identity() - H * C;
        const Mat<NX, NX> TA = T * A;
        // Candidate selection.  Unlike the rest of the family the UIO cannot be
        // held to a *designer-chosen* decay rate: the invariant zeros of
        // (A, E, C) are fixed modes of F = TA - K1 C and no K1 can move them
        // (|z| = 0.99973, tau ~ 375 s, for the cell).  The gate is therefore
        // nominal stability, and the scheduling-family certification is used as
        // a *preference* between nominally stable candidates.
        const Real rho_max = Real(1) - opts.rho_margin;
        X K1best; bool have_best = false, best_robust = false;
        auto consider = [&](const X& Kc) {
            if (!Kc.is_finite() || Kc.norm_inf() > opts.max_gain) return;
            const Mat<NX, NX> Fc = TA - Kc * C;
            if (!Fc.is_finite()) return;
            if (!(obs::spectral_radius<NX>(Fc) < Real(1) - Real(1e-6))) return;
            const bool rob = f_is_robust(A, C, T, Kc, rho_max);
            if (!have_best || (rob && !best_robust)) { K1best = Kc; have_best = true; best_robust = rob; }
        };
        if (opts.design == Design::PolePlacement) {
            // K1 enters F = TA - K1 C directly, i.e. it is a *prediction*-form
            // gain, so the placement is used without the A^{-1} conversion.
            X Kp;
            if (obs::place_gain_pred<NX>(TA, C, opts.pole, Kp)) consider(Kp);
        }
        if (opts.design == Design::Riccati || !best_robust) {
            Mat<NX, NX> Qd = m_.Q(x_, u) * opts.q_scale;
            for (int i = 0; i < NX; ++i) Qd(i, i) += sq(opts.q_extra[i]);
            const Mat<NY, NY> Rd = m_.R(x_, u) * opts.r_scale;
            if (!riccati_init_) { P_ = obs::dare_design<NX, NY>(TA, C, Qd, Rd, opts.riccati_exact, opts.riccati_iters_init); riccati_init_ = true; }
            // K1 for the *prediction* form F = TA - K1 C is TA times the
            // current-estimator (filter) gain of the pair (TA, C).
            const Mat<NX, NY> Kf = obs::riccati_refine<NX, NY>(TA, C, Qd, Rd, P_, opts.riccati_iters_refine);
            consider(TA * Kf.col(0));
        }
        if (!have_best) return;
        // Re-base the internal state.  z estimates T x = x - H C x, so its
        // meaning changes with H; without the shift below, a change of H makes
        // xhat = z + H ytilde jump by (H_new - H_old) ytilde, i.e. by several
        // volts times the change of the decoupling.
        if (have_design_) { const X dz = (H_ - H) * ytil; if (dz.is_finite()) z_ = z_ + dz; }
        H_ = H; T_ = T; K1_ = K1best; F_ = TA - K1best * C; K_ = K1best + F_ * H;
        have_design_ = true;
    }

    // Certification of F = T A - K1 C over the same scheduling family used by
    // the rest of the family (obs::gain_is_robust).  T is invariant under a
    // uniform scaling of C because H carries the reciprocal factor, so only the
    // state matrix and the loop gain have to be varied.
    bool f_is_robust(const Mat<NX, NX>& A, const Mat<NY, NX>& C, const Mat<NX, NX>& T,
                     const X& K1, Real rho_max) const {
        const Mat<NX, NX> I = Mat<NX, NX>::identity();
        const Real sv[3] = {Real(1), opts.robust_lo, opts.robust_hi};
        for (int a = 0; a < 3; ++a) {
            const Mat<NX, NX> As = I + (A - I) * sv[a];
            for (int g = 0; g < 3; ++g)
                if (!(obs::spectral_radius<NX>(T * As - (K1 * sv[g]) * C) < rho_max)) return false;
        }
        return true;
    }

    Model m_;
    X x_, z_, H_, K1_, K_;
    Y ytil_, residual_;
    Mat<NX, NX> T_, F_, A_ref_, P_;
    Mat<NY, NX> C_ref_;
    Real d_ = Real(0), ce_ = Real(0), theta_ = Real(0), dmag_ = Real(0), gate_ = Real(1);
    bool have_design_ = false, riccati_init_ = false, first_ = true, force_design_ = false, decoupled_ = false;
};

}  // namespace estkit
