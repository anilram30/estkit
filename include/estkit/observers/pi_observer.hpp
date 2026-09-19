// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/pi_observer.hpp — Proportional-Integral (PI) observer.
//
//  References
//    * Wojciechowski, B. (1978). Analysis and synthesis of proportional-integral
//      observers for single-input single-output time-invariant continuous systems.
//      Ph.D. dissertation, Gliwice Technical University, Gliwice, Poland.
//    * Beale, S. & Shafai, B. (1989). Robust control system design with a
//      proportional integral observer. International Journal of Control 50(1), 97-111.
//    * Shafai, B., Beale, S., Niemann, H. H. & Stoustrup, J. L. (1996). LTR design
//      of discrete-time proportional-integral observers. IEEE Transactions on
//      Automatic Control 41(7), 1056-1062.
//    * Luenberger, D. G. (1971). An introduction to observers. IEEE TAC 16(6), 596-602.
//
//  Structure.  The plain Luenberger observer leaves a steady-state estimation
//  error whenever the state equation is corrupted by a constant (or slowly
//  varying) disturbance -- for a battery, an offset on the integrated current.
//  The PI observer adds an integral of the output error,
//
//      xi_{k+1} = xi_k + (y_k - yhat_k),         (integral state, NY-dimensional)
//      xhat^+_k = xhat^-_k + L_P (y_k - yhat_k) + L_I xi_k ,
//
//  which is equivalent to the Luenberger observer of the plant augmented with a
//  constant unknown state disturbance d entering through a distribution matrix
//  G (in R^{NX x NY}):
//
//      x_{k+1} = f(x_k, u_k) + G d_k ,  d_{k+1} = d_k ,   y_k = h(x_k, u_k).   (1)
//
//  Writing the observer for (1) in current-estimator form,
//
//      xhat^-_k = f(xhat^+_{k-1}, u_{k-1}) + G dhat^+_{k-1},   dhat^-_k = dhat^+_{k-1},
//      r_k      = y_k - h(xhat^-_k, u_k),
//      xhat^+_k = xhat^-_k + L_P r_k ,   dhat^+_k = dhat^-_k + L_I r_k ,        (2)
//
//  gives dhat^+_k = sum_{j<=k} L_I r_j, i.e. dhat is precisely the integral state
//  xi scaled by L_I, and G dhat is the integral correction of the PI form.
//
//  Error dynamics.  With eta = [xhat - x ; dhat - d] and Abar = [[A, G],[0, I]],
//  Cbar = [C, 0], Lbar = [L_P; L_I],
//
//      eta_k = (I - Lbar Cbar) Abar eta_{k-1} + noise,                          (3)
//
//  so the NX + NY poles are assigned by placing eig(Abar - (Abar Lbar) Cbar).
//  The augmented pair is observable iff (A, C) is observable and
//      rank [ A - I   G ;  C   0 ] = NX + NY                                    (4)
//  (no transmission zero of (A, G, C) at z = 1).
//
//  Choice of G.  When no physical disturbance model is available, G defaults to
//  the unit vector e_{j*} of the state with the largest DC gain from a constant
//  disturbance, j* = argmax_j ||(I - A)^{-1} e_j||_1 (obs::slowest_state) -- the
//  state that cannot correct itself.  For the cell model A = diag(1, a1, a2, ah)
//  and j* is the SOC integrator, so the integral action is applied exactly where
//  the coulomb-counting drift accumulates.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "luenberger.hpp"

namespace estkit {

template <class Model>
class PiObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NA = NX + NY;    // augmented dimension
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    using Design = obs::Design;

    struct Options {
        Design design = Design::PolePlacement;
        Real pole[NX];        // poles of the NX "plant" modes (z-plane radii)
        Real pole_i[NY];      // poles of the NY integral modes
        Real g[NX];           // disturbance distribution column (NY = 1); all zero => auto
        Real q_scale = Real(1), r_scale = Real(1);
        Real d_max = Real(1e30);         // anti-windup bound on the integral/disturbance state
        Real q_d = Real(1e-6);           // random-walk std of d, Riccati design only
        Real regain_tol = Real(1e-3);
        int  riccati_iters_init = 30000, riccati_iters_refine = 40;
        bool riccati_exact = true;         // exact DARE (doubling) instead of the truncated recursion
        // Certified decay rate: a design is accepted only if rho <= 1 - rho_margin
        // over the whole scheduling family (see obs::gain_is_robust).
        Real rho_margin = Real(1e-4);
        bool apply_constraints = true;
        bool fallback_to_riccati = true;
        Real gain_smooth = Real(1);        // exponential blending of a re-designed gain (1 = replace)
        Real robust_lo = Real(0.8), robust_hi = Real(1.25);  // scheduling-uncertainty range for the gain check
        Real max_gain = Real(1e4);

        Options() {
            for (int i = 0; i < NX; ++i) pole[i] = Real(0.996672);   // tau = 30 s at dt = 0.1 s
            for (int i = 0; i < NY; ++i) pole_i[i] = Real(0.999);    // tau = 100 s at dt = 0.1 s
            for (int i = 0; i < NX; ++i) g[i] = Real(0);
        }
    } opts;

    explicit PiObserver(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0;
        xi_ = Y();
        d_ = Y();
        Pa_ = Mat<NA, NA>();
        Pa_.template set_block<NX, NX>(0, 0, P0);
        for (int i = 0; i < NY; ++i) Pa_(NX + i, NX + i) = sq(Real(1e-2));
        have_design_ = false; riccati_init_ = false; La_ = Vec<NA>();
        Lp_ = Mat<NX, NY>();
        Li_ = Mat<NY, NY>();
    }

    void predict(const U& u) { x_ = m_.f(x_, u) + G_ * d_; }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NX, NX> A = jac_F(m_, x_, u);
        const Mat<NY, NX> C = jac_H(m_, x_, u);
        schedule_gain(A, C, u);
        const Y r = y - m_.h(x_, u);
        residual_ = r;
        xi_ = xi_ + r;                       // integral state (bookkeeping / accessor)
        const X dx = Lp_ * r;
        const Y dd = Li_ * r;
        if (dx.is_finite() && dd.is_finite()) {
            x_ = x_ + dx;
            for (int i = 0; i < NY; ++i) d_[i] = clampr(d_[i] + dd[i], -opts.d_max, opts.d_max);
        }
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Y& disturbance() const { return d_; }   // estimated constant state disturbance
    const Y& integral() const { return xi_; }     // raw integral of the output error
    const Y& residual() const { return residual_; }
    const Mat<NX, NY>& gainP() const { return Lp_; }
    const Mat<NY, NY>& gainI() const { return Li_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void build_G(const Mat<NX, NX>& A) {
        Real nrm = Real(0);
        for (int i = 0; i < NX; ++i) nrm += std::fabs(opts.g[i]);
        G_ = Mat<NX, NY>();
        if (nrm > Real(0)) {
            for (int i = 0; i < NX; ++i) G_(i, 0) = opts.g[i];
        } else {
            const int j = obs::slowest_state<NX>(A);
            G_(j, 0) = Real(1);
        }
    }

    void schedule_gain(const Mat<NX, NX>& A, const Mat<NY, NX>& C, const U& u) {
        if (have_design_ && obs::lin_change<NX, NY>(A_ref_, C_ref_, A, C) < opts.regain_tol) return;
        A_ref_ = A; C_ref_ = C;
        build_G(A);
        // augmented pair
        Mat<NA, NA> Aa = Mat<NA, NA>::identity();
        Aa.template set_block<NX, NX>(0, 0, A);
        Aa.template set_block<NX, NY>(0, NX, G_);
        Mat<NY, NA> Ca;
        Ca.template set_block<NY, NX>(0, 0, C);
        Vec<NA> L;
        bool ok = false;
        if (opts.design == Design::PolePlacement) {
            if constexpr (NY == 1) {
                Real p[NA];
                for (int i = 0; i < NX; ++i) p[i] = opts.pole[i];
                for (int i = 0; i < NY; ++i) p[NX + i] = opts.pole_i[i];
                ok = obs::place_gain<NA>(Aa, Ca, p, L);
                if (ok && L.norm_inf() > opts.max_gain) ok = false;
            }
        }
        // Certify the placement over the whole scheduling family before using it.
        // The augmented pair (Abar, Cbar) is even worse conditioned than (A, C)
        // -- det O ~ 1e-30 for the cell -- so an uncertified assignment here is
        // exactly what blew the integral state up on the cold scenario.
        const Real rho_max = Real(1) - opts.rho_margin;
        if (ok && !obs::gain_is_robust<NA, NY>(Aa, Ca, L, opts.robust_lo, opts.robust_hi, rho_max)) ok = false;
        if (opts.design == Design::Riccati || (!ok && opts.fallback_to_riccati)) {
            Mat<NA, NA> Qa;
            Qa.template set_block<NX, NX>(0, 0, m_.Q(x_, u) * opts.q_scale);
            for (int i = 0; i < NY; ++i) Qa(NX + i, NX + i) = sq(opts.q_d);
            const Mat<NY, NY> Ra = m_.R(x_, u) * opts.r_scale;
            if (!riccati_init_) { Pa_ = obs::dare_design<NA, NY>(Aa, Ca, Qa, Ra, opts.riccati_exact, opts.riccati_iters_init); riccati_init_ = true; }
            const Mat<NA, NY> K = obs::riccati_refine<NA, NY>(Aa, Ca, Qa, Ra, Pa_, opts.riccati_iters_refine);
            const Vec<NA> Lr = K.col(0);
            if (Lr.is_finite() && (obs::gain_is_robust<NA, NY>(Aa, Ca, Lr, opts.robust_lo, opts.robust_hi, rho_max) || !have_design_)) { L = Lr; ok = true; }
        }
        if (ok) {
            obs::blend_gain<NA>(La_, L, opts.gain_smooth, have_design_);
            for (int i = 0; i < NX; ++i) Lp_(i, 0) = La_[i];
            for (int i = 0; i < NY; ++i) Li_(i, 0) = La_[NX + i];
            have_design_ = true;
        }
    }

    Model m_;
    X x_;
    Y xi_, d_, residual_;
    Mat<NX, NY> G_, Lp_;
    Mat<NY, NY> Li_;
    Mat<NX, NX> A_ref_;
    Mat<NY, NX> C_ref_;
    Vec<NA> La_;
    Mat<NA, NA> Pa_;
    bool have_design_ = false, riccati_init_ = false;
};

}  // namespace estkit
