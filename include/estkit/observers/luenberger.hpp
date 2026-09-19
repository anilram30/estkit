// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/luenberger.hpp — Luenberger state observer in
//  predictor-corrector ("current estimator") form, with gain scheduling on the
//  linearisation of the model, plus the shared gain-design utilities used by the
//  whole `observers` family.
//
//  References
//    * Luenberger, D. G. (1964). Observing the state of a linear system.
//      IEEE Transactions on Military Electronics 8(2), 74-80.
//    * Luenberger, D. G. (1971). An introduction to observers. IEEE Transactions
//      on Automatic Control 16(6), 596-602.
//    * Ackermann, J. (1972). Der Entwurf linearer Regelungssysteme im
//      Zustandsraum. Regelungstechnik und Prozess-Datenverarbeitung 20(7), 297-300.
//    * Ogata, K. (2010). Modern Control Engineering, 5th ed., Prentice Hall,
//      Section 10-6 (state observers, Ackermann's formula).
//    * Anderson, B. D. O. & Moore, J. B. (1979). Optimal Filtering, Prentice-Hall,
//      Section 4.4 (steady-state / algebraic Riccati solution -> LQE gain).
//    * Franklin, G. F., Powell, J. D. & Workman, M. L. (1998). Digital Control of
//      Dynamic Systems, 3rd ed., Addison-Wesley, Section 8.3 (prediction vs
//      current estimator, L = A^{-1} Lbar).
//
//  Algorithm (one sampling period, current-estimator form)
//      time update      x^- _k     = f(x^+_{k-1}, u_{k-1})
//      output residual  r_k        = y_k - h(x^-_k, u_k)
//      measurement upd. x^+_k      = x^-_k + L r_k
//
//  Error dynamics.  With e_k = x^+_k - x_k, A = df/dx, C = dh/dx evaluated at the
//  current estimate,
//      e_k = (I - L C) A e_{k-1} + (I - L C) w_{k-1} - L v_k ,
//  so the observer poles are the eigenvalues of (I - L C) A.  Because
//      eig((I - L C) A) = eig(A (I - L C)) = eig(A - (A L) C) ,
//  the *current-estimator* gain L and the classical *prediction* gain Lbar of
//  Luenberger (1971) are related by Lbar = A L.  The design therefore places the
//  poles of (A - Lbar C) with Ackermann's formula and returns L = A^{-1} Lbar.
//
//  Two designs are offered through Options::design
//    PolePlacement : Lbar from Ackermann's formula for the desired z-plane poles
//                    p_i = exp(-omega_i dt) (single-output models only).
//    Riccati       : L is the steady-state Kalman / LQE gain obtained from the
//                    discrete algebraic Riccati equation for (A, C, Q, R).  This
//                    is the "steady-state Kalman filter" (SSKF) registration.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

// -----------------------------------------------------------------------------
//  Shared design utilities for the deterministic-observer family.
// -----------------------------------------------------------------------------
namespace obs {

enum class Design { PolePlacement, Riccati };

// Spectral radius estimate rho(M) ~ ||M^(2^s)||^(1/2^s) by repeated squaring with
// renormalisation (robust to complex eigenvalue pairs, no eigen-solver needed).
// 16 steps (M^65536) are used by default because the closed-loop matrices of this
// family are strongly non-normal: a transient growth factor of 10^3 would bias a
// short-horizon estimate by e^(ln 10^3 / 2^s), which is 11 % at s = 6 but only
// 0.01 % at s = 16.
template <int N>
inline Real spectral_radius(Mat<N, N> M, int steps = 16) {
    Real acc = Real(0), w = Real(1);
    for (int s = 0; s < steps; ++s) {
        M = M * M;
        w *= Real(0.5);
        const Real c = M.norm_inf();
        if (!std::isfinite(c)) return std::numeric_limits<Real>::infinity();
        if (!(c > Real(0))) return Real(0);
        acc += w * std::log(c);
        M /= c;
    }
    return std::exp(acc);
}

// Pole placement, prediction form: returns Lbar with eig(A - Lbar C) = poles.
//
// Ackermann's formula is applied to the SHIFTED AND SCALED pair
//      A' = (A - sigma I) / gamma ,   p'_i = (p_i - sigma) / gamma ,
// which has the same solution because
//      A - (gamma Lbar') C = gamma (A' - Lbar' C) + sigma I
// has eigenvalues gamma p'_i + sigma = p_i, so Lbar = gamma Lbar'.
// The transformation is essential for finely sampled plants.  The rows of the
// observability matrix of (A, C) are C, C A, C A^2, ... ; when dt is small every
// eigenvalue of A is close to 1, those rows are nearly parallel and O is nearly
// rank deficient (det O ~ 1e-17 for the 2-RC cell model at 10 Hz, ~1e-26 for its
// disturbance-augmented version).  Shifting by sigma = mean(p_i) and scaling by
// gamma = max(1 - sigma, max_i |p_i - sigma|) maps the eigenvalues of A onto
// O(1)-separated values, and the achieved characteristic polynomial then matches
// the requested one to machine precision instead of 1e-4 (measured).
template <int N>
inline bool place_gain_pred(const Mat<N, N>& A, const RowVec<N>& C, const Real (&poles)[N], Vec<N>& Lbar) {
    Real sigma = Real(0);
    for (int i = 0; i < N; ++i) sigma += poles[i];
    sigma /= Real(N);
    Real gamma = std::fabs(Real(1) - sigma);
    for (int i = 0; i < N; ++i) gamma = std::max(gamma, std::fabs(poles[i] - sigma));
    if (!(gamma > Real(0))) gamma = Real(1);
    Mat<N, N> As = A;
    for (int i = 0; i < N; ++i) As(i, i) -= sigma;
    As /= gamma;
    Real ps[N], a[N];
    for (int i = 0; i < N; ++i) ps[i] = (poles[i] - sigma) / gamma;
    poly_from_roots<N>(ps, a);
    Vec<N> Lp;
    if (!ackermann_observer<N>(As, C, a, Lp)) return false;
    Lbar = Lp * gamma;
    return Lbar.is_finite();
}

// Pole placement for the current-estimator gain: returns L = A^{-1} Lbar so that
// eig((I - L C) A) = poles.  Single-output only (Ackermann's formula).
template <int N>
inline bool place_gain(const Mat<N, N>& A, const RowVec<N>& C, const Real (&poles)[N], Vec<N>& L) {
    Vec<N> Lbar;
    if (!place_gain_pred<N>(A, C, poles, Lbar)) return false;
    const Vec<N> Lc = solve(A, Lbar);
    if (!Lc.is_finite()) return false;
    L = Lc;
    return true;
}

// Warm-started fixed-point iteration of the discrete algebraic Riccati equation
//     P <- A (P - K C P) A^T + Q ,  K = P C^T (C P C^T + R)^{-1}
// (Anderson & Moore 1979 Sec. 4.4).  P is the *prior* covariance, so K is directly
// the current-estimator gain used as x^+ = x^- + K r.  `P` is updated in place,
// which lets a gain-scheduled observer re-converge in a handful of iterations
// after a small change of the linearisation instead of restarting from scratch.
template <int N, int M>
inline Mat<N, M> riccati_refine(const Mat<N, N>& A, const Mat<M, N>& C, const Mat<N, N>& Q,
                                const Mat<M, M>& R, Mat<N, N>& P, int iters) {
    for (int it = 0; it < iters; ++it) {
        const Mat<M, M> S = C * P * C.t() + R;
        const Mat<N, M> K = P * C.t() * inverse(S);
        Mat<N, N> Pn = A * (P - K * C * P) * A.t() + Q;
        if (!Pn.is_finite()) break;
        Pn.symmetrize();
        P = Pn;
    }
    const Mat<M, M> S = C * P * C.t() + R;
    return P * C.t() * inverse(S);
}

// Robust validation of a scheduled gain ("gain certification").
//
// Checking rho((I - L C) A) < 1 only at the *design* linearisation is not enough
// for a gain-scheduled observer: a pole-assignment design can be stabilising for
// the frozen pair and violently unstable a few seconds later, because the plant
// modes move with the operating point.  Two independent uncertainties matter and
// both are covered here by a small structured family:
//
//   (i) state-matrix scheduling,  A(s) = I + s (A - I) ,  s in [lo, hi] ,
//       i.e. the same eigenvalue *directions* with time constants scaled by 1/s.
//       For the cell model a_1, a_2 vary with temperature through the Arrhenius
//       law (+90 % on R0 at -10 C) and a_h varies with the current between hover
//       and rest, each by a factor of several.
//
//  (ii) output-map (loop-gain) scheduling,  C(g) = g C ,  g in [lo, hi].  The
//       dominant entry of C is the OCV slope dOCV/dz, which varies by a factor
//       of ~2.5 over one eVTOL mission (0.75 -> 1.8 V per unit SOC).  Because
//       (I - L (g C)) A = (I - (g L) C) A this is simultaneously a *gain-margin*
//       test: it rejects designs that are stable only at the nominal loop gain.
//
// The gain must achieve a guaranteed *decay rate*, not mere stability:
// rho <= rho_max with rho_max < 1.  A design whose certified pole sits at
// 1 - 1e-5 is formally stable but leaves the SOC channel effectively in open
// loop for the length of a mission, which is exactly the failure mode observed
// when an ill-conditioned single-output placement is accepted (see the chapter).
template <int N, int M>
inline bool gain_is_robust(const Mat<N, N>& A, const Mat<M, N>& C, const Vec<N>& L,
                           Real lo, Real hi, Real rho_max) {
    const Mat<N, N> I = Mat<N, N>::identity();
    const Real ss[3] = {Real(1), lo, hi};
    const Real gs[3] = {Real(1), lo, hi};
    for (int a = 0; a < 3; ++a) {
        const Mat<N, N> As = I + (A - I) * ss[a];
        for (int g = 0; g < 3; ++g) {
            const Mat<N, N> ILC = I - (L * gs[g]) * C;
            if (!(spectral_radius<N>(ILC * As) < rho_max)) return false;
        }
    }
    return true;
}

// Steady-state LQE (DARE) design used by every observer of this family.
//   exact = true  -> structure-preserving doubling algorithm, quadratic
//                    convergence, the true stabilising solution;
//   exact = false -> plain fixed-point recursion truncated at `iters`, i.e. the
//                    finite-horizon gain after `iters` samples (a deliberately
//                    "soft" start-up design).
// The doubling solution is the default because the fixed-point recursion needs
// O(1e4...1e5) sweeps on a 10 Hz battery model (its rate is rho((I-KC)A)^2 ~
// 0.9994 per sweep) and a truncated run can return a *wrong-signed* gain.
template <int N, int M>
inline Mat<N, N> dare_design(const Mat<N, N>& A, const Mat<M, N>& C, const Mat<N, N>& Q,
                             const Mat<M, M>& R, bool exact, int iters) {
    if (exact) return dare_doubling<N, M>(A, C, Q, R, nullptr, 120, Real(1e-14));
    return dare_iterate<N, M>(A, C, Q, R, nullptr, iters, Real(1e-14));
}

// Exponential blending of a re-designed gain into the running one.
//
// Gain scheduling is only legitimate when the schedule varies slowly compared
// with the closed-loop dynamics (Shamma & Athans 1990).  For the cell model the
// frozen-time design is *not* smooth in the operating point: the hysteresis
// eigenvalue a_h = exp(-|eta i gamma dt / (3600 Q)|) tends to 1 as the current
// tends to zero, where it collides with the SOC integrator, so the gain that
// assigns a prescribed pole set changes by orders of magnitude (and flips sign
// on the weakly observable channels) between hover and rest.  Feeding that
// directly into the observer produces exactly the "fast scheduling" instability
// the theory warns about.  Blending with beta << 1 restores the time-scale
// separation at the cost of a lag in the schedule.
template <int N>
inline void blend_gain(Vec<N>& L, const Vec<N>& Lnew, Real beta, bool have) {
    if (!have || !(beta < Real(1))) { L = Lnew; return; }
    for (int i = 0; i < N; ++i) L[i] += beta * (Lnew[i] - L[i]);
}

// Relative change of the linearisation, used as the gain-scheduling trigger.
template <int N, int M>
inline Real lin_change(const Mat<N, N>& A0, const Mat<M, N>& C0, const Mat<N, N>& A1, const Mat<M, N>& C1) {
    const Real d = (A1 - A0).norm_inf() + (C1 - C0).norm_inf();
    return d / (Real(1) + A1.norm_inf() + C1.norm_inf());
}

// Index of the state direction with the largest DC gain from a constant state
// disturbance, i.e. argmax_j || (I - A + mu I)^{-1} e_j ||_1.  Used by the
// integral / disturbance observers of this family to place the augmentation on
// the state that cannot correct itself (for the cell model this is the SOC
// integrator, whose open-loop eigenvalue is exactly one).  Fully model-agnostic.
template <int N>
inline int slowest_state(const Mat<N, N>& A, Real mu = Real(1e-6)) {
    Mat<N, N> Mreg = Mat<N, N>::identity() - A;
    for (int i = 0; i < N; ++i) Mreg(i, i) += mu;
    const Mat<N, N> G = inverse(Mreg);
    if (!G.is_finite()) return 0;
    int best = 0;
    Real bestv = Real(-1);
    for (int j = 0; j < N; ++j) {
        Real s = Real(0);
        for (int i = 0; i < N; ++i) s += std::fabs(G(i, j));
        if (s > bestv) { bestv = s; best = j; }
    }
    return best;
}

}  // namespace obs

// -----------------------------------------------------------------------------
//  LuenbergerObserver
// -----------------------------------------------------------------------------
template <class Model>
class LuenbergerObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    using Design = obs::Design;

    struct Options {
        Design design = Design::PolePlacement;
        // Desired closed-loop observer poles, given as z-plane radii
        //   p_i = exp(-omega_i dt) = exp(-dt / tau_i).
        // Default: a repeated ("binomial") pole set at tau = 40 s for dt = 0.1 s.
        // A repeated set is used deliberately: the 2-RC cell model sampled at
        // 10 Hz has three eigenvalues within 2 % of each other, so demanding a
        // *spread* pole set costs gains of order 10^2-10^3 (see the chapter).
        Real pole[NX];
        Real q_scale = Real(1);        // multiplies Model::Q in the Riccati design
        Real r_scale = Real(1);        // multiplies Model::R in the Riccati design
        Real q_extra[NX];              // extra process-noise std-dev per state (Riccati design only)
        Real regain_tol = Real(1e-3);  // relative change of (A,C) that triggers a re-design
        bool riccati_exact = true;         // exact DARE (doubling) instead of the truncated recursion
        int  riccati_iters_init = 4000;    // DARE sweeps for the first design when !riccati_exact
        int  riccati_iters_refine = 40;    // warm-start iterations on re-scheduling
        bool apply_constraints = true;     // project with Model::constrain after the update
        bool fallback_to_riccati = true;   // use the LQE gain if pole placement fails/destabilises
        Real gain_smooth = Real(1);        // exponential blending of a re-designed gain (1 = replace)
        Real robust_lo = Real(0.8), robust_hi = Real(1.25);  // scheduling-uncertainty range for the gain check
        // Certified decay rate: a design is accepted only if rho <= 1 - rho_margin
        // over the whole scheduling family (see obs::gain_is_robust).  1e-4 at
        // dt = 0.1 s corresponds to a guaranteed closed-loop time constant of
        // 1000 s, i.e. no slower than half a mission.
        Real rho_margin = Real(1e-4);
        Real max_gain = Real(1e4);         // reject a design whose ||L||_inf exceeds this

        Options() {
            for (int i = 0; i < NX; ++i) pole[i] = Real(0.9975);
            for (int i = 0; i < NX; ++i) q_extra[i] = Real(0);
        }
        // convenience: all poles at exp(-dt/tau)
        void set_poles_tau(Real tau, Real dt) {
            const Real p = std::exp(-dt / tau);
            for (int i = 0; i < NX; ++i) pole[i] = p;
        }
    } opts;

    explicit LuenbergerObserver(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0;
        P_ = P0;                 // only used to warm-start the Riccati design
        have_design_ = false; riccati_init_ = false;
        L_ = Vec<NX>();
        n_design_ = 0; n_placed_ = 0;
    }

    void predict(const U& u) { x_ = m_.f(x_, u); }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NX, NX> A = jac_F(m_, x_, u);
        const Mat<NY, NX> C = jac_H(m_, x_, u);
        schedule_gain(A, C, u);
        const Y r = y - m_.h(x_, u);
        residual_ = r;
        const X dx = L_ * r;
        if (dx.is_finite()) x_ = x_ + dx;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Vec<NX>& gain() const { return L_; }
    const Y& residual() const { return residual_; }
    int redesign_count() const { return n_design_; }
    int placement_count() const { return n_placed_; }   // designs accepted from pole placement
    Model& model() { return m_; }
    const Model& model() const { return m_; }

protected:
    // Re-design L if the linearisation moved by more than opts.regain_tol.
    void schedule_gain(const Mat<NX, NX>& A, const Mat<NY, NX>& C, const U& u) {
        if (have_design_ && obs::lin_change<NX, NY>(A_ref_, C_ref_, A, C) < opts.regain_tol) return;
        A_ref_ = A; C_ref_ = C;
        Vec<NX> L;
        bool ok = false, placed = false;
        if (opts.design == Design::PolePlacement) {
            if constexpr (NY == 1) {
                ok = obs::place_gain<NX>(A, C, opts.pole, L);
                if (ok && L.norm_inf() > opts.max_gain) ok = false;
                placed = ok;
            }
        }
        const Real rho_max = Real(1) - opts.rho_margin;
        // Certify the placement before accepting it.  On a finely sampled plant
        // the single-output assignment problem is severely ill-conditioned
        // (cond(O) ~ 1e14 for the 2-RC cell at 10 Hz), so the returned gain may
        // be stabilising at the frozen pair yet unstable a few seconds later.
        if (ok && !obs::gain_is_robust<NX, NY>(A, C, L, opts.robust_lo, opts.robust_hi, rho_max)) { ok = false; placed = false; }
        // LQE fallback.  This must be available at *every* re-design, not only
        // on the first one: keeping a stale certified gain while the operating
        // point moves is what let the scheduled observer drift on `combined`.
        if (opts.design == Design::Riccati || (!ok && opts.fallback_to_riccati)) {
            Mat<NX, NX> Qd = m_.Q(x_, u) * opts.q_scale;
            for (int i = 0; i < NX; ++i) Qd(i, i) += sq(opts.q_extra[i]);
            const Mat<NY, NY> Rd = m_.R(x_, u) * opts.r_scale;
            if (!riccati_init_) { P_ = obs::dare_design<NX, NY>(A, C, Qd, Rd, opts.riccati_exact, opts.riccati_iters_init); riccati_init_ = true; }
            const Mat<NX, NY> K = obs::riccati_refine<NX, NY>(A, C, Qd, Rd, P_, opts.riccati_iters_refine);
            const Vec<NX> Lr = K.col(0);
            if (Lr.is_finite() && obs::gain_is_robust<NX, NY>(A, C, Lr, opts.robust_lo, opts.robust_hi, rho_max)) { L = Lr; ok = true; }
            else if (Lr.is_finite() && !have_design_) { L = Lr; ok = true; }   // nothing better exists yet
        }
        if (ok) { obs::blend_gain<NX>(L_, L, opts.gain_smooth, have_design_); have_design_ = true; ++n_design_; if (placed) ++n_placed_; }
    }

    Model m_;
    X x_;
    Vec<NX> L_;
    Y residual_;
    Mat<NX, NX> A_ref_;
    Mat<NY, NX> C_ref_;
    Mat<NX, NX> P_;          // Riccati working covariance (warm start)
    bool have_design_ = false, riccati_init_ = false;
    int n_design_ = 0, n_placed_ = 0;
};

}  // namespace estkit
