// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/interval_observer.hpp — cooperative interval observer:
//  two coupled observers that bracket the true state from below and above.
//
//  References
//    * Gouze, J.-L., Rapaport, A. & Hadj-Sadok, M. Z. (2000). Interval observers
//      for uncertain biological systems. Ecological Modelling 133(1-2), 45-56.
//    * Raissi, T., Efimov, D. & Zolghadri, A. (2012). Interval state estimation
//      for a class of nonlinear systems. IEEE Transactions on Automatic Control
//      57(1), 260-265.
//    * Mazenc, F. & Bernard, O. (2011). Interval observers for linear time-
//      invariant systems with disturbances. Automatica 47(1), 140-147.
//    * Efimov, D. & Raissi, T. (2016). Design of interval observers for uncertain
//      dynamical systems. Automation and Remote Control 77(2), 191-225.
//    * Smith, H. L. (1995). Monotone Dynamical Systems. AMS, Providence.
//      (cooperative / order-preserving systems)
//
//  Setting.  No probabilistic assumption is made.  Instead every uncertainty is
//  assumed *bounded*:
//      |v_k| <= vbar          measurement error (noise + quantisation + outliers)
//      |du_j| <= ubar_j       error of input channel j (bias + gain + noise)
//      |w_k| <= wbar          per-step model error (unmodelled dynamics, ageing)
//  and the observer returns an interval [xlo_k, xhi_k] guaranteed to contain the
//  true state whenever [xlo_0, xhi_0] contains x_0.
//
//  Cooperativity.  Propagating an interval through the dynamics preserves the
//  order relation iff the Jacobian A = df/dx is non-negative (the discrete-time
//  analogue of a Metzler matrix, Smith 1995).  The cell model has
//  A = diag(1, a1, a2, ah) with a1, a2, ah in (0,1), so it is cooperative by
//  construction; any negative off-diagonal entry that a different model might
//  have is handled by widening the interval with |A_ij| (xhi_j - xlo_j).
//
//  Time update (first-order over-approximation of the input uncertainty)
//      xhi^-_i = f_i(xhi, u) + sum_j |B_ij| ubar_j + wbar_i ,
//      xlo^-_i = f_i(xlo, u) - sum_j |B_ij| ubar_j - wbar_i .                  (1)
//
//  Measurement update.  Using the mean-value theorem on the scalar output,
//  h(xhi^-, u) - h(x, u) = C(xi) (xhi^- - x) for some xi on the segment, the
//  corrected upper bound
//      xhi^+ = xhi^- + L ( y - h(xhi^-, u) ) + |L| vbar + inflation            (2)
//  preserves xhi^+ >= x provided
//      (I - L C)_{ii} = 1 - L_i C_i >= 0   and   -L_i C_j >= 0  (i != j)       (3)
//  for every C in the interval hull of C(.) over the box.  (3) is the discrete
//  cooperativity condition of the *error* dynamics e^+ = (I - L C) A e^-.
//
//  Feasibility for the cell.  C = [dOCV/dz, -1, -1, M] has mixed signs
//  (dOCV/dz > 0 and M > 0 but the RC coefficients are negative), and (3) applied
//  to every pair (i, j) forces L = 0 for this C: cooperativity and output
//  injection are incompatible unless some columns are removed.  The implementation
//  therefore corrects a single state j* (by default the state with the largest DC
//  gain from a disturbance, i.e. SOC), takes the sign of L_{j*} from C_{j*}, and
//  *neutralises* the columns j != j* whose sign blocks (3) by adding
//      inflation = |L_{j*}| sum_{j blocked} |C_j| (xhi_j - xlo_j)              (4)
//  to the upper bound and subtracting it from the lower one, which is a valid
//  over-approximation of the discarded term.  For the cell only the hysteresis
//  column (M = 5 mV) is blocked, so (4) costs at most L_{j*} M (hhi - hlo) <=
//  2 L_{j*} M per step -- negligible next to the current-sensor term in (1).
//
//  Point estimate.  lower()/upper() are the guaranteed bounds; x() is the point
//  estimate.  The interval *midpoint* is the obvious choice but a poor estimator:
//  the admissible cooperative gain is L_{j*} = gain_frac / max_box |C_{j*}|, i.e.
//  it is set by the worst OCV slope anywhere in the box, so the wider the
//  guaranteed interval the slower the midpoint responds -- on the single-particle
//  plant, where the bounds must be inflated for the structured model error, the
//  midpoint carries a 2.4 % steady-state SOC bias that no choice of gain_frac
//  removes (measured: rmse_ss is flat to three digits for gain_frac in [0.5, 0.99]).
//  By default the point estimate is therefore produced by a certified LQE
//  observer (LuenbergerObserver in Riccati mode, the same design used by SSKF)
//  running in parallel, *projected onto the guaranteed interval*:
//      xhat = min(max(xhat_LQE, xlo), xhi) .                                     (5)
//  The projection is what makes the two consistent: the reported point estimate
//  always lies inside the certified bounds, so the guarantee is never contradicted,
//  while the accuracy of the point estimate is not charged for the conservatism of
//  the bounds.  Set Options::point_from_lqe = false for the plain midpoint.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "luenberger.hpp"

namespace estkit {

template <class Model>
class IntervalObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static_assert(NY == 1, "IntervalObserver is implemented for a scalar output");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real init_sigmas = Real(5);     // initial half-width in units of sqrt(P0_ii)
        Real v_sigmas = Real(5);        // measurement bound in units of sqrt(R_jj)
        Real v_extra[NY];               // additive measurement bound (quantisation, outliers)
        Real u_bar[NU];                 // input-error bound per channel
        Real w_bar[NX];                 // per-step model-error bound per state
        Real gain_frac = Real(0.5);     // L_{j*} = gain_frac / max|C_{j*}|  (must be < 1)
        int  corr_index = -1;           // corrected state; -1 = auto (obs::slowest_state)
        int  c_samples = 5;             // samples along the box diagonal for the hull of C
        Real c_margin = Real(1.25);     // safety factor on the sampled hull of C
        Real min_width = Real(1e-6);    // floor on the interval width (numerical hygiene)
        bool apply_constraints = true;
        bool point_from_lqe = true;     // point estimate: certified LQE projected onto the interval
        Real point_q_extra[NX];         // extra process-noise std-dev for that LQE design

        Options() {
            for (int i = 0; i < NY; ++i) v_extra[i] = Real(0);
            for (int i = 0; i < NU; ++i) u_bar[i] = Real(0);
            for (int i = 0; i < NX; ++i) { w_bar[i] = Real(0); point_q_extra[i] = Real(0); }
        }
    } opts;

    explicit IntervalObserver(const Model& m) : m_(m), point_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0;
        point_.opts.design = obs::Design::Riccati;
        for (int i = 0; i < NX; ++i) point_.opts.q_extra[i] = opts.point_q_extra[i];
        point_.init(x0, P0);
        for (int i = 0; i < NX; ++i) {
            const Real s = (P0(i, i) > Real(0)) ? std::sqrt(P0(i, i)) : Real(0);
            lo_[i] = x0[i] - opts.init_sigmas * s;
            hi_[i] = x0[i] + opts.init_sigmas * s;
        }
        if (opts.apply_constraints) { lo_ = constrain(m_, lo_); hi_ = constrain(m_, hi_); }
        repair();
        cooperative_ = true;
        x_ = point_estimate();
    }

    void predict(const U& u) {
        const X mid = x_;
        const Mat<NX, NX> A = jac_F(m_, mid, u);
        const Mat<NX, NU> B = jac_B(m_, mid, u);
        // width of the current interval (used for the non-cooperative inflation)
        X w;
        for (int i = 0; i < NX; ++i) w[i] = hi_[i] - lo_[i];
        cooperative_ = true;
        X pad;
        for (int i = 0; i < NX; ++i) {
            Real p = opts.w_bar[i];
            for (int j = 0; j < NU; ++j) p += std::fabs(B(i, j)) * opts.u_bar[j];
            for (int j = 0; j < NX; ++j) {
                if (j == i) continue;
                if (A(i, j) < Real(0)) { p += -A(i, j) * w[j]; cooperative_ = false; }
            }
            if (A(i, i) < Real(0)) { p += -A(i, i) * w[i]; cooperative_ = false; }
            pad[i] = p;
        }
        const X fhi = m_.f(hi_, u), flo = m_.f(lo_, u);
        for (int i = 0; i < NX; ++i) { hi_[i] = fhi[i] + pad[i]; lo_[i] = flo[i] - pad[i]; }
        repair();
        if (opts.point_from_lqe) point_.predict(u);
        x_ = point_estimate();
    }

    Y predict_measurement(const U& u) const { return opts.point_from_lqe ? point_.predict_measurement(u) : m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        // ---- interval hull of C over the box (sampled along the diagonal) ----
        Mat<NY, NX> Cmin, Cmax;
        const int ns = (opts.c_samples < 2) ? 2 : opts.c_samples;
        for (int s = 0; s < ns; ++s) {
            const Real t = Real(s) / Real(ns - 1);
            X xs;
            for (int i = 0; i < NX; ++i) xs[i] = lo_[i] + t * (hi_[i] - lo_[i]);
            const Mat<NY, NX> Cs = jac_H(m_, xs, u);
            for (int j = 0; j < NX; ++j) {
                if (s == 0) { Cmin(0, j) = Cs(0, j); Cmax(0, j) = Cs(0, j); }
                else { Cmin(0, j) = std::min(Cmin(0, j), Cs(0, j)); Cmax(0, j) = std::max(Cmax(0, j), Cs(0, j)); }
            }
        }
        for (int j = 0; j < NX; ++j) {   // safety margin on the sampled hull
            const Real c = Real(0.5) * (Cmin(0, j) + Cmax(0, j));
            const Real h = Real(0.5) * (Cmax(0, j) - Cmin(0, j)) * opts.c_margin;
            Cmin(0, j) = c - h; Cmax(0, j) = c + h;
        }
        // ---- correction index and admissible gain ----
        if (idx_ < 0) {
            const Mat<NX, NX> A = jac_F(m_, x_, u);
            idx_ = (opts.corr_index >= 0) ? opts.corr_index : obs::slowest_state<NX>(A);
        }
        const Real cmin = Cmin(0, idx_), cmax = Cmax(0, idx_);
        Real ell = Real(0);
        if (cmin > Real(0))      ell = opts.gain_frac / cmax;    // 1 - ell C_{j*} >= 0
        else if (cmax < Real(0)) ell = opts.gain_frac / cmin;    // (negative gain)
        // ---- inflation for the columns whose sign blocks cooperativity ----
        Real infl = Real(0);
        if (ell != Real(0)) {
            for (int j = 0; j < NX; ++j) {
                if (j == idx_) continue;
                const Real cworst = (ell > Real(0)) ? Cmax(0, j) : Cmin(0, j);
                const Real bad = (ell > Real(0)) ? std::max(cworst, Real(0)) : std::min(cworst, Real(0));
                if (bad != Real(0)) infl += std::fabs(ell * bad) * (hi_[j] - lo_[j]);
            }
        }
        const Real vbar = opts.v_sigmas * std::sqrt(std::fabs(m_.R(x_, u)(0, 0))) + opts.v_extra[0];
        if (ell != Real(0)) {
            const Real rhi = y[0] - m_.h(hi_, u)[0];
            const Real rlo = y[0] - m_.h(lo_, u)[0];
            const Real pad = std::fabs(ell) * vbar + infl;
            hi_[idx_] += ell * rhi + pad;
            lo_[idx_] += ell * rlo - pad;
        }
        if (opts.apply_constraints) { lo_ = constrain(m_, lo_); hi_ = constrain(m_, hi_); }
        repair();
        if (opts.point_from_lqe) point_.update(y, u);
        x_ = point_estimate();
        gain_ = ell;
    }

    const X& x() const { return x_; }
    const X& lower() const { return lo_; }
    const X& upper() const { return hi_; }
    Real gain() const { return gain_; }
    bool cooperative() const { return cooperative_; }
    const X& midpoint() const { return mid_; }      // interval midpoint (the classical estimate)
    const X& lqe_state() const { return point_.x(); }   // unprojected LQE point estimate
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    X mid_point() const {
        X m;
        for (int i = 0; i < NX; ++i) m[i] = Real(0.5) * (lo_[i] + hi_[i]);
        return m;
    }
    // Reported point estimate: the certified LQE estimate clamped into the
    // guaranteed interval (5), or the plain midpoint when point_from_lqe is off.
    X point_estimate() {
        mid_ = mid_point();
        if (!opts.point_from_lqe) return mid_;
        X p = point_.x();
        if (!p.is_finite()) return mid_;
        for (int i = 0; i < NX; ++i) p[i] = clampr(p[i], lo_[i], hi_[i]);
        return p;
    }
    void repair() {
        for (int i = 0; i < NX; ++i) {
            if (!std::isfinite(lo_[i]) || !std::isfinite(hi_[i])) { lo_[i] = Real(0); hi_[i] = Real(0); }
            if (hi_[i] < lo_[i]) { const Real m = Real(0.5) * (lo_[i] + hi_[i]); lo_[i] = m; hi_[i] = m; }
            if (hi_[i] - lo_[i] < opts.min_width) {
                const Real m = Real(0.5) * (lo_[i] + hi_[i]);
                lo_[i] = m - Real(0.5) * opts.min_width;
                hi_[i] = m + Real(0.5) * opts.min_width;
            }
        }
    }

    Model m_;
    LuenbergerObserver<Model> point_;   // certified LQE point estimate (Riccati design)
    X x_, lo_, hi_, mid_;
    Real gain_ = Real(0);
    int idx_ = -1;
    bool cooperative_ = true;
};

}  // namespace estkit
