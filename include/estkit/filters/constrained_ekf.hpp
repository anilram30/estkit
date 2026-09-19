// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/constrained_ekf.hpp — EKF with state constraints enforced by
//  ESTIMATE PROJECTION (active-set solution of the projection QP).
//
//  References
//    * Simon, D. (2010). Kalman filtering with state constraints: a survey of
//      linear and nonlinear algorithms. IET Control Theory & Applications 4(8),
//      1303-1318, Section 2 ("estimate projection").
//    * Simon, D. & Chia, T. L. (2002). Kalman filtering with state equality
//      constraints. IEEE Trans. Aerospace and Electronic Systems 38(1), 128-136.
//    * Nocedal, J. & Wright, S. J. (2006). Numerical Optimization, 2nd ed.,
//      Springer, Ch. 16 (active-set method for inequality-constrained QPs).
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, Ch. 8.                                (EKF linearisation)
//
//  Method -----------------------------------------------------------------
//  After the ordinary EKF measurement update, the estimate is projected onto the
//  feasible set by solving
//
//      x_p = argmin_x (x - xhat)^T W (x - xhat)   s.t.  lo <= x <= hi,        (1)
//
//  with W = (P^+)^{-1}, which is the maximum-probability choice: for a Gaussian
//  posterior, (1) with W = P^{-1} returns the constrained maximum-likelihood
//  estimate (Simon 2010, §2.1) and it is unbiased-optimal among projections.
//
//  For a fixed ACTIVE SET A = {i_1..i_m} of bounds treated as equalities
//  D x = d (D = rows e_{i_a}^T, d_a the violated bound), the KKT system of (1)
//  has the closed-form solution
//
//      x_p = xhat - W^{-1} D^T (D W^{-1} D^T)^{-1} (D xhat - d)
//          = xhat - P D^T (D P D^T)^{-1} (D xhat - d)  =  xhat - sum_a mu_a P e_{i_a},  (2)
//      mu  = (D P D^T)^{-1} (D xhat - d)                       (Lagrange multipliers).
//
//  Because W = P^{-1}, (2) only needs P: NO inversion of P is required.  Note
//  that (2) moves ALL states, not only the violated one: a clipped SOC drags
//  v_1, v_2 and h along through the correlations in P - this is precisely the
//  "covariance-consistent correction of the other states" that distinguishes
//  projection from naive clipping.  For a diagonal P, (2) degenerates to
//  clipping.
//
//  The active set is found by the standard primal active-set loop: repeatedly
//  (i) drop an active bound whose multiplier has the wrong sign (mu_a < 0 for an
//  upper bound, mu_a > 0 for a lower bound - such a constraint is pushing the
//  estimate the wrong way), (ii) otherwise add the most violated inactive bound.
//  With n_x box constraints the loop terminates in at most O(n_x) exchanges; it
//  is hard-capped at 2 n_x + 4 iterations so the execution time stays bounded.
//
//  The covariance is by default left unchanged, as recommended in Simon (2010,
//  §2.1): treating (2) as a perfect (noise-free) pseudo-measurement would make
//  P singular in the constrained direction and the filter would lock up.  The
//  deflation P_p = P - P D^T (D P D^T)^{-1} D P is available through
//  opts.deflate_covariance for the readers who want the "perfect measurement"
//  interpretation.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class ConstrainedEkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    // Box constraints; a non-finite entry (kNaN) means "unconstrained".
    struct Constraints {
        Real lo[NX], hi[NX];
        Constraints() { for (int k = 0; k < NX; ++k) { lo[k] = kNaN; hi[k] = kNaN; } }
        void set(int k, Real l, Real h) { lo[k] = l; hi[k] = h; }
    };

    struct Options {
        Constraints con;
        bool joseph = true;                 // Joseph-stabilised covariance update
        bool project_after_update = true;
        bool project_after_predict = true;
        bool deflate_covariance = false;    // P <- P - P D^T (D P D^T)^{-1} D P
        bool use_model_constrain = false;   // additionally apply Model::constrain (clipping)
        int  max_exchanges = 2 * NX + 4;
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    explicit ConstrainedEkf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; n_active_ = 0; }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        P_ = F * P_ * F.t() + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
        if (opts.project_after_predict) project();
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> R = m_.R(x_, u) * opts.r_scale;
        Mat<NY, NY> S = H * P_ * H.t() + R; S.symmetrize();
        const Mat<NY, NY> Sinv = inverse(S);
        if (!Sinv.is_finite()) return;
        const Mat<NX, NY> K = P_ * H.t() * Sinv;
        if (!K.is_finite()) return;
        innovation_ = y - m_.h(x_, u); S_ = S; K_ = K;
        x_ = x_ + K * innovation_;
        if (opts.joseph) {
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_ = IKH * P_ * IKH.t() + K * R * K.t();
        } else {
            P_ = (Mat<NX, NX>::identity() - K * H) * P_;
        }
        P_.symmetrize();
        if (opts.use_model_constrain) x_ = constrain(m_, x_);
        if (opts.project_after_update) project();
    }

    // Estimate projection (1)-(2).  Returns the number of active constraints.
    int project() {
        int idx[NX] = {};        // active coordinates
        Real bnd[NX] = {};       // the bound each of them is held at
        bool upper[NX] = {};     // true if it is an upper bound
        int na = 0;
        X xp = x_;
        Vec<NX> mu;         // multipliers of the active constraints (padded with 0)
        Mat<NX, NX> G;      // padded  D P D^T
        for (int pass = 0; pass < opts.max_exchanges; ++pass) {
            // ---- solve the equality-constrained projection for the current set
            G = Mat<NX, NX>::identity();
            Vec<NX> r;
            for (int a = 0; a < na; ++a) {
                for (int b = 0; b < na; ++b) G(a, b) = P_(idx[a], idx[b]);
                for (int b = na; b < NX; ++b) { G(a, b) = Real(0); G(b, a) = Real(0); }
                r[a] = x_[idx[a]] - bnd[a];
            }
            for (int a = na; a < NX; ++a) r[a] = Real(0);
            mu = (na > 0) ? solve_spd(G, r) : Vec<NX>();
            if (!mu.is_finite()) { na = 0; xp = x_; break; }
            xp = x_;
            for (int a = 0; a < na; ++a) xp = xp - P_.col(idx[a]) * mu[a];
            // ---- drop a constraint whose multiplier has the wrong sign
            int drop = -1; Real worst = Real(0);
            for (int a = 0; a < na; ++a) {
                const Real s = upper[a] ? mu[a] : -mu[a];   // feasibility of the KKT sign
                if (s < worst) { worst = s; drop = a; }
            }
            if (drop >= 0) {
                for (int a = drop; a + 1 < na; ++a) { idx[a] = idx[a + 1]; bnd[a] = bnd[a + 1]; upper[a] = upper[a + 1]; }
                --na;
                continue;
            }
            // ---- add the most violated inactive bound
            int add = -1; Real viol = Real(0); Real addb = Real(0); bool addu = false;
            for (int k = 0; k < NX; ++k) {
                bool in_set = false;
                for (int a = 0; a < na; ++a) if (idx[a] == k) { in_set = true; break; }
                if (in_set) continue;
                if (std::isfinite(opts.con.lo[k]) && xp[k] < opts.con.lo[k]) {
                    const Real v = opts.con.lo[k] - xp[k];
                    if (v > viol) { viol = v; add = k; addb = opts.con.lo[k]; addu = false; }
                } else if (std::isfinite(opts.con.hi[k]) && xp[k] > opts.con.hi[k]) {
                    const Real v = xp[k] - opts.con.hi[k];
                    if (v > viol) { viol = v; add = k; addb = opts.con.hi[k]; addu = true; }
                }
            }
            if (add < 0) break;                         // KKT satisfied: done
            idx[na] = add; bnd[na] = addb; upper[na] = addu; ++na;
        }
        if (opts.deflate_covariance && na > 0) {
            Mat<NX, NX> Gi = inverse(G);
            if (Gi.is_finite()) {
                Mat<NX, NX> dP;
                for (int a = 0; a < na; ++a)
                    for (int b = 0; b < na; ++b) dP += outer(P_.col(idx[a]), P_.col(idx[b])) * Gi(a, b);
                P_ = P_ - dP; P_.symmetrize();
            }
        }
        // Final safety clip: absorbs round-off left by the QP (never more than ~1e-12).
        for (int k = 0; k < NX; ++k) {
            if (std::isfinite(opts.con.lo[k]) && xp[k] < opts.con.lo[k]) xp[k] = opts.con.lo[k];
            if (std::isfinite(opts.con.hi[k]) && xp[k] > opts.con.hi[k]) xp[k] = opts.con.hi[k];
        }
        if (xp.is_finite()) x_ = xp;
        n_active_ = na;
        return na;
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    int active_constraints() const { return n_active_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    int n_active_ = 0;
};

}  // namespace estkit
