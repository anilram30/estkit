// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/hinf.hpp — discrete-time extended H-infinity (minimax) filter
//
//  References
//    * Simon, D. (2006). Optimal State Estimation: Kalman, H-infinity, and
//      Nonlinear Approaches. Wiley, Hoboken, NJ, Ch. 11 (game-theoretic
//      H-infinity filter, Eqs. (11.72)-(11.76)).
//    * Shen, X. & Deng, L. (1997). Game theory approach to discrete H-infinity
//      filter design. IEEE Trans. Signal Processing 45(4), 1092-1095.
//    * Hassibi, B., Sayed, A. H. & Kailath, T. (1996). Linear estimation in
//      Krein spaces - Part II: Applications. IEEE Trans. Autom. Control 41(1),
//      34-49.
//    * Kailath, T., Sayed, A. H. & Hassibi, B. (2000). Linear Estimation.
//      Prentice Hall, Ch. 11-12.
//
//  Problem ----------------------------------------------------------------
//  With  x_{k+1} = f(x_k,u_k) + w_k,  y_k = h(x_k,u_k) + v_k  and the quantity
//  to be estimated  z_k = L_k x_k, the H-infinity filter makes the worst-case
//  energy gain from the disturbances (x_0 - x_0hat, w, v) to the estimation
//  error z - zhat smaller than 1/theta:
//
//      sup  sum_k ||z_k - zhat_k||^2_{S_k}
//      ---------------------------------------------------------------  <  1/theta
//      ||x_0-x_0hat||^2_{P_0^{-1}} + sum_k (||w_k||^2_{Q^{-1}} + ||v_k||^2_{R^{-1}})
//
//  The saddle point of the associated dynamic game (Simon 2006 §11.3) gives the
//  recursion (with Sbar_k = L_k^T S_k L_k; here L = I so Sbar = S):
//
//      K_k   = P_k [ I - theta Sbar_k P_k + H_k^T R_k^{-1} H_k P_k ]^{-1} H_k^T R_k^{-1}
//      xhat_{k+1} = F_k xhat_k + F_k K_k (y_k - h(xhat_k,u_k))
//      P_{k+1}    = F_k P_k [ I - theta Sbar_k P_k + H_k^T R_k^{-1} H_k P_k ]^{-1} F_k^T + Q_k
//
//  subject to the feasibility (existence) condition
//
//      W_k = P_k^{-1} - theta Sbar_k + H_k^T R_k^{-1} H_k  >  0.                (*)
//
//  Implementation form ----------------------------------------------------
//  Splitting the recursion into the predict/update convention of this library
//  and using  P_k [I + (H^T R^{-1} H - theta Sbar) P_k]^{-1} = W_k^{-1}
//  (Sherman-Morrison-Woodbury / information form) gives the numerically direct
//
//      update:   W = P^- ^{-1} - theta Sbar + H^T R^{-1} H       (check W > 0)
//                P^+ = W^{-1},   K = P^+ H^T R^{-1}
//                x^+ = x^- + K (y - h(x^-,u))
//      predict:  x^- = f(x^+,u),   P^- = F P^+ F^T + Q
//
//  theta = 0 recovers exactly the (information-form) extended Kalman filter.
//  Condition (*) is verified at every step by a Cholesky factorisation of
//  W - eps P^{-1} (eps = opts.feas_margin, so that P^+ <= P^-/eps is enforced
//  with a margin); if it fails, theta is halved until the test passes, and
//  theta = 0 is the guaranteed fall-back.  The number of back-offs is counted
//  and exposed through backoff_count() for diagnostics.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class HinfFilter {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        // Performance bound.  theta has units of (state unit)^-2: it is compared
        // against P^{-1} + H^T R^{-1} H, so it must be chosen per application
        // (see the chapter for the cell model).  theta = 0 gives the EKF.
        Real theta = Real(30);
        // Diagonal of S in Sbar = L^T S L with L = I (which states are penalised).
        Real s_diag[NX];
        // Require W >= feas_margin * P^{-1}, i.e. P^+ <= P^-/feas_margin.  This is
        // a strict form of the existence condition (*) with a safety margin.
        Real feas_margin = Real(0.9);
        Real backoff = Real(0.5);     // factor applied to theta when (*) fails
        int  max_backoff = 40;
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
        Options() { for (int k = 0; k < NX; ++k) s_diag[k] = Real(1); }
    } opts;

    explicit HinfFilter(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0; theta_used_ = opts.theta; backoff_count_ = 0;
    }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        P_ = F * P_ * F.t() + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> R = m_.R(x_, u) * opts.r_scale;
        const Mat<NY, NY> Rinv = inverse(R);
        innovation_ = y - m_.h(x_, u);
        Mat<NY, NY> Sinn = H * P_ * H.t() + R; Sinn.symmetrize(); S_ = Sinn;

        Mat<NX, NX> Pinv = inverse(P_);
        if (!Pinv.is_finite() || !Rinv.is_finite()) {   // singular prior: plain Joseph EKF step
            const Mat<NX, NY> K = P_ * H.t() * inverse(Sinn);
            if (!K.is_finite()) return;
            x_ = x_ + K * innovation_; K_ = K;
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_ = IKH * P_ * IKH.t() + K * R * K.t(); P_.symmetrize();
            if (opts.apply_constraints) x_ = constrain(m_, x_);
            return;
        }
        Pinv.symmetrize();
        const Mat<NX, NX> HtRiH = H.t() * Rinv * H;
        Mat<NX, NX> Sbar;
        for (int k = 0; k < NX; ++k) Sbar(k, k) = opts.s_diag[k];

        Real th = opts.theta;
        Mat<NX, NX> Pp;
        bool ok = false;
        for (int t = 0; t < opts.max_backoff && th > Real(0); ++t) {
            Mat<NX, NX> W = Pinv - Sbar * th + HtRiH; W.symmetrize();
            Mat<NX, NX> test = W - Pinv * opts.feas_margin; test.symmetrize();
            Mat<NX, NX> L;
            if (cholesky(test, L)) {
                Pp = solve_spd(W, Mat<NX, NX>::identity());
                if (Pp.is_finite()) { ok = true; break; }
            }
            th *= opts.backoff;
            ++backoff_count_;
        }
        if (!ok) {                                   // theta = 0 : Kalman (always feasible)
            th = Real(0);
            Mat<NX, NX> W = Pinv + HtRiH; W.symmetrize();
            Pp = solve_spd(W, Mat<NX, NX>::identity());
            if (!Pp.is_finite()) return;
        }
        theta_used_ = th;
        Pp.symmetrize();
        const Mat<NX, NY> K = Pp * H.t() * Rinv;
        if (!K.is_finite()) return;
        K_ = K;
        x_ = x_ + K * innovation_;
        P_ = Pp;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    Real theta_used() const { return theta_used_; }     // theta actually applied at the last step
    long backoff_count() const { return backoff_count_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    Real theta_used_ = Real(0);
    long backoff_count_ = 0;
};

}  // namespace estkit
