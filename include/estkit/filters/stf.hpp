// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/stf.hpp — Strong Tracking Filter (suboptimal fading factor)
//
//  References
//    * Zhou, D. H. & Frank, P. M. (1996). Strong tracking filtering of nonlinear
//      time-varying stochastic systems with coloured noise: application to
//      parameter estimation and empirical robustness analysis. International
//      Journal of Control 65(2), 295-307.
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, New York, Ch. 8.                  (EKF this builds on)
//    * Simon, D. (2006). Optimal State Estimation: Kalman, H-infinity, and
//      Nonlinear Approaches. Wiley, Hoboken, NJ, Sec. 7.4. (covariance inflation)
//
//  Idea.  An EKF whose model is wrong produces innovations that are no longer
//  white: they stay correlated because the covariance P has already collapsed
//  and the gain is too small to react.  Zhou & Frank impose, in addition to the
//  minimum-variance condition, the orthogonality condition
//
//      E[ e_{k+j} e_k^T ] = 0,   j = 1, 2, ...                             (1)
//
//  on the innovation sequence e_k = y_k - h(x^-_k, u_k), and satisfy it
//  (suboptimally, in the trace sense) by scaling the propagated part of the
//  prediction covariance with a scalar fading factor lambda_k >= 1:
//
//      P^-_k = lambda_k F_{k-1} P^+_{k-1} F_{k-1}^T + Q_{k-1}.             (2)
//
//  Requiring the actual innovation covariance to equal the theoretical one,
//  E[e_k e_k^T] = H_k P^-_k H_k^T + R_k, and substituting (2) gives
//
//      lambda_k = max( 1, tr(N_k)/tr(M_k) ),                               (3)
//      N_k = V_k - H_k Q_{k-1} H_k^T - beta R_k,                           (4)
//      M_k = H_k F_{k-1} P^+_{k-1} F_{k-1}^T H_k^T,                        (5)
//
//  where beta >= 1 is a softening factor (beta = 4.5 in the original paper) that
//  makes the estimate smoother, and V_k is a recursive estimate of the actual
//  innovation covariance with forgetting factor rho in (0,1]:
//
//      V_0 = e_0 e_0^T,   V_k = (rho V_{k-1} + e_k e_k^T)/(1 + rho).       (6)
//
//  Implementation note on the timing convention.  lambda_k needs e_k, which is
//  only available after y_k arrives, i.e. inside update().  predict() therefore
//  stores F P^+ F^T and Q, and update() re-forms P^- with (2) before computing
//  the gain; this is algebraically identical to the original formulation and
//  costs one extra n_x-by-n_x add.
//
//  Covariance limiting.  The scalar lambda_k of (3) inflates EVERY direction of
//  the state space, including directions the measurement barely constrains: for
//  those, M_k in (5) does not grow with P, so the negative feedback that keeps
//  lambda_k bounded is absent and P diverges geometrically.  The prediction
//  covariance is therefore bounded by
//
//      P^-_ii <= p_max_scale * P_0,ii ,                                     (7)
//
//  with the off-diagonal entries of the affected row/column rescaled by the
//  same square-root factor so that the correlation structure and positive
//  semi-definiteness are preserved: no state's predicted standard deviation may
//  exceed sqrt(p_max_scale) times its value at initialisation (10 % by
//  default).  This is the classical remedy for the
//  divergence of covariance-inflating filters (Anderson & Moore 1979, Sec. 4.4;
//  Simon 2006, Sec. 7.4).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class StrongTrackingFilter {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real beta = Real(4.5);        // softening factor (Zhou & Frank 1996)
        Real rho = Real(0.95);        // forgetting factor of the innovation-covariance recursion
        Real lambda_max = Real(1e4);  // numerical cap on the fading factor
        Real p_max_scale = Real(0.01);  // ceiling P^-_ii <= p_max_scale*P0_ii (sigma <= 10 % of sigma_0)
        bool enable = true;
        bool joseph = true;
        bool apply_constraints = true;
        Real q_scale = Real(1);
        Real r_scale = Real(1);
    } opts;

    explicit StrongTrackingFilter(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0;
        for (int i = 0; i < NX; ++i) p_max_[i] = P0(i, i);
        FPFt_ = Mat<NX, NX>(); Q_ = Mat<NX, NX>(); have_pred_ = false;
        V_ = Mat<NY, NY>(); have_V_ = false; lambda_ = Real(1);
        innovation_ = Y(); S_ = Mat<NY, NY>(); K_ = Mat<NX, NY>();
    }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        FPFt_ = F * P_ * F.t();
        Q_ = m_.Q(x_, u) * opts.q_scale;
        P_ = FPFt_ + Q_;               // provisional (lambda = 1); refined in update()
        P_.symmetrize();
        limit_covariance(P_, p_max_, opts.p_max_scale);
        have_pred_ = true;
    }

    // P_ii <- min(P_ii, s*pmax_i), row/column rescaled by sqrt so that the
    // correlation coefficients (and positive semi-definiteness) are preserved.
    static void limit_covariance(Mat<NX, NX>& P, const Real (&pmax)[NX], Real s) {
        if (!(s > Real(0))) return;
        for (int i = 0; i < NX; ++i) {
            const Real lim = s * pmax[i];
            if (P(i, i) > lim && P(i, i) > Real(0)) {
                const Real f = std::sqrt(lim / P(i, i));
                for (int j = 0; j < NX; ++j) { P(i, j) *= f; P(j, i) *= f; }
                P(i, i) = lim;
            }
        }
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> R = m_.R(x_, u) * opts.r_scale;
        const Y e = y - m_.h(x_, u);                    // a-priori innovation e_k

        // ---- (6) recursive estimate of the actual innovation covariance ----
        if (!have_V_) { V_ = outer(e, e); have_V_ = true; }
        else {
            const Real rho = clampr(opts.rho, Real(0), Real(1));
            V_ = (V_ * rho + outer(e, e)) / (Real(1) + rho);
        }

        // ---- (3)-(5) suboptimal fading factor ------------------------------
        lambda_ = Real(1);
        if (opts.enable && have_pred_) {
            const Mat<NY, NY> N = V_ - H * Q_ * H.t() - R * opts.beta;
            const Mat<NY, NY> M = H * FPFt_ * H.t();
            const Real trN = N.trace(), trM = M.trace();
            if (trM > Real(1e-300) && trN > trM) {
                const Real lam = trN / trM;
                lambda_ = (std::isfinite(lam)) ? clampr(lam, Real(1), opts.lambda_max) : Real(1);
            }
            if (lambda_ > Real(1)) {
                P_ = FPFt_ * lambda_ + Q_;
                P_.symmetrize();
                limit_covariance(P_, p_max_, opts.p_max_scale);   // eq. (7)
            }
        }

        // ---- ordinary EKF measurement update -------------------------------
        const Mat<NX, NX> Pminus = P_;
        const Mat<NY, NY> S = H * Pminus * H.t() + R;
        const Mat<NX, NY> K = Pminus * H.t() * inverse(S);
        if (!K.is_finite()) return;                     // keep the previous estimate
        innovation_ = e; S_ = S; K_ = K;
        x_ = x_ + K * e;
        if (opts.joseph) {
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_ = IKH * Pminus * IKH.t() + K * R * K.t();
        } else {
            P_ = (Mat<NX, NX>::identity() - K * H) * Pminus;
        }
        P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    Real lambda() const { return lambda_; }
    const Mat<NY, NY>& V() const { return V_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Mat<NX, NX> FPFt_, Q_;
    Real p_max_[NX] = {};
    Mat<NY, NY> V_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    Real lambda_ = Real(1);
    bool have_pred_ = false, have_V_ = false;
};

}  // namespace estkit
