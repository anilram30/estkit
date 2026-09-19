// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/sage_husa_akf.hpp — Sage-Husa adaptive extended Kalman filter
//  (recursive maximum-a-posteriori estimation of the noise statistics).
//
//  References
//    * Sage, A. P. & Husa, G. W. (1969). Adaptive filtering with unknown prior
//      statistics. Proc. Joint Automatic Control Conference, 760-769.
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, New York, Ch. 8.                    (EKF this builds on)
//    * Maybeck, P. S. (1979). Stochastic Models, Estimation, and Control,
//      Volume 1, Academic Press, New York, Ch. 10.   (adaptive filtering survey)
//    * Mohamed, A. H. & Schwarz, K. P. (1999). Adaptive Kalman filtering for
//      INS/GPS. Journal of Geodesy 73(4), 193-203.       (relation to IAE/RAE)
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs - Part 3. State and parameter
//      estimation. J. Power Sources 134(2), 277-292.       (timing convention)
//
//  Algorithm (one cycle; y in R^{n_y}, x in R^{n_x})
//    weight            d_k = (1-b)/(1-b^{k+1}),  b in (0,1) the forgetting factor
//    time update       x^- = f(x^+_{k-1}, u_{k-1}),   P^- = F P^+ F^T + Q
//    innovation        e_k = y_k - h(x^-, u_k)
//    gain              S = H P^- H^T + R_hat,  K = P^- H^T S^{-1}
//    state update      x^+ = x^- + K e_k,  P^+ = (I-KH) P^- (I-KH)^T + K R_hat K^T
//    residual          r_k = y_k - h(x^+, u_k)          (a-posteriori residual)
//    R adaptation      R_hat <- (1-d_k) R_hat + d_k ( r_k r_k^T + H P^+ H^T )
//    (optional, off)   Q_hat <- (1-d_k) Q_hat + d_k ( K e_k e_k^T K^T
//                                                     + P^+ - F P^+_{k-1} F^T )
//
//  The residual ("a-posteriori") form of the R recursion is used because both
//  terms it adds are positive semi-definite, so R_hat stays positive definite
//  for every k once it is initialised positive definite -- this is the standard
//  safeguard against the well-known loss of definiteness of the innovation form
//  R_hat <- (1-d) R_hat + d (e e^T - H P^- H^T).  Both forms are implemented;
//  the innovation form is additionally projected onto the PD cone by a diagonal
//  floor.  Q adaptation is DISABLED by default (see opts.adapt_Q).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class SageHusaAkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        // --- Sage-Husa knobs -------------------------------------------------
        Real b = Real(0.995);        // forgetting factor; memory length ~ 1/(1-b) samples
        bool adapt_R = true;         // recursive estimation of the measurement-noise covariance
        bool adapt_Q = false;        // recursive estimation of the process-noise covariance (see header)
        bool residual_form = true;   // true: R_hat from the a-posteriori residual (PD by construction)
        int  warmup = 20;            // samples during which R_hat is frozen at the model value
        // --- positive-definiteness safeguard ---------------------------------
        Real r_min_scale = Real(0.05);   // diagonal floor,  R_hat_ii >= r_min_scale * R_ii(model)
        Real r_max_scale = Real(5e3);    // diagonal ceiling, R_hat_ii <= r_max_scale * R_ii(model)
        Real q_min_scale = Real(0.1), q_max_scale = Real(1e3);   // same for Q_hat when adapt_Q
        // --- plain EKF knobs -------------------------------------------------
        bool joseph = true;
        bool apply_constraints = true;
        Real q_scale = Real(1);
        Real r_scale = Real(1);
    } opts;

    explicit SageHusaAkf(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0; k_ = 0; initialised_ = false;
        b_ = clampr(opts.b, Real(0), Real(1) - Real(1e-6));
        bpow_ = b_;                       // b^{k+1} with k = 0
        R_hat_ = Mat<NY, NY>(); Q_hat_ = Mat<NX, NX>();
        FPFt_ = Mat<NX, NX>(); have_pred_ = false;
        innovation_ = Y(); S_ = Mat<NY, NY>(); K_ = Mat<NX, NY>();
    }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        const Mat<NX, NX> Pp = P_;              // P^+_{k-1}, needed by the Q recursion
        x_ = m_.f(x_, u);
        FPFt_ = F * Pp * F.t();
        Q_used_ = opts.adapt_Q && initialised_ ? Q_hat_ : m_.Q(x_, u) * opts.q_scale;
        P_ = FPFt_ + Q_used_;
        P_.symmetrize();
        have_pred_ = true;
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> R_model = m_.R(x_, u) * opts.r_scale;
        if (!initialised_) {                     // lazy: the model R needs (x,u)
            R_hat_ = R_model;
            Q_hat_ = m_.Q(x_, u) * opts.q_scale;
            initialised_ = true;
        }
        const Mat<NY, NY> R_use = opts.adapt_R ? R_hat_ : R_model;

        const Mat<NX, NX> Pminus = P_;
        const Mat<NY, NY> S = H * Pminus * H.t() + R_use;
        const Mat<NX, NY> K = Pminus * H.t() * inverse(S);
        if (!K.is_finite()) { ++k_; return; }     // keep the previous estimate (API rule 4)

        innovation_ = y - m_.h(x_, u);            // a-priori innovation e_k
        S_ = S; K_ = K;
        const X x_prior = x_;
        x_ = x_ + K * innovation_;
        if (opts.joseph) {
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_ = IKH * Pminus * IKH.t() + K * R_use * K.t();
        } else {
            P_ = (Mat<NX, NX>::identity() - K * H) * Pminus;
        }
        P_.symmetrize();

        // ---- Sage-Husa recursions (weight d_k = (1-b)/(1-b^{k+1})) ----------
        const Real d = weight();
        if (opts.adapt_R && k_ >= opts.warmup) {
            Mat<NY, NY> incr;
            if (opts.residual_form) {
                // r_k = y - h(x^+, u); both terms below are PSD -> R_hat stays PD
                const Y r = y - m_.h(x_, u);
                incr = outer(r, r) + H * P_ * H.t();
            } else {
                incr = outer(innovation_, innovation_) - H * Pminus * H.t();
            }
            Mat<NY, NY> Rn = R_hat_ * (Real(1) - d) + incr * d;
            Rn.symmetrize();
            clamp_diag(Rn, R_model, opts.r_min_scale, opts.r_max_scale);
            if (Rn.is_finite()) R_hat_ = Rn;
        }
        if (opts.adapt_Q && have_pred_ && k_ >= opts.warmup) {
            const Y e = innovation_;
            const X dx = K * e;
            Mat<NX, NX> incr = outer(dx, dx) + P_ - FPFt_;
            Mat<NX, NX> Qn = Q_hat_ * (Real(1) - d) + incr * d;
            Qn.symmetrize();
            const Mat<NX, NX> Q_model = m_.Q(x_prior, u) * opts.q_scale;
            clamp_diag(Qn, Q_model, opts.q_min_scale, opts.q_max_scale);
            if (Qn.is_finite()) Q_hat_ = Qn;
        }
        ++k_;
        bpow_ *= b_;                      // b^{k+1}, updated once per sample
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    // ---- accessors ----------------------------------------------------------
    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    const Mat<NY, NY>& R_hat() const { return R_hat_; }
    const Mat<NX, NX>& Q_hat() const { return Q_hat_; }
    Real sigma_v_hat() const { return std::sqrt(std::fabs(R_hat_(0, 0))); }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    // d_k = (1-b)/(1-b^{k+1}); an unbiased running average for small k that
    // saturates at 1-b (exponential window of length 1/(1-b) samples).
    // b^{k+1} is propagated incrementally, so the weight costs one multiply.
    Real weight() const {
        const Real den = Real(1) - bpow_;
        return (den > Real(1e-12)) ? (Real(1) - b_) / den : Real(1);
    }
    template <int N>
    static void clamp_diag(Mat<N, N>& A, const Mat<N, N>& ref, Real lo, Real hi) {
        for (int i = 0; i < N; ++i) {
            const Real r = ref(i, i) > Real(0) ? ref(i, i) : Real(1e-12);
            const Real v = clampr(A(i, i), lo * r, hi * r);
            // rescale the whole row/column so that correlations stay consistent
            if (A(i, i) > Real(0) && v != A(i, i)) {
                const Real s = std::sqrt(v / A(i, i));
                for (int j = 0; j < N; ++j) { A(i, j) *= s; A(j, i) *= s; }
            }
            A(i, i) = v;
        }
    }

    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Mat<NX, NX> FPFt_;          // F P^+_{k-1} F^T, kept for the Q recursion
    Mat<NX, NX> Q_used_;        // Q actually used in the last time update
    Mat<NY, NY> R_hat_;
    Mat<NX, NX> Q_hat_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    int k_ = 0;
    Real b_ = Real(0.995), bpow_ = Real(0.995);
    bool initialised_ = false;
    bool have_pred_ = false;
};

}  // namespace estkit
