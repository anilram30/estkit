// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/fuzzy_akf.hpp — Fuzzy-logic adaptive extended Kalman filter
//  (covariance matching by a Mamdani fuzzy inference system).
//
//  References
//    * Loebis, D., Sutton, R., Chudley, J. & Naeem, W. (2004). Adaptive tuning
//      of a Kalman filter via fuzzy logic for an intelligent AUV navigation
//      system. Control Engineering Practice 12(12), 1531-1539.
//    * Mohamed, A. H. & Schwarz, K. P. (1999). Adaptive Kalman filtering for
//      INS/GPS. Journal of Geodesy 73(4), 193-203.       (covariance matching)
//    * Mehra, R. K. (1970). On the identification of variances and adaptive
//      Kalman filtering. IEEE Trans. Autom. Control 15(2), 175-184.
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, New York, Ch. 8.                  (EKF this builds on)
//
//  Idea.  A consistent filter has innovations e_k = y_k - h(x^-_k, u_k) whose
//  sample covariance over a window of N samples,
//
//      C_k = (1/N) sum_{j=k-N+1}^{k} e_j e_j^T ,                           (1)
//
//  matches the theoretical value S_k = H_k P^-_k H_k^T + R_hat_k.  Loebis et al.
//  feed the scalar "degree of matching"
//
//      DoM_k = ( tr C_k - tr S_k ) / tr S_k                                 (2)
//
//  into a fuzzy inference system whose output is an incremental change of the
//  measurement-noise scaling s_k (R_hat_k = s_k R_k^{model}):
//
//      IF DoM is N (negative) THEN Delta is D (decrease s)
//      IF DoM is Z (zero)     THEN Delta is H (hold)
//      IF DoM is P (positive) THEN Delta is I (increase s)
//      s_k = clamp( s_{k-1} exp(Delta_k), s_min, s_max ) .                  (3)
//
//  Three triangular/shoulder membership functions partition the DoM universe
//  (sum-to-one, so at most two rules fire), Mamdani min-implication and
//  max-aggregation are used, and the crisp output is the centroid of the
//  aggregated consequent set, evaluated on a fixed grid of NG points (no heap,
//  no transcendental functions except one exp per step).
//
//  The equilibrium of (3) is exactly the covariance-matching (IAE) solution
//  s* R^model = C_k - H P^- H^T, so the fuzzy system acts as a rate-limited and
//  strongly smoothed version of innovation-based adaptive estimation; the rate
//  limit is what makes it robust to the short bursts of large innovations
//  produced by measurement outliers.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model, int NWIN = 50, int NG = 41>
class FuzzyAkf {
public:
    static_assert(NWIN >= 2, "fuzzy window must contain at least two innovations");
    static_assert(NG >= 5, "centroid grid too coarse");
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int WIN = NWIN;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        bool enable = true;
        Real dom_scale = Real(0.5);   // |DoM| at which the N / P membership saturates
        Real step = Real(0.3);        // maximum |Delta log s| per sample (rule consequent centre)
        Real s_min = Real(0.2);       // bounds on the measurement-noise scaling s
        Real s_max = Real(2e3);
        int  min_samples = NWIN;      // fill the window before adapting
        bool joseph = true;
        bool apply_constraints = true;
        Real q_scale = Real(1);
        Real r_scale = Real(1);
    } opts;

    explicit FuzzyAkf(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0;
        head_ = 0; count_ = 0; sum_ = Mat<NY, NY>();
        for (int j = 0; j < NWIN; ++j) buf_[j] = Y();
        s_ = Real(1); dom_ = Real(0);
        innovation_ = Y(); S_ = Mat<NY, NY>(); K_ = Mat<NX, NY>(); R_hat_ = Mat<NY, NY>();
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
        const Mat<NY, NY> R_model = m_.R(x_, u) * opts.r_scale;
        const Mat<NX, NX> Pminus = P_;
        const Y e = y - m_.h(x_, u);

        // ---- (1) windowed innovation covariance (ring buffer, O(n_y^2)) ----
        if (count_ == NWIN) sum_ -= outer(buf_[head_], buf_[head_]);
        buf_[head_] = e;
        sum_ += outer(e, e);
        head_ = (head_ + 1) % NWIN;
        if (count_ < NWIN) ++count_;

        // ---- (2)-(3) fuzzy covariance matching -----------------------------
        Mat<NY, NY> R_use = R_model * s_;
        if (opts.enable && count_ >= opts.min_samples) {
            const Mat<NY, NY> C = sum_ / Real(count_);
            const Mat<NY, NY> S_th = H * Pminus * H.t() + R_use;
            const Real trS = S_th.trace();
            if (trS > Real(1e-300)) {
                dom_ = (C.trace() - trS) / trS;
                const Real delta = infer(dom_);
                s_ = clampr(s_ * safe_exp(delta), opts.s_min, opts.s_max);
                R_use = R_model * s_;
            }
        }
        R_hat_ = R_use;

        // ---- ordinary EKF measurement update -------------------------------
        const Mat<NY, NY> S = H * Pminus * H.t() + R_use;
        const Mat<NX, NY> K = Pminus * H.t() * inverse(S);
        if (!K.is_finite()) return;                 // keep the previous estimate
        innovation_ = e; S_ = S; K_ = K;
        x_ = x_ + K * e;
        if (opts.joseph) {
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_ = IKH * Pminus * IKH.t() + K * R_use * K.t();
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
    const Mat<NY, NY>& R_hat() const { return R_hat_; }
    Real r_scaling() const { return s_; }
    Real dom() const { return dom_; }
    Real sigma_v_hat() const { return std::sqrt(std::fabs(R_hat_(0, 0))); }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    // ---- Mamdani fuzzy inference system (3 rules, centroid defuzzification) --
    //  Antecedent MFs on the normalised input d = DoM / dom_scale:
    //     mu_N(d) = 1 (d <= -1), -d (-1 < d < 0), 0 (d >= 0)
    //     mu_Z(d) = 1 - |d| for |d| < 1, else 0
    //     mu_P(d) = 1 (d >=  1),  d ( 0 < d < 1), 0 (d <= 0)
    //  Consequent MFs on the normalised output v in [-1.5, 1.5]: triangles of
    //  unit half-width centred at -1 (decrease), 0 (hold), +1 (increase).
    Real infer(Real dom) const {
        const Real c = (opts.dom_scale > Real(0)) ? opts.dom_scale : Real(0.5);
        const Real d = clampr(dom / c, Real(-2), Real(2));
        Real w[3];
        w[0] = (d <= Real(-1)) ? Real(1) : (d < Real(0) ? -d : Real(0));         // N -> decrease
        w[1] = (d > Real(-1) && d < Real(1)) ? (Real(1) - std::fabs(d)) : Real(0);  // Z -> hold
        w[2] = (d >= Real(1)) ? Real(1) : (d > Real(0) ? d : Real(0));           // P -> increase
        const Real centre[3] = { Real(-1), Real(0), Real(1) };
        Real num = Real(0), den = Real(0);
        for (int g = 0; g < NG; ++g) {
            const Real v = Real(-1.5) + Real(3) * Real(g) / Real(NG - 1);
            Real mu = Real(0);
            for (int r = 0; r < 3; ++r) {
                const Real tri = Real(1) - std::fabs(v - centre[r]);            // unit half-width
                const Real clipped = std::min(w[r], tri > Real(0) ? tri : Real(0));  // min-implication
                if (clipped > mu) mu = clipped;                                  // max-aggregation
            }
            num += v * mu; den += mu;
        }
        const Real vstar = (den > Real(1e-12)) ? num / den : Real(0);
        return vstar * opts.step;
    }

    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y buf_[NWIN];
    Mat<NY, NY> sum_;
    Mat<NY, NY> R_hat_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    Real s_ = Real(1), dom_ = Real(0);
    int head_ = 0, count_ = 0;
};

}  // namespace estkit
