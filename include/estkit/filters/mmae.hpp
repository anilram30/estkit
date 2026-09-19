// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/mmae.hpp — Multiple-Model Adaptive Estimation (static bank).
//
//  References
//    * Magill, D. T. (1965). Optimal adaptive estimation of sampled stochastic
//      processes. IEEE Trans. Autom. Control 10(4), 434-439.        (the method)
//    * Bar-Shalom, Y., Li, X. R. & Kirubarajan, T. (2001). Estimation with
//      Applications to Tracking and Navigation: Theory, Algorithms and Software.
//      Wiley, New York, Sec. 11.6.1.                    (static MM estimation)
//    * Maybeck, P. S. (1979). Stochastic Models, Estimation, and Control,
//      Volume 1, Academic Press, New York, Ch. 10.
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs - Part 3. J. Power Sources
//      134(2), 277-292.                    (state and parameter estimation)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, New York, Ch. 8.               (mode-matched EKF cycle)
//
//  Idea.  The unknown parameter vector theta is discretised onto a fixed grid
//  { theta^(1), ..., theta^(r) }.  One EKF is run per hypothesis; because the
//  hypotheses do not switch, Bayes' rule gives the exact recursive posterior
//
//      w_j(k) = w_j(k-1) L_j(k) / sum_l w_l(k-1) L_l(k),                    (1)
//      L_j(k) = N( e_j(k); 0, S_j(k) )
//             = |2 pi S_j|^{-1/2} exp( -1/2 e_j^T S_j^{-1} e_j ),           (2)
//
//  with e_j, S_j the innovation and innovation covariance of filter j, and the
//  minimum-mean-square estimates are the posterior mixtures
//
//      x(k)     = sum_j w_j(k) x_j(k),                                      (3)
//      theta(k) = sum_j w_j(k) theta^(j),                                   (4)
//      P(k)     = sum_j w_j [ P_j + (x_j - x)(x_j - x)^T ].                 (5)
//
//  Because (1) integrates the whole data history, the posterior collapses
//  exponentially fast onto the single closest hypothesis ("identifiability
//  lock-out"): all other weights underflow and the bank can never react to a
//  later change.  Two standard safeguards are implemented: a lower bound w_min
//  on every weight (renormalised afterwards), and an optional exponential
//  forgetting factor gamma in (0,1] applied to the log-posterior, which limits
//  the effective memory to 1/(1-gamma) samples.
//
//  Likelihood inflation.  (1)-(2) assume that the ONLY discrepancy between the
//  data and hypothesis j is the parameter error; every un-modelled effect (here:
//  a growing series resistance, a current-sensor bias, a noisier voltage sensor)
//  enters every L_j identically in the mean but is divided by the very small
//  S_j, so a few hundred samples of un-modelled error can swamp the parameter
//  information and lock the posterior onto an arbitrary hypothesis.  Following
//  the standard practice of evaluating multiple-model residuals with an
//  artificially increased noise covariance (Maybeck 1979, Ch. 10), the
//  likelihood is computed with
//
//      S_j^lik = H_j P_j^- H_j^T + c_lik R_j ,   c_lik >= 1,                (6)
//
//  which leaves the mode-matched filters themselves untouched (they keep the
//  correct R and hence the correct gain) but makes the weight recursion see the
//  innovations through a covariance large enough to absorb the un-modelled
//  part.  c_lik = 1 recovers the textbook MMAE.
//
//  Genericity.  The bank is built from the model's own parameter interface
//  (Model::NP, params(), set_params()); Options::scale[j] multiplies the
//  parameter with index Options::p_index, so the same class realises a capacity
//  (SOH) bank for the battery ECM and, say, a drag-coefficient bank for a
//  navigation model.  No heap, no exceptions.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model, int NM = 3>
class Mmae {
public:
    static_assert(NM >= 2, "an MMAE bank needs at least two hypotheses");
    static_assert(has_params<Model>::value, "Mmae requires Model::NP, params(), set_params()");
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY, NP = Model::NP;
    static constexpr int NHYP = NM;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real scale[NM];             // multiplicative hypotheses on parameter p_index
        int  p_index = 0;           // parameter varied by the bank (EcmModel: 0 = capacity Q_Ah)
        Real w_min = Real(0.01);    // floor on the posterior weights (anti lock-out)
        Real gamma = Real(1);       // exponential forgetting of the log-posterior (1 = pure Bayes)
        Real lik_inflate = Real(100);   // c_lik in eq. (6); 1 = textbook MMAE likelihood
        bool joseph = true;
        bool apply_constraints = true;
        Real q_scale = Real(1);
        Real r_scale = Real(1);
        Options() { for (int j = 0; j < NM; ++j) scale[j] = Real(1) - Real(0.1) * Real(j); }
    } opts;

    explicit Mmae(const Model& m) : theta0_(m.params()) { for (int j = 0; j < NM; ++j) mm_[j] = m; }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        const int pi = clampi(opts.p_index, 0, NP - 1);
        for (int j = 0; j < NM; ++j) {
            Vec<NP> th = theta0_;
            th[pi] = theta0_[pi] * opts.scale[j];
            mm_[j].set_params(th);
            theta_[j] = mm_[j].params();       // read back (set_params may clamp)
            xm_[j] = x0; Pm_[j] = P0;
            w_[j] = Real(1) / Real(NM);
            logw_[j] = std::log(w_[j]);
            loglik_[j] = Real(0);
        }
        x_ = x0; P_ = P0;
        theta_hat_ = mixture_theta();
        innovation_ = Y(); S_ = Mat<NY, NY>();
    }

    void predict(const U& u) {
        for (int j = 0; j < NM; ++j) {
            const Mat<NX, NX> F = jac_F(mm_[j], xm_[j], u);
            xm_[j] = mm_[j].f(xm_[j], u);
            Pm_[j] = F * Pm_[j] * F.t() + mm_[j].Q(xm_[j], u) * opts.q_scale;
            Pm_[j].symmetrize();
        }
        combine();
    }

    Y predict_measurement(const U& u) const {
        Y ym;
        for (int j = 0; j < NM; ++j) ym += mm_[j].h(xm_[j], u) * w_[j];
        return ym;
    }

    void update(const Y& y, const U& u) {
        Real logl[NM];
        for (int j = 0; j < NM; ++j) {
            const Mat<NY, NX> H = jac_H(mm_[j], xm_[j], u);
            const Mat<NY, NY> R = mm_[j].R(xm_[j], u) * opts.r_scale;
            const Mat<NX, NX> Pminus = Pm_[j];
            const Mat<NY, NY> S = H * Pminus * H.t() + R;
            const Mat<NX, NY> K = Pminus * H.t() * inverse(S);
            const Y e = y - mm_[j].h(xm_[j], u);
            if (!K.is_finite()) { logl[j] = -Real(1e30); continue; }
            xm_[j] = xm_[j] + K * e;
            if (opts.joseph) {
                const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
                Pm_[j] = IKH * Pminus * IKH.t() + K * R * K.t();
            } else {
                Pm_[j] = (Mat<NX, NX>::identity() - K * H) * Pminus;
            }
            Pm_[j].symmetrize();
            if (opts.apply_constraints) xm_[j] = constrain(mm_[j], xm_[j]);
            const Real c = (opts.lik_inflate > Real(1)) ? opts.lik_inflate : Real(1);
            logl[j] = gauss_loglik(e, S + R * (c - Real(1)));     // eq. (6)
            if (j == 0) { innovation_ = e; S_ = S; }
        }
        // ---- (1) recursive Bayesian weights, evaluated in the log domain ----
        const Real g = clampr(opts.gamma, Real(0), Real(1));
        Real logmax = -std::numeric_limits<Real>::infinity();
        for (int j = 0; j < NM; ++j) {
            loglik_[j] = logl[j];
            logw_[j] = g * logw_[j] + logl[j];
            if (logw_[j] > logmax) logmax = logw_[j];
        }
        Real sum = Real(0);
        for (int j = 0; j < NM; ++j) {
            const Real w = safe_exp(logw_[j] - logmax);
            w_[j] = std::isfinite(w) ? w : Real(0);
            sum += w_[j];
        }
        if (!(sum > Real(1e-300))) { for (int j = 0; j < NM; ++j) w_[j] = Real(1) / Real(NM); }
        else                       { for (int j = 0; j < NM; ++j) w_[j] /= sum; }
        // floor + renormalise, then fold back so the recursion cannot lock out
        const Real wmin = clampr(opts.w_min, Real(0), Real(1) / Real(2 * NM));
        Real s2 = Real(0);
        for (int j = 0; j < NM; ++j) { w_[j] = std::max(w_[j], wmin); s2 += w_[j]; }
        for (int j = 0; j < NM; ++j) { w_[j] /= s2; logw_[j] = std::log(std::max(w_[j], Real(1e-300))); }
        combine();
    }

    // ---- accessors ----------------------------------------------------------
    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    Real weight(int j) const { return w_[j]; }
    Real loglik(int j) const { return loglik_[j]; }
    Real hypothesis(int j) const { return theta_[j][clampi(opts.p_index, 0, NP - 1)]; }
    const Vec<NP>& theta_hat() const { return theta_hat_; }
    Real param_hat(int i) const { return theta_hat_[i]; }
    // posterior-mean estimate of the parameter the bank varies; for the battery
    // ECM with p_index = 0 this is the cell capacity in Ah (state of health)
    Real capacity() const { return theta_hat_[clampi(opts.p_index, 0, NP - 1)]; }
    Model& model() { return mm_[0]; }
    const Model& model() const { return mm_[0]; }

private:
    static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
    Vec<NP> mixture_theta() const {
        Vec<NP> th;
        for (int j = 0; j < NM; ++j) th += theta_[j] * w_[j];
        return th;
    }
    void combine() {
        X xs;
        for (int j = 0; j < NM; ++j) xs += xm_[j] * w_[j];
        Mat<NX, NX> Ps;
        for (int j = 0; j < NM; ++j) { const X d = xm_[j] - xs; Ps += (Pm_[j] + outer(d, d)) * w_[j]; }
        Ps.symmetrize();
        x_ = xs; P_ = Ps; theta_hat_ = mixture_theta();
    }
    static Real gauss_loglik(const Y& e, const Mat<NY, NY>& S) {
        const Mat<NY, NY> L = cholesky_safe(S);
        Real logdet = Real(0);
        for (int i = 0; i < NY; ++i) logdet += Real(2) * std::log(std::max(L(i, i), Real(1e-300)));
        const Y w = solve_lower(L, e);
        return Real(-0.5) * (dot(w, w) + logdet + Real(NY) * std::log(Real(2) * kPi));
    }

    Model mm_[NM];
    Vec<NP> theta0_;
    Vec<NP> theta_[NM];
    X xm_[NM];
    Mat<NX, NX> Pm_[NM];
    Real w_[NM], logw_[NM], loglik_[NM];
    X x_;
    Mat<NX, NX> P_;
    Vec<NP> theta_hat_;
    Y innovation_;
    Mat<NY, NY> S_;
};

}  // namespace estkit
