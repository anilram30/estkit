// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/imm.hpp — Interacting Multiple Model (IMM) estimator with a
//  bank of extended Kalman filters that differ in their assumed noise levels.
//
//  References
//    * Blom, H. A. P. & Bar-Shalom, Y. (1988). The interacting multiple model
//      algorithm for systems with Markovian switching coefficients. IEEE Trans.
//      Autom. Control 33(8), 780-783.
//    * Bar-Shalom, Y., Li, X. R. & Kirubarajan, T. (2001). Estimation with
//      Applications to Tracking and Navigation: Theory, Algorithms and Software.
//      Wiley, New York, Sec. 11.6.
//    * Magill, D. T. (1965). Optimal adaptive estimation of sampled stochastic
//      processes. IEEE Trans. Autom. Control 10(4), 434-439.  (static MM bank)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, New York, Ch. 8.               (mode-matched EKF cycle)
//
//  Model.  The plant is assumed to switch between r = NM operating modes
//  according to a first-order Markov chain with transition probabilities
//  pi_ij = P{ mode_k = j | mode_{k-1} = i }.  Mode j uses the same f, h and the
//  same nominal Q, R as the others but scales them,
//      Q^(j) = q_scale_j Q(x,u),     R^(j) = r_scale_j R(x,u),
//  so that the bank spans "nominal", "noisy sensor" and "model mismatch".
//
//  One IMM cycle (Bar-Shalom et al. 2001, Sec. 11.6.6)
//   1. mixing probabilities   cbar_j = sum_i pi_ij mu_i,
//                             mu_{i|j} = pi_ij mu_i / cbar_j
//   2. mixed initial cond.    x0_j = sum_i mu_{i|j} x_i
//                             P0_j = sum_i mu_{i|j} [ P_i + (x_i-x0_j)(x_i-x0_j)^T ]
//   3. mode-matched filtering EKF time + measurement update of mode j started
//                             from (x0_j, P0_j); innovation e_j, covariance S_j
//   4. likelihoods            L_j = N(e_j; 0, S_j)
//   5. mode probabilities     mu_j = L_j cbar_j / sum_l L_l cbar_l
//   6. combination (output)   x = sum_j mu_j x_j,
//                             P = sum_j mu_j [ P_j + (x_j-x)(x_j-x)^T ]
//
//  Implementation notes.  The bank shares ONE model instance (the modes differ
//  only in the noise scalings), so the memory cost over a single EKF is
//  NM (n_x + n_x^2) reals.  Likelihoods are evaluated in the log domain and
//  shifted by their maximum before exponentiation, which makes the mode update
//  immune to under/overflow in the float32 build.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model, int NM = 3>
class Imm {
public:
    static_assert(NM >= 2, "an IMM needs at least two modes");
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NMODES = NM;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real q_scale[NM];        // process-noise scaling of every mode
        Real r_scale[NM];        // measurement-noise scaling of every mode
        Real p_switch = Real(1e-4);   // off-diagonal Markov transition probability
        Real mu_min = Real(1e-6);     // floor on the mode probabilities
        bool joseph = true;
        bool apply_constraints = true;
        Options() {
            for (int j = 0; j < NM; ++j) { q_scale[j] = Real(1); r_scale[j] = Real(1); }
            // mode 1: degraded voltage sensor  (measurement noise 100x nominal)
            if (NM > 1) r_scale[1] = Real(100);
            // mode 2: gross model mismatch — neither the transition model nor the
            //         output model can be trusted (see the chapter for why the
            //         "high Q only" hypothesis performs badly on an aged cell)
            if (NM > 2) { q_scale[2] = Real(100); r_scale[2] = Real(100); }
            for (int j = 3; j < NM; ++j) { q_scale[j] = Real(10); r_scale[j] = Real(10); }
        }
    } opts;

    explicit Imm(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        for (int j = 0; j < NM; ++j) {
            xm_[j] = x0; Pm_[j] = P0;
            mu_[j] = Real(1) / Real(NM); cbar_[j] = mu_[j]; loglik_[j] = Real(0);
        }
        x_ = x0; P_ = P0;
        innovation_ = Y(); S_ = Mat<NY, NY>();
        build_transition();
    }

    // ---- steps 1-3 (mixing + mode-matched time update) ----------------------
    void predict(const U& u) {
        // 1. mixing probabilities
        Real mix[NM][NM];                      // mix[i][j] = mu_{i|j}
        for (int j = 0; j < NM; ++j) {
            Real c = Real(0);
            for (int i = 0; i < NM; ++i) c += pi_[i][j] * mu_[i];
            cbar_[j] = (c > Real(1e-300)) ? c : Real(1e-300);
            for (int i = 0; i < NM; ++i) mix[i][j] = pi_[i][j] * mu_[i] / cbar_[j];
        }
        // 2. mixed initial conditions
        X x0[NM]; Mat<NX, NX> P0[NM];
        for (int j = 0; j < NM; ++j) {
            X xs;
            for (int i = 0; i < NM; ++i) xs += xm_[i] * mix[i][j];
            x0[j] = xs;
        }
        for (int j = 0; j < NM; ++j) {
            Mat<NX, NX> Ps;
            for (int i = 0; i < NM; ++i) {
                const X d = xm_[i] - x0[j];
                Ps += (Pm_[i] + outer(d, d)) * mix[i][j];
            }
            Ps.symmetrize(); P0[j] = Ps;
        }
        // 3a. mode-matched time update
        for (int j = 0; j < NM; ++j) {
            const Mat<NX, NX> F = jac_F(m_, x0[j], u);
            xm_[j] = m_.f(x0[j], u);
            Pm_[j] = F * P0[j] * F.t() + m_.Q(xm_[j], u) * opts.q_scale[j];
            Pm_[j].symmetrize();
        }
        combine_prior();
    }

    // mixture mean of the predicted measurement, weighted by the predicted mode
    // probabilities cbar_j (the mode probabilities after the Markov transition)
    Y predict_measurement(const U& u) const {
        Y ym;
        for (int j = 0; j < NM; ++j) ym += m_.h(xm_[j], u) * cbar_[j];
        return ym;
    }

    // ---- steps 3b-6 (mode-matched measurement update, mode probabilities) ---
    void update(const Y& y, const U& u) {
        Real logl[NM];
        Real logmax = -std::numeric_limits<Real>::infinity();
        for (int j = 0; j < NM; ++j) {
            const Mat<NY, NX> H = jac_H(m_, xm_[j], u);
            const Mat<NY, NY> R = m_.R(xm_[j], u) * opts.r_scale[j];
            const Mat<NX, NX> Pminus = Pm_[j];
            const Mat<NY, NY> S = H * Pminus * H.t() + R;
            const Mat<NX, NY> K = Pminus * H.t() * inverse(S);
            const Y e = y - m_.h(xm_[j], u);
            if (!K.is_finite()) { logl[j] = -Real(1e30); continue; }
            xm_[j] = xm_[j] + K * e;
            if (opts.joseph) {
                const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
                Pm_[j] = IKH * Pminus * IKH.t() + K * R * K.t();
            } else {
                Pm_[j] = (Mat<NX, NX>::identity() - K * H) * Pminus;
            }
            Pm_[j].symmetrize();
            if (opts.apply_constraints) xm_[j] = constrain(m_, xm_[j]);
            logl[j] = gauss_loglik(e, S);
            if (j == 0) { innovation_ = e; S_ = S; }
            if (logl[j] > logmax) logmax = logl[j];
        }
        loglik_for_debug(logl);
        // 5. mode probability update
        Real sum = Real(0);
        for (int j = 0; j < NM; ++j) {
            const Real w = cbar_[j] * safe_exp(logl[j] - logmax);
            mu_[j] = std::isfinite(w) ? w : Real(0);
            sum += mu_[j];
        }
        if (!(sum > Real(1e-300))) { for (int j = 0; j < NM; ++j) mu_[j] = Real(1) / Real(NM); }
        else                       { for (int j = 0; j < NM; ++j) mu_[j] /= sum; }
        Real s2 = Real(0);
        for (int j = 0; j < NM; ++j) { mu_[j] = std::max(mu_[j], opts.mu_min); s2 += mu_[j]; }
        for (int j = 0; j < NM; ++j) mu_[j] /= s2;
        // 6. combination
        X xs;
        for (int j = 0; j < NM; ++j) xs += xm_[j] * mu_[j];
        Mat<NX, NX> Ps;
        for (int j = 0; j < NM; ++j) { const X d = xm_[j] - xs; Ps += (Pm_[j] + outer(d, d)) * mu_[j]; }
        Ps.symmetrize();
        x_ = xs; P_ = Ps;
    }

    // ---- accessors ----------------------------------------------------------
    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    Real mode_prob(int j) const { return mu_[j]; }
    Real mode_loglik(int j) const { return loglik_[j]; }
    const X& mode_x(int j) const { return xm_[j]; }
    const Mat<NX, NX>& mode_P(int j) const { return Pm_[j]; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void build_transition() {
        const Real p = clampr(opts.p_switch, Real(0), Real(1) / Real(NM));
        for (int i = 0; i < NM; ++i)
            for (int j = 0; j < NM; ++j)
                pi_[i][j] = (i == j) ? (Real(1) - Real(NM - 1) * p) : p;
    }
    void combine_prior() {
        X xs;
        for (int j = 0; j < NM; ++j) xs += xm_[j] * cbar_[j];
        Mat<NX, NX> Ps;
        for (int j = 0; j < NM; ++j) { const X d = xm_[j] - xs; Ps += (Pm_[j] + outer(d, d)) * cbar_[j]; }
        Ps.symmetrize();
        x_ = xs; P_ = Ps;
    }
    void loglik_for_debug(const Real (&l)[NM]) { for (int j = 0; j < NM; ++j) loglik_[j] = l[j]; }
    // log N(e; 0, S) via the Cholesky factor of S (robust for small S)
    static Real gauss_loglik(const Y& e, const Mat<NY, NY>& S) {
        const Mat<NY, NY> L = cholesky_safe(S);
        Real logdet = Real(0);
        for (int i = 0; i < NY; ++i) logdet += Real(2) * std::log(std::max(L(i, i), Real(1e-300)));
        const Y w = solve_lower(L, e);
        return Real(-0.5) * (dot(w, w) + logdet + Real(NY) * std::log(Real(2) * kPi));
    }

    Model m_;
    X xm_[NM];
    Mat<NX, NX> Pm_[NM];
    Real mu_[NM], cbar_[NM], loglik_[NM];
    Real pi_[NM][NM];
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Mat<NY, NY> S_;
};

}  // namespace estkit
