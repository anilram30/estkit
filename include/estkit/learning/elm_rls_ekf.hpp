// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/learning/elm_rls_ekf.hpp — extended Kalman filter with an ON-LINE
//  learned output residual: an extreme learning machine (random hidden layer,
//  linear output) whose output weights are adapted by recursive least squares
//  with exponential forgetting, driven by the filter innovation.
//
//  References
//    * Huang, G.-B., Zhu, Q.-Y. & Siew, C.-K. (2006). Extreme learning machine:
//      Theory and applications. Neurocomputing 70(1-3), 489-501.
//      (random, fixed input weights + least-squares output weights)
//    * Pao, Y.-H., Park, G.-H. & Sobajic, D. J. (1994). Learning and generalization
//      characteristics of the random vector functional-link net. Neurocomputing
//      6(2), 163-180.                       (random features + direct linear link)
//    * Liang, N.-Y., Huang, G.-B., Saratchandran, P. & Sundararajan, N. (2006).
//      A fast and accurate online sequential learning algorithm for feedforward
//      networks. IEEE Transactions on Neural Networks 17(6), 1411-1423.
//      (recursive least-squares training of the ELM output layer)
//    * Ljung, L. (1999). System Identification: Theory for the User, 2nd ed.
//      Prentice Hall, Ch. 11.        (RLS with exponential forgetting, windup)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory.
//      Academic Press, Ch. 8.                                          (the EKF)
//    * Friedland, B. (1969). Treatment of bias in recursive filtering. IEEE
//      Transactions on Automatic Control 14(4), 359-367.
//      (separating a slowly varying output bias from the state estimate)
//
//  Method.  The output equation of the model is corrected by a learned term
//
//      y_k = h(x_k, u_k) + rho_k + v_k,     rho_k = w_k^T phi(xi_k),      (1)
//      phi(xi) = [ tanh(a_1^T xi + b_1) ... tanh(a_{n_h}^T xi + b_{n_h}), xi^T ]^T, (2)
//
//  with a_m, b_m drawn ONCE from a fixed pseudo-random stream (deterministic for
//  a given seed) and never changed - this is the extreme-learning-machine idea:
//  a random nonlinear feature map turns the residual regression into a LINEAR
//  least-squares problem in w, which can be solved recursively.  The regressor xi
//  is appended to the random features unchanged (the "direct link" of the random
//  vector functional-link net, Pao, Park & Sobajic 1994): the physically dominant
//  residual of a resistance error, -Delta R0 * i, is then represented EXACTLY
//  instead of being approximated by a lucky draw of tanh ridges, which removes
//  most of the variance the random layer would otherwise introduce.
//
//  The regressor xi is built from the model input u (and optionally the state x),
//  centred and scaled,
//
//      xi = [ (u - u_c) .* u_s ;  m_x .* (x - x_c) .* x_s ],              (3)
//
//  where the mask m_x is zero unless `use_state_features` is set.
//
//  Adaptation.  Let nu_k = y_k - h(xhat^-_k, u_k) be the raw innovation. The
//  output weights follow the forgetting RLS recursion with a leak,
//
//      g_k = P_{k-1} phi_k / ( lambda + phi_k^T P_{k-1} phi_k ),          (4)
//      w_k = (1 - varrho) w_{k-1} + g_k ( nu_k - phi_k^T w_{k-1} ),       (5)
//      P_k = ( P_{k-1} - g_k phi_k^T P_{k-1} ) / lambda,                  (6)
//
//  and the EKF is run with the CORRECTED innovation nu_k - rho_k, where rho_k
//  uses the a-priori weights w_{k-1}. Using w_{k-1} (not w_k) is essential: with
//  the a-posteriori weights the ELM would fit the innovation exactly at every
//  sample, the filter would see zero innovation and the state estimate would
//  freeze at open-loop Coulomb counting.
//
//  Identifiability.  An output offset and a state-of-charge error are locally
//  indistinguishable: to first order a SOC error delta_z perturbs the predicted
//  output by
//
//      delta_y = s_k delta_z,     s_k = dh/dz|_{xhat^-_k} = dOCV/dz + ...     (7)
//
//  If the ELM can reproduce the time profile s_k it will ABSORB the SOC error,
//  the corrected innovation will vanish, the filter will stop correcting and the
//  estimate degenerates into open-loop Coulomb counting.  This is not a
//  hypothetical failure: with an unconstrained regressor it is what the closed
//  loop actually converges to (the SOC-error direction is the one direction in
//  which the RLS gradient vanishes once the state has compensated, so nothing
//  pulls the weights back).  Four mechanisms keep the two apart:
//
//    (i)   PROJECTION.  Before it enters the regression the feature vector is
//          orthogonalised against s_k in the same exponentially weighted inner
//          product the RLS uses,
//
//              q_k   = lambda q_{k-1} + s_k^2,
//              p_k   = lambda p_{k-1} + s_k phi_k,   c_k = p_k / q_k,         (8)
//              phit_k = phi_k - c_k s_k,
//
//          i.e. one step of Gram-Schmidt.  The learned residual rho = w^T phit
//          is then, by construction, orthogonal to the SOC-error signature (7):
//          the ELM can only explain structure that a SOC error CANNOT explain -
//          exactly the current- and temperature-dependent structure it is meant
//          to capture.  (`project_soc_direction`, on by default.)
//    (ii)  The regressor excludes the state by default (`use_state_features =
//          false`), so the residual depends on the exogenous inputs only.
//    (iii) The leak varrho in (5) pulls w towards zero with time constant
//          1/varrho, giving the residual model a finite DC gain, so any
//          direction that is not persistently excited decays.
//    (iv)  Adaptation is gated: it starts after `warmup` samples and only runs
//          while the normalised innovation is inside `gate_sigma` standard
//          deviations, so transients (initial-SOC errors, outliers, sensor
//          dropouts) are never learned as model structure.
//  In addition |rho| is saturated at r_max.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../core/rng.hpp"

namespace estkit {

template <class Model, int NH = 20>
class ElmRlsEkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NF   = NU + NX;          // regressor width, eq. (3)
    static constexpr int NPHI = NH + NF;          // random features + direct linear link
    static_assert(NY == 1, "ElmRlsEkf assumes a scalar measurement");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        // --- EKF ---
        bool joseph = true;
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
        // --- regressor construction, eq. (3) ---
        bool use_state_features = false;   // see the identifiability discussion above
        Vec<NU> u_center, u_scale;
        Vec<NX> x_center, x_scale;
        // --- ELM random layer, eq. (2) ---
        Real a_range = Real(1.5);          // a_m ~ U(-a_range, a_range)
        Real b_range = Real(1.0);          // b_m ~ U(-b_range, b_range)
        uint64_t elm_seed = 0xE1B0C0DEull;
        // --- RLS, eqs. (4)-(6) ---
        Real lambda = Real(0.999);         // forgetting factor (memory ~ 1/(1-lambda) samples)
        Real leak   = Real(1e-3);          // varrho: leak of the output weights
        Real p0     = Real(1e-2);          // initial output-weight covariance
        Real p_max  = Real(1e3);           // covariance windup bound on trace(P)
        Real r_max  = Real(0.05);          // saturation of rho [V]
        Real gate_sigma = Real(4);         // adapt only if |nu - rho| < gate_sigma sqrt(S)
        int  warmup = 600;                 // samples of pure EKF before adaptation starts
        // --- identifiability projection, eq. (8) ---
        bool project_soc_direction = true; // orthogonalise phi against dh/dx[proj_index]
        int  proj_index = 0;               // the slowly varying state whose error is
                                           // locally degenerate with an output bias
                                           // (the SOC for the cell model)
    } opts;

    explicit ElmRlsEkf(const Model& m) : m_(m) {
        for (int j = 0; j < NU; ++j) { opts.u_center[j] = Real(0); opts.u_scale[j] = Real(1); }
        for (int i = 0; i < NX; ++i) { opts.x_center[i] = Real(0); opts.x_scale[i] = Real(1); }
    }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0;
        // draw the random hidden layer once, deterministically
        Rng rng(opts.elm_seed);
        for (int m = 0; m < NH; ++m) {
            for (int j = 0; j < NF; ++j) A_(m, j) = rng.uniform(-opts.a_range, opts.a_range);
            bh_[m] = rng.uniform(-opts.b_range, opts.b_range);
        }
        w_ = Vec<NPHI>();
        Pw_ = Mat<NPHI, NPHI>::identity() * opts.p0;
        cproj_ = Vec<NPHI>(); pcorr_ = Vec<NPHI>(); qproj_ = Real(0);
        k_ = 0; rho_ = Real(0);
    }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        P_ = F * P_ * F.t() + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    // The predicted measurement includes the learned residual, eq. (1).
    Y predict_measurement(const U& u) const {
        Y y = m_.h(x_, u);
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Vec<NPHI> phi = project(features(x_, u), H(0, opts.proj_index));
        y[0] += clampr(dot(w_, phi), -opts.r_max, opts.r_max);
        return y;
    }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Real s_soc = H(0, opts.proj_index);               // eq. (7)
        const Vec<NPHI> phi_raw = features(x_, u);
        const Vec<NPHI> phi = project(phi_raw, s_soc);          // eq. (8)
        rho_ = clampr(dot(w_, phi), -opts.r_max, opts.r_max);   // a-priori residual

        const Mat<NY, NY> R = m_.R(x_, u) * opts.r_scale;
        const Mat<NY, NY> S = H * P_ * H.t() + R;
        const Mat<NX, NY> K = P_ * H.t() * inverse(S);
        const Real nu = y[0] - m_.h(x_, u)[0];                  // raw innovation
        innovation_[0] = nu - rho_;                             // corrected innovation
        S_ = S; K_ = K;

        x_ = x_ + K * innovation_;
        if (opts.joseph) {
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_ = IKH * P_ * IKH.t() + K * R * K.t();
        } else {
            P_ = (Mat<NX, NX>::identity() - K * H) * P_;
        }
        P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);

        // --- gated RLS update of the output weights, eqs. (4)-(6) ------------
        const Real sd = std::sqrt(std::max(S(0, 0), Real(1e-12)));
        const bool gate = (k_ >= opts.warmup) && (std::fabs(innovation_[0]) < opts.gate_sigma * sd);
        if (gate) {
            update_projection(phi_raw, s_soc);   // eq. (8): advance q, p, c
            rls_update(phi, nu);
        }
        ++k_;
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    Real residual_now() const { return rho_; }
    const Vec<NPHI>& weights() const { return w_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    // eq. (3) then eq. (2)
    Vec<NPHI> features(const X& x, const U& u) const {
        Vec<NF> xi;
        for (int j = 0; j < NU; ++j) xi[j] = (u[j] - opts.u_center[j]) * opts.u_scale[j];
        for (int i = 0; i < NX; ++i)
            xi[NU + i] = opts.use_state_features ? (x[i] - opts.x_center[i]) * opts.x_scale[i] : Real(0);
        Vec<NPHI> phi;
        for (int m = 0; m < NH; ++m) {
            Real s = bh_[m];
            for (int j = 0; j < NF; ++j) s += A_(m, j) * xi[j];
            phi[m] = std::tanh(s);
        }
        for (int j = 0; j < NF; ++j) phi[NH + j] = xi[j];     // direct linear link
        return phi;
    }
    // eq. (8): remove the component of phi that lies along the SOC-error
    // signature s, using the exponentially weighted regression coefficient c.
    Vec<NPHI> project(const Vec<NPHI>& phi, Real s) const {
        if (!opts.project_soc_direction) return phi;
        Vec<NPHI> out = phi;
        for (int m = 0; m < NPHI; ++m) out[m] -= cproj_[m] * s;
        return out;
    }
    void update_projection(const Vec<NPHI>& phi_raw, Real s) {
        qproj_ = opts.lambda * qproj_ + s * s;
        for (int m = 0; m < NPHI; ++m) pcorr_[m] = opts.lambda * pcorr_[m] + s * phi_raw[m];
        const Real q = std::max(qproj_, Real(1e-12));
        for (int m = 0; m < NPHI; ++m) cproj_[m] = pcorr_[m] / q;
    }
    void rls_update(const Vec<NPHI>& phi, Real target) {
        const Vec<NPHI> Pphi = Pw_ * phi;
        const Real den = opts.lambda + dot(phi, Pphi);
        if (!(den > Real(1e-12))) return;
        const Vec<NPHI> g = Pphi / den;
        const Real err = target - dot(w_, phi);
        for (int m = 0; m < NPHI; ++m) w_[m] = (Real(1) - opts.leak) * w_[m] + g[m] * err;
        Pw_ = (Pw_ - outer<NPHI, NPHI>(g, Pphi)) / opts.lambda;   // g (P phi)^T = g phi^T P
        Pw_.symmetrize();
        const Real tr = Pw_.trace();
        if (!(tr > Real(0)) || !Pw_.is_finite()) { Pw_ = Mat<NPHI, NPHI>::identity() * opts.p0; return; }
        if (tr > opts.p_max) Pw_ *= (opts.p_max / tr);          // covariance windup guard
    }

    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Mat<NH, NF> A_;        // random input weights a_m^T (rows)
    Vec<NH> bh_;           // random hidden biases
    Vec<NPHI> w_;             // output weights (RLS)
    Mat<NPHI, NPHI> Pw_;      // RLS covariance
    Vec<NPHI> cproj_, pcorr_; // Gram-Schmidt coefficients of eq. (8)
    Real qproj_ = Real(0);
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    Real rho_ = Real(0);
    int  k_ = 0;
};

}  // namespace estkit
