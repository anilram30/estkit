// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/svsf.hpp — Smooth Variable Structure Filter (SVSF), in the
//  SVSF-KF ("combined") form for systems with more states than measurements.
//
//  References
//    * Habibi, S. (2007). The smooth variable structure filter. Proceedings of
//      the IEEE 95(5), 1026-1059.           (SVSF gain, boundary layer, stability)
//    * Gadsden, S. A. & Habibi, S. R. (2010). A new form of the smooth variable
//      structure filter with a covariance derivation. Proc. 49th IEEE Conference
//      on Decision and Control (CDC), 7389-7394.     (SVSF with a covariance /
//                                                     SVSF-KF combination)
//    * Utkin, V. I. (1992). Sliding Modes in Control and Optimization. Springer.
//                                             (variable-structure / sliding mode)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, Ch. 8.                                (EKF linearisation)
//
//  SVSF gain --------------------------------------------------------------
//  The SVSF is a predictor-corrector whose corrective action is a *discontinuous*
//  (sliding-mode) function of the a-priori output error, smoothed inside a
//  boundary layer of half-width psi ("existence subspace"):
//
//      e_{k+1|k}   = y_{k+1} - h(x_{k+1|k}, u_{k+1})          a-priori residual
//      e_{k|k}                                                a-posteriori residual
//                                                             of the PREVIOUS step
//      k_svsf = ( |e_{k+1|k}|_abs + gamma |e_{k|k}|_abs ) o sat( e_{k+1|k} / psi ), (1)
//
//  with o the Hadamard product, |.|_abs elementwise absolute value and
//  gamma in [0,1) the "SVSF convergence rate".  Outside the boundary layer
//  sat(.) = sgn(.) and, because the correction is applied so that
//  H dx = k_svsf exactly,
//
//      e_{k+1|k+1} = e_{k+1|k} - k_svsf = -gamma |e_{k|k}| sgn(e_{k+1|k}),      (2)
//
//  i.e. |e| contracts by the factor gamma every step - the sliding-mode
//  convergence result of Habibi (2007, §III.B), obtained WITHOUT any knowledge
//  of the noise statistics and robust to bounded modelling error, provided psi
//  covers the "existence subspace" (noise + model error).  Inside the layer the
//  correction becomes smooth (and quadratic in e), which removes chattering at
//  the price of a low gain for small residuals.
//
//  n_x > n_y : the SVSF-KF combination ------------------------------------
//  For the cell model n_y = 1 < n_x = 4, so (1) only determines the correction
//  inside the one-dimensional row space of H.  We use the covariance-weighted
//  right inverse of H (which is the Kalman gain direction, normalised)
//
//      H^+ = P H^T (H P H^T)^{-1}   (opts.covariance_pinv, default)
//      H^+ = H^T (H H^T)^{-1}       (Moore-Penrose, opts.covariance_pinv = false)
//
//  - both satisfy H H^+ = I, so (2) holds either way - and give the unobserved
//  complement (the null space of H, dimension n_x - n_y = 3) the ordinary EKF
//  gain, as proposed by Gadsden & Habibi:
//
//      dx = H^+ k_svsf + (I - H^+ H) K_EKF e_{k+1|k}.                            (3)
//
//  Writing (3) as dx = K_eff e with
//      K_eff = H^+ diag(ratio) + (I - H^+ H) K_EKF,
//      ratio_j = (|e_j| + gamma|e_j^{k|k}|)/psi_j      if |e_j| <= psi_j
//              = (|e_j| + gamma|e_j^{k|k}|)/|e_j|      otherwise                 (4)
//  (no division by zero: psi_j > 0 and |e_j| > psi_j in the second branch)
//  lets the covariance be propagated with the Joseph form of the gain that was
//  actually applied - this is the "SVSF with a covariance derivation" of
//  Gadsden & Habibi (2010) and gives the filter a usable P (and hence a SOC
//  standard deviation) without changing the SVSF state update.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class Svsf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real gamma = Real(0.20);        // SVSF convergence rate, 0 <= gamma < 1
        // Boundary layer.  psi_j = max(psi_abs[j], psi_sigma * sqrt(R_jj)), so the
        // default scales itself with the measurement-noise level of any model.
        Real psi_sigma = Real(5);
        Real psi_abs[NY];
        bool covariance_pinv = true;    // P-weighted right inverse of H (SVSF-KF)
        bool kf_nullspace = true;       // EKF gain on the unobserved complement
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
        Options() { for (int j = 0; j < NY; ++j) psi_abs[j] = Real(0); }
    } opts;

    explicit Svsf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; eprev_ = Y(); }

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
        const Y e = y - m_.h(x_, u);                       // a-priori residual e_{k+1|k}
        innovation_ = e;

        Mat<NY, NY> S = H * P_ * H.t() + R; S.symmetrize(); S_ = S;
        const Mat<NY, NY> Sinv = inverse(S);
        if (!Sinv.is_finite()) return;
        const Mat<NX, NY> Kkf = P_ * H.t() * Sinv;

        // right inverse of H
        Mat<NX, NY> Hp;
        if (opts.covariance_pinv) {
            Mat<NY, NY> HPHt = H * P_ * H.t(); HPHt.symmetrize();
            for (int j = 0; j < NY; ++j) HPHt(j, j) += Real(1e-30);
            const Mat<NY, NY> Ginv = inverse(HPHt);
            if (!Ginv.is_finite()) return;
            Hp = P_ * H.t() * Ginv;
        } else {
            Mat<NY, NY> HHt = H * H.t(); HHt.symmetrize();
            for (int j = 0; j < NY; ++j) HHt(j, j) += Real(1e-30);
            const Mat<NY, NY> Ginv = inverse(HHt);
            if (!Ginv.is_finite()) return;
            Hp = H.t() * Ginv;
        }
        if (!Hp.is_finite()) return;

        // SVSF gain ratios (4)
        Mat<NY, NY> Dr;
        for (int j = 0; j < NY; ++j) {
            Real psi = opts.psi_sigma * std::sqrt(std::max(R(j, j), Real(1e-30)));
            if (opts.psi_abs[j] > psi) psi = opts.psi_abs[j];
            if (!(psi > Real(0))) psi = Real(1e-12);
            const Real ae = std::fabs(e[j]);
            const Real num = ae + opts.gamma * std::fabs(eprev_[j]);
            Dr(j, j) = (ae <= psi) ? (num / psi) : (num / ae);
            psi_[j] = psi;
        }
        Mat<NX, NY> Keff = Hp * Dr;
        if (opts.kf_nullspace) Keff += (Mat<NX, NX>::identity() - Hp * H) * Kkf;
        if (!Keff.is_finite()) return;
        K_ = Keff;

        x_ = x_ + Keff * e;
        const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - Keff * H;
        P_ = IKH * P_ * IKH.t() + Keff * R * Keff.t();
        P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
        eprev_ = y - m_.h(x_, u);                          // a-posteriori residual e_{k+1|k+1}
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    const Y& innovation() const { return innovation_; }
    const Y& post_residual() const { return eprev_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    Real psi(int j) const { return psi_[j]; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Y eprev_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    Real psi_[NY] = {};
};

}  // namespace estkit
