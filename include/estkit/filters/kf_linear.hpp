// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/kf_linear.hpp — the plain (linear) Kalman filter applied to a
//  nonlinear model that is linearised ONCE, at the initial condition.
//
//  References
//    * Kalman, R. E. (1960). A new approach to linear filtering and prediction
//      problems. ASME Journal of Basic Engineering 82(1), 35-45.
//    * Kalman, R. E. & Bucy, R. S. (1961). New results in linear filtering and
//      prediction theory. ASME Journal of Basic Engineering 83(1), 95-108.
//    * Anderson, B. D. O. & Moore, J. B. (1979). Optimal Filtering. Prentice-Hall,
//      Englewood Cliffs, NJ, Ch. 4 (time-invariant filter, Riccati convergence).
//    * Bucy, R. S. & Joseph, P. D. (1968). Filtering for Stochastic Processes with
//      Applications to Guidance. Interscience.        (Joseph-form covariance update)
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs - Part 3. Journal of Power Sources
//      134(2), 277-292.            (the SOC problem this baseline is compared on)
//
//  Idea
//    The Kalman filter is optimal for the linear-Gaussian model
//        x_{k+1} = F x_k + G u_k + w_k,     y_k = H x_k + v_k .
//    Applying it to a nonlinear model requires a linear surrogate.  The EKF
//    re-linearises at every step (x^+_{k-1}, x^-_k); the filter implemented here
//    performs the first-order Taylor expansion ONCE, at the initial estimate
//    (x_0, u_lin), and keeps the resulting constant matrices for the whole run:
//
//        F = df/dx |_(x0,u_lin),        H = dh/dx |_(x0,u_lin)
//        f(x,u) ~ f(x0,u) + F (x - x0)              (affine in x, exact in u)
//        h(x,u) ~ h(x0,u) + H (x - x0)
//
//  Algorithm (one cycle)
//    time update:      x^- = F (x^+ - x0) + f(x0, u_{k-1}),   P^- = F P^+ F^T + Q
//    measurement:      yhat = h(x0,u_k) + H (x^- - x0)
//                      S = H P^- H^T + R,   K = P^- H^T S^{-1}
//                      x^+ = x^- + K (y_k - yhat)
//                      P^+ = (I-KH) P^- (I-KH)^T + K R K^T
//    With `freeze_noise` the pair (Q,R) is also frozen at (x0,u_lin); the filter is
//    then strictly linear time-invariant, its Riccati recursion is data independent
//    and P_k -> P_inf, K_k -> K_inf (the DARE solution, Anderson & Moore 1979 §4.4),
//    so the whole gain schedule can be computed offline.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class LinearKf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        U    u_lin;                    // input at which F, H (and Q, R) are linearised
        bool freeze_noise = true;      // Q, R evaluated once at (x0, u_lin) -> strictly LTI filter
        bool joseph = true;            // Joseph stabilised covariance update
        bool apply_constraints = true; // project the estimate with Model::constrain (if defined)
        Real q_scale = Real(1);        // multiplicative tuning of Q
        Real r_scale = Real(1);        // multiplicative tuning of R
    } opts;

    explicit LinearKf(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0;
        x_lin_ = x0;
        F_ = jac_F(m_, x_lin_, opts.u_lin);
        H_ = jac_H(m_, x_lin_, opts.u_lin);
        Q_ = m_.Q(x_lin_, opts.u_lin) * opts.q_scale;
        R_ = m_.R(x_lin_, opts.u_lin) * opts.r_scale;
    }

    void predict(const U& u) {
        // x^- = f(x0,u) + F (x^+ - x0):  first-order Taylor about the frozen state x0
        x_ = m_.f(x_lin_, u) + F_ * (x_ - x_lin_);
        const Mat<NX, NX> Q = opts.freeze_noise ? Q_ : Mat<NX, NX>(m_.Q(x_lin_, u) * opts.q_scale);
        P_ = F_ * P_ * F_.t() + Q;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const { return m_.h(x_lin_, u) + H_ * (x_ - x_lin_); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NY> R = opts.freeze_noise ? R_ : Mat<NY, NY>(m_.R(x_lin_, u) * opts.r_scale);
        const Mat<NY, NY> S = H_ * P_ * H_.t() + R;
        const Mat<NX, NY> K = P_ * H_.t() * inverse(S);
        innovation_ = y - predict_measurement(u);
        S_ = S; K_ = K;
        x_ = x_ + K * innovation_;
        if (opts.joseph) {
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H_;
            P_ = IKH * P_ * IKH.t() + K * R * K.t();
        } else {
            P_ = (Mat<NX, NX>::identity() - K * H_) * P_;
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
    // the frozen linearisation (for inspection / offline gain computation)
    const Mat<NX, NX>& F() const { return F_; }
    const Mat<NY, NX>& H() const { return H_; }
    const X& x_lin() const { return x_lin_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    X x_lin_;                 // frozen linearisation point
    Mat<NX, NX> P_;
    Mat<NX, NX> F_;           // frozen df/dx
    Mat<NY, NX> H_;           // frozen dh/dx
    Mat<NX, NX> Q_;           // frozen process-noise covariance
    Mat<NY, NY> R_;           // frozen measurement-noise covariance
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
};

}  // namespace estkit
