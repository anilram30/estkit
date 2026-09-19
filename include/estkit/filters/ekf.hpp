// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/ekf.hpp — Extended Kalman filter (generic over the model concept)
//
//  References
//    * Kalman, R. E. (1960). A new approach to linear filtering and prediction
//      problems. ASME J. Basic Eng. 82(1), 35-45.                  (linear KF)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, Ch. 8.                                        (EKF)
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs — Part 3. J. Power Sources 134,
//      277-292.                                                       (EKF for SOC)
//    * Bucy, R. S. & Joseph, P. D. (1968). Filtering for Stochastic Processes
//      with Applications to Guidance. Wiley.        (Joseph-form covariance update)
//
//  Algorithm (one cycle)
//    time update:      x^- = f(x^+, u_{k-1}),  P^- = F P^+ F^T + Q
//    measurement:      S = H P^- H^T + R,  K = P^- H^T S^{-1}
//                      x^+ = x^- + K (y - h(x^-, u_k)),  P^+ = (I-KH) P^- (I-KH)^T + K R K^T
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class Ekf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        bool joseph = true;          // Joseph stabilised covariance update
        bool apply_constraints = true;   // project the estimate with Model::constrain (if defined)
        Real q_scale = Real(1);      // multiplicative tuning of Q
        Real r_scale = Real(1);      // multiplicative tuning of R
    } opts;

    explicit Ekf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; }

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
        const Mat<NY, NY> S = H * P_ * H.t() + R;
        const Mat<NX, NY> K = P_ * H.t() * inverse(S);
        innovation_ = y - m_.h(x_, u);
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
    }
    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
};

}  // namespace estkit
