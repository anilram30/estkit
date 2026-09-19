// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/iekf.hpp — Iterated extended Kalman filter (IEKF)
//
//  References
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory.
//      Academic Press, New York, Section 8.3 ("iterated filter").
//    * Bell, B. M. & Cathey, F. W. (1993). The iterated Kalman filter update as a
//      Gauss-Newton method. IEEE Transactions on Automatic Control 38(2), 294-297.
//    * Gelb, A. (ed., 1974). Applied Optimal Estimation. MIT Press, Section 6.1.
//    * Bucy, R. S. & Joseph, P. D. (1968). Filtering for Stochastic Processes with
//      Applications to Guidance. Interscience.        (Joseph-form covariance update)
//
//  Idea
//    The EKF measurement update linearises h at the PRIOR mean x^-.  When the
//    innovation is large (poor initialisation, sparse measurements) this point is a
//    bad expansion centre.  The IEKF re-linearises at the current posterior iterate
//    and repeats the update; Bell & Cathey (1993) show that the recursion
//
//      x_{j+1} = x^- + K_j [ y - h(x_j,u) - H_j (x^- - x_j) ],   H_j = dh/dx|_{x_j}
//      K_j     = P^- H_j^T (H_j P^- H_j^T + R)^{-1}
//
//    is exactly the Gauss-Newton iteration for the maximum-a-posteriori problem
//
//      min_x  (x - x^-)^T (P^-)^{-1} (x - x^-) + (y - h(x,u))^T R^{-1} (y - h(x,u)),
//
//    so it inherits local quadratic-in-the-residual convergence; one iteration
//    reproduces the EKF exactly.  The covariance is evaluated with the Jacobian at
//    the converged point (the Gauss-Newton approximation of the Hessian of the cost).
//
//  Algorithm (one cycle)
//    time update:     x^- = f(x^+,u_{k-1}),  P^- = F P^+ F^T + Q       (as the EKF)
//    measurement:     iterate the Gauss-Newton step above until
//                     ||x_{j+1} - x_j|| <= tol * (1 + ||x_j||) or j = max_iter
//                     P^+ = (I-K H)P^-(I-K H)^T + K R K^T with the final (H,K)
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class Iekf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        int  max_iter = 5;             // Gauss-Newton iterations (1 == plain EKF)
        Real tol = Real(1e-6);         // relative step-size stopping tolerance
        Real step = Real(1);           // damping in (0,1]: x_{j+1} = x_j + step*(GN step)
        bool joseph = true;            // Joseph stabilised covariance update
        bool apply_constraints = true; // project the estimate with Model::constrain (if defined)
        Real q_scale = Real(1);        // multiplicative tuning of Q
        Real r_scale = Real(1);        // multiplicative tuning of R
    } opts;

    explicit Iekf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; iters_ = 0; }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        P_ = F * P_ * F.t() + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const X xprior = x_;                 // x^-  (expansion centre of the prior term)
        const Mat<NY, NY> R = m_.R(x_, u) * opts.r_scale;
        innovation_ = y - m_.h(xprior, u);   // a-priori innovation (diagnostics)
        X xj = xprior;                       // current iterate
        Mat<NY, NX> H = jac_H(m_, xj, u);    // Jacobian of the last executed step
        Mat<NX, NY> K;                       // gain of the last executed step
        const int it_max = opts.max_iter < 1 ? 1 : opts.max_iter;
        iters_ = 0;
        for (int j = 0; j < it_max; ++j) {
            const Mat<NY, NX> Hj = jac_H(m_, xj, u);
            const Mat<NY, NY> Sj = Hj * P_ * Hj.t() + R;
            const Mat<NX, NY> Kj = P_ * Hj.t() * inverse(Sj);
            if (!Kj.is_finite()) break;
            // residual of the linearised measurement seen from the prior mean
            const Y res = y - m_.h(xj, u) - Hj * (xprior - xj);
            const X d = (xprior + Kj * res - xj) * opts.step;
            if (!d.is_finite()) break;
            xj = xj + d; H = Hj; K = Kj; S_ = Sj;
            ++iters_;
            if (d.norm() <= opts.tol * (Real(1) + xj.norm())) break;
        }
        x_ = xj;
        // Covariance with the Jacobian/gain of the last Gauss-Newton step (Gelb 1974,
        // Section 6.1).  With max_iter = 1 this reproduces the EKF exactly; at
        // convergence H(x_{N-1}) = H(x_N), so the choice is immaterial.
        if (K.is_finite()) {
            if (opts.joseph) {
                const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
                P_ = IKH * P_ * IKH.t() + K * R * K.t();
            } else {
                P_ = (Mat<NX, NX>::identity() - K * H) * P_;
            }
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
    int iterations() const { return iters_; }   // iterations used in the last update
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Mat<NY, NY> S_;
    int iters_ = 0;
};

}  // namespace estkit
