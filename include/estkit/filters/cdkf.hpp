// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/cdkf.hpp — Second-order divided-difference filter (DD2), also
//  known as the central-difference Kalman filter (CDKF).
//
//  References
//    * Norgaard, M., Poulsen, N. K. & Ravn, O. (2000). New developments in state
//      estimation for nonlinear systems. Automatica 36(11), 1627-1638.  (primary;
//      DD1 = first-order, DD2 = second-order divided-difference filter)
//    * Ito, K. & Xiong, K. (2000). Gaussian filters for nonlinear filtering
//      problems. IEEE Transactions on Automatic Control 45(5), 910-927.
//      (independently derived "central difference filter")
//    * van der Merwe, R. & Wan, E. A. (2001). The square-root unscented Kalman
//      filter for state and parameter-estimation. Proc. IEEE ICASSP, vol. 6,
//      3461-3464.                       (CDKF as a member of the sigma-point family)
//    * Schei, T. S. (1997). A finite-difference method for linearization in
//      nonlinear estimation algorithms. Automatica 33(11), 2053-2058.
//
//  Instead of a Taylor expansion the DD2 filter uses Stirling's polynomial
//  interpolation of f and h around the mean, with a step length h along the
//  columns s_i of the Cholesky factor S (P = S S^T).  For a function g and a
//  decorrelated coordinate z (x = xbar + S z) define
//
//      g_i^+ = g(xbar + h s_i),   g_i^- = g(xbar - h s_i),   g_0 = g(xbar),
//      D_i^{(1)} = (g_i^+ - g_i^-) / (2h),
//      D_i^{(2)} = sqrt(h^2-1)/(2h^2) * (g_i^+ + g_i^- - 2 g_0).
//
//  Then (Norgaard et al. 2000, Eqs. 30-34)
//
//      E[g] ~ (h^2-n)/h^2 g_0 + 1/(2h^2) sum_i (g_i^+ + g_i^-)                 (1)
//      Cov[g] ~ D^{(1)} D^{(1)T} + D^{(2)} D^{(2)T}                            (2)
//      Cov[x,g] ~ S D^{(1)T}                                                   (3)
//
//  where n is the state dimension.  (1)-(2) reproduce the true mean and covariance
//  of a Gaussian to second order in the Taylor sense; the interpolation step that
//  additionally matches the fourth moment of a Gaussian is h^2 = 3, the default
//  used here and recommended in the primary reference.  The second divided
//  difference D^{(2)} is the term that distinguishes DD2 from DD1 (and from the
//  EKF); it makes the predicted covariance strictly larger, which is exactly the
//  "extra" uncertainty caused by the curvature of f and h.
//
//  Only 2n+1 model evaluations per stage are needed and no Jacobians at all.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class Cdkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        Real h_step = Real(0);             // interpolation step; 0 -> sqrt(3) (Gaussian-optimal)
        bool second_order = true;          // include the D^(2) terms (false -> DD1)
        bool apply_constraints = true;     // project with Model::constrain (if defined)
        Real q_scale = Real(1);            // multiplicative tuning of Q
        Real r_scale = Real(1);            // multiplicative tuning of R
    } opts;

    explicit Cdkf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; }

    Real step_h() const { return opts.h_step > Real(0) ? opts.h_step : std::sqrt(Real(3)); }

    void predict(const U& u) {
        const Real h = step_h(), h2 = h * h;
        const Mat<NX, NX> S = cholesky_safe(P_);
        const X f0 = m_.f(x_, u);
        X fp[NX], fm[NX];
        for (int i = 0; i < NX; ++i) {
            const X c = S.col(i) * h;
            fp[i] = m_.f(x_ + c, u); fm[i] = m_.f(x_ - c, u);
        }
        X xm = f0 * ((h2 - Real(NX)) / h2);
        for (int i = 0; i < NX; ++i) xm += (fp[i] + fm[i]) * (Real(1) / (Real(2) * h2));
        Mat<NX, NX> Pm;
        const Real c1 = Real(1) / (Real(4) * h2);                   // weight of D^(1) D^(1)T
        const Real c2 = (h2 - Real(1)) / (Real(4) * h2 * h2);       // weight of D^(2) D^(2)T
        for (int i = 0; i < NX; ++i) {
            const X d1 = fp[i] - fm[i];
            Pm += outer(d1, d1) * c1;
            if (opts.second_order) { const X d2 = fp[i] + fm[i] - f0 * Real(2); Pm += outer(d2, d2) * c2; }
        }
        x_ = xm;
        P_ = Pm + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const {
        const Real h = step_h(), h2 = h * h;
        const Mat<NX, NX> S = cholesky_safe(P_);
        Y ym = m_.h(x_, u) * ((h2 - Real(NX)) / h2);
        for (int i = 0; i < NX; ++i) {
            const X c = S.col(i) * h;
            ym += (m_.h(x_ + c, u) + m_.h(x_ - c, u)) * (Real(1) / (Real(2) * h2));
        }
        return ym;
    }

    void update(const Y& y, const U& u) {
        const Real h = step_h(), h2 = h * h;
        const Mat<NX, NX> S = cholesky_safe(P_);
        const Y z0 = m_.h(x_, u);
        Y zp[NX], zm[NX];
        for (int i = 0; i < NX; ++i) {
            const X c = S.col(i) * h;
            zp[i] = m_.h(x_ + c, u); zm[i] = m_.h(x_ - c, u);
        }
        Y ym = z0 * ((h2 - Real(NX)) / h2);
        for (int i = 0; i < NX; ++i) ym += (zp[i] + zm[i]) * (Real(1) / (Real(2) * h2));
        Mat<NY, NY> Pyy = m_.R(x_, u) * opts.r_scale;
        Mat<NX, NY> Pxy;
        const Real c1 = Real(1) / (Real(4) * h2);
        const Real c2 = (h2 - Real(1)) / (Real(4) * h2 * h2);
        for (int i = 0; i < NX; ++i) {
            const Y e1 = zp[i] - zm[i];
            Pyy += outer(e1, e1) * c1;
            if (opts.second_order) { const Y e2 = zp[i] + zm[i] - z0 * Real(2); Pyy += outer(e2, e2) * c2; }
            Pxy += outer(X(S.col(i)), e1) * (Real(1) / (Real(2) * h));   // S D^(1)T
        }
        const Mat<NX, NY> K = Pxy * inverse(Pyy);
        if (!K.is_finite()) return;                   // keep the prior rather than produce NaN
        innovation_ = y - ym; S_ = Pyy;
        x_ = x_ + K * innovation_;
        P_ = P_ - K * Pyy * K.t();
        P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Mat<NY, NY> S_;
};

}  // namespace estkit
