// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/enkf.hpp — stochastic ensemble Kalman filter (perturbed
//  observations)
//
//  References
//    * Evensen, Geir (1994). Sequential data assimilation with a nonlinear
//      quasi-geostrophic model using Monte Carlo methods to forecast error
//      statistics. Journal of Geophysical Research: Oceans 99(C5), 10143-10162.
//    * Burgers, Gerrit, van Leeuwen, Peter Jan & Evensen, Geir (1998). Analysis
//      scheme in the ensemble Kalman filter. Monthly Weather Review 126(6),
//      1719-1724.                       (why the observations must be perturbed)
//    * Evensen, Geir (2003). The ensemble Kalman filter: theoretical formulation
//      and practical implementation. Ocean Dynamics 53(4), 343-367.
//    * Anderson, Jeffrey L. & Anderson, Stephen L. (1999). A Monte Carlo
//      implementation of the nonlinear filtering problem to produce ensemble
//      assimilations and forecasts. Monthly Weather Review 127(12), 2741-2758.
//      (multiplicative covariance inflation)
//
//  The EnKF replaces the covariance propagation of the Kalman filter by the
//  sample covariance of an ensemble {x^i}_{i=1..N} that is integrated through the
//  true nonlinear dynamics:
//
//      forecast   x^{f,i}_k = f(x^{a,i}_{k-1}, u_{k-1}) + w^i,  w^i ~ N(0,Q)    (1)
//      xbar^f = (1/N) sum_i x^{f,i},   X' = [x^{f,i} - xbar^f] / sqrt(N-1)      (2)
//      y^i    = h(x^{f,i}, u_k),       ybar = (1/N) sum_i y^i,
//               Y' = [y^i - ybar] / sqrt(N-1)                                   (3)
//      P_xy = X' Y'^T,   P_yy = Y' Y'^T + R                                     (4)
//      K    = P_xy P_yy^{-1}                                                    (5)
//      analysis   x^{a,i} = x^{f,i} + K ( y_k + v^i - y^i ),  v^i ~ N(0,R)      (6)
//
//  Burgers, van Leeuwen & Evensen (1998) showed that the observation
//  perturbations v^i in (6) are not optional: updating every member with the same
//  y_k produces an analysis ensemble whose covariance is (I-KH)P^f(I-KH)^T, i.e.
//  systematically too small by K R K^T, and the filter then diverges.  With
//  perturbed observations the analysis covariance is (I-KH)P^f in expectation,
//  exactly as in the Kalman filter.  The perturbations are centred
//  (sum_i v^i = 0) to remove the leading sampling error (Evensen 2003 §4.3).
//
//  Nothing in (1)-(6) requires a Jacobian: H never appears, only the ensemble in
//  observation space.  The cost is O(N (n_x + cost(f) + cost(h)) + N n_x n_y +
//  n_y^3) per step and the storage is N n_x reals — no n_x x n_x covariance
//  propagation, which is why the method scales to the very large n_x of
//  geophysical assimilation.  For small n_x (here 4) it is simply a Monte-Carlo
//  approximation of the UKF/EKF with sampling error O(N^{-1/2}).
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../core/rng.hpp"
#include "pf_common.hpp"

namespace estkit {

template <class Model, int N = 50>
class EnKf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NENS = N;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real inflation = Real(1.001);     // multiplicative prior inflation per step (Anderson 1999)
        Real q_scale = Real(1);
        Real r_scale = Real(1);
        bool center_perturbations = true; // enforce sum_i v^i = 0
        bool apply_constraints = true;
    } opts;

    explicit EnKf(const Model& m) : m_(m) {}
    void seed(uint64_t s) { rng_.reseed(s); }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        const Mat<NX, NX> L = cholesky_safe(P0);
        for (int i = 0; i < N; ++i) {
            X xi = x0 + L * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xi = constrain(m_, xi);
            e_[i] = xi;
        }
        refresh_moments();
    }

    // ---- forecast step (1) ---------------------------------------------------
    void predict(const U& u) {
        const Mat<NX, NX> Lq = cholesky_safe(m_.Q(x_, u) * opts.q_scale);
        for (int i = 0; i < N; ++i) {
            X xn = m_.f(e_[i], u) + Lq * rng_.normal_vec<NX>();
            if (opts.apply_constraints) xn = constrain(m_, xn);
            e_[i] = xn;
        }
        refresh_moments();
    }

    Y predict_measurement(const U& u) const {
        Y ym;
        for (int i = 0; i < N; ++i) ym += m_.h(e_[i], u);
        return ym / Real(N);
    }

    // ---- analysis step (2)-(6) ----------------------------------------------
    void update(const Y& y, const U& u) {
        // prior inflation about the ensemble mean
        if (opts.inflation != Real(1)) {
            const X xb = mean();
            for (int i = 0; i < N; ++i) e_[i] = xb + (e_[i] - xb) * opts.inflation;
        }
        const X xb = mean();
        Y yb;
        for (int i = 0; i < N; ++i) { ys_[i] = m_.h(e_[i], u); yb += ys_[i]; }
        yb = yb / Real(N);
        const Real inv = Real(1) / Real(N - 1);
        Mat<NX, NY> Pxy;
        Mat<NY, NY> Pyy = m_.R(xb, u) * opts.r_scale;
        for (int i = 0; i < N; ++i) {
            const X dx = e_[i] - xb;
            const Y dy = ys_[i] - yb;
            Pxy += outer(dx, dy) * inv;
            Pyy += outer(dy, dy) * inv;
        }
        const Mat<NX, NY> K = Pxy * inverse(Pyy);
        // perturbed observations (6), centred
        const Mat<NY, NY> Lr = cholesky_safe(m_.R(xb, u) * opts.r_scale);
        Y vbar;
        for (int i = 0; i < N; ++i) { vs_[i] = Lr * rng_.normal_vec<NY>(); vbar += vs_[i]; }
        if (opts.center_perturbations) { vbar = vbar / Real(N); for (int i = 0; i < N; ++i) vs_[i] -= vbar; }
        for (int i = 0; i < N; ++i) {
            X xa = e_[i] + K * (y + vs_[i] - ys_[i]);
            if (opts.apply_constraints) xa = constrain(m_, xa);
            e_[i] = xa;
        }
        K_ = K;
        refresh_moments();
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    const Mat<NX, NY>& K() const { return K_; }
    const X* ensemble() const { return e_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    X mean() const {
        X s;
        for (int i = 0; i < N; ++i) s += e_[i];
        return s / Real(N);
    }
    void refresh_moments() {
        x_ = mean();
        P_ = Mat<NX, NX>();
        const Real inv = Real(1) / Real(N - 1);
        for (int i = 0; i < N; ++i) { const X d = e_[i] - x_; P_ += outer(d, d) * inv; }
        P_.symmetrize();
    }

    Model m_;
    Rng rng_{1};
    X e_[N];          // the ensemble
    Y ys_[N];         // ensemble in observation space
    Y vs_[N];         // observation perturbations
    X x_;
    Mat<NX, NX> P_;
    Mat<NX, NY> K_;
};

}  // namespace estkit
