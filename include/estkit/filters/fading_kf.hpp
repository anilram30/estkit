// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/fading_kf.hpp — Fading-memory (exponentially weighted)
//  extended Kalman filter.
//
//  References
//    * Sorenson, H. W. & Sacks, J. E. (1971). Recursive fading memory filtering.
//      Information Sciences 3(2), 101-119.
//    * Simon, D. (2006). Optimal State Estimation: Kalman, H-infinity, and
//      Nonlinear Approaches. Wiley, Hoboken, NJ, Sec. 7.4 (fading-memory filter).
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, New York, Ch. 8.                  (EKF this builds on)
//    * Anderson, B. D. O. & Moore, J. B. (1979). Optimal Filtering,
//      Prentice-Hall, Englewood Cliffs, NJ, Sec. 4.4.  (divergence, exp. data
//      weighting)
//
//  Idea.  The Kalman filter weights all past data equally, so P^+ collapses and
//  the gain goes to its steady-state value; any un-modelled bias then produces a
//  slowly decaying estimation error that the filter cannot correct ("divergence
//  through model mismatch").  Sorenson & Sacks minimise instead the
//  exponentially weighted cost
//
//      J_k = sum_{j=1}^{k} alpha^{-2(k-j)} ( e_j^T R_j^{-1} e_j + ... )     (1)
//
//  with alpha >= 1, i.e. data j steps old are discounted by alpha^{-2j}.  The
//  resulting recursion is the ordinary Kalman filter with the single change
//
//      P^-_k = alpha^2 F_{k-1} P^+_{k-1} F_{k-1}^T + Q_{k-1} ,              (2)
//
//  which bounds the filter memory to roughly
//
//      tau = 1 / (2 ln alpha)   samples                                     (3)
//
//  (alpha = 1.01 -> tau ~ 50 samples; alpha = 1 recovers the plain EKF).  The
//  gain therefore never decays below a floor, and the filter keeps tracking
//  slowly varying parameters and un-modelled biases at the price of a higher
//  noise-induced variance.  NOTE that alpha is a PER-SAMPLE factor: the textbook
//  values 1.01-1.05 assume a filter running at roughly the time scale of the
//  plant.  At the 10 Hz sampling rate of this benchmark they correspond to only
//  0.5-5 s of memory, far shorter than the cell's slow RC time constant
//  (tau_2 = 120 s); the default alpha = 1.002 gives tau = 250 samples = 25 s.
//
//  Covariance limiting (essential on weakly observable models).  Equation (2)
//  multiplies EVERY direction of the state space by alpha^2, including the ones
//  the measurement barely constrains.  For a direction whose observability
//  Gramian grows more slowly than alpha^{2k}, P diverges geometrically; the
//  gain in that direction then changes sign and the filter blows up.  The
//  implementation therefore bounds the prediction covariance,
//
//      P^-_ii <= p_max_scale * P_0,ii ,                                     (4)
//
//  i.e. no state's predicted standard deviation may exceed sqrt(p_max_scale)
//  times its value at initialisation (10 % by default), with the off-diagonal
//  entries of the affected row/column
//  rescaled by the same square-root factor so that the correlation structure
//  (and hence positive semi-definiteness) is preserved.  Covariance limiting of
//  this kind is the classical remedy for the divergence of exponentially
//  weighted filters (Anderson & Moore 1979, Sec. 4.4; Simon 2006, Sec. 7.4).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class FadingMemoryKf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real alpha = Real(1.002);    // fading factor, >= 1; memory 1/(2 ln alpha) samples (250 here)
        Real p_max_scale = Real(0.01);  // ceiling P^-_ii <= p_max_scale*P0_ii (sigma <= 10 % of sigma_0)
        bool joseph = true;
        bool apply_constraints = true;
        Real q_scale = Real(1);
        Real r_scale = Real(1);
    } opts;

    explicit FadingMemoryKf(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0;
        for (int i = 0; i < NX; ++i) p_max_[i] = P0(i, i);
        innovation_ = Y(); S_ = Mat<NY, NY>(); K_ = Mat<NX, NY>();
    }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        const Real a2 = opts.alpha * opts.alpha;
        x_ = m_.f(x_, u);
        P_ = (F * P_ * F.t()) * a2 + m_.Q(x_, u) * opts.q_scale;   // eq. (2)
        P_.symmetrize();
        limit_covariance(P_, p_max_, opts.p_max_scale);            // eq. (4)
    }

    // P_ii <- min(P_ii, s*pmax_i), row/column rescaled by sqrt so that the
    // correlation coefficients (and positive semi-definiteness) are preserved.
    static void limit_covariance(Mat<NX, NX>& P, const Real (&pmax)[NX], Real s) {
        if (!(s > Real(0))) return;
        for (int i = 0; i < NX; ++i) {
            const Real lim = s * pmax[i];
            if (P(i, i) > lim && P(i, i) > Real(0)) {
                const Real f = std::sqrt(lim / P(i, i));
                for (int j = 0; j < NX; ++j) { P(i, j) *= f; P(j, i) *= f; }
                P(i, i) = lim;
            }
        }
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> R = m_.R(x_, u) * opts.r_scale;
        const Mat<NX, NX> Pminus = P_;
        const Mat<NY, NY> S = H * Pminus * H.t() + R;
        const Mat<NX, NY> K = Pminus * H.t() * inverse(S);
        if (!K.is_finite()) return;                 // keep the previous estimate
        innovation_ = y - m_.h(x_, u);
        S_ = S; K_ = K;
        x_ = x_ + K * innovation_;
        if (opts.joseph) {
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_ = IKH * Pminus * IKH.t() + K * R * K.t();
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
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Real p_max_[NX] = {};
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
};

}  // namespace estkit
