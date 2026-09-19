// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/smoothers/fixed_lag.hpp — real-time fixed-lag smoother by state
//  augmentation (extended/linearised form), lag L.
//
//  References
//    * Moore, J. B. (1973). Discrete-time fixed-lag smoothing algorithms.
//      Automatica 9(2), 163-173.
//    * Anderson, B. D. O. & Moore, J. B. (1979). Optimal Filtering,
//      Prentice-Hall, Ch. 7 (fixed-lag smoothing as filtering of an augmented
//      system).
//    * Rauch, H. E., Tung, F. & Striebel, C. T. (1965). Maximum likelihood
//      estimates of linear dynamic systems. AIAA Journal 3(8), 1445-1450.
//    * Simon, D. (2006). Optimal State Estimation, Wiley, §9.4.
//
//  Construction.  Stack the last L+1 states,
//
//      xi_k = [ x_k ; x_{k-1} ; ... ; x_{k-L} ] in R^{n_x (L+1)},
//
//  whose dynamics are the original dynamics on the leading block and a pure
//  delay line on the rest:
//
//      Fa = [ F 0 ... 0 0 ;    Qa = blkdiag(Q, 0, ..., 0),
//             I 0 ... 0 0 ;    Ha = [ H 0 ... 0 ].
//             0 I ... 0 0 ;
//             ...        ]
//
//  An ordinary EKF on xi therefore produces, in its last block, the fixed-lag
//  smoothed estimate xhat_{k-L|k} (Anderson & Moore 1979, §7.3): the delayed
//  copies are corrected by every measurement that arrives after them, which is
//  exactly the smoothing operation, but the algorithm stays RECURSIVE and
//  causal, so it can run on the vehicle with a fixed L-sample output delay.
//
//  Block implementation.  Writing the augmented covariance as blocks
//  Pi_{ij} = cov(x_{k-i}, x_{k-j}), i,j = 0..L, the time update is
//
//      Pi'_{00} = F Pi_{00} F^T + Q,   Pi'_{0j} = F Pi_{0,j-1} (j >= 1),
//      Pi'_{ij} = Pi_{i-1,j-1}  (i,j >= 1),
//
//  i.e. O(L n_x^3) instead of the O(L^3 n_x^3) of a dense (L+1)n_x filter, and
//  the (i,j >= 1) blocks are a pure memory shift.  With
//  hP_j = H Pi_{0j} (n_y x n_x) the measurement update is
//
//      S    = hP_0 H^T + R,        K_i = hP_i^T S^{-1},
//      x_i += K_i (y - h(x_0,u)),
//      Pi_{ij} <- Pi_{ij} - K_i hP_j - (K_j hP_i)^T + K_i S K_j^T,
//
//  the last line being the Joseph form specialised to Ha = [H 0 ... 0]; it costs
//  O(L^2 n_x^2 n_y).  For the cell model (n_x = 4, n_y = 1, L = 10) the
//  augmented state has 44 components but one cycle is only a few thousand flops
//  and the memory is (L+1)^2 n_x^2 = 1936 reals (~15 kB) — an order of magnitude
//  less work than a dense 44x44 Kalman filter.
//
//  OUTPUT CONVENTION.  x() returns the L-step-DELAYED smoothed state, i.e. the
//  estimate of x_{k-L} given data up to k.  soc_delayed_index() reports how many
//  samples of delay are currently in effect (it ramps from 0 to L during the
//  first L steps).  predict_measurement() uses the CURRENT block x_0 so the
//  voltage residual reported by the benchmark remains a genuine one-step-ahead
//  prediction.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model, int L = 10>
class FixedLagSmoother {
public:
    static_assert(L >= 1, "fixed-lag smoother needs L >= 1");
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int LAG = L;
    static constexpr int NAUG = NX * (L + 1);      // augmented dimension (documentation)
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    explicit FixedLagSmoother(const Model& m) : m_(m) {}

    // All delayed copies start equal to x0 with full correlation, which is the
    // correct statement that x_{-1} = ... = x_{-L} = x_0 a priori.
    void init(const X& x0, const Mat<NX, NX>& P0) {
        for (int idx = 0; idx <= L; ++idx) {
            x_[idx] = x0;
            for (int j = 0; j <= L; ++j) P_[idx][j] = P0;
        }
        n_steps_ = 0;
    }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_[0], u);
        // --- new leading row of the covariance, from the OLD blocks ---
        Mat<NX, NX> row0[L + 1];
        const X xn = m_.f(x_[0], u);
        row0[0] = F * P_[0][0] * F.t() + m_.Q(xn, u) * opts.q_scale;
        for (int j = 1; j <= L; ++j) row0[j] = F * P_[0][j - 1];
        // --- shift the delay line (descending indices: read before write) ---
        for (int idx = L; idx >= 1; --idx)
            for (int j = L; j >= 1; --j) P_[idx][j] = P_[idx - 1][j - 1];
        for (int idx = L; idx >= 1; --idx) x_[idx] = x_[idx - 1];
        x_[0] = xn;
        for (int j = 0; j <= L; ++j) { P_[0][j] = row0[j]; P_[j][0] = row0[j].t(); }
        P_[0][0].symmetrize();
        if (n_steps_ < L) ++n_steps_;
    }

    Y predict_measurement(const U& u) const { return m_.h(x_[0], u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_[0], u);
        const Mat<NY, NY> R = m_.R(x_[0], u) * opts.r_scale;
        Mat<NY, NX> hP[L + 1];
        for (int j = 0; j <= L; ++j) hP[j] = H * P_[0][j];
        const Mat<NY, NY> S = hP[0] * H.t() + R;
        const Mat<NY, NY> Sinv = inverse(S);
        if (!Sinv.is_finite()) return;
        Mat<NX, NY> K[L + 1];
        for (int idx = 0; idx <= L; ++idx) K[idx] = hP[idx].t() * Sinv;
        innovation_ = y - m_.h(x_[0], u);
        S_ = S;
        for (int idx = 0; idx <= L; ++idx) x_[idx] = x_[idx] + K[idx] * innovation_;
        for (int idx = 0; idx <= L; ++idx)
            for (int j = 0; j <= L; ++j)
                P_[idx][j] = P_[idx][j] - K[idx] * hP[j] - (K[j] * hP[idx]).t() + K[idx] * S * K[j].t();
        for (int idx = 0; idx <= L; ++idx) P_[idx][idx].symmetrize();
        if (opts.apply_constraints)
            for (int idx = 0; idx <= L; ++idx) x_[idx] = constrain(m_, x_[idx]);
    }

    // --- generic accessors: the reported estimate is the DELAYED smoothed one ---
    const X& x() const { return x_[lag_index_()]; }
    const Mat<NX, NX>& P() const { const int d = lag_index_(); return P_[d][d]; }
    // the filtered (zero-lag) estimate, for reference
    const X& x_current() const { return x_[0]; }
    const Mat<NX, NX>& P_current() const { return P_[0][0]; }
    // how many samples of delay the reported estimate carries right now
    int soc_delayed_index() const { return lag_index_(); }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    int lag_index_() const { return n_steps_ < L ? n_steps_ : L; }

    Model m_;
    X x_[L + 1];
    Mat<NX, NX> P_[L + 1][L + 1];
    Y innovation_;
    Mat<NY, NY> S_;
    int n_steps_ = 0;
};

}  // namespace estkit
