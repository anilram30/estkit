// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/particle/rbpf.hpp — marginalised (Rao-Blackwellised) particle filter
//  for mixed linear/nonlinear state-space models.
//
//  References
//    * Schon, Thomas, Gustafsson, Fredrik & Nordlund, Per-Johan (2005).
//      Marginalized particle filters for mixed linear/nonlinear state-space
//      models. IEEE Transactions on Signal Processing 53(7), 2279-2289.
//    * Doucet, Arnaud, de Freitas, Nando, Murphy, Kevin & Russell, Stuart (2000).
//      Rao-Blackwellised particle filtering for dynamic Bayesian networks. In:
//      Proc. 16th Conference on Uncertainty in Artificial Intelligence (UAI),
//      pp. 176-183.
//    * Chen, Rong & Liu, Jun S. (2000). Mixture Kalman filters. Journal of the
//      Royal Statistical Society B 62(3), 493-508.
//    * Casella, George & Robert, Christian P. (1996). Rao-Blackwellisation of
//      sampling schemes. Biometrika 83(1), 81-94.   (the variance argument)
//
//  ---------------------------------------------------------------------------
//  Model class ("conditionally linear", see battery/ecm_mixed_model.hpp for the
//  full concept documentation).  With x = [x_n ; x_l], n_n = NXN, n_l = NXL:
//
//      x_n(k+1) = f_n(x_n(k), u(k)) + w_n(k),              w_n ~ N(0, Q_n)     (1)
//      x_l(k+1) = A_l(u(k)) x_l(k) + b_l(u(k)) + w_l(k),   w_l ~ N(0, Q_l)     (2)
//      y(k)     = h_n(x_n(k),u(k)) + C_l(x_n(k),u(k)) x_l(k) + v(k), v ~ N(0,R) (3)
//
//  Crucially (1) does NOT depend on x_l, and (3) is affine in x_l.  Therefore,
//  conditioned on the trajectory x_n(0:k), the remaining system is linear and
//  Gaussian, so p(x_l(k) | x_n(0:k), y(1:k)) = N(xbar_l^i, P_l^i) is available in
//  closed form from a Kalman filter attached to each particle, and the joint
//  posterior factorises (Schon et al. eq. 6; Doucet et al. 2000 §3):
//
//      p(x_n(0:k), x_l(k) | y(1:k))
//            = p(x_l(k) | x_n(0:k), y(1:k)) * p(x_n(0:k) | y(1:k)).            (4)
//
//  Only the SECOND factor is approximated by particles, so the Monte-Carlo
//  integration happens in an n_n-dimensional space instead of n_n + n_l.
//
//  Rao-Blackwell variance reduction.  For any statistic g, the identity
//      Var[g] = Var[ E(g | x_n) ] + E[ Var(g | x_n) ]  >=  Var[ E(g | x_n) ]   (5)
//  says that replacing the Monte-Carlo average of g by the Monte-Carlo average of
//  its conditional expectation can never increase the variance (Casella & Robert
//  1996).  The marginalised filter estimates E(x_l|x_n) exactly (Kalman), so the
//  entire second term of (5) — the sampling noise of the linear states — is
//  removed, and the sampling noise of x_n is reduced because the importance
//  weights use the exactly marginalised likelihood
//      p(y_k | x_n^i, y_{1:k-1}) = N( y_k ; h_n + C_l xbar_l^i , C_l P_l^i C_l^T + R ).
//
//  ---------------------------------------------------------------------------
//  One cycle for particle i (block-diagonal process noise, Q_nl = 0)
//
//    measurement update (uses y_k)
//      e^i  = y_k - h_n(x_n^i,u_k) - C_l^i xbar_l^i                            (6)
//      S^i  = C_l^i P_l^i (C_l^i)^T + R                                        (7)
//      w^i <- w^i * N(e^i; 0, S^i)                                             (8)
//      K^i  = P_l^i (C_l^i)^T (S^i)^{-1}                                       (9)
//      xbar_l^i <- xbar_l^i + K^i e^i,   P_l^i <- P_l^i - K^i S^i (K^i)^T     (10)
//    time update
//      x_n^i    <- f_n(x_n^i,u_k) + w_n^i,   w_n^i ~ N(0,Q_n)                 (11)
//      xbar_l^i <- A_l xbar_l^i + b_l,  P_l^i <- A_l P_l^i A_l^T + Q_l        (12)
//
//  (In the general model of Schon et al. the nonlinear transition (11) contains a
//  term A_n x_l; the realised x_n(k+1) is then an extra measurement of x_l and a
//  second Kalman update appears between (10) and (12).  A_n = 0 here, so that
//  term is absent — see battery/ecm_mixed_model.hpp.)
// =============================================================================
#pragma once
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../core/rng.hpp"
#include "pf_common.hpp"

namespace estkit {
namespace pf {
template <class M, class = void> struct has_constrain_n : std::false_type {};
template <class M> struct has_constrain_n<M, std::void_t<decltype(std::declval<const M&>().constrain_n(std::declval<Vec<M::NXN>>()))>> : std::true_type {};
template <class M> inline Vec<M::NXN> constrain_nl(const M& m, Vec<M::NXN> x) {
    if constexpr (has_constrain_n<M>::value) return m.constrain_n(x); else return x;
}
}  // namespace pf

template <class MixedModel, int N = 200>
class MarginalizedPf {
public:
    static constexpr int NXN = MixedModel::NXN, NXL = MixedModel::NXL;
    static constexpr int NX = MixedModel::NX, NU = MixedModel::NU, NY = MixedModel::NY;
    static constexpr int NPART = N;
    static_assert(NX == NXN + NXL, "the full state must be ordered [x_n ; x_l]");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    using Xn = Vec<NXN>; using Xl = Vec<NXL>;

    struct Options {
        Real ess_ratio = Real(0.5);
        Real roughen_K = Real(0.2);           // Gordon 1993 §4.2, applied to x_n only
        Real roughen_floor_frac = Real(0.01);   // jitter floor, fraction of sqrt(diag(P0)); only n_n states are sampled
        Real roughen_cap_frac = Real(0.02);     // jitter ceiling, fraction of sqrt(diag(P0))
        Real q_scale = Real(1);
        Real r_scale = Real(1);
        bool apply_constraints = true;
    } opts;

    explicit MarginalizedPf(const MixedModel& m) : m_(m) {}
    void seed(uint64_t s) { rng_.reseed(s); }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        const Xn xn0 = x0.template block<NXN, 1>(0, 0);
        const Xl xl0 = x0.template block<NXL, 1>(NXN, 0);
        const Mat<NXN, NXN> Pnn = P0.template block<NXN, NXN>(0, 0);
        const Mat<NXL, NXL> Pll = P0.template block<NXL, NXL>(NXN, NXN);
        const Mat<NXN, NXN> L = cholesky_safe(Pnn);
        for (int i = 0; i < N; ++i) {
            Xn xi = xn0 + L * rng_.normal_vec<NXN>();
            if (opts.apply_constraints) xi = pf::constrain_nl(m_, xi);
            xn_[i] = xi; xl_[i] = xl0; Pl_[i] = Pll;
            w_[i] = Real(1) / Real(N);
            lw_[i] = -std::log(Real(N));
        }
        for (int k = 0; k < NXN; ++k)
            { const Real sd = std::sqrt(Pnn(k, k) > Real(0) ? Pnn(k, k) : Real(0));
              floor_[k] = opts.roughen_floor_frac * sd; cap_[k] = opts.roughen_cap_frac * sd; }
        spacing_ = pf::mean_spacing(N, NXN);     // only NXN dimensions are sampled
        refresh_moments();
    }

    // ---- time update: (11) for the particles, (12) for the attached KFs -------
    void predict(const U& u) {
        const Mat<NXL, NXL> A = m_.A_l(u);
        const Xl b = m_.b_l(u);
        const Mat<NXL, NXL> Ql = m_.Q_l(u) * opts.q_scale;
        for (int i = 0; i < N; ++i) {
            const Mat<NXN, NXN> Ln = cholesky_safe(m_.Q_n(xn_[i], u) * opts.q_scale);
            Xn xi = m_.f_n(xn_[i], u) + Ln * rng_.normal_vec<NXN>();
            if (opts.apply_constraints) xi = pf::constrain_nl(m_, xi);
            xn_[i] = xi;
            xl_[i] = A * xl_[i] + b;
            Mat<NXL, NXL> Pl = A * Pl_[i] * A.t() + Ql;
            Pl.symmetrize();
            Pl_[i] = Pl;
        }
        refresh_moments();
    }

    // ---- prior predictive measurement: sum_i w^i (h_n + C_l xbar_l^i) --------
    Y predict_measurement(const U& u) const {
        Y ym;
        for (int i = 0; i < N; ++i) ym += (m_.h_n(xn_[i], u) + m_.C_l(xn_[i], u) * xl_[i]) * w_[i];
        return ym;
    }

    // ---- measurement update: (6)-(10) then weights, resampling ---------------
    void update(const Y& y, const U& u) {
        const Mat<NY, NY> R = m_.R(u) * opts.r_scale;
        pf::GaussLogLik<NY> lik;
        for (int i = 0; i < N; ++i) {
            const Mat<NY, NXL> C = m_.C_l(xn_[i], u);
            const Y e = y - m_.h_n(xn_[i], u) - C * xl_[i];
            Mat<NY, NY> S = C * Pl_[i] * C.t() + R;
            S.symmetrize();
            lik.set(S);                                   // marginalised likelihood (7)
            lw_[i] += lik.logpdf(e);                      // (8)
            const Mat<NXL, NY> K = Pl_[i] * C.t() * inverse(S);          // (9)
            xl_[i] = xl_[i] + K * e;                                      // (10)
            Mat<NXL, NXL> Pl = Pl_[i] - K * S * K.t();
            Pl.symmetrize();
            Pl_[i] = Pl;
        }
        const Real lse = pf::normalize_log_weights(lw_, w_);
        if (std::isfinite(lse)) for (int i = 0; i < N; ++i) lw_[i] -= lse;
        else                    for (int i = 0; i < N; ++i) lw_[i] = -std::log(Real(N));
        ess_ = pf::ess(w_);
        if (ess_ < opts.ess_ratio * Real(N)) resample();
        refresh_moments();
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    Real ess() const { return ess_; }
    MixedModel& model() { return m_; }
    const MixedModel& model() const { return m_; }

private:
    void resample() {
        pf::systematic_indices(w_, rng_.uniform(), idx_);
        for (int i = 0; i < N; ++i) { sn_[i] = xn_[idx_[i]]; sl_[i] = xl_[idx_[i]]; sP_[i] = Pl_[idx_[i]]; }
        for (int i = 0; i < N; ++i) {
            xn_[i] = sn_[i]; xl_[i] = sl_[i]; Pl_[i] = sP_[i];
            w_[i] = Real(1) / Real(N); lw_[i] = -std::log(Real(N));
        }
        const Xn s = pf::roughening_sigma(xn_, opts.roughen_K, spacing_, floor_, cap_);
        pf::roughen(xn_, s, rng_);
        if (opts.apply_constraints) for (int i = 0; i < N; ++i) xn_[i] = pf::constrain_nl(m_, xn_[i]);
    }

    // Mean and covariance of the Gaussian MIXTURE (4): the linear block carries
    //   P_ll = sum_i w^i [ P_l^i + (xbar_l^i - xbar_l)(xbar_l^i - xbar_l)^T ].
    void refresh_moments() {
        const Xn mn = pf::weighted_mean(xn_, w_);
        const Xl ml = pf::weighted_mean(xl_, w_);
        for (int k = 0; k < NXN; ++k) x_[k] = mn[k];
        for (int k = 0; k < NXL; ++k) x_[NXN + k] = ml[k];
        Mat<NXN, NXN> Pnn; Mat<NXN, NXL> Pnl; Mat<NXL, NXL> Pll;
        for (int i = 0; i < N; ++i) {
            const Real wi = w_[i];
            const Xn dn = xn_[i] - mn;
            const Xl dl = xl_[i] - ml;
            Pnn += outer(dn, dn) * wi;
            Pnl += outer(dn, dl) * wi;
            Pll += (Pl_[i] + outer(dl, dl)) * wi;
        }
        P_.set_block(0, 0, Pnn);
        P_.set_block(0, NXN, Pnl);
        P_.set_block(NXN, 0, Pnl.t());
        P_.set_block(NXN, NXN, Pll);
    }

    MixedModel m_;
    Rng rng_{1};
    Xn xn_[N];                 // sampled nonlinear states
    Xl xl_[N];                 // conditional means of the linear states
    Mat<NXL, NXL> Pl_[N];      // conditional covariances of the linear states
    Xn sn_[N]; Xl sl_[N]; Mat<NXL, NXL> sP_[N];   // resampling scratch
    Real w_[N] = {};
    Real lw_[N] = {};
    int idx_[N] = {};
    X x_;
    Mat<NX, NX> P_;
    Xn floor_, cap_;
    Real spacing_ = Real(0);
    Real ess_ = Real(N);
};

}  // namespace estkit
