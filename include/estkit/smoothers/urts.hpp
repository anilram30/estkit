// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/smoothers/urts.hpp — Unscented Rauch-Tung-Striebel (URTS) smoother.
//
//  References
//    * Sarkka, S. (2008). Unscented Rauch-Tung-Striebel smoother. IEEE Trans.
//      Autom. Control 53(3), 845-849.
//    * Sarkka, S. (2013). Bayesian Filtering and Smoothing, Cambridge University
//      Press, Alg. 9.1 (unscented RTS smoother).
//    * Rauch, H. E., Tung, F. & Striebel, C. T. (1965). Maximum likelihood
//      estimates of linear dynamic systems. AIAA Journal 3(8), 1445-1450.
//    * Julier, S. J. & Uhlmann, J. K. (2004). Unscented filtering and nonlinear
//      estimation. Proc. IEEE 92(3), 401-422.
//
//  Idea.  The RTS gain of the linear/extended smoother,
//      G_k = P^+_k F_k^T (P^-_{k+1})^{-1},
//  contains the cross-covariance between the filtered state at k and the
//  predicted state at k+1, because P^+_k F_k^T = cov(x_k, x_{k+1} | y_{1:k}) for
//  a linear model.  Sarkka's insight is to compute that cross-covariance
//  DIRECTLY with the unscented transform instead of through a Jacobian:
//
//      X^(i)_k   = sigma points of  N(xhat^+_k, P^+_k)
//      Xf^(i)    = f(X^(i)_k, u_k),      xhat^-_{k+1} = sum_i w_m^(i) Xf^(i)
//      C_k       = sum_i w_c^(i) (X^(i)_k - xhat^+_k)(Xf^(i) - xhat^-_{k+1})^T
//      D_k       = C_k (P^-_{k+1})^{-1}                                      (gain)
//      xhat^s_k  = xhat^+_k + D_k ( xhat^s_{k+1} - xhat^-_{k+1} )
//      P^s_k     = P^+_k + D_k ( P^s_{k+1} - P^-_{k+1} ) D_k^T
//
//  With f linear this reduces exactly to the RTS gain, so the URTS is a strict
//  generalisation of the extended RTS smoother and, like the UKF, is accurate to
//  second order for a general smooth f.
//
//  The sigma set used for C_k is regenerated from (xhat^+_k, P^+_k) immediately
//  before the filter's own predict() call, so it is bit-identical to the set the
//  UKF uses internally; the only cost is one extra unscented propagation per
//  step.  The gain is formed as D_k^T = (P^-_{k+1})^{-1} C_k^T with solve_spd.
//
//  OFFLINE method (estkit::BatchEstimator); std::vector is permitted (§1 of
//  docs/ESTIMATOR_API.md) because it never runs on the vehicle.
// =============================================================================
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../battery/ecm_model.hpp"
#include "../filters/ukf.hpp"

namespace estkit {

template <class Model>
class UnscentedRtsSmoother final : public BatchEstimator {
public:
    using Filter = Ukf<Model>;
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NS = 2 * NX + 1;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real alpha = Real(1e-1), beta = Real(2), kappa = Real(0);
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    UnscentedRtsSmoother(std::string nm, std::string grp) : name_(std::move(nm)), group_(std::move(grp)) {}

    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }
    size_t state_bytes() const override {
        return sizeof(*this) + (xm_.capacity() + xp_.capacity() + xs_.capacity()) * sizeof(X)
             + (Pm_.capacity() + Pp_.capacity() + Ps_.capacity() + C_.capacity()) * sizeof(Mat<NX, NX>)
             + u_.capacity() * sizeof(U);
    }
    void reset(const EstimatorConfig& c) override { cfg_ = c; m_ = Model(c); }

    void run(const Real* i, const Real* v, const Real* T, int n, Real* soc_out, Real* v_pred_out) override {
        if (n <= 0) return;
        const size_t N = static_cast<size_t>(n);
        xm_.assign(N, X()); xp_.assign(N, X()); xs_.assign(N, X());
        Pm_.assign(N, Mat<NX, NX>()); Pp_.assign(N, Mat<NX, NX>()); Ps_.assign(N, Mat<NX, NX>());
        C_.assign(N, Mat<NX, NX>()); u_.assign(N, U());

        // ---------------- forward UKF pass (storing the sigma cross-covariance) -------
        Filter f(m_);
        f.opts.alpha = opts.alpha; f.opts.beta = opts.beta; f.opts.kappa = opts.kappa;
        f.opts.apply_constraints = opts.apply_constraints;
        f.opts.q_scale = opts.q_scale; f.opts.r_scale = opts.r_scale;
        f.init(m_.x0(cfg_), m_.P0(cfg_));
        U u_prev;
        for (int k = 0; k < n; ++k) {
            const U u = Model::u_of(i[k], T[k]);
            const size_t sk = static_cast<size_t>(k);
            if (k > 0) {
                // C_{k-1} = sum_s w_c (X_s - xhat^+_{k-1})(f(X_s) - xhat^-_k)^T
                const X xplus = f.x();
                f.sigma_points(xplus, f.P());
                X Xf[NS]; X xmean;
                for (int s = 0; s < NS; ++s) { Xf[s] = f.model().f(f.sigma()[s], u_prev); xmean += Xf[s] * f.wm(s); }
                Mat<NX, NX> C;
                for (int s = 0; s < NS; ++s) C += outer(f.sigma()[s] - xplus, Xf[s] - xmean) * f.wc(s);
                C_[sk - 1] = C;
                f.predict(u_prev);
            }
            xm_[sk] = f.x(); Pm_[sk] = f.P();
            Y y; y[0] = v[k];
            f.update(y, u);
            xp_[sk] = f.x(); Pp_[sk] = f.P();
            u_[sk] = u; u_prev = u;
        }

        // ---------------- backward URTS recursion ----------------
        xs_[N - 1] = xp_[N - 1]; Ps_[N - 1] = Pp_[N - 1];
        for (int k = n - 2; k >= 0; --k) {
            const size_t sk = static_cast<size_t>(k), s1 = sk + 1;
            const Mat<NX, NX> Dt = solve_spd(Pm_[s1], C_[sk].t());   // D^T = (P^-)^{-1} C^T
            const Mat<NX, NX> D = Dt.t();
            if (!D.is_finite()) { xs_[sk] = xp_[sk]; Ps_[sk] = Pp_[sk]; continue; }
            X xs = xp_[sk] + D * (xs_[s1] - xm_[s1]);
            Mat<NX, NX> Ps = Pp_[sk] + D * (Ps_[s1] - Pm_[s1]) * D.t();
            Ps.symmetrize();
            if (!xs.is_finite() || !Ps.is_finite()) { xs = xp_[sk]; Ps = Pp_[sk]; }
            if (opts.apply_constraints) xs = constrain(f.model(), xs);
            xs_[sk] = xs; Ps_[sk] = Ps;
        }

        // ---------------- outputs ----------------
        const Model& mm = f.model();
        for (int k = 0; k < n; ++k) {
            const size_t sk = static_cast<size_t>(k);
            soc_out[k] = xs_[sk][Model::IZ];
            v_pred_out[k] = mm.h(xs_[sk], u_[sk])[0];
        }
    }

    const std::vector<X>& smoothed_states() const { return xs_; }
    const std::vector<Mat<NX, NX>>& smoothed_cov() const { return Ps_; }

private:
    std::string name_, group_;
    EstimatorConfig cfg_;
    Model m_;
    std::vector<X> xm_, xp_, xs_;
    std::vector<Mat<NX, NX>> Pm_, Pp_, Ps_, C_;
    std::vector<U> u_;
};

}  // namespace estkit
