// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/smoothers/rts.hpp — Rauch-Tung-Striebel fixed-interval smoother
//  (extended / linearised form) for the whole recorded mission.
//
//  References
//    * Rauch, H. E., Tung, F. & Striebel, C. T. (1965). Maximum likelihood
//      estimates of linear dynamic systems. AIAA Journal 3(8), 1445-1450.
//    * Sarkka, S. (2013). Bayesian Filtering and Smoothing, Cambridge University
//      Press, Alg. 8.2 (extended RTS smoother) and §8.1 (RTS lemma).
//    * Anderson, B. D. O. & Moore, J. B. (1979). Optimal Filtering,
//      Prentice-Hall, Ch. 7.
//    * Gelb, A. (Ed.) (1974). Applied Optimal Estimation, MIT Press, §5.3.
//
//  Algorithm.  A forward EKF pass stores, for every sample k,
//      xhat^-_k, P^-_k   (prior),   xhat^+_k, P^+_k   (posterior),
//      F_k = df/dx |_(xhat^+_k, u_k)   (the Jacobian used to go from k to k+1).
//  The backward pass then runs, from k = N-2 down to 0,
//
//      G_k        = P^+_k F_k^T (P^-_{k+1})^{-1}                          (gain)
//      xhat^s_k   = xhat^+_k + G_k ( xhat^s_{k+1} - xhat^-_{k+1} )
//      P^s_k      = P^+_k + G_k ( P^s_{k+1} - P^-_{k+1} ) G_k^T
//
//  initialised with xhat^s_{N-1} = xhat^+_{N-1}, P^s_{N-1} = P^+_{N-1}.
//
//  The gain is evaluated as G_k^T = (P^-_{k+1})^{-1} F_k P^+_k with a Cholesky
//  solve (solve_spd) rather than an explicit inverse.
//
//  This is an OFFLINE method (estkit::BatchEstimator): it needs the whole record,
//  which is exactly the ground-station post-flight situation.  std::vector is
//  permitted here (docs/ESTIMATOR_API.md §1) because the class never runs on the
//  vehicle.
//
//  Templated on the forward filter type, which must expose x(), P(), model(),
//  init(), predict(), update() — i.e. Ekf<Model> or any drop-in replacement.
// =============================================================================
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../battery/ecm_model.hpp"

namespace estkit {

template <class Filter>
class RtsSmoother final : public BatchEstimator {
public:
    using Model = typename std::decay<decltype(std::declval<Filter&>().model())>::type;
    static constexpr int NX = Filter::NX, NU = Filter::NU, NY = Filter::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        bool apply_constraints = true;   // project the smoothed states with Model::constrain
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    RtsSmoother(std::string nm, std::string grp) : name_(std::move(nm)), group_(std::move(grp)) {}

    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }
    size_t state_bytes() const override {
        return sizeof(*this) + (xm_.capacity() + xp_.capacity() + xs_.capacity()) * sizeof(X)
             + (Pm_.capacity() + Pp_.capacity() + Ps_.capacity() + F_.capacity()) * sizeof(Mat<NX, NX>)
             + u_.capacity() * sizeof(U);
    }
    void reset(const EstimatorConfig& c) override { cfg_ = c; m_ = Model(c); }

    void run(const Real* i, const Real* v, const Real* T, int n, Real* soc_out, Real* v_pred_out) override {
        if (n <= 0) return;
        const size_t N = static_cast<size_t>(n);
        xm_.assign(N, X()); xp_.assign(N, X()); xs_.assign(N, X());
        Pm_.assign(N, Mat<NX, NX>()); Pp_.assign(N, Mat<NX, NX>()); Ps_.assign(N, Mat<NX, NX>());
        F_.assign(N, Mat<NX, NX>()); u_.assign(N, U());

        // ---------------- forward EKF pass ----------------
        Filter f(m_);
        f.opts.apply_constraints = opts.apply_constraints;
        f.opts.q_scale = opts.q_scale; f.opts.r_scale = opts.r_scale;
        f.init(m_.x0(cfg_), m_.P0(cfg_));
        U u_prev;
        for (int k = 0; k < n; ++k) {
            const U u = Model::u_of(i[k], T[k]);
            if (k > 0) {
                F_[static_cast<size_t>(k - 1)] = jac_F(f.model(), f.x(), u_prev);
                f.predict(u_prev);
            }
            const size_t sk = static_cast<size_t>(k);
            xm_[sk] = f.x(); Pm_[sk] = f.P();
            Y y; y[0] = v[k];
            f.update(y, u);
            xp_[sk] = f.x(); Pp_[sk] = f.P();
            u_[sk] = u; u_prev = u;
        }

        // ---------------- backward RTS recursion ----------------
        xs_[N - 1] = xp_[N - 1]; Ps_[N - 1] = Pp_[N - 1];
        for (int k = n - 2; k >= 0; --k) {
            const size_t sk = static_cast<size_t>(k), s1 = sk + 1;
            // G^T = (P^-_{k+1})^{-1} F_k P^+_k   (Cholesky solve, no explicit inverse)
            const Mat<NX, NX> Gt = solve_spd(Pm_[s1], F_[sk] * Pp_[sk]);
            const Mat<NX, NX> G = Gt.t();
            if (!G.is_finite()) { xs_[sk] = xp_[sk]; Ps_[sk] = Pp_[sk]; continue; }
            X xs = xp_[sk] + G * (xs_[s1] - xm_[s1]);
            Mat<NX, NX> Ps = Pp_[sk] + G * (Ps_[s1] - Pm_[s1]) * G.t();
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
            v_pred_out[k] = mm.h(xs_[sk], u_[sk])[0];   // SMOOTHED voltage reconstruction
        }
    }

    // access for analysis / plotting
    const std::vector<X>& smoothed_states() const { return xs_; }
    const std::vector<Mat<NX, NX>>& smoothed_cov() const { return Ps_; }

private:
    std::string name_, group_;
    EstimatorConfig cfg_;
    Model m_;
    std::vector<X> xm_, xp_, xs_;
    std::vector<Mat<NX, NX>> Pm_, Pp_, Ps_, F_;
    std::vector<U> u_;
};

}  // namespace estkit
