// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/parameter/joint_ukf.hpp — Joint state/parameter estimation with a
//  sigma-point (unscented) Kalman filter ("joint SPKF").
//
//  References
//    * Plett, G. L. (2006). Sigma-point Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs — Part 2: Simultaneous state and
//      parameter estimation. J. Power Sources 161(2), 1369-1384.
//    * Plett, G. L. (2006). Sigma-point Kalman filtering ... Part 1:
//      Introduction and state estimation. J. Power Sources 161(2), 1356-1368.
//    * Wan, E. A. & van der Merwe, R. (2000). The unscented Kalman filter for
//      nonlinear estimation. Proc. IEEE AS-SPCC, 153-158.   (scaled UT)
//    * Ljung, L. (1979). Asymptotic behavior of the extended Kalman filter as a
//      parameter estimator for linear systems. IEEE Trans. Autom. Control 24(1),
//      36-50.
//
//  Idea
//    Identical augmentation to the joint EKF (parameters as a random walk)
//
//        xa_k = [ x_k ; theta_k ],   theta_{k+1} = theta_k + r_k,
//
//    but the moments of the augmented state are propagated with the scaled
//    unscented transform instead of a first-order Taylor expansion, so no
//    Jacobian with respect to theta is ever formed.  For 2*n_a+1 sigma points
//    Xa^(i) with weights w_m^(i), w_c^(i),
//
//        Xa^(i)_{k|k-1} = fa(Xa^(i)_{k-1}, u_{k-1}),
//        xa^-_k = sum_i w_m^(i) Xa^(i)_{k|k-1},
//        Pa^-_k = sum_i w_c^(i) (Xa^(i)-xa^-)(Xa^(i)-xa^-)^T + Qa,
//
//    with fa the augmented dynamics (identity on the theta block) and
//    Qa = blkdiag(Q, diag(q_theta)).  The measurement update uses
//    ya^(i) = h(x^(i), u_k; theta^(i)), i.e. every sigma point evaluates the
//    output with *its own* parameter vector, which is exactly what makes the
//    SPKF capture the state/parameter cross-covariance without linearisation.
//
//  This class is a thin, allocation-free wrapper around Ukf<AugmentedModel<Model>>
//  with the same interface conventions as JointEkf (see joint_ekf.hpp).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../filters/ukf.hpp"

namespace estkit {

template <class Model>
class JointUkf {
public:
    using Aug   = AugmentedModel<Model>;
    using Inner = Ukf<Aug>;

    static constexpr int NB = Model::NX;
    static constexpr int NP = Model::NP;
    static constexpr int NX = Aug::NX;
    static constexpr int NU = Model::NU, NY = Model::NY;
    static constexpr int NS = 2 * NX + 1;
    using X  = Vec<NX>;
    using XB = Vec<NB>;
    using U  = Vec<NU>; using Y = Vec<NY>;
    using Theta = Vec<NP>;

    struct Options {
        Theta q_theta;      // per-step random-walk variance; <= 0 -> (q_rel*|theta_0|)^2
        Theta p0_theta;     // initial variance;              <= 0 -> (p0_rel*|theta_0|)^2
        Real q_rel  = Real(1e-5);
        Real p0_rel = Real(0.1);
        Theta theta_lo, theta_hi;    // box constraints (inactive when hi <= lo)
        Real alpha = Real(1e-1), beta = Real(2), kappa = Real(0);   // scaled UT
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    explicit JointUkf(const Model& m) : inner_(Aug(m, Theta())) {}

    void init(const XB& x0, const Mat<NB, NB>& P0) {
        const Theta th0 = inner_.model().base.params();
        X xa; Mat<NX, NX> Pa;
        xa.set_block(0, 0, x0);
        Pa.set_block(0, 0, P0);
        Theta q;
        for (int k = 0; k < NP; ++k) {
            const Real scale = std::fabs(th0[k]);
            xa[NB + k] = th0[k];
            q[k] = (opts.q_theta[k] > Real(0)) ? opts.q_theta[k] : sq(opts.q_rel * scale);
            Pa(NB + k, NB + k) = (opts.p0_theta[k] > Real(0)) ? opts.p0_theta[k] : sq(opts.p0_rel * scale);
        }
        inner_.model().q_theta = q;
        inner_.opts.alpha = opts.alpha;
        inner_.opts.beta  = opts.beta;
        inner_.opts.kappa = opts.kappa;
        inner_.opts.apply_constraints = opts.apply_constraints;
        inner_.opts.q_scale = opts.q_scale;
        inner_.opts.r_scale = opts.r_scale;
        inner_.init(xa, Pa);
    }

    void predict(const U& u) { inner_.predict(u); project_(); }
    Y    predict_measurement(const U& u) const { return inner_.predict_measurement(u); }
    void update(const Y& y, const U& u) { inner_.update(y, u); project_(); }

    const X& x() const { return inner_.x(); }
    const Mat<NX, NX>& P() const { return inner_.P(); }
    const Y& innovation() const { return inner_.innovation(); }
    const Mat<NY, NY>& S() const { return inner_.S(); }
    Aug& model() { return inner_.model(); }
    const Aug& model() const { return inner_.model(); }
    Inner& inner() { return inner_; }
    const Inner& inner() const { return inner_; }

    Theta params() const { Theta th; for (int k = 0; k < NP; ++k) th[k] = inner_.x()[NB + k]; return th; }
    Real  param(int k) const { return inner_.x()[NB + k]; }
    Real  param_std(int k) const { const Real p = inner_.P()(NB + k, NB + k); return p > Real(0) ? std::sqrt(p) : Real(0); }
    Real  capacity() const { return param(0); }
    Model tuned_model() const { return inner_.model().with(inner_.x()); }

private:
    void project_() {
        bool changed = false;
        X xa = inner_.x();
        for (int k = 0; k < NP; ++k) {
            if (opts.theta_hi[k] > opts.theta_lo[k]) {
                const Real c = clampr(xa[NB + k], opts.theta_lo[k], opts.theta_hi[k]);
                if (c != xa[NB + k]) { xa[NB + k] = c; changed = true; }
            }
        }
        if (changed) inner_.x_mut() = xa;
    }

    Inner inner_;
};

}  // namespace estkit
