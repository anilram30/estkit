// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/parameter/joint_ekf.hpp — Joint state/parameter estimation with an
//  extended Kalman filter ("joint EKF").
//
//  References
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs — Part 3. State and parameter
//      estimation. J. Power Sources 134(2), 277-292.   (Section 3, joint estimation)
//    * Ljung, L. (1979). Asymptotic behavior of the extended Kalman filter as a
//      parameter estimator for linear systems. IEEE Trans. Autom. Control 24(1),
//      36-50.                                   (convergence of the joint EKF)
//    * Wan, E. A. & Nelson, A. T. (2001). Dual extended Kalman filter methods.
//      In S. Haykin (Ed.), Kalman Filtering and Neural Networks, Ch. 5,
//      pp. 123-173. Wiley, New York.            (joint vs dual comparison)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, Ch. 8.                    (EKF)
//
//  Idea
//    The unknown parameter vector theta in R^{n_p} is modelled as a random walk
//    and appended to the state:
//
//        xa_k = [ x_k ; theta_k ],
//        x_{k+1}     = f(x_k, u_k; theta_k) + w_k,      cov(w)     = Q
//        theta_{k+1} = theta_k + r_k,                   cov(r)     = diag(q_theta)
//        y_k         = h(x_k, u_k; theta_k) + v_k,      cov(v)     = R
//
//    A single EKF of dimension n_x + n_p is then run on the augmented model.  The
//    augmented Jacobians are
//
//        Fa = [ df/dx   df/dtheta ;  0   I ],     Ha = [ dh/dx   dh/dtheta ],
//
//    with df/dtheta and dh/dtheta obtained by central differences through
//    set_params() (see core/model.hpp, AugmentedModel<M>).
//
//    This class is a thin, allocation-free wrapper that (i) composes
//    Ekf<AugmentedModel<Model>>, (ii) builds the augmented initial condition from
//    the base-model initial condition plus the parameter priors in Options, and
//    (iii) exposes the identified parameters.  It therefore satisfies the generic
//    filter interface of docs/ESTIMATOR_API.md with init() taking the *base*
//    state dimension, so that it can be registered with the standard
//    CellEstimator adapter.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../filters/ekf.hpp"

namespace estkit {

template <class Model>
class JointEkf {
public:
    using Aug   = AugmentedModel<Model>;
    using Inner = Ekf<Aug>;

    static constexpr int NB = Model::NX;          // base state dimension
    static constexpr int NP = Model::NP;          // number of identified parameters
    static constexpr int NX = Aug::NX;            // NB + NP (augmented dimension)
    static constexpr int NU = Model::NU, NY = Model::NY;
    using X  = Vec<NX>;                           // augmented state
    using XB = Vec<NB>;                           // base state
    using U  = Vec<NU>; using Y = Vec<NY>;
    using Theta = Vec<NP>;

    struct Options {
        // Per-parameter random-walk VARIANCE per sampling period (Sigma_r in Plett's
        // notation).  An entry <= 0 falls back to (q_rel * |theta_0,k|)^2.
        Theta q_theta;
        // Per-parameter initial VARIANCE.  An entry <= 0 falls back to
        // (p0_rel * |theta_0,k|)^2.
        Theta p0_theta;
        Real q_rel  = Real(1e-5);   // fallback random-walk std as a fraction of |theta_0|
        Real p0_rel = Real(0.1);    // fallback initial std as a fraction of |theta_0|
        // Box constraints on the parameters; inactive for component k when
        // theta_hi[k] <= theta_lo[k] (the default, both zero).
        Theta theta_lo, theta_hi;
        bool joseph = true;             // Joseph-stabilised covariance update
        bool apply_constraints = true;  // project the base state with Model::constrain
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    explicit JointEkf(const Model& m) : inner_(Aug(m, Theta())) {}

    // x0/P0 are the *base*-model initial condition; the parameter block is taken
    // from the model's own params() with the prior variances from Options.
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
        inner_.opts.joseph = opts.joseph;
        inner_.opts.apply_constraints = opts.apply_constraints;
        inner_.opts.q_scale = opts.q_scale;
        inner_.opts.r_scale = opts.r_scale;
        inner_.init(xa, Pa);
    }

    void predict(const U& u) { inner_.predict(u); project_(); }
    Y    predict_measurement(const U& u) const { return inner_.predict_measurement(u); }
    void update(const Y& y, const U& u) { inner_.update(y, u); project_(); }

    // --- generic accessors (augmented state: [x; theta]) ---
    const X& x() const { return inner_.x(); }
    const Mat<NX, NX>& P() const { return inner_.P(); }
    const Y& innovation() const { return inner_.innovation(); }
    const Mat<NY, NY>& S() const { return inner_.S(); }
    Aug& model() { return inner_.model(); }
    const Aug& model() const { return inner_.model(); }
    Inner& inner() { return inner_; }
    const Inner& inner() const { return inner_; }

    // --- identified parameters ---
    Theta params() const { Theta th; for (int k = 0; k < NP; ++k) th[k] = inner_.x()[NB + k]; return th; }
    Real  param(int k) const { return inner_.x()[NB + k]; }
    Real  param_std(int k) const { const Real p = inner_.P()(NB + k, NB + k); return p > Real(0) ? std::sqrt(p) : Real(0); }
    // Convenience alias for param(0).  By the estkit ECM convention (ecm_model.hpp)
    // theta = [Q_Ah, R0], so param(0) is the cell total capacity in Ah.
    Real capacity() const { return param(0); }
    // A copy of the base model with the currently identified parameters applied.
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
