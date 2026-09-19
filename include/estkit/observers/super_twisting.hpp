// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/super_twisting.hpp — second-order (super-twisting)
//  sliding-mode observer.
//
//  References
//    * Levant, A. (1993). Sliding order and sliding accuracy in sliding mode
//      control. International Journal of Control 58(6), 1247-1263.
//      (the super-twisting algorithm, finite-time convergence)
//    * Davila, J., Fridman, L. & Levant, A. (2005). Second-order sliding-mode
//      observer for mechanical systems. IEEE Transactions on Automatic Control
//      50(11), 1785-1789.
//    * Moreno, J. A. & Osorio, M. (2008). A Lyapunov approach to second-order
//      sliding mode controllers and observers. Proc. 47th IEEE Conference on
//      Decision and Control, 2856-2861.  (strict Lyapunov function, gain conditions)
//    * Utkin, V. I. (1992). Sliding Modes in Control and Optimization. Springer.
//
//  Algorithm (one sampling period, current-estimator form, NY = 1)
//      x^-_k  = f(x^+_{k-1}, u_{k-1})
//      r_k    = y_k - h(x^-_k, u_k)
//      nu_k   = k1 |r_k|^{1/2} sgn(r_k) + w_k
//      w_{k+1}= w_k + k2 sgn(r_k)                     (k2 already contains dt)
//      x^+_k  = x^-_k + Lambda nu_k
//
//  Lambda in R^{NX} is the injection direction, normalised so that C Lambda = 1;
//  the default puts the whole injection on the state with the largest DC gain
//  from a constant disturbance (the SOC integrator for the cell model), the other
//  states being driven open-loop by the known input.  With that normalisation the
//  output error obeys, to first order,
//      r_{k+1} = r_k - (k1 |r_k|^{1/2} sgn r_k + w_k) + rho_k ,
//      w_{k+1} = w_k + k2 sgn(r_k) ,
//  which is the exact discretisation of the continuous super-twisting pair
//      s_dot  = -k1c |s|^{1/2} sgn(s) + z + rho(t),   z_dot = -k2c sgn(s),      (1)
//  with k1 = k1c dt, k2 = k2c dt^2.
//
//  Gain conditions (Moreno & Osorio 2008, Thm. 1).  If the perturbation satisfies
//      |rho(t)| <= delta |s|^{1/2} ,                                           (2)
//  then the origin of (1) is finite-time stable for
//      k1c > 2 delta ,   k2c > k1c (5 delta k1c + 4 delta^2) / (2 (k1c - 2 delta)) , (3)
//  and V(s,z) = 2 k2c |s| + z^2/2 + (k1c|s|^{1/2} sgn s - z)^2 / 2 is a strict
//  Lyapunov function.  A convenient sufficient choice used here is the classical
//  Levant tuning k1c = 1.5 sqrt(delta), k2c = 1.1 delta.
//
//  Unlike the first-order SMO, the injection nu is *continuous* (the discontinuity
//  is hidden in the integrator w), so the estimate is chattering-free while
//  retaining exact finite-time rejection of bounded, Lipschitz-in-time model
//  errors.  A boundary layer phi > 0 may still be used in the sgn() of the
//  integrator to stop the measurement noise from random-walking w.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "luenberger.hpp"

namespace estkit {

template <class Model>
class SuperTwistingObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static_assert(NY == 1, "SuperTwistingObserver assumes a scalar sliding variable (NY == 1)");
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real k1 = Real(2.5e-4);     // per step; k1 = k1_continuous * dt
        Real k2 = Real(2e-8);       // per step; k2 = k2_continuous * dt^2
        Real w_max = Real(2e-3);    // anti-windup clamp on the integral injection
        Real phi = Real(0.002);     // boundary layer inside the integrator's sgn (0 = pure sgn)
        Real lam[NX];               // injection direction; all zero => auto (see below)
        int  inject_index = -1;     // -1 = auto (obs::slowest_state)
        Real c_floor = Real(1e-6);  // guard on |C_j| when normalising C Lambda = 1
        bool apply_constraints = true;

        Options() { for (int i = 0; i < NX; ++i) lam[i] = Real(0); }
    } opts;

    explicit SuperTwistingObserver(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        (void)P0;
        x_ = x0; w_ = Real(0); residual_ = Y(); lam_ = X(); have_dir_ = false;
    }

    void predict(const U& u) { x_ = m_.f(x_, u); }
    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> C = jac_H(m_, x_, u);
        build_direction(C, u);
        const Real r = (y - m_.h(x_, u))[0];
        residual_[0] = r;
        // continuous part + integral part
        const Real nu = opts.k1 * std::sqrt(std::fabs(r)) * sgn(r) + w_;
        const X dx = lam_ * nu;
        if (dx.is_finite()) x_ = x_ + dx;
        // integrator: sgn(r), optionally regularised by a boundary layer
        w_ += opts.k2 * ((opts.phi > Real(0)) ? sat(r, opts.phi) : sgn(r));
        w_ = clampr(w_, -opts.w_max, opts.w_max);
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    Real integral_injection() const { return w_; }
    const X& direction() const { return lam_; }
    const Y& residual() const { return residual_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void build_direction(const Mat<NY, NX>& C, const U& u) {
        Real user = Real(0);
        for (int i = 0; i < NX; ++i) user += std::fabs(opts.lam[i]);
        if (user > Real(0)) {
            if (!have_dir_) { for (int i = 0; i < NX; ++i) lam_[i] = opts.lam[i]; have_dir_ = true; }
            return;
        }
        // auto: put the injection on one state and normalise so that C Lambda = 1
        if (!have_dir_) {
            const Mat<NX, NX> A = jac_F(m_, x_, u);
            idx_ = (opts.inject_index >= 0) ? opts.inject_index : obs::slowest_state<NX>(A);
            have_dir_ = true;
        }
        const Real cj = C(0, idx_);
        const Real den = (std::fabs(cj) > opts.c_floor) ? cj : (cj >= Real(0) ? opts.c_floor : -opts.c_floor);
        lam_ = X();
        lam_[idx_] = Real(1) / den;
    }

    Model m_;
    X x_, lam_;
    Y residual_;
    Real w_ = Real(0);
    int idx_ = 0;
    bool have_dir_ = false;
};

}  // namespace estkit
