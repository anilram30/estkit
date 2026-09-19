// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/observers/dob.hpp — augmented-state disturbance observer (DOB) for an
//  additive bias on one input channel.
//
//  References
//    * Ohnishi, K., Shibata, M. & Murakami, T. (1996). Motion control for advanced
//      mechatronics. IEEE/ASME Transactions on Mechatronics 1(1), 56-67.
//      (the disturbance observer as an inverse-model + low-pass estimator)
//    * Chen, W.-H., Yang, J., Guo, L. & Li, S. (2016). Disturbance-observer-based
//      control and related methods -- An overview. IEEE Transactions on Industrial
//      Electronics 63(2), 1083-1095.
//    * Johnson, C. D. (1971). Accommodation of external disturbances in linear
//      regulator and servomechanism problems. IEEE Transactions on Automatic
//      Control 16(6), 635-644.  (disturbance-accommodating observer)
//    * Friedland, B. (1969). Treatment of bias in recursive filtering. IEEE
//      Transactions on Automatic Control 14(4), 359-367.
//    * Zhao, S., Duncan, S. R. & Howey, D. A. (2017). Observability analysis and
//      state estimation of lithium-ion batteries in the presence of sensor biases.
//      IEEE Transactions on Control Systems Technology 25(1), 326-333.
//
//  Model.  The measured input is biased, u_true = u_meas - d e_j, and the bias is
//  a random walk.  Unlike the ESO, the disturbance here is *physical*: it is
//  expressed in input units (amperes for the cell) and is fed back into BOTH the
//  state equation and the output equation, which is what removes the ohmic
//  feed-through error R0 * d that a state-only disturbance model leaves behind:
//
//      x_{k+1} = f(x_k, u_k - dhat_k e_j) ,     d_{k+1} = d_k + w_d ,
//      y_k     = h(x_k, u_k - dhat_k e_j) + v_k .                               (1)
//
//  Linearising (1) about the current estimate gives the augmented pair
//      Abar = [ A   -E ]        Cbar = [ C   -D_j ] ,                           (2)
//             [ 0    1 ]
//  with E = df/du_j and D_j = dh/du_j (both obtained generically by finite
//  differences through jac_B and a central difference of h).  The observer is the
//  Luenberger observer of (2) in current-estimator form,
//      x^+_k = x^-_k + L_x r_k ,   d^+_k = d^-_k + l_d r_k ,                    (3)
//  with [L_x; l_d] from the discrete Riccati equation (default) or from pole
//  placement.
//
//  Observability of the bias.  The Hautus test at z = 1 requires
//      rank [ A - I   -E ; C   -D_j ] = NX + 1 ,                                (4)
//  which for the cell model holds because E_z = -eta dt / (3600 Q) != 0 (the bias
//  slowly corrupts the coulomb count) *and* because D_j = -R0 != 0 (it instantly
//  offsets the ohmic drop).  The second path is the fast one and is what makes
//  the bias identifiable in a mission of a few thousand seconds; with R0 -> 0 the
//  problem degenerates to the weak, purely integral observability discussed by
//  Zhao et al. (2017).
//
//  Numerical note.  Ackermann's formula on the augmented pair is ill-conditioned
//  for this plant (det of the augmented observability matrix ~ 4e-30 at dt =
//  0.1 s), so the Riccati/LQE design is the default; pole placement remains
//  available through Options::design and is guarded by a spectral-radius check.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "luenberger.hpp"

namespace estkit {

template <class Model>
class DisturbanceObserver {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NA = NX + 1;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    using Design = obs::Design;

    struct Options {
        int  u_index = 0;                    // input channel carrying the bias
        Design design = Design::Riccati;
        Real pole[NA];                       // used when design == PolePlacement
        Real q_scale = Real(1), r_scale = Real(1);
        Real q_extra[NX];                    // extra process-noise std per state
        Real q_d = Real(3e-3);               // random-walk std of the bias, per step
        Real d_max = Real(1e3);              // clamp on the bias estimate (input units)
        Real regain_tol = Real(1e-3);
        int  riccati_iters_init = 30000, riccati_iters_refine = 40;
        bool riccati_exact = true;         // exact DARE (doubling) instead of the truncated recursion
        // Certified decay rate: a design is accepted only if rho <= 1 - rho_margin
        // over the whole scheduling family (see obs::gain_is_robust).
        Real rho_margin = Real(1e-4);
        Real gain_smooth = Real(1);        // exponential blending of a re-designed gain (1 = replace)
        Real robust_lo = Real(0.8), robust_hi = Real(1.25);  // scheduling-uncertainty range for the gain check
        Real max_gain = Real(1e4);
        bool fallback_to_riccati = true;
        bool compensate_output = true;       // feed dhat back into h() as well as f()
        bool apply_constraints = true;

        Options() {
            for (int i = 0; i < NA; ++i) pole[i] = Real(0.998);
            for (int i = 0; i < NX; ++i) q_extra[i] = Real(0);
        }
    } opts;

    explicit DisturbanceObserver(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; d_ = Real(0); have_design_ = false; riccati_init_ = false; La_ = Vec<NA>(); Lx_ = X(); ld_ = Real(0); residual_ = Y();
        Pa_ = Mat<NA, NA>();
        Pa_.template set_block<NX, NX>(0, 0, P0);
        Pa_(NX, NX) = sq(Real(0.3));
    }

    void predict(const U& u) { x_ = m_.f(x_, ueff(u)); }
    Y predict_measurement(const U& u) const { return m_.h(x_, ueff(u)); }

    void update(const Y& y, const U& u) {
        const U ue = ueff(u);
        const Mat<NX, NX> A = jac_F(m_, x_, ue);
        const Mat<NY, NX> C = jac_H(m_, x_, ue);
        schedule_gain(A, C, ue);
        const Y r = y - m_.h(x_, ue);
        residual_ = r;
        const X dx = Lx_ * r[0];
        const Real dd = ld_ * r[0];
        if (dx.is_finite() && std::isfinite(dd)) {
            x_ = x_ + dx;
            d_ = clampr(d_ + dd, -opts.d_max, opts.d_max);
        }
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    Real disturbance() const { return d_; }   // estimated bias on channel u_index (input units)
    const X& gain() const { return Lx_; }
    Real disturbance_gain() const { return ld_; }
    const Y& residual() const { return residual_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    U ueff(const U& u) const {
        U ue = u;
        ue[opts.u_index] -= d_;
        return ue;
    }
    // D_j = dh/du_j by central differences (generic: the model concept has no D)
    Y dh_du(const U& u) const {
        U up = u, um = u;
        const Real h = Real(1e-5) * (Real(1) + std::fabs(u[opts.u_index]));
        up[opts.u_index] += h; um[opts.u_index] -= h;
        return (m_.h(x_, up) - m_.h(x_, um)) / (Real(2) * h);
    }

    void schedule_gain(const Mat<NX, NX>& A, const Mat<NY, NX>& C, const U& ue) {
        if (have_design_ && obs::lin_change<NX, NY>(A_ref_, C_ref_, A, C) < opts.regain_tol) return;
        A_ref_ = A; C_ref_ = C;
        const X E = jac_B(m_, x_, ue).col(opts.u_index);
        const Y D = opts.compensate_output ? dh_du(ue) : Y();
        Mat<NA, NA> Aa = Mat<NA, NA>::identity();
        Aa.template set_block<NX, NX>(0, 0, A);
        for (int i = 0; i < NX; ++i) Aa(i, NX) = -E[i];
        Mat<NY, NA> Ca;
        Ca.template set_block<NY, NX>(0, 0, C);
        for (int i = 0; i < NY; ++i) Ca(i, NX) = -D[i];
        Vec<NA> L;
        bool ok = false;
        if (opts.design == Design::PolePlacement) {
            if constexpr (NY == 1) {
                ok = obs::place_gain<NA>(Aa, Ca, opts.pole, L);
                if (ok && L.norm_inf() > opts.max_gain) ok = false;
            }
        }
        const Real rho_max = Real(1) - opts.rho_margin;
        if (ok && !obs::gain_is_robust<NA, NY>(Aa, Ca, L, opts.robust_lo, opts.robust_hi, rho_max)) ok = false;
        if (opts.design == Design::Riccati || (!ok && opts.fallback_to_riccati)) {
            Mat<NA, NA> Qa;
            Mat<NX, NX> Qx = m_.Q(x_, ue) * opts.q_scale;
            for (int i = 0; i < NX; ++i) Qx(i, i) += sq(opts.q_extra[i]);
            Qa.template set_block<NX, NX>(0, 0, Qx);
            Qa(NX, NX) = sq(opts.q_d);
            const Mat<NY, NY> Ra = m_.R(x_, ue) * opts.r_scale;
            if (!riccati_init_) { Pa_ = obs::dare_design<NA, NY>(Aa, Ca, Qa, Ra, opts.riccati_exact, opts.riccati_iters_init); riccati_init_ = true; }
            const Vec<NA> Lr = obs::riccati_refine<NA, NY>(Aa, Ca, Qa, Ra, Pa_, opts.riccati_iters_refine).col(0);
            if (Lr.is_finite() && (obs::gain_is_robust<NA, NY>(Aa, Ca, Lr, opts.robust_lo, opts.robust_hi, rho_max) || !have_design_)) { L = Lr; ok = true; }
        }
        if (ok) {
            obs::blend_gain<NA>(La_, L, opts.gain_smooth, have_design_);
            for (int i = 0; i < NX; ++i) Lx_[i] = La_[i];
            ld_ = La_[NX];
            have_design_ = true;
        }
    }

    Model m_;
    X x_, Lx_;
    Y residual_;
    Real d_ = Real(0), ld_ = Real(0);
    Mat<NX, NX> A_ref_;
    Mat<NY, NX> C_ref_;
    Vec<NA> La_;
    Mat<NA, NA> Pa_;
    bool have_design_ = false, riccati_init_ = false;
};

}  // namespace estkit
