// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/core/model.hpp — the model "concept" every generic estimator is
//  written against, plus adapters (numeric Jacobians, augmented-state models).
//
//  A model is any class that provides
//
//     static constexpr int NX;   // number of states
//     static constexpr int NU;   // number of inputs (incl. scheduling variables such as temperature)
//     static constexpr int NY;   // number of measurements
//     Vec<NX>     f(const Vec<NX>& x, const Vec<NU>& u) const;   // x_{k+1} = f(x_k, u_k)
//     Vec<NY>     h(const Vec<NX>& x, const Vec<NU>& u) const;   // y_k     = h(x_k, u_k)
//     Mat<NX,NX>  Q(const Vec<NX>& x, const Vec<NU>& u) const;   // process-noise covariance
//     Mat<NY,NY>  R(const Vec<NX>& x, const Vec<NU>& u) const;   // measurement-noise covariance
//
//  Optional members (detected at compile time, with fall-backs):
//     Mat<NX,NX>  F(x,u)         Jacobian df/dx   (default: central differences)
//     Mat<NY,NX>  H(x,u)         Jacobian dh/dx   (default: central differences)
//     Mat<NX,NU>  B(x,u)         Jacobian df/du   (default: central differences)
//     Vec<NX>     constrain(x)   projection of x onto the feasible set (default: identity)
//     static constexpr int NP;   number of tunable parameters (for joint/dual estimation)
//     Vec<NP>     params() const / void set_params(const Vec<NP>&)
//
//  The battery ECM (estkit/battery/ecm_model.hpp) is one instance of this
//  concept; navigation, attitude or any other discrete-time model can be
//  plugged into every filter/observer in the same way.
// =============================================================================
#pragma once
#include <type_traits>
#include <utility>
#include "linalg.hpp"

namespace estkit {

// ---- compile-time detection of optional model members -------------------------
template <class M, class = void> struct has_F : std::false_type {};
template <class M> struct has_F<M, std::void_t<decltype(std::declval<const M&>().F(std::declval<const Vec<M::NX>&>(), std::declval<const Vec<M::NU>&>()))>> : std::true_type {};
template <class M, class = void> struct has_H : std::false_type {};
template <class M> struct has_H<M, std::void_t<decltype(std::declval<const M&>().H(std::declval<const Vec<M::NX>&>(), std::declval<const Vec<M::NU>&>()))>> : std::true_type {};
template <class M, class = void> struct has_B : std::false_type {};
template <class M> struct has_B<M, std::void_t<decltype(std::declval<const M&>().B(std::declval<const Vec<M::NX>&>(), std::declval<const Vec<M::NU>&>()))>> : std::true_type {};
template <class M, class = void> struct has_constrain : std::false_type {};
template <class M> struct has_constrain<M, std::void_t<decltype(std::declval<const M&>().constrain(std::declval<Vec<M::NX>>()))>> : std::true_type {};
template <class M, class = void> struct has_params : std::false_type {};
template <class M> struct has_params<M, std::void_t<decltype(M::NP), decltype(std::declval<const M&>().params())>> : std::true_type {};

// ---- numeric Jacobians (central differences) ------------------------------------
// default relative step: eps^(1/3) of the machine precision (GVL §5.x guidance)
constexpr Real kJacEps = (sizeof(Real) == 4) ? Real(2e-3) : Real(1e-6);
template <class M>
inline Mat<M::NX, M::NX> numeric_F(const M& m, const Vec<M::NX>& x, const Vec<M::NU>& u, Real eps = kJacEps) {
    Mat<M::NX, M::NX> J;
    for (int j = 0; j < M::NX; ++j) {
        Vec<M::NX> xp = x, xm = x; const Real d = eps * (Real(1) + std::fabs(x[j]));
        xp[j] += d; xm[j] -= d;
        const Vec<M::NX> fp = m.f(xp, u), fm = m.f(xm, u);
        for (int i = 0; i < M::NX; ++i) J(i, j) = (fp[i] - fm[i]) / (Real(2) * d);
    }
    return J;
}
template <class M>
inline Mat<M::NY, M::NX> numeric_H(const M& m, const Vec<M::NX>& x, const Vec<M::NU>& u, Real eps = kJacEps) {
    Mat<M::NY, M::NX> J;
    for (int j = 0; j < M::NX; ++j) {
        Vec<M::NX> xp = x, xm = x; const Real d = eps * (Real(1) + std::fabs(x[j]));
        xp[j] += d; xm[j] -= d;
        const Vec<M::NY> hp = m.h(xp, u), hm = m.h(xm, u);
        for (int i = 0; i < M::NY; ++i) J(i, j) = (hp[i] - hm[i]) / (Real(2) * d);
    }
    return J;
}
template <class M>
inline Mat<M::NX, M::NU> numeric_B(const M& m, const Vec<M::NX>& x, const Vec<M::NU>& u, Real eps = kJacEps) {
    Mat<M::NX, M::NU> J;
    for (int j = 0; j < M::NU; ++j) {
        Vec<M::NU> up = u, um = u; const Real d = eps * (Real(1) + std::fabs(u[j]));
        up[j] += d; um[j] -= d;
        const Vec<M::NX> fp = m.f(x, up), fm = m.f(x, um);
        for (int i = 0; i < M::NX; ++i) J(i, j) = (fp[i] - fm[i]) / (Real(2) * d);
    }
    return J;
}

// ---- uniform accessors: analytic member if present, numeric otherwise -----------
template <class M> inline Mat<M::NX, M::NX> jac_F(const M& m, const Vec<M::NX>& x, const Vec<M::NU>& u) {
    if constexpr (has_F<M>::value) return m.F(x, u); else return numeric_F(m, x, u);
}
template <class M> inline Mat<M::NY, M::NX> jac_H(const M& m, const Vec<M::NX>& x, const Vec<M::NU>& u) {
    if constexpr (has_H<M>::value) return m.H(x, u); else return numeric_H(m, x, u);
}
template <class M> inline Mat<M::NX, M::NU> jac_B(const M& m, const Vec<M::NX>& x, const Vec<M::NU>& u) {
    if constexpr (has_B<M>::value) return m.B(x, u); else return numeric_B(m, x, u);
}
template <class M> inline Vec<M::NX> constrain(const M& m, Vec<M::NX> x) {
    if constexpr (has_constrain<M>::value) return m.constrain(x); else return x;
}

// ---- Augmented model: appends NP slowly varying parameters as random-walk states
//      (joint state/parameter estimation, Plett 2004 Part 3 §3; Ljung 1979) ------
//  x_aug = [x; theta],  theta_{k+1} = theta_k + w_theta,  cov(w_theta) = diag(q_theta)
template <class M>
struct AugmentedModel {
    static_assert(has_params<M>::value, "AugmentedModel requires M::NP, params(), set_params()");
    static constexpr int NP = M::NP;
    static constexpr int NX = M::NX + NP;
    static constexpr int NU = M::NU;
    static constexpr int NY = M::NY;
    M base;
    Vec<NP> q_theta;   // random-walk variances of the parameters (per step)

    explicit AugmentedModel(const M& m = M(), const Vec<NP>& q = Vec<NP>()) : base(m), q_theta(q), scratch_(m) {}
    AugmentedModel(const AugmentedModel& o) : base(o.base), q_theta(o.q_theta), scratch_(o.base) {}
    AugmentedModel& operator=(const AugmentedModel& o) { base = o.base; q_theta = o.q_theta; scratch_ = o.base; return *this; }

    // The base model with the parameters of x_aug applied.  A single scratch copy is
    // re-parameterised in place (set_params only touches the parameter fields), which
    // avoids copying the whole model — e.g. the OCV table of the battery model — on
    // every call.  Single-threaded use assumed (the scratch object is shared).
    const M& with(const Vec<NX>& x) const { scratch_.set_params(x.template block<NP, 1>(M::NX, 0)); return scratch_; }
    static Vec<M::NX> xs(const Vec<NX>& x) { return x.template block<M::NX, 1>(0, 0); }

    Vec<NX> f(const Vec<NX>& x, const Vec<NU>& u) const {
        Vec<NX> xn;
        xn.set_block(0, 0, with(x).f(xs(x), u));
        for (int k = 0; k < NP; ++k) xn[M::NX + k] = x[M::NX + k];
        return xn;
    }
    Vec<NY> h(const Vec<NX>& x, const Vec<NU>& u) const { return with(x).h(xs(x), u); }
    Mat<NX, NX> F(const Vec<NX>& x, const Vec<NU>& u) const {
        Mat<NX, NX> Fa = Mat<NX, NX>::identity();
        Fa.set_block(0, 0, jac_F(with(x), xs(x), u));
        // df/dtheta by central differences
        for (int k = 0; k < NP; ++k) {
            Vec<NX> xp = x, xm = x; const Real d = kJacEps * (Real(1) + std::fabs(x[M::NX + k]));
            xp[M::NX + k] += d; xm[M::NX + k] -= d;
            const Vec<M::NX> fp = with(xp).f(xs(x), u);
            const Vec<M::NX> fm = with(xm).f(xs(x), u);
            for (int i = 0; i < M::NX; ++i) Fa(i, M::NX + k) = (fp[i] - fm[i]) / (Real(2) * d);
        }
        return Fa;
    }
    Mat<NY, NX> H(const Vec<NX>& x, const Vec<NU>& u) const {
        Mat<NY, NX> Ha;
        Ha.set_block(0, 0, jac_H(with(x), xs(x), u));
        for (int k = 0; k < NP; ++k) {
            Vec<NX> xp = x, xm = x; const Real d = kJacEps * (Real(1) + std::fabs(x[M::NX + k]));
            xp[M::NX + k] += d; xm[M::NX + k] -= d;
            const Vec<NY> hp = with(xp).h(xs(x), u);
            const Vec<NY> hm = with(xm).h(xs(x), u);
            for (int i = 0; i < NY; ++i) Ha(i, M::NX + k) = (hp[i] - hm[i]) / (Real(2) * d);
        }
        return Ha;
    }
    Mat<NX, NX> Q(const Vec<NX>& x, const Vec<NU>& u) const {
        Mat<NX, NX> Qa; Qa.set_block(0, 0, with(x).Q(xs(x), u));
        for (int k = 0; k < NP; ++k) Qa(M::NX + k, M::NX + k) = q_theta[k];
        return Qa;
    }
    Mat<NY, NY> R(const Vec<NX>& x, const Vec<NU>& u) const { return with(x).R(xs(x), u); }
    Vec<NX> constrain(Vec<NX> x) const {
        Vec<NX> xc = x; xc.set_block(0, 0, estkit::constrain(with(x), xs(x))); return xc;
    }
private:
    mutable M scratch_;
};

}  // namespace estkit
