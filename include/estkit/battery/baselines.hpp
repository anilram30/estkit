// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/battery/baselines.hpp — non-Kalman reference methods every BMS ships:
//    * Coulomb counting (ampere-hour integration)  — Ng et al. (2009), Applied
//      Energy 86, 1506-1511; Plett (2015) Vol. I §3.3.
//    * OCV inversion with a low-pass filtered, IR-compensated voltage — Plett
//      (2015) Vol. I §3.2 (the "voltage lookup" method).
//    * Coulomb counting with OCV re-anchoring at rest (industry hybrid).
// =============================================================================
#pragma once
#include "ecm_model.hpp"

namespace estkit {

class CoulombCounting final : public Estimator {
public:
    const char* name() const override { return "CoulombCounting"; }
    const char* group() const override { return "baselines"; }
    size_t state_bytes() const override { return sizeof(*this); }
    void reset(const EstimatorConfig& c) override { m_ = EcmModel(c); z_ = c.z0; have_prev_ = false; x_ = m_.x0(c); }
    void step(Real i, Real v, Real T) override {
        (void)v;
        const Vec<2> u = EcmModel::u_of(i, T);
        if (have_prev_) x_ = m_.f(x_, u_prev_);
        ypred_ = m_.h(x_, u)[0];
        z_ = x_[EcmModel::IZ];
        u_prev_ = u; have_prev_ = true;
    }
    Real soc() const override { return z_; }
    Real voltage_pred() const override { return ypred_; }
private:
    EcmModel m_; Vec<4> x_; Vec<2> u_prev_; bool have_prev_ = false; Real z_ = 0, ypred_ = kNaN;
};

// Invert the OCV curve from the IR/RC-compensated terminal voltage, then low-pass.
class OcvLookup final : public Estimator {
public:
    const char* name() const override { return "OCVLookup"; }
    const char* group() const override { return "baselines"; }
    size_t state_bytes() const override { return sizeof(*this); }
    void reset(const EstimatorConfig& c) override { m_ = EcmModel(c); z_ = c.z0; x_ = m_.x0(c); have_prev_ = false; tau_lp_ = Real(30); }
    void step(Real i, Real v, Real T) override {
        const Vec<2> u = EcmModel::u_of(i, T);
        if (have_prev_) x_ = m_.f(x_, u_prev_);           // open-loop RC states from the model
        ypred_ = m_.h(x_, u)[0];
        // OCV estimate = v + R0 i + v1 + v2 - M h
        const Real ocv = v + m_.R0(T) * i + x_[1] + x_[2] - m_.p.M * x_[3];
        const Real z_inst = invert_ocv(ocv, T);
        const Real a = std::exp(-m_.dt / tau_lp_);
        z_ = a * z_ + (Real(1) - a) * z_inst;
        x_[EcmModel::IZ] = z_;
        u_prev_ = u; have_prev_ = true;
    }
    Real soc() const override { return z_; }
    Real voltage_pred() const override { return ypred_; }
private:
    Real invert_ocv(Real ocv, Real T) const {   // bisection on the monotone OCV table
        Real lo = Real(-0.05), hi = Real(1.05);
        for (int it = 0; it < 40; ++it) { const Real mid = Real(0.5) * (lo + hi); if (m_.ocv.ocv(mid, T) < ocv) lo = mid; else hi = mid; }
        return Real(0.5) * (lo + hi);
    }
    EcmModel m_; Vec<4> x_; Vec<2> u_prev_; bool have_prev_ = false; Real z_ = 0, ypred_ = kNaN, tau_lp_ = 30;
};

// Coulomb counting with slow OCV correction under low load ("OCV-corrected Coulomb
// counting", the usual industrial hybrid — Plett 2015 Vol. I §3.4; Ng et al. 2009):
// the SOC is integrated from the current at all times; whenever the load is light
// (|i| < i_thr, where the IR/RC-compensated terminal voltage is a trustworthy OCV
// proxy) the count is pulled towards the OCV-inverted SOC with a first-order gain
// (time constant tau_corr).  Under heavy load the OCV proxy is unreliable and the
// correction is frozen.
class CoulombOcvHybrid final : public Estimator {
public:
    struct Options { Real i_thr_C = Real(0.3); Real tau_corr = Real(60); } opts;
    const char* name() const override { return "CoulombOCVHybrid"; }
    const char* group() const override { return "baselines"; }
    size_t state_bytes() const override { return sizeof(*this); }
    void reset(const EstimatorConfig& c) override { m_ = EcmModel(c); x_ = m_.x0(c); have_prev_ = false; ypred_ = kNaN; }
    void step(Real i, Real v, Real T) override {
        const Vec<2> u = EcmModel::u_of(i, T);
        if (have_prev_) x_ = m_.f(x_, u_prev_);                 // Coulomb counting + open-loop RC/hysteresis states
        ypred_ = m_.h(x_, u)[0];
        if (std::fabs(i) < opts.i_thr_C * m_.p.Q_Ah) {
            const Real ocv = v + m_.R0(T) * i + x_[1] + x_[2] - m_.p.M * x_[3];   // compensated OCV proxy
            Real lo = Real(-0.05), hi = Real(1.05);
            for (int it = 0; it < 40; ++it) { const Real mid = Real(0.5) * (lo + hi); if (m_.ocv.ocv(mid, T) < ocv) lo = mid; else hi = mid; }
            const Real z_ocv = Real(0.5) * (lo + hi);
            const Real a = std::exp(-m_.dt / opts.tau_corr);
            x_[0] = a * x_[0] + (Real(1) - a) * z_ocv;
        }
        u_prev_ = u; have_prev_ = true;
    }
    Real soc() const override { return x_[0]; }
    Real voltage_pred() const override { return ypred_; }
private:
    EcmModel m_; Vec<4> x_; Vec<2> u_prev_; bool have_prev_ = false; Real ypred_ = kNaN;
};

}  // namespace estkit
