// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/battery/plant_ecm.hpp — the "true" cell used to generate benchmark data
//  (enhanced equivalent-circuit plant).
//
//  Compared with the estimator model (ecm_model.hpp) the plant additionally has
//    * closed-form OCV (not a lookup table) with entropic temperature term
//    * lumped thermal ODE with irreversible + reversible (entropic) heating
//    * Arrhenius-scaled R0, R1, R2, tau1, tau2
//    * one-state hysteresis (Plett) OR a discrete Preisach operator
//    * capacity fade and resistance growth (cycle + calendar ageing)
//    * coulombic efficiency on charge
//  Current sign convention: discharge > 0.
// =============================================================================
#pragma once
#include "cell.hpp"

namespace estkit {

class PlantBase {
public:
    virtual ~PlantBase() = default;
    virtual void reset(Real z0, Real T0, Real T_amb) = 0;
    virtual void step(Real i, Real dt) = 0;             // advance the state by dt under current i
    virtual Real voltage(Real i) const = 0;             // terminal voltage at the current state under current i
    virtual Real soc() const = 0;
    virtual Real temperature() const = 0;               // cell temperature [K]
    virtual Real capacity_Ah() const = 0;               // present total capacity
    virtual Real ocv_now() const = 0;                   // equilibrium voltage at present SOC/T
    virtual void set_ambient(Real T_amb) = 0;
    virtual void age_calendar(Real days) {}             // optional: storage ageing between flights
    virtual Real ah_throughput() const { return Real(0); }
    virtual Real hysteresis_voltage() const { return Real(0); }
};

class EcmPlant final : public PlantBase {
public:
    CellParams p;
    bool use_preisach = false;
    Real preisach_M = Real(0.015);   // half-width of the major hysteresis loop [V]
    bool enable_aging = true;
    bool enable_thermal = true;

    explicit EcmPlant(const CellParams& params = CellParams::nominal()) : p(params) {}

    void reset(Real z0, Real T0, Real T_amb) override {
        z_ = z0; v1_ = v2_ = Real(0); h_ = (z0 > Real(0.5)) ? Real(1) : Real(-1); T_ = T0; T_amb_ = T_amb;   // arrives from a charge (h=+1) or a discharge (h=-1)
        Q_loss_ = Real(0); Ah_ = Real(0); t_days_ = Real(0);
        // Preisach memory consistent with the arrival direction: from a charge (ascending SOC) if z0 > 0.5
        if (use_preisach) { pr_.init(preisach_M); pr_.set_saturated(z0, z0 > Real(0.5)); }
    }
    void set_ambient(Real T_amb) override { T_amb_ = T_amb; }
    // start from a pre-aged condition (fraction of capacity lost)
    void set_capacity_loss(Real frac) { Q_loss_ = frac; }
    Real capacity_Ah() const override { return p.Q_Ah * (Real(1) - Q_loss_); }
    Real R0_now() const { return p.R0_T(T_) * (Real(1) + p.age_R_gain * Q_loss_); }
    Real soc() const override { return z_; }
    Real temperature() const override { return T_; }
    Real ah_throughput() const override { return Ah_; }
    Real capacity_loss() const { return Q_loss_; }
    // Battery sign convention: the charging (ascending-SOC) branch lies ABOVE the discharging branch.
    // The classical Preisach relay (switch up at u >= alpha, down at u <= beta) yields the opposite
    // (magnetic B-H) orientation, hence the minus sign.
    Real hysteresis_voltage() const override { return use_preisach ? -pr_.output() : p.M * h_; }
    Real ocv_now() const override { return Chen2020::ocv(z_) + (T_ - kTref) * entropic_dUdT(z_); }
    Real v1() const { return v1_; } Real v2() const { return v2_; } Real h() const { return h_; }

    Real voltage(Real i) const override {
        return ocv_now() + hysteresis_voltage() - v1_ - v2_ - R0_now() * i;
    }

    void step(Real i, Real dt) override {
        const Real Q = capacity_Ah();
        const Real eta = (i >= Real(0)) ? Real(1) : p.eta_c;
        // --- heat generation at the current state (Bernardi 1985; discharge-positive form) ---
        const Real v = voltage(i);
        const Real q_irr = i * (ocv_now() - v);                 // >= 0
        const Real q_rev = -i * T_ * entropic_dUdT(z_);          // entropic (reversible) heat
        // --- electrical states (ZOH discretisation) ---
        const Real a1 = std::exp(-dt / p.tau1_T(T_)), a2 = std::exp(-dt / p.tau2_T(T_));
        const Real dz = -eta * i * dt / (Real(3600) * Q);
        v1_ = a1 * v1_ + p.R1_T(T_) * (Real(1) - a1) * i;
        v2_ = a2 * v2_ + p.R2_T(T_) * (Real(1) - a2) * i;
        const Real ah = std::exp(-std::fabs(eta * i * p.gamma * dt / (Real(3600) * Q)));
        h_ = ah * h_ + (ah - Real(1)) * sgn(i);
        z_ += dz;
        if (use_preisach) pr_.update(z_);
        // --- thermal ---
        if (enable_thermal) {
            const Real dT = dt / p.C_th * (q_irr + q_rev - (T_ - T_amb_) / p.R_th);
            T_ += dT;
        }
        // --- ageing (rate form of Q_loss = B exp(-Ea/RT) Ah^z) ---
        const Real dAh = std::fabs(i) * dt / Real(3600);
        Ah_ += dAh;
        if (enable_aging) {
            const Real arr = safe_exp(-p.age_Ea / (kRgas * T_));
            const Real ahz = std::pow(std::max(Ah_, Real(1e-6)), p.age_z - Real(1));
            Q_loss_ += p.age_B * arr * p.age_z * ahz * dAh;
            t_days_ += dt / Real(86400);
        }
    }
    // storage ageing between missions (calendar term, sqrt(t) law)
    void age_calendar(Real days) override {
        if (!enable_aging) return;
        const Real arr = safe_exp(-p.age_cal_Ea / (kRgas * T_amb_));
        const Real t0 = t_days_, t1 = t_days_ + days;
        Q_loss_ += p.age_cal_A * arr * (std::sqrt(t1) - std::sqrt(t0));
        t_days_ = t1;
        // rest: RC states and thermal state relax to equilibrium
        v1_ = v2_ = Real(0); T_ = T_amb_;
    }

private:
    Real z_ = 1, v1_ = 0, v2_ = 0, h_ = 0, T_ = kTref, T_amb_ = kTref;
    Real Q_loss_ = 0, Ah_ = 0, t_days_ = 0;
    Preisach<32> pr_;
};

}  // namespace estkit
