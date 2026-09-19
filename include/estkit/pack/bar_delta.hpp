// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/pack/bar_delta.hpp — BAR-DELTA pack state estimation.
//
//  References
//    * Plett, G. L. (2009). Efficient battery pack state estimation using
//      bar-delta filtering. Proc. 24th International Battery, Hybrid and Fuel
//      Cell Electric Vehicle Symposium (EVS24), Stavanger, Norway.
//    * Plett, G. L. (2016). Battery Management Systems, Volume II:
//      Equivalent-Circuit Methods. Artech House, Norwood MA, Ch. 4.
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs - Part 3. State and parameter
//      estimation. J. Power Sources 134(2), 277-292.       (the "bar" filter)
//
//  ---------------------------------------------------------------------------
//  Idea
//  ---------------------------------------------------------------------------
//  Running one full nx-state nonlinear filter per cell costs NC times one cell
//  filter and gives NC nearly identical state trajectories, because all cells
//  carry the same current and therefore share their fast dynamics.  Bar-delta
//  filtering exploits this: ONE full filter (the "bar" filter) is run on the
//  pack-AVERAGE cell,
//
//      xbar = [zbar, v1bar, v2bar, hbar]^T ,   ybar = (1/NC) sum_j v_j ,
//
//  with the nominal (average) cell parameters, and NC cheap scalar/2-state
//  "delta" filters track only the deviations of each cell from that average:
//
//      z_j = zbar + dz_j ,        v_j - vbar = OCV'(zbar) dz_j - drho_j w + e_j ,
//
//  with w = R0 i + v1 + v2 the total overpotential of the average cell and
//  drho_j the relative impedance deviation of cell j (drho_j R0 = dR0_j).  The
//  delta filters are run at a REDUCED rate (Plett: ~1 Hz against a 10 Hz bar
//  filter) because the deviations change only through capacity and resistance
//  differences, i.e. very slowly.
//
//  Cost:   1 x (nx-state EKF at f_s)  +  NC x (2-state KF at f_s/rate_div)
//  versus  NC x (nx-state EKF at f_s) for the decentralised baseline: the
//  saving is close to a factor NC in both time and memory.
//
//  Noise of the average measurement: averaging NC independent voltage channels
//  divides the sensor variance by NC, but NOT the common-mode error (current
//  sensor, model error), so the bar filter uses
//      sigma_v,bar^2 = sigma_v^2 / NC + sigma_model^2 ,
//  the second term being the residual error of representing the pack by one
//  average cell (Jensen gap of the nonlinear OCV over the SOC spread).
// =============================================================================
#pragma once
#include <string>
#include "pack_common.hpp"

namespace estkit {

template <class Filter, class Model = EcmModel>
class BarDeltaPack final : public PackEstimator {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;

    struct Options {
        bool average_noise_reduction = true;   // sigma_v -> sigma_v/sqrt(NC) for the bar filter
        Real sigma_model = Real(0.002);        // residual "average-cell" model error [V]
        bool jensen_correction = true;         // remove mean_j[OCV(zbar+dz_j)] - OCV(zbar) from vbar
        bool deltas_on_slaves = false;         // delta filters run on the cell-monitor slaves (needs vbar broadcast)
        DeltaBank::Options delta;              // delta-filter tuning (see pack_common.hpp)
    } opts;

    explicit BarDeltaPack(std::string name, std::string group = "pack")
        : name_(std::move(name)), group_(std::move(group)), model_(), bar_(model_) {
        opts.delta.reference = DeltaReference::MeanVoltage;
    }

    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }

    void reset(const EstimatorConfig& cfg) override {
        sigma_v_ = cfg.sigma_v;
        model_ = Model(cfg);
        if (opts.average_noise_reduction)
            model_.sigma_v = std::sqrt(sq(cfg.sigma_v) / Real(kPackCells) + sq(opts.sigma_model));
        bar_ = Filter(model_);
        bar_.init(model_.x0(cfg), model_.P0(cfg));
        delta_.opts = opts.delta;
        delta_.opts.Q_Ah = cfg.params.Q_Ah;
        delta_.opts.eta_c = cfg.params.eta_c;
        delta_.reset(cfg.dt);
        have_prev_ = false; jensen_ = Real(0);
    }

    void step(Real i, const Real* v, const Real* T) override {
        Real vbar = Real(0), Tbar = Real(0);
        for (int c = 0; c < kPackCells; ++c) { vbar += v[c]; Tbar += T[c]; }
        vbar /= Real(kPackCells); Tbar /= Real(kPackCells);

        // ---- bar filter: one full nonlinear filter on the average cell ----
        const Vec<NU> u = Model::u_of(i, Tbar);
        if (have_prev_) bar_.predict(u_prev_);
        Vec<NY> y; y[0] = vbar - jensen_;      // Jensen gap of the nonlinear OCV over the SOC spread
        bar_.update(y, u);
        u_prev_ = u; have_prev_ = true;

        // ---- delta filters: NC 3-state filters at f_s / rate_div ----
        const Model& m = bar_.model();
        const Vec<NX>& x = bar_.x();
        const Real wtot = ecm_overpotential(m, x, u);
        const Real w = opts.delta.include_rc_in_regressor ? wtot : m.R0(Tbar) * i;
        Real ref[kPackCells], zc[kPackCells];
        for (int c = 0; c < kPackCells; ++c) zc[c] = x[Model::IZ];
        if (opts.delta.reference == DeltaReference::MeanVoltage) {
            for (int c = 0; c < kPackCells; ++c) ref[c] = vbar;
        } else {
            for (int c = 0; c < kPackCells; ++c) ref[c] = m.h(x, Model::u_of(i, T[c]))[0];
        }
        if (delta_.step(m, zc, Tbar, i, v, ref, w, wtot, sigma_v_) && opts.jensen_correction) {
            // mean_j OCV(zbar + dz_j) - OCV(zbar): the pack-average voltage is not
            // the voltage of the average cell because OCV is curved.  Refreshed at
            // the (low) delta rate, where it is essentially free.
            const Real zb = clampr(x[Model::IZ], Real(-0.05), Real(1.05));
            const Real ocv0 = m.ocv.ocv(zb, Tbar);
            Real acc = Real(0);
            for (int c = 0; c < kPackCells; ++c)
                acc += m.ocv.ocv(clampr(zb + delta_.dz(c), Real(-0.05), Real(1.05)), Tbar) - ocv0;
            jensen_ = acc / Real(kPackCells);
        }
    }

    Real cell_soc(int c) const override {
        return clampr(bar_.x()[Model::IZ] + delta_.dz(c), Real(-0.05), Real(1.05));
    }
    Real pack_soc_mean() const override { return bar_.x()[Model::IZ]; }

    size_t state_bytes() const override { return sizeof(Filter) + sizeof(DeltaBank); }
    // Architecture: the NC cell-monitor slaves send their voltage to the module
    // master every sample (the bar filter needs vbar); the master runs the bar
    // and the delta filters.  If the delta filters are distributed to the
    // slaves, vbar (one number) must additionally be broadcast.
    int messages_per_step() const override { return kPackCells + (opts.deltas_on_slaves ? 1 : 0); }

    // --- inspection ---
    const Filter& bar() const { return bar_; }
    Filter& bar() { return bar_; }
    const DeltaBank& deltas() const { return delta_; }
    Real cell_dR0(int c) const { return delta_.dR0(c, model_.p.R0); }

private:
    std::string name_, group_;
    Model model_;
    Filter bar_;
    DeltaBank delta_;
    Vec<NU> u_prev_;
    bool have_prev_ = false;
    Real sigma_v_ = Real(0.002);
    Real jensen_ = Real(0);
};

}  // namespace estkit
