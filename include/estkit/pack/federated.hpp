// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/pack/federated.hpp — FEDERATED (information-sharing) pack filter.
//
//  References
//    * Carlson, N. A. (1990). Federated square root filter for decentralized
//      parallel processes. IEEE Trans. Aerospace and Electronic Systems 26(3),
//      517-525.
//    * Carlson, N. A. & Berarducci, M. P. (1994). Federated Kalman filter
//      simulation results. Navigation: J. Inst. of Navigation 41(3), 297-321.
//    * Mutambara, A. G. O. (1998). Decentralized Estimation and Control for
//      Multisensor Systems. CRC Press, Ch. 4.
//    * Plett, G. L. (2016). Battery Management Systems, Volume II. Artech
//      House, Ch. 4.                        (pack architecture, delta filters)
//
//  ---------------------------------------------------------------------------
//  Structure
//  ---------------------------------------------------------------------------
//  The NC cells of a series module share one current, hence one COMMON state
//      xbar = [zbar, v1bar, v2bar, hbar]^T ,
//  which every cell-monitor slave observes through its own voltage channel
//      v_j = h(xbar, u_j) + OCV'(zbar) dz_j - drho_j w + noise .
//  Slave j runs a LOCAL filter LF_j on xbar with measurement v_j; the cell's own
//  deviation acts as an extra, zero-mean measurement error, so
//      R_j = R + (OCV'(zbar) sigma_dz)^2 + (eps_R w)^2 .                     (1)
//
//  Because all NC local filters are driven by the SAME process noise, their
//  errors are correlated and a naive information sum would be optimistic.
//  Carlson's information-sharing principle removes the correlation by
//  DIVIDING the common information among the local filters:
//      Q_j = Q / beta_j ,   P_j(0) = P(0) / beta_j ,   sum_j beta_j = 1 .     (2)
//  Each LF_j is then a conservative (information-deflated) filter, its errors
//  are mutually uncorrelated by construction, and the master fusion
//      P_m^-1 = sum_j P_j^-1 ,      x_m = P_m sum_j P_j^-1 x_j                (3)
//  reconstructs exactly the information that a centralised filter would have.
//  With equal sensors beta_j = 1/NC.
//
//  Reset modes (Carlson 1990, Tab. I):
//    * FR (fusion reset, default): after (3) the locals are re-initialised with
//      the fused estimate,  x_j <- x_m ,  P_j <- P_m / beta_j .  For identical
//      local dynamics this makes the federation ALGEBRAICALLY EQUAL to the
//      centralised filter, which the benchmark verifies.
//    * NR (no reset):  the locals keep their own estimate; the master output is
//      still (3) but the federation is strictly conservative.
//  Fusion may be run every `fuse_every` samples to trade bus load for accuracy;
//  between fusions the master dead-reckons its mean through the model.
//
//  Per-cell SOC:  z_j = zbar + dz_j with dz_j from the shared 2-state delta
//  filters of pack_common.hpp, driven by the LOCAL residual v_j - h(x_m, u_j)
//  (no extra communication: the slave already has v_j and receives x_m).
//
//  Required filter interface (Ekf, Ukf, ... all satisfy it):
//      Filter(const Model&), init, predict, update, x(), P(), x_mut(), P_mut(),
//      model(), and (optionally) opts.q_scale for the information deflation.
// =============================================================================
#pragma once
#include <string>
#include "pack_common.hpp"

namespace estkit {

template <class Filter, class Model = EcmModel>
class FederatedPack final : public PackEstimator {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int kCov = NX * (NX + 1) / 2;   // numbers needed to ship a covariance

    struct Options {
        Real beta = Real(-1);              // information-sharing factor; < 0 -> 1/NC
        bool fusion_reset = true;          // FR mode (false: NR mode)
        int fuse_every = 1;                // master fusion period [samples]
        Real sigma_dz = Real(0.03);        // assumed cell-to-cell SOC deviation entering (1)
        Real eps_R = Real(0.08);           // assumed relative impedance spread entering (1)
        bool compensate_deviation = true;  // subtract the estimated deviation from v_j before the local update
        DeltaBank::Options delta;
    } opts;

    explicit FederatedPack(std::string name, std::string group = "pack")
        : name_(std::move(name)), group_(std::move(group)), model_(),
          lf_(fill_array<Filter, kPackCells>(Filter(model_))) {
        opts.delta.reference = DeltaReference::ModelPrediction;
    }

    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }

    void reset(const EstimatorConfig& cfg) override {
        sigma_v_ = cfg.sigma_v;
        model_ = Model(cfg);
        const Real beta = (opts.beta > Real(0)) ? opts.beta : Real(1) / Real(kPackCells);
        const Real inv_beta = Real(1) / beta;
        const Vec<NX> x0 = model_.x0(cfg);
        const Mat<NX, NX> P0 = model_.P0(cfg) * inv_beta;      // prior information sharing, eq. (2)
        for (int c = 0; c < kPackCells; ++c) {
            lf_[c] = Filter(model_);
            if constexpr (has_q_scale<Filter>::value) lf_[c].opts.q_scale = inv_beta;   // Q_j = Q/beta_j
            lf_[c].init(x0, P0);
        }
        xm_ = x0; Pm_ = model_.P0(cfg);
        delta_.opts = opts.delta;
        delta_.opts.Q_Ah = cfg.params.Q_Ah;
        delta_.opts.eta_c = cfg.params.eta_c;
        delta_.reset(cfg.dt);
        inv_beta_ = inv_beta;
        have_prev_ = false; k_ = 0;
    }

    void step(Real i, const Real* v, const Real* T) override {
        // ---- local filters (one per cell-monitor slave) ----
        Real w_bar = Real(0);
        for (int c = 0; c < kPackCells; ++c) {
            const Vec<NU> u = Model::u_of(i, T[c]);
            Filter& f = lf_[c];
            if (have_prev_) f.predict(u_prev_[c]);
            // measurement-noise inflation (1): the deviation of this cell from
            // the common state is treated as an extra white measurement error.
            const Real g = ecm_docv(f.model(), f.x(), T[c]);
            const Real wtot = ecm_overpotential(f.model(), f.x(), u);
            const Real w = opts.delta.include_rc_in_regressor ? wtot : f.model().R0(T[c]) * i;
            f.model().sigma_v = std::sqrt(sq(sigma_v_) + sq(g * opts.sigma_dz) + sq(opts.eps_R * wtot));
            Vec<NY> y; y[0] = v[c];
            if (opts.compensate_deviation)
                y[0] -= delta_.deviation_voltage(f.model(), c, f.x()[Model::IZ], T[c], w);
            f.update(y, u);
            u_prev_[c] = u;
            w_bar += wtot;
        }
        w_bar /= Real(kPackCells);

        // ---- master fusion (3), every `fuse_every` samples ----
        Real Tbar = Real(0);
        for (int c = 0; c < kPackCells; ++c) Tbar += T[c];
        Tbar /= Real(kPackCells);
        ++k_;
        const int fuse_every = opts.fuse_every > 0 ? opts.fuse_every : 1;
        if (k_ % fuse_every == 0) {
            Mat<NX, NX> Y;            // fused information matrix
            Vec<NX> yv;               // fused information vector
            bool ok = true;
            for (int c = 0; c < kPackCells; ++c) {
                const Mat<NX, NX> Yj = inverse(lf_[c].P());
                if (!Yj.is_finite()) { ok = false; break; }
                Y += Yj; yv += Yj * lf_[c].x();
            }
            if (ok) {
                Y.symmetrize();
                const Mat<NX, NX> Pm = inverse(Y);
                if (Pm.is_finite()) {
                    Pm_ = Pm; Pm_.symmetrize();
                    xm_ = Pm_ * yv;
                    if (opts.fusion_reset) {               // FR mode: redistribute the information
                        const Mat<NX, NX> Pj = Pm_ * inv_beta_;
                        for (int c = 0; c < kPackCells; ++c) { lf_[c].x_mut() = xm_; lf_[c].P_mut() = Pj; }
                    }
                }
            }
        } else if (have_prev_) {
            xm_ = model_.f(xm_, Model::u_of(i, Tbar));      // master dead-reckons between fusions
        }
        xm_ = constrain(model_, xm_);
        have_prev_ = true;

        // ---- per-cell delta filters ----
        Real ref[kPackCells], zc[kPackCells];
        for (int c = 0; c < kPackCells; ++c) zc[c] = xm_[Model::IZ];
        if (opts.delta.reference == DeltaReference::MeanVoltage) {
            Real vbar = Real(0);
            for (int c = 0; c < kPackCells; ++c) vbar += v[c];
            vbar /= Real(kPackCells);
            for (int c = 0; c < kPackCells; ++c) ref[c] = vbar;
        } else {
            for (int c = 0; c < kPackCells; ++c) ref[c] = model_.h(xm_, Model::u_of(i, T[c]))[0];
        }
        const Real w_use = opts.delta.include_rc_in_regressor ? w_bar : model_.R0(Tbar) * i;
        delta_.step(model_, zc, Tbar, i, v, ref, w_use, w_bar, sigma_v_);
    }

    Real cell_soc(int c) const override {
        return clampr(xm_[Model::IZ] + delta_.dz(c), Real(-0.05), Real(1.05));
    }
    Real pack_soc_mean() const override { return xm_[Model::IZ]; }

    size_t state_bytes() const override { return kPackCells * sizeof(Filter) + sizeof(DeltaBank) + sizeof(xm_) + sizeof(Pm_); }

    // Numbers crossing the module bus per sample (each broadcast counted once):
    //   every fusion   NC x (x_j, P_j) up  + (x_m, P_m) down in FR mode
    //   every delta    the zero-mean projection needs dz_j up and its mean down
    int messages_per_step() const override {
        const Real fuse = Real(kPackCells * (NX + kCov) + (opts.fusion_reset ? NX + kCov : 0))
                        / Real(opts.fuse_every > 0 ? opts.fuse_every : 1);
        const Real dlt = Real((opts.delta.zero_mean ? kPackCells + 1 : 0) + (opts.fusion_reset ? 0 : NX))
                       / Real(opts.delta.rate_div > 0 ? opts.delta.rate_div : 1);
        return int(fuse + dlt + Real(0.5));
    }

    const Vec<NX>& common_state() const { return xm_; }
    const Mat<NX, NX>& common_cov() const { return Pm_; }
    const DeltaBank& deltas() const { return delta_; }

private:
    std::string name_, group_;
    Model model_;
    std::array<Filter, kPackCells> lf_;
    DeltaBank delta_;
    Vec<NX> xm_;
    Mat<NX, NX> Pm_;
    Vec<NU> u_prev_[kPackCells];
    Real inv_beta_ = Real(kPackCells);
    Real sigma_v_ = Real(0.002);
    bool have_prev_ = false;
    long k_ = 0;
};

}  // namespace estkit
