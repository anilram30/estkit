// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/pack/distributed_information.hpp — CENTRALISED INFORMATION-FILTER
//  FUSION of all NC cell-voltage channels onto the common cell state.
//
//  References
//    * Mutambara, A. G. O. (1998). Decentralized Estimation and Control for
//      Multisensor Systems. CRC Press, Boca Raton, Ch. 3-4.
//    * Grime, S. & Durrant-Whyte, H. F. (1994). Data fusion in decentralized
//      sensor networks. Control Engineering Practice 2(5), 849-863.
//    * Sun, S.-L. & Deng, Z.-L. (2004). Multi-sensor optimal information fusion
//      Kalman filter. Automatica 40(6), 1017-1023.
//    * Anderson, B. D. O. & Moore, J. B. (1979). Optimal Filtering.
//      Prentice-Hall, Ch. 6 (information form).
//    * Plett, G. L. (2016). Battery Management Systems, Volume II. Artech
//      House, Ch. 4.                          (pack architecture, delta filters)
//
//  ---------------------------------------------------------------------------
//  The centralised optimum of the common-state formulation
//  ---------------------------------------------------------------------------
//  Stacking all NC cell voltages into one measurement vector
//      y_k = [v_1, ..., v_NC]^T in R^NC ,   y = h_stack(xbar, u) + e ,
//      e ~ N(0, R),   R = blockdiag(R_1, ..., R_NC)
//  (the channels are independent: separate ADCs, separate references), the
//  minimum-variance estimate of the common state is the ordinary Kalman update
//  with n_y = NC = 12.  Because R is block diagonal, the measurement update
//  collapses into a SUM of per-cell information contributions:
//
//      Y_k^+ = (P_k^-)^-1 + sum_j H_j^T R_j^-1 H_j ,                       (1)
//      xhat_k^+ = xhat_k^- + P_k^+ sum_j H_j^T R_j^-1 ( v_j - h(xhat_k^-, u_j) ).  (2)
//
//  (1)-(2) are algebraically identical to the n_y = 12 covariance-form update
//      K = P^- H^T (H P^- H^T + R)^-1 ,
//  but need only n_x x n_x inversions (two 4x4) instead of one 12x12 inverse:
//  O(NC n_x^2 + n_x^3) instead of O(NC^2 n_x + NC^3).  This is the classical
//  argument for the information filter in multi-sensor fusion (Mutambara 1998;
//  Grime & Durrant-Whyte 1994).  Set `covariance_form = true` to run the literal
//  n_y = 12 update instead and verify the identity numerically.
//
//  This estimator is the CENTRALISED BENCHMARK that the federated filter
//  (federated.hpp) reproduces exactly in fusion-reset mode and that the
//  Kalman-consensus filter (consensus.hpp) approaches only asymptotically,
//  because a consensus node aggregates the information of its own
//  neighbourhood (3 of 12 channels on a ring) rather than of the whole module.
//
//  Modelling caveat.  The deviation of cell j from the common state,
//      v_j - h(xbar, u_j) = OCV'(zbar) dz_j - drho_j w + ... ,
//  is systematic, not white; representing it by the inflated
//      R_j = R + (OCV'(zbar) sigma_dz)^2 + (eps_R w)^2
//  is a Schmidt-type "consider" approximation shared by all four common-state
//  estimators of this family, so their comparison remains meaningful.  The
//  residual part is recovered by the per-cell delta filters.
// =============================================================================
#pragma once
#include <string>
#include "pack_common.hpp"

namespace estkit {

template <class Model = EcmModel>
class InformationFusionPack final : public PackEstimator {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NYP = kPackCells;          // stacked pack measurement dimension
    static_assert(NY == 1, "one voltage channel per cell");

    struct Options {
        bool covariance_form = false;      // literal n_y = NC update instead of (1)-(2)
        Real sigma_dz = Real(0.03);        // assumed cell-to-cell SOC deviation -> R_j inflation
        Real eps_R = Real(0.08);           // assumed relative impedance spread -> R_j inflation
        bool compensate_deviation = true;  // remove the estimated deviation from v_j before the update
        bool joseph = true;                // Joseph form in the covariance-form branch
        DeltaBank::Options delta;
    } opts;

    explicit InformationFusionPack(std::string name, std::string group = "pack")
        : name_(std::move(name)), group_(std::move(group)) {
        opts.delta.reference = DeltaReference::ModelPrediction;
    }

    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }

    void reset(const EstimatorConfig& cfg) override {
        sigma_v_ = cfg.sigma_v;
        model_ = Model(cfg);
        x_ = model_.x0(cfg); P_ = model_.P0(cfg);
        delta_.opts = opts.delta;
        delta_.opts.Q_Ah = cfg.params.Q_Ah;
        delta_.opts.eta_c = cfg.params.eta_c;
        delta_.reset(cfg.dt);
        have_prev_ = false;
    }

    void step(Real i, const Real* v, const Real* T) override {
        Real Tbar = Real(0);
        for (int c = 0; c < kPackCells; ++c) Tbar += T[c];
        Tbar /= Real(kPackCells);

        // ---- time update (covariance form: one F P F^T, cheaper than the
        //      information-form time update which needs F^-1 and Q^-1) ----
        if (have_prev_) {
            const Mat<NX, NX> F = jac_F(model_, x_, u_prev_);
            x_ = model_.f(x_, u_prev_);
            P_ = F * P_ * F.t() + model_.Q(x_, u_prev_);
            P_.symmetrize();
        }
        u_prev_ = Model::u_of(i, Tbar); have_prev_ = true;

        // ---- per-channel linearisation ----
        Mat<NYP, NX> H;
        Real innov[kPackCells], Rd[kPackCells];
        Real w_bar = Real(0);
        for (int c = 0; c < kPackCells; ++c) {
            const Vec<NU> u = Model::u_of(i, T[c]);
            const Mat<NY, NX> Hc = jac_H(model_, x_, u);
            H.set_row(c, Hc.row(0));
            const Real g = ecm_docv(model_, x_, T[c]);
            const Real wtot = ecm_overpotential(model_, x_, u);
            const Real w = opts.delta.include_rc_in_regressor ? wtot : model_.R0(T[c]) * i;
            Rd[c] = model_.R(x_, u)(0, 0) + sq(g * opts.sigma_dz) + sq(opts.eps_R * wtot);
            Real yj = v[c];
            if (opts.compensate_deviation)
                yj -= delta_.deviation_voltage(model_, c, x_[Model::IZ], T[c], w);
            innov[c] = yj - model_.h(x_, u)(0, 0);
            w_bar += wtot;
        }
        w_bar /= Real(kPackCells);

        if (!opts.covariance_form) {
            // ---- information-form fusion, eqs. (1)-(2) ----
            Mat<NX, NX> Y = inverse(P_);
            Vec<NX> s;                                   // sum_j H_j^T R_j^-1 innov_j
            if (Y.is_finite()) {
                for (int c = 0; c < kPackCells; ++c) {
                    const Vec<NX> Ht = H.row(c).t();
                    const Real invR = (Rd[c] > Real(0)) ? Real(1) / Rd[c] : Real(0);
                    Y += outer(Ht, Ht) * invR;
                    s += Ht * (invR * innov[c]);
                }
                Y.symmetrize();
                Mat<NX, NX> Pp = inverse(Y);
                if (Pp.is_finite()) {
                    Pp.symmetrize();
                    const Vec<NX> xn = x_ + Pp * s;
                    if (xn.is_finite()) { x_ = xn; P_ = Pp; }
                }
            }
        } else {
            // ---- literal n_y = NC covariance-form update (identical result) ----
            Mat<NYP, NYP> R;
            Vec<NYP> e;
            for (int c = 0; c < kPackCells; ++c) { R(c, c) = Rd[c]; e[c] = innov[c]; }
            const Mat<NYP, NYP> S = H * P_ * H.t() + R;
            const Mat<NX, NYP> K = P_ * H.t() * inverse(S);
            if (K.is_finite()) {
                const Vec<NX> xn = x_ + K * e;
                Mat<NX, NX> Pn;
                if (opts.joseph) {
                    const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
                    Pn = IKH * P_ * IKH.t() + K * R * K.t();
                } else {
                    Pn = (Mat<NX, NX>::identity() - K * H) * P_;
                }
                Pn.symmetrize();
                if (xn.is_finite() && Pn.is_finite()) { x_ = xn; P_ = Pn; }
            }
        }
        x_ = constrain(model_, x_);

        // ---- per-cell delta filters ----
        Real ref[kPackCells], zc[kPackCells];
        for (int c = 0; c < kPackCells; ++c) zc[c] = x_[Model::IZ];
        if (opts.delta.reference == DeltaReference::MeanVoltage) {
            Real vbar = Real(0);
            for (int c = 0; c < kPackCells; ++c) vbar += v[c];
            vbar /= Real(kPackCells);
            for (int c = 0; c < kPackCells; ++c) ref[c] = vbar;
        } else {
            for (int c = 0; c < kPackCells; ++c) ref[c] = model_.h(x_, Model::u_of(i, T[c]))(0, 0);
        }
        const Real w_use = opts.delta.include_rc_in_regressor ? w_bar : model_.R0(Tbar) * i;
        delta_.step(model_, zc, Tbar, i, v, ref, w_use, w_bar, sigma_v_);
    }

    Real cell_soc(int c) const override {
        return clampr(x_[Model::IZ] + delta_.dz(c), Real(-0.05), Real(1.05));
    }
    Real pack_soc_mean() const override { return x_[Model::IZ]; }

    size_t state_bytes() const override {
        return sizeof(Model) + sizeof(x_) + sizeof(P_) + sizeof(DeltaBank);
    }
    // fully centralised: every slave ships its voltage to the module master
    int messages_per_step() const override { return kPackCells; }

    const Vec<NX>& common_state() const { return x_; }
    const Mat<NX, NX>& common_cov() const { return P_; }
    const DeltaBank& deltas() const { return delta_; }

private:
    std::string name_, group_;
    Model model_;
    DeltaBank delta_;
    Vec<NX> x_;
    Mat<NX, NX> P_;
    Vec<NU> u_prev_;
    Real sigma_v_ = Real(0.002);
    bool have_prev_ = false;
};

}  // namespace estkit
