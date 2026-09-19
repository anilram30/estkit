// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/pack/consensus.hpp — KALMAN-CONSENSUS pack filter (distributed
//  estimation of the common cell state over a cell-monitor network).
//
//  References
//    * Olfati-Saber, R. (2007). Distributed Kalman filtering for sensor
//      networks. Proc. 46th IEEE Conference on Decision and Control (CDC),
//      New Orleans, 5492-5498.        (Algorithm 1: DKF; Algorithm 2: KCF)
//    * Olfati-Saber, R., Fax, J. A. & Murray, R. M. (2007). Consensus and
//      cooperation in networked multi-agent systems. Proc. IEEE 95(1), 215-233.
//                                     (graph Laplacian, convergence conditions)
//    * Olfati-Saber, R. (2009). Kalman-consensus filter: optimality, stability
//      and performance. Proc. 48th IEEE CDC, 7036-7042.
//    * Mutambara, A. G. O. (1998). Decentralized Estimation and Control for
//      Multisensor Systems. CRC Press.            (information-form updates)
//
//  ---------------------------------------------------------------------------
//  Setting
//  ---------------------------------------------------------------------------
//  Each cell j of the series module is a NODE of a communication graph G
//  (a ring or a daisy-chained line in a real module: node j talks to j-1 and
//  j+1 only).  Every node estimates the SAME common state
//      xbar = [zbar, v1bar, v2bar, hbar]^T
//  from its own voltage channel, exchanging with its neighbours N_j
//      xhat_j          (nx numbers)   -- its current prior estimate
//      u_j = H_j^T R_j^-1 ytilde_j    (nx numbers)   -- its measurement information vector
//      U_j = H_j^T R_j^-1 H_j         (nx(nx+1)/2)   -- its measurement information matrix
//
//  Kalman-Consensus Filter (Olfati-Saber 2007, Alg. 2), one period at node i:
//
//      u_i^agg = sum_{j in J_i} u_j ,   U_i^agg = sum_{j in J_i} U_j ,   J_i = N_i U {i}   (1)
//      M_i     = ( P_i^-1 + U_i^agg )^-1                                                   (2)
//      xhat_i  = xbar_i + M_i ( u_i^agg - U_i^agg xbar_i )
//                       + eps M_i sum_{j in N_i} ( xbar_j - xbar_i )                       (3)
//      xbar_i  <- f(xhat_i, u) ,   P_i <- F M_i F^T + Q                                    (4)
//
//  with the covariance-weighted consensus gain of the paper
//      eps = gamma / ( ||M_i||_F + 1 ) ,   gamma > 0 small.                                (5)
//
//  Stability.  Writing the consensus increment in (3) as -eps M_i (L xbar)_i with L
//  the graph Laplacian, the consensus part of the error recursion is
//  (I - eps M L).  Since eps ||M_i|| = gamma ||M_i||/(||M_i||+1) < gamma and
//  lambda_max(L) <= 2 d_max, the discrete consensus iteration is stable whenever
//      gamma < 1 / d_max ,
//  i.e. gamma < 1/2 for a ring/line (d_max = 2).  Equation (5) is precisely the
//  normalisation that makes this bound independent of the covariance scale, so
//  no re-tuning is needed as P_i shrinks.  With a connected graph the KCF is
//  asymptotically unbiased and all nodes reach consensus on xbar
//  (Olfati-Saber 2009); with the COMPLETE graph (1)-(3) collapse onto the
//  centralised information filter of distributed_information.hpp.
//
//  Per-cell SOC:  z_j = xhat_j[z] + dz_j with the 2-state delta filters of
//  pack_common.hpp driven by node j's own residual v_j - h(xhat_j, u_j).  The
//  zero-mean projection sum_j dz_j = 0 is NOT available locally, so it is off by
//  default for this estimator (the delta filter then acts as node j's slow
//  scalar corrector of its own common estimate).
//
//  Heap-free: all per-node storage is in fixed arrays of size kPackCells.
// =============================================================================
#pragma once
#include <string>
#include "pack_common.hpp"

namespace estkit {

enum class PackTopology { Ring, Line, Complete };

template <class Model = EcmModel>
class ConsensusPack final : public PackEstimator {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int kCov = NX * (NX + 1) / 2;
    static_assert(NY == 1, "one voltage channel per cell");

    struct Options {
        PackTopology topology = PackTopology::Ring;
        Real gamma = Real(0.4);            // consensus gain constant, eq. (5); gamma < 1/d_max
        Real sigma_dz = Real(0.03);        // assumed cell-to-cell SOC deviation -> R_j inflation
        Real eps_R = Real(0.08);           // assumed relative impedance spread -> R_j inflation
        bool compensate_deviation = true;  // remove the estimated deviation from v_j before the update
        DeltaBank::Options delta;
    } opts;

    explicit ConsensusPack(std::string name, std::string group = "pack")
        : name_(std::move(name)), group_(std::move(group)) {
        opts.delta.reference = DeltaReference::ModelPrediction;
        opts.delta.zero_mean = false;        // sum_j dz_j = 0 is not computable at a local node
    }

    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }

    void reset(const EstimatorConfig& cfg) override {
        sigma_v_ = cfg.sigma_v;
        model_ = Model(cfg);
        const Vec<NX> x0 = model_.x0(cfg);
        const Mat<NX, NX> P0 = model_.P0(cfg);
        for (int c = 0; c < kPackCells; ++c) { xb_[c] = x0; xh_[c] = x0; P_[c] = P0; M_[c] = P0; }
        delta_.opts = opts.delta;
        delta_.opts.Q_Ah = cfg.params.Q_Ah;
        delta_.opts.eta_c = cfg.params.eta_c;
        delta_.reset(cfg.dt);
        have_prev_ = false;
    }

    void step(Real i, const Real* v, const Real* T) override {
        // ---- (4) time update, carried out at the start of the period so that
        //      the harness convention predict(u_{k-1}) / update(y_k, u_k) holds
        if (have_prev_) {
            for (int c = 0; c < kPackCells; ++c) {
                const Mat<NX, NX> F = jac_F(model_, xh_[c], u_prev_[c]);
                xb_[c] = model_.f(xh_[c], u_prev_[c]);
                Mat<NX, NX> P = F * M_[c] * F.t() + model_.Q(xb_[c], u_prev_[c]);
                P.symmetrize();
                P_[c] = P;
            }
        }

        // ---- local measurement information of every node ----
        Real w_bar = Real(0);
        for (int c = 0; c < kPackCells; ++c) {
            const Vec<NU> u = Model::u_of(i, T[c]);
            const Mat<NY, NX> H = jac_H(model_, xb_[c], u);
            const Real g = ecm_docv(model_, xb_[c], T[c]);
            const Real wtot = ecm_overpotential(model_, xb_[c], u);
            const Real w = opts.delta.include_rc_in_regressor ? wtot : model_.R0(T[c]) * i;
            const Real R = model_.R(xb_[c], u)(0, 0) + sq(g * opts.sigma_dz) + sq(opts.eps_R * wtot);
            Real yj = v[c];
            if (opts.compensate_deviation)
                yj -= delta_.deviation_voltage(model_, c, xb_[c][Model::IZ], T[c], w);
            // linearised measurement referred to the origin: ytilde = y - h(xbar) + H xbar
            const Real ytil = yj - model_.h(xb_[c], u)(0, 0) + (H * xb_[c])(0, 0);
            const Real invR = (R > Real(0)) ? Real(1) / R : Real(0);
            const Vec<NX> Ht = H.t();
            ui_[c] = Ht * (invR * ytil);          // u_j = H^T R^-1 ytilde
            Ui_[c] = outer(Ht, Ht) * invR;        // U_j = H^T R^-1 H   (rank one)
            w_bar += wtot;
            u_prev_[c] = u;
        }
        w_bar /= Real(kPackCells);

        // ---- (1)-(3) neighbourhood aggregation, information update, consensus ----
        int nb[kPackCells];
        for (int c = 0; c < kPackCells; ++c) {
            const int nn = neighbours(c, nb);
            Vec<NX> uagg = ui_[c];
            Mat<NX, NX> Uagg = Ui_[c];
            Vec<NX> dx;                                   // sum_{j in N_i} (xbar_j - xbar_i)
            for (int q = 0; q < nn; ++q) {
                const int j = nb[q];
                uagg += ui_[j]; Uagg += Ui_[j];
                dx += xb_[j] - xb_[c];
            }
            Mat<NX, NX> Yi = inverse(P_[c]);
            if (!Yi.is_finite()) Yi = Mat<NX, NX>::identity() * Real(1e6);
            Mat<NX, NX> S = Yi + Uagg; S.symmetrize();
            Mat<NX, NX> M = inverse(S);
            if (!M.is_finite()) M = P_[c];
            M.symmetrize();
            const Real eps = opts.gamma / (M.norm() + Real(1));          // eq. (5)
            Vec<NX> xh = xb_[c] + M * (uagg - Uagg * xb_[c]) + (M * dx) * eps;
            if (!xh.is_finite()) xh = xb_[c];
            xh_[c] = constrain(model_, xh);
            M_[c] = M;
        }
        have_prev_ = true;

        // ---- per-cell delta filters (node j uses its own common estimate) ----
        Real ref[kPackCells], zc[kPackCells];
        for (int c = 0; c < kPackCells; ++c) zc[c] = xh_[c][Model::IZ];
        if (opts.delta.reference == DeltaReference::MeanVoltage) {
            Real vbar = Real(0);
            for (int c = 0; c < kPackCells; ++c) vbar += v[c];
            vbar /= Real(kPackCells);
            for (int c = 0; c < kPackCells; ++c) ref[c] = vbar;
        } else {
            for (int c = 0; c < kPackCells; ++c) ref[c] = model_.h(xh_[c], Model::u_of(i, T[c]))(0, 0);
        }
        Real Tbar = Real(0);
        for (int c = 0; c < kPackCells; ++c) Tbar += T[c];
        Tbar /= Real(kPackCells);
        const Real w_use = opts.delta.include_rc_in_regressor ? w_bar : model_.R0(Tbar) * i;
        delta_.step(model_, zc, Tbar, i, v, ref, w_use, w_bar, sigma_v_);
    }

    Real cell_soc(int c) const override {
        return clampr(xh_[c][Model::IZ] + delta_.dz(c), Real(-0.05), Real(1.05));
    }

    size_t state_bytes() const override {
        return sizeof(Model) + kPackCells * (2 * sizeof(Vec<NX>) + 2 * sizeof(Mat<NX, NX>)) + sizeof(DeltaBank);
    }
    // per node: xbar (nx) + u (nx) + U (nx(nx+1)/2) broadcast to its neighbours
    int messages_per_step() const override { return kPackCells * (2 * NX + kCov); }

    int node_degree() const {
        switch (opts.topology) {
            case PackTopology::Ring: return 2;
            case PackTopology::Line: return 2;
            default: return kPackCells - 1;
        }
    }
    const Vec<NX>& node_state(int c) const { return xh_[c]; }
    const Mat<NX, NX>& node_cov(int c) const { return M_[c]; }
    const DeltaBank& deltas() const { return delta_; }

private:
    // fills nb[] with the neighbours of node c and returns their number
    int neighbours(int c, int* nb) const {
        int n = 0;
        switch (opts.topology) {
            case PackTopology::Ring:
                nb[n++] = (c + kPackCells - 1) % kPackCells;
                nb[n++] = (c + 1) % kPackCells;
                break;
            case PackTopology::Line:
                if (c > 0) nb[n++] = c - 1;
                if (c < kPackCells - 1) nb[n++] = c + 1;
                break;
            case PackTopology::Complete:
                for (int j = 0; j < kPackCells; ++j) if (j != c) nb[n++] = j;
                break;
        }
        return n;
    }

    std::string name_, group_;
    Model model_;
    DeltaBank delta_;
    Vec<NX> xb_[kPackCells], xh_[kPackCells];      // prior / posterior node estimates
    Mat<NX, NX> P_[kPackCells], M_[kPackCells];    // prior / posterior node covariances
    Vec<NX> ui_[kPackCells];
    Mat<NX, NX> Ui_[kPackCells];
    Vec<NU> u_prev_[kPackCells];
    Real sigma_v_ = Real(0.002);
    bool have_prev_ = false;
};

}  // namespace estkit
