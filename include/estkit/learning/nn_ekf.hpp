// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/learning/nn_ekf.hpp — hybrid model/data estimator: an extended Kalman
//  filter on the equivalent-circuit model whose OUTPUT EQUATION is augmented by
//  a small neural network trained offline on the model residual.
//
//  References
//    * Charkhgard, M. & Farrokhi, M. (2010). State-of-charge estimation for
//      lithium-ion batteries using neural networks and EKF. IEEE Transactions on
//      Industrial Electronics 57(12), 4178-4187.
//      (neural network + EKF hybrid; the network supplies what the lumped model
//       cannot represent, the EKF supplies the recursive state estimate)
//    * Hornik, K., Stinchcombe, M. & White, H. (1989). Multilayer feedforward
//      networks are universal approximators. Neural Networks 2(5), 359-366.
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs - Part 3. State and parameter
//      estimation. J. Power Sources 134(2), 277-292.
//    * Mayergoyz, I. D. (1991). Mathematical Models of Hysteresis. Springer.
//      (the Preisach operator whose effect the residual network learns)
//
//  Model.  The estimator model is the 2-RC ESC model of battery/ecm_model.hpp,
//
//      x = [z, v1, v2, h]^T,   u = [i, T]^T,
//      h_ECM(x,u) = OCV(z,T) + M h - v1 - v2 - R0(T) i,                   (1)
//
//  augmented by a learned residual r: R^3 -> R,
//
//      h(x,u) = h_ECM(x,u) + sat_{r_max}( g * r(z, i, T; theta) ).        (2)
//
//  The network is trained offline to reproduce the mismatch between the TRUE
//  terminal voltage of the plant and the nominal ECM prediction evaluated at the
//  true SOC,
//
//      r_k = v_k - h_ECM( [z_k, v1_k, v2_k, h_k], u_k ),                  (3)
//
//  where v1, v2, h are propagated open-loop by the nominal model from the
//  measured current, so that r carries exactly the part of the output error that
//  the lumped model cannot express: the difference between the one-state
//  hysteresis M h and the Preisach operator of the truth cell, the entropic /
//  thermal terms, and the resistance growth of an aged cell.
//
//  Filtering.  With (2) the model still satisfies the estkit model concept, so
//  the ordinary EKF of filters/ekf.hpp is used unchanged; only the measurement
//  Jacobian picks up the network's sensitivity,
//
//      H(x,u) = H_ECM(x,u) + [ dr/dz, 0, 0, 0 ],                          (4)
//      dr/dz  ~= ( r(z+delta,i,T) - r(z-delta,i,T) ) / (2 delta),         (5)
//
//  evaluated by central differences with delta = 1e-4 (the network is smooth, so
//  the truncation error is O(delta^2) ~ 1e-8 and the cancellation error
//  O(eps/delta) ~ 1e-12 in double precision).
// =============================================================================
#pragma once
#include "../battery/ecm_model.hpp"
#include "../core/model.hpp"
#include "../filters/ekf.hpp"
#include "mlp.hpp"
#include "train_data.hpp"

namespace estkit {

// -----------------------------------------------------------------------------
//  The residual network: r(z, i, T) with one hidden tanh layer.
//  Kept deliberately small (3-10-1, 51 parameters): the residual is a smooth,
//  low-dimensional surface and a larger network would only fit sensor noise.
// -----------------------------------------------------------------------------
using NnResidualNet = Mlp<3, 10, 1>;

struct NnResidualOptions {
    TrainingSetSpec data;
    MlpTrainOptions train;
    uint64_t init_seed = 0xA11CE007ull;
};

inline NnResidualOptions nn_residual_default_options() {
    NnResidualOptions o;
    o.data.profiles  = {"fixed_wing", "hover_hold"};
    o.data.scenarios = {"nominal", "cold", "hot"};   // three ambient temperatures, so that
                                                     // the residual's T-dependence is covered
    o.data.n_seeds   = 2;
    o.data.seed0     = 211;
    o.data.preisach  = 1;      // the characterisation cells DO carry the Preisach operator:
                               // the residual model is fitted to the cell chemistry the ECM
                               // cannot represent. See the chapter for what this costs on a
                               // cell that does NOT show the extra hysteresis.
    o.data.subsample = 10;
    o.train.epochs = 12;
    o.train.batch  = 32;
    o.train.lr     = Real(5e-3);
    o.train.l2     = Real(1e-6);
    return o;
}

// Offline fit of eq. (3). Propagates the nominal ECM open-loop with the measured
// current while pinning z to the ground truth, and regresses the voltage error
// on (z, i, T).
inline NnResidualNet train_nn_residual() {
    const NnResidualOptions o = nn_residual_default_options();
    std::vector<Vec<3>> X;
    std::vector<Vec<1>> Y;
    X.reserve(40000); Y.reserve(40000);
    for_each_training_dataset(o.data, [&](const Dataset& d) {
        EcmModel m(d.est_cfg);
        Vec<4> x = m.x0(d.est_cfg);
        x[EcmModel::IZ] = d.soc_true[0];
        x[EcmModel::IH] = (d.soc_true[0] > Real(0.5)) ? Real(1) : Real(-1);  // as the plant is reset
        for (int k = 0; k < d.n; ++k) {
            x[EcmModel::IZ] = d.soc_true[k];                 // pin the SOC to the truth
            const Vec<2> u = EcmModel::u_of(d.i_meas[k], d.T_meas[k]);
            const Real r = d.v_meas[k] - m.h(x, u)[0];       // eq. (3)
            if (k % o.data.subsample == 0) {
                Vec<3> f;
                f[0] = d.soc_true[k];
                f[1] = d.i_meas[k];
                f[2] = d.T_meas[k] - kT0;
                X.push_back(f);
                Vec<1> y; y[0] = r;
                Y.push_back(y);
            }
            x = m.f(x, u);                                   // open-loop v1, v2, h
        }
    });
    NnResidualNet net;
    Rng rng(o.init_seed);
    net.init_glorot(rng);
    train_mlp(net, X.data(), Y.data(), int(X.size()), o.train, rng);
    return net;
}

// The shared, lazily trained residual network (one instance per process).
inline const NnResidualNet& nn_residual_net() {
    static const NnResidualNet net = train_nn_residual();
    return net;
}

// -----------------------------------------------------------------------------
//  NnEcmModel — the ECM with the learned output residual, eqs. (2), (4).
//  Satisfies the estkit model concept, so any generic filter accepts it; here it
//  is used with the plain EKF.
//
//  A default-constructed NnEcmModel has NO network attached (it is then exactly
//  the nominal ECM). The network is bound only by the EstimatorConfig
//  constructor, i.e. inside CellEstimator::reset(), which is where the one-off
//  training is triggered.
// -----------------------------------------------------------------------------
struct NnEcmModel {
    static constexpr int NX = EcmModel::NX, NU = EcmModel::NU, NY = EcmModel::NY;
    static constexpr int IZ = EcmModel::IZ, IV1 = EcmModel::IV1, IV2 = EcmModel::IV2, IH = EcmModel::IH;
    static constexpr int UI = EcmModel::UI, UT = EcmModel::UT;

    EcmModel base;
    const NnResidualNet* net = nullptr;
    Real r_gain = Real(1);        // trust placed in the learned residual, 0 = pure EKF
    Real r_max  = Real(0.06);     // saturation of eq. (2) [V] -- bounds the damage a
                                  // badly extrapolating network can do
    Real fd_eps = Real(1e-4);     // central-difference step of eq. (5)
    Real sigma_nn = Real(0.004);  // 1-sigma error of the residual model [V]; inflates R

    NnEcmModel() = default;
    explicit NnEcmModel(const EstimatorConfig& c) : base(c), net(&nn_residual_net()) {}

    // eq. (2), the learned part alone
    Real residual(Real z, Real i, Real T) const {
        if (net == nullptr || r_gain == Real(0)) return Real(0);
        Vec<3> f; f[0] = z; f[1] = i; f[2] = T - kT0;
        return clampr(r_gain * (*net)(f)[0], -r_max, r_max);
    }

    // --- model concept -------------------------------------------------------
    Vec<NX> f(const Vec<NX>& x, const Vec<NU>& u) const { return base.f(x, u); }
    Mat<NX, NX> F(const Vec<NX>& x, const Vec<NU>& u) const { return base.F(x, u); }
    Mat<NX, NU> B(const Vec<NX>& x, const Vec<NU>& u) const { return base.B(x, u); }
    Vec<NY> h(const Vec<NX>& x, const Vec<NU>& u) const {
        Vec<NY> y = base.h(x, u);
        y[0] += residual(x[IZ], u[UI], u[UT]);
        return y;
    }
    Mat<NY, NX> H(const Vec<NX>& x, const Vec<NU>& u) const {
        Mat<NY, NX> Hm = base.H(x, u);
        if (net != nullptr && r_gain != Real(0)) {
            const Real d = fd_eps;
            const Real rp = residual(x[IZ] + d, u[UI], u[UT]);
            const Real rm = residual(x[IZ] - d, u[UI], u[UT]);
            Hm(0, IZ) += (rp - rm) / (Real(2) * d);                 // eq. (5)
        }
        return Hm;
    }
    Mat<NX, NX> Q(const Vec<NX>& x, const Vec<NU>& u) const { return base.Q(x, u); }
    // The measurement-noise covariance is inflated by the residual model's own
    // uncertainty sigma_nn^2 (the network is fitted to a *population* of cells,
    // so its prediction error on any single specimen is not zero).
    Mat<NY, NY> R(const Vec<NX>& x, const Vec<NU>& u) const {
        Mat<NY, NY> Rm = base.R(x, u);
        Rm(0, 0) += sq(sigma_nn);
        return Rm;
    }
    Vec<NX> constrain(Vec<NX> x) const { return base.constrain(x); }

    // --- helpers used by the benchmark adapter -------------------------------
    static Vec<NU> u_of(Real i, Real T) { return EcmModel::u_of(i, T); }
    Vec<NX> x0(const EstimatorConfig& c) const { return base.x0(c); }
    Mat<NX, NX> P0(const EstimatorConfig& c) const { return base.P0(c); }
};

// The registered estimator is just the ordinary EKF on the augmented model.
using NnEkf = Ekf<NnEcmModel>;

}  // namespace estkit
