// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  Joint / dual state-parameter estimation and state-of-health (SOH) family.
//  Headers live in include/estkit/parameter/.
//
//  Tuning rationale for the ECM parameter vector theta = [Q_Ah, R0]
//  (see report/chapters/soh/*.tex for the full discussion):
//
//   * P0(Q_Ah) = (0.10 * Q_nom)^2.  A BMS knows the nameplate capacity of the
//     pack it was commissioned with; 10 % (0.5 Ah on a 5 Ah cell) covers a cell
//     that has aged to its 80 % end-of-life threshold within 2 sigma and is the
//     order of magnitude Plett (2004, Part 3) uses for the capacity prior.
//   * q(Q_Ah) = (5e-5 Ah)^2 per 0.1 s step.  Capacity fade is a calendar/cycle
//     process: 5e-5 Ah/step accumulates to 9.5 mAh (0.19 % of 5 Ah) of allowed
//     random walk per hour of operation, i.e. the filter is told the capacity is
//     essentially constant within a flight but must never lock up completely.
//   * P0(R0) = (0.30 * R0_nom)^2.  The `aged_cell` scenario grows R0 by 37.5 %
//     and `param_mismatch` starts it 30 % low, so both are ~1 sigma events.
//   * q(R0) = (2e-6 ohm)^2 per step -> 0.28 mohm (2.3 % of R0) per hour, which
//     lets the estimate track the slow drift that the Arrhenius law does not
//     already account for.
//   * Box constraints keep the augmented state physical if the mission is
//     unexciting (Q in [0.5, 1.5] Q_nom, R0 in [0.2, 5] R0_nom).
// =============================================================================
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/parameter/joint_ekf.hpp"
#include "estkit/parameter/joint_ukf.hpp"
#include "estkit/parameter/dual_ekf.hpp"
#include "estkit/parameter/dual_ukf.hpp"
#include "estkit/parameter/rls_ekf.hpp"
#include "estkit/parameter/awtls_capacity.hpp"

namespace estkit {

namespace {
// common theta prior / random-walk settings for the ECM (theta = [Q_Ah, R0])
template <class Opts>
inline void set_theta_priors(Opts& o, const EstimatorConfig& c) {
    o.q_theta[0]  = sq(Real(5e-5));                      // [Ah^2 / step]
    o.q_theta[1]  = sq(Real(2e-6));                      // [ohm^2 / step]
    o.p0_theta[0] = sq(Real(0.10) * c.params.Q_Ah);      // [Ah^2]
    o.p0_theta[1] = sq(Real(0.30) * c.params.R0);        // [ohm^2]
    o.theta_lo[0] = Real(0.5) * c.params.Q_Ah;  o.theta_hi[0] = Real(1.5) * c.params.Q_Ah;
    o.theta_lo[1] = Real(0.2) * c.params.R0;    o.theta_hi[1] = Real(5.0) * c.params.R0;
}
}  // namespace

void register_soh(EstimatorList& list) {
    // ---- joint (augmented-state) estimation ---------------------------------
    using JE = JointEkf<EcmModel>;
    add_cell_estimator<JE>(list, "JointEKF", "soh",
        [](JE& f, const EstimatorConfig& c) { set_theta_priors(f.opts, c); })
        .with_capacity([](const JE& f) { return f.capacity(); });

    using JU = JointUkf<EcmModel>;
    add_cell_estimator<JU>(list, "JointUKF", "soh",
        [](JU& f, const EstimatorConfig& c) { set_theta_priors(f.opts, c); })
        .with_capacity([](const JU& f) { return f.capacity(); });

    // ---- dual (two coupled filters) -----------------------------------------
    // `innov_gate` is available on both dual filters but is left OFF: on this
    // mission the residual is dominated by ECM model error during the high-current
    // transients, not by sensor noise, so any gate tight enough to catch a 50 mV
    // voltage outlier also rejects the most informative samples (measured: with a
    // 5 sigma gate the nominal SOC RMSE_ss of DualEKF rises from 0.0011 to 0.013).
    using DE = DualEkf<EcmModel>;
    add_cell_estimator<DE>(list, "DualEKF", "soh",
        [](DE& f, const EstimatorConfig& c) { set_theta_priors(f.opts, c); })
        .with_capacity([](const DE& f) { return f.capacity(); });

    // The dual SPKF needs its measurement-noise covariance tuned independently of
    // the state filter's (Plett 2006 Part 2, §5): the residual that drives the
    // parameter filter contains the state error and the correlated ECM model error
    // as well as the sensor noise.  R_theta is inflated by 9 (3 sigma); measured on
    // `aged_cell` this takes the SOC RMSE_ss from 0.0170 to 0.0013 and the capacity
    // error from -0.154 Ah to +0.018 Ah.  A larger factor (16) was also tried and
    // degrades `sensor_dropout`.
    using DU = DualUkf<EcmModel>;
    add_cell_estimator<DU>(list, "DualUKF", "soh",
        [](DU& f, const EstimatorConfig& c) { set_theta_priors(f.opts, c); f.opts.r_theta_scale = Real(9); })
        .with_capacity([](const DU& f) { return f.capacity(); });

    // ---- battery-specific SOH methods (implement Estimator directly) --------
    list.push_back(std::make_unique<RlsEkf>());
    list.push_back(std::make_unique<AwtlsCapacity>());
}

}  // namespace estkit
