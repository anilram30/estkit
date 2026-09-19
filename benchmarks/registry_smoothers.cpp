// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  Smoothing family: fixed-interval (RTS / URTS), fixed-lag and full-information
//  (batch nonlinear least squares) estimators.  Headers in
//  include/estkit/smoothers/.
//
//  ERTS, URTS and Batch-NLS are OFFLINE (estkit::BatchEstimator): the benchmark
//  hands them the whole recorded mission at once, which is the ground-station
//  post-flight situation.  FixedLag-EKF is recursive and is registered through
//  the ordinary CellEstimator adapter.
// =============================================================================
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/filters/ekf.hpp"
#include "estkit/smoothers/rts.hpp"
#include "estkit/smoothers/urts.hpp"
#include "estkit/smoothers/fixed_lag.hpp"
#include "estkit/smoothers/batch_nls.hpp"

namespace estkit {

void register_smoothers(EstimatorList& list) {
    // fixed-interval smoothers (offline)
    list.push_back(std::make_unique<RtsSmoother<Ekf<EcmModel>>>("ERTS", "smoothers"));
    list.push_back(std::make_unique<UnscentedRtsSmoother<EcmModel>>("URTS", "smoothers"));
    // full-information nonlinear least squares (offline)
    list.push_back(std::make_unique<BatchNls<EcmModel>>("Batch-NLS", "smoothers"));
    // real-time fixed-lag smoother, lag L = 10 samples = 1 s at 10 Hz
    add_cell_estimator<FixedLagSmoother<EcmModel, 10>>(list, "FixedLag-EKF", "smoothers");
}

}  // namespace estkit
