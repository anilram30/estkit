// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// Classical Kalman family — registered here (see include/estkit/filters/)
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/filters/ekf.hpp"
#include "estkit/filters/ukf.hpp"
#include "estkit/filters/kf_linear.hpp"
#include "estkit/filters/iekf.hpp"
#include "estkit/filters/ckf.hpp"
#include "estkit/filters/sckf.hpp"
#include "estkit/filters/ckf_high_degree.hpp"
#include "estkit/filters/cdkf.hpp"
#include "estkit/filters/ghkf.hpp"
#include "estkit/filters/iplf.hpp"
#include "estkit/filters/eif.hpp"
#include "estkit/filters/srekf.hpp"
#include "estkit/filters/srukf.hpp"
#include "estkit/filters/udkf.hpp"
#include "estkit/filters/soekf.hpp"
namespace estkit {
void register_kalman(EstimatorList& list) {
    add_cell_estimator<Ekf<EcmModel>>(list, "EKF", "kalman");
    add_cell_estimator<Ukf<EcmModel>>(list, "UKF", "kalman");
    // Plain (time-invariant) Kalman filter: the model is linearised once, at the
    // initial estimate and at rest (i = 0) with the temperature reported at reset.
    add_cell_estimator<LinearKf<EcmModel>>(list, "LKF", "kalman",
        [](LinearKf<EcmModel>& f, const EstimatorConfig& c) { f.opts.u_lin = EcmModel::u_of(Real(0), c.T0); });
    add_cell_estimator<Iekf<EcmModel>>(list, "IEKF", "kalman");
    add_cell_estimator<Ckf<EcmModel>>(list, "CKF", "kalman");
    add_cell_estimator<SqrtCkf<EcmModel>>(list, "SCKF", "kalman");
    add_cell_estimator<Ckf5<EcmModel>>(list, "CKF5", "kalman");
    add_cell_estimator<Cdkf<EcmModel>>(list, "CDKF", "kalman");
    add_cell_estimator<Ghkf<EcmModel, 3>>(list, "GHKF", "kalman");
    add_cell_estimator<Iplf<EcmModel>>(list, "IPLF", "kalman");
    add_cell_estimator<Eif<EcmModel>>(list, "EIF", "kalman");
    add_cell_estimator<SqrtEkf<EcmModel>>(list, "SR-EKF", "kalman");
    add_cell_estimator<SqrtUkf<EcmModel>>(list, "SR-UKF", "kalman");
    add_cell_estimator<UdKf<EcmModel>>(list, "UD-EKF", "kalman");
    add_cell_estimator<SecondOrderEkf<EcmModel>>(list, "SO-EKF", "kalman");
}
}
