// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// Robust / embedded Kalman family — registered here (see include/estkit/filters/)
//   Schmidt-KF        consider filter over theta = [Q_Ah, R0]   (schmidt_kf.hpp)
//   Hinf-EKF          game-theoretic H-infinity filter          (hinf.hpp)
//   Huber-EKF         Huber M-estimation measurement update     (huber_kf.hpp)
//   MC-EKF            maximum-correntropy Kalman filter         (mckf.hpp)
//   StudentT-KF       Student's t filter                        (student_t_kf.hpp)
//   Constrained-EKF   estimate projection onto box constraints  (constrained_ekf.hpp)
//   ReducedOrder-EKF  EKF on the 3-state reduced ECM            (battery/ecm_model_reduced.hpp)
//   SVSF              smooth variable structure filter (SVSF-KF) (svsf.hpp)
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/battery/ecm_model_reduced.hpp"
#include "estkit/filters/constrained_ekf.hpp"
#include "estkit/filters/ekf.hpp"
#include "estkit/filters/hinf.hpp"
#include "estkit/filters/huber_kf.hpp"
#include "estkit/filters/mckf.hpp"
#include "estkit/filters/schmidt_kf.hpp"
#include "estkit/filters/student_t_kf.hpp"
#include "estkit/filters/svsf.hpp"

namespace estkit {

void register_robust_kalman(EstimatorList& list) {
    const char* G = "robust_kalman";

    // --- Schmidt-Kalman "consider" filter -----------------------------------
    // theta = [Q_Ah, R0]: 5 % capacity (a typical cell-to-cell + ageing spread
    // before re-identification) and 20 % series resistance (Arrhenius scatter +
    // ageing growth, Plett 2015 Vol. I §2.10).
    add_cell_estimator<SchmidtKf<EcmModel>>(list, "Schmidt-KF", G,
        [](SchmidtKf<EcmModel>& f, const EstimatorConfig&) {
            f.opts.rel_std[0] = Real(0.05);   // Q_Ah
            f.opts.rel_std[1] = Real(0.20);   // R0
        });

    // --- game-theoretic H-infinity filter ------------------------------------
    // L = I and S = diag(1,0,0,0): the performance output is the SOC only.
    // theta = 30 (SOC^-2) is about 3e-4 of the SOC information P^-1 + H^T R^-1 H
    // in steady state; larger values make the filter track the R0 mismatch.
    add_cell_estimator<HinfFilter<EcmModel>>(list, "Hinf-EKF", G,
        [](HinfFilter<EcmModel>& f, const EstimatorConfig&) {
            for (int k = 0; k < EcmModel::NX; ++k) f.opts.s_diag[k] = Real(0);
            f.opts.s_diag[EcmModel::IZ] = Real(1);
            f.opts.theta = Real(30);
            f.opts.feas_margin = Real(0.9);
        });

    // --- Huber M-estimation EKF ----------------------------------------------
    add_cell_estimator<HuberKf<EcmModel>>(list, "Huber-EKF", G);

    // --- maximum-correntropy EKF ---------------------------------------------
    add_cell_estimator<MaxCorrentropyKf<EcmModel>>(list, "MC-EKF", G);

    // --- Student's t filter ---------------------------------------------------
    add_cell_estimator<StudentTKf<EcmModel>>(list, "StudentT-KF", G);

    // --- constrained EKF (estimate projection) --------------------------------
    // Physical box constraints of the ECM state: SOC in [0,1], hysteresis in [-1,1].
    add_cell_estimator<ConstrainedEkf<EcmModel>>(list, "Constrained-EKF", G,
        [](ConstrainedEkf<EcmModel>& f, const EstimatorConfig&) {
            f.opts.con.set(EcmModel::IZ, Real(0), Real(1));
            f.opts.con.set(EcmModel::IH, Real(-1), Real(1));
        });

    // --- reduced-order EKF (3 states, quasi-static slow RC branch) -------------
    add_cell_estimator<Ekf<EcmModelReduced>, EcmModelReduced>(list, "ReducedOrder-EKF", G);

    // --- smooth variable structure filter (SVSF-KF form) ----------------------
    // The boundary layer psi must cover the "existence subspace", i.e. the bound
    // on the VOLTAGE modelling error, not only the sensor noise: a 20 % R0
    // uncertainty at the 4.5 C hover current (22.5 A) is already ~54 mV, and the
    // OCV/RC/hysteresis residuals add to it -> psi = 0.10 V (Habibi 2007, §IV).
    add_cell_estimator<Svsf<EcmModel>>(list, "SVSF", G,
        [](Svsf<EcmModel>& f, const EstimatorConfig&) {
            f.opts.psi_abs[0] = Real(0.10);
            f.opts.gamma = Real(0.20);
        });
}

}  // namespace estkit
