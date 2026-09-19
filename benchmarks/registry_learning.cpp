// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  Learning (data-driven) family — registered here.
//    NN-Direct    include/estkit/learning/nn_direct.hpp     (Chemali et al. 2018)
//    NN-EKF       include/estkit/learning/nn_ekf.hpp        (Charkhgard & Farrokhi 2010)
//    ELM-RLS-EKF  include/estkit/learning/elm_rls_ekf.hpp   (Huang et al. 2006 + RLS)
//
//  The two offline-trained networks are fitted lazily, ONCE per process, on the
//  first reset() of the corresponding estimator (fixed-wing / hover-hold missions
//  only; the eVTOL test mission is never used for training).
// =============================================================================
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/learning/elm_rls_ekf.hpp"
#include "estkit/learning/nn_direct.hpp"
#include "estkit/learning/nn_ekf.hpp"

namespace estkit {

using ElmEkfCell = ElmRlsEkf<EcmModel, 20>;

// Regressor scaling for the cell model: u = [i, T], x = [z, v1, v2, h].
// The inputs are mapped to O(1) so that the random tanh features of the ELM
// operate in their non-saturated range.
static void configure_elm(ElmEkfCell& f, const EstimatorConfig& c) {
    // NOTE: the random feature map is deliberately NOT reseeded per run -- a fixed,
    // frozen hidden layer is part of the delivered artefact (determinism/qualification).
    f.opts.u_center[EcmModel::UI] = Real(0);
    f.opts.u_scale [EcmModel::UI] = Real(1) / Real(8);        // current  [A]
    f.opts.u_center[EcmModel::UT] = c.T0;
    f.opts.u_scale [EcmModel::UT] = Real(1) / Real(15);       // temperature [K]
    f.opts.x_center[EcmModel::IZ] = Real(0.5);
    f.opts.x_scale [EcmModel::IZ] = Real(2);
    f.opts.x_center[EcmModel::IV1] = Real(0);
    f.opts.x_scale [EcmModel::IV1] = Real(50);
    f.opts.x_center[EcmModel::IV2] = Real(0);
    f.opts.x_scale [EcmModel::IV2] = Real(50);
    f.opts.x_center[EcmModel::IH] = Real(0);
    f.opts.x_scale [EcmModel::IH] = Real(1);
}

void register_learning(EstimatorList& list) {
    list.push_back(std::make_unique<NnDirectSoc>());          // implements Estimator directly
    add_cell_estimator<NnEkf, NnEcmModel>(list, "NN-EKF", "learning");
    add_cell_estimator<ElmEkfCell>(list, "ELM-RLS-EKF", "learning", configure_elm);
}

}  // namespace estkit
