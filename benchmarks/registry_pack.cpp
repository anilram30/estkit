// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  registry_pack — pack-level (multi-cell, distributed) estimators.
//
//  Baseline already registered by pack_estimator.hpp: `Decentralized-EKF`
//  (one independent 4-state EKF per cell, no information exchange).
//
//  Registered here
//    BarDelta                 Plett (2009, 2016): one full filter on the pack
//                             average + NC cheap 2-state delta filters at 1 Hz.
//    Federated                Carlson (1990): NC local filters on the common
//                             state with information sharing beta_j = 1/NC and
//                             a fusion-reset master.
//    Consensus                Olfati-Saber (2007) Kalman-consensus filter on a
//                             ring communication graph.
//    InfoFusion               Mutambara (1998) / Grime & Durrant-Whyte (1994):
//                             centralised information-form fusion of all NC
//                             voltage channels -- the common-state optimum.
//    Decentralized-UKF        the baseline with a sigma-point cell filter.
//    Decentralized-ReducedEKF the baseline with the 3-state reduced ECM model.
// =============================================================================
#include "estkit/battery/ecm_model_reduced.hpp"
#include "estkit/filters/ukf.hpp"
#include "estkit/pack/bar_delta.hpp"
#include "estkit/pack/consensus.hpp"
#include "estkit/pack/distributed_information.hpp"
#include "estkit/pack/federated.hpp"
#include "estkit/pack/pack_common.hpp"

namespace estkit {

void register_pack(PackList& list) {
    // --- bar-delta (Plett 2009) -------------------------------------------------
    {
        auto e = std::make_unique<BarDeltaPack<Ekf<EcmModel>>>("BarDelta");
        list.push_back(std::move(e));
    }
    // --- federated filter with information sharing (Carlson 1990) ---------------
    {
        auto e = std::make_unique<FederatedPack<Ekf<EcmModel>>>("Federated");
        list.push_back(std::move(e));
    }
    // --- Kalman-consensus filter on a ring graph (Olfati-Saber 2007) -----------
    {
        auto e = std::make_unique<ConsensusPack<EcmModel>>("Consensus");
        list.push_back(std::move(e));
    }
    // --- centralised information-form fusion (Mutambara 1998) ------------------
    {
        auto e = std::make_unique<InformationFusionPack<EcmModel>>("InfoFusion");
        list.push_back(std::move(e));
    }
    // --- decentralised baseline variants ---------------------------------------
    list.push_back(std::make_unique<DecentralizedPack<Ukf<EcmModel>>>("Decentralized-UKF"));
    list.push_back(std::make_unique<DecentralizedPack<Ekf<EcmModelReduced>, EcmModelReduced>>("Decentralized-ReducedEKF"));
}

}  // namespace estkit
