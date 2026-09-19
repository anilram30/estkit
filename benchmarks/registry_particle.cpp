// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  Particle / ensemble / Gaussian-sum family — registration.
//  Headers: include/estkit/particle/ ; battery adapter for the marginalised
//  filter: include/estkit/battery/ecm_mixed_model.hpp
//
//  Every stochastic filter takes its RNG stream from EstimatorConfig::seed so
//  that each benchmark run is bit-exactly reproducible.
// =============================================================================
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/battery/ecm_mixed_model.hpp"
#include "estkit/particle/enkf.hpp"
#include "estkit/particle/etkf.hpp"
#include "estkit/particle/gsf.hpp"
#include "estkit/particle/pf_auxiliary.hpp"
#include "estkit/particle/pf_bootstrap.hpp"
#include "estkit/particle/pf_fast.hpp"
#include "estkit/particle/pf_regularized.hpp"
#include "estkit/particle/rbpf.hpp"
#include "estkit/particle/upf.hpp"

namespace estkit {

void register_particle(EstimatorList& list) {
    using BootPf  = BootstrapPf<EcmModel, 500>;
    using FastPf_  = FastPf<EcmModel, 100>;
    using RegPf   = RegularizedPf<EcmModel, 500>;
    using AuxPf   = AuxiliaryPf<EcmModel, 500>;
    using Rbpf    = MarginalizedPf<EcmMixedModel, 200>;
    using Upf     = UnscentedPf<EcmModel, 30>;
    using EnKfT   = EnKf<EcmModel, 50>;
    using EtkfT   = Etkf<EcmModel, 24>;
    using GsfT    = GaussianSumFilter<EcmModel, 3>;

    add_cell_estimator<BootPf>(list, "PF-Bootstrap", "particle",
        [](BootPf& f, const EstimatorConfig& c) { f.seed(c.seed); });
    add_cell_estimator<FastPf_>(list, "PF-Fast", "particle",
        [](FastPf_& f, const EstimatorConfig& c) { f.seed(c.seed); });
    add_cell_estimator<RegPf>(list, "PF-Regularized", "particle",
        [](RegPf& f, const EstimatorConfig& c) { f.seed(c.seed); });
    add_cell_estimator<AuxPf>(list, "PF-Auxiliary", "particle",
        [](AuxPf& f, const EstimatorConfig& c) { f.seed(c.seed); });
    add_cell_estimator<Rbpf, EcmMixedModel>(list, "RBPF", "particle",
        [](Rbpf& f, const EstimatorConfig& c) { f.seed(c.seed); });
    add_cell_estimator<Upf>(list, "UPF", "particle",
        [](Upf& f, const EstimatorConfig& c) { f.seed(c.seed); });
    add_cell_estimator<EnKfT>(list, "EnKF", "particle",
        [](EnKfT& f, const EstimatorConfig& c) { f.seed(c.seed); });
    add_cell_estimator<EtkfT>(list, "ETKF", "particle",
        [](EtkfT& f, const EstimatorConfig& c) { f.seed(c.seed); });
    add_cell_estimator<GsfT>(list, "GSF", "particle",
        [](GsfT& f, const EstimatorConfig& c) { f.seed(c.seed); f.opts.spread_index = EcmModel::IZ; });
}

}  // namespace estkit
