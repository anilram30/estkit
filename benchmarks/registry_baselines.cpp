// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/battery/baselines.hpp"
namespace estkit {
void register_baselines(EstimatorList& list) {
    list.push_back(std::make_unique<CoulombCounting>());
    list.push_back(std::make_unique<OcvLookup>());
    list.push_back(std::make_unique<CoulombOcvHybrid>());
}
}
