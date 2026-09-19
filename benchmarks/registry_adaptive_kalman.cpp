// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// Adaptive Kalman family — registered here (see include/estkit/filters/)
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/filters/fading_kf.hpp"
#include "estkit/filters/fuzzy_akf.hpp"
#include "estkit/filters/iae_akf.hpp"
#include "estkit/filters/imm.hpp"
#include "estkit/filters/mmae.hpp"
#include "estkit/filters/sage_husa_akf.hpp"
#include "estkit/filters/stf.hpp"
#include "estkit/filters/vb_akf.hpp"

namespace estkit {

void register_adaptive_kalman(EstimatorList& list) {
    const char* g = "adaptive_kalman";
    add_cell_estimator<SageHusaAkf<EcmModel>>(list, "SageHusa-AKF", g);
    add_cell_estimator<IaeAkf<EcmModel>>(list, "IAE-AKF", g);
    add_cell_estimator<VbAkf<EcmModel>>(list, "VB-AKF", g);
    add_cell_estimator<StrongTrackingFilter<EcmModel>>(list, "STF", g);
    add_cell_estimator<FadingMemoryKf<EcmModel>>(list, "Fading-EKF", g);
    add_cell_estimator<FuzzyAkf<EcmModel>>(list, "Fuzzy-AKF", g);
    add_cell_estimator<Imm<EcmModel>>(list, "IMM", g);
    add_cell_estimator<Mmae<EcmModel>>(list, "MMAE", g)
        .with_capacity([](const Mmae<EcmModel>& f) { return f.capacity(); });
}

}  // namespace estkit
