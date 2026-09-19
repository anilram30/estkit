// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  Optimization (moving-horizon) family — registered here.
//    MHE-RTI   include/estkit/optimization/mhe.hpp   one Gauss-Newton iteration
//                                                    per sample (Diehl et al. 2005)
//    MHE       include/estkit/optimization/mhe.hpp   iterated to convergence
//                                                    (<= 5 GN iterations)
// =============================================================================
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/optimization/mhe.hpp"

namespace estkit {

using MheCell = Mhe<EcmModel, 20>;     // N = 20 samples = 2.0 s at dt = 0.1 s

// Box constraints of the cell model: z in [0,1], h in [-1,1]; the RC voltages are
// left free (they are physically bounded by the current but not by a known box).
static void configure_box(MheCell& f) {
    f.opts.lo[EcmModel::IZ] = Real(0);   f.opts.hi[EcmModel::IZ] = Real(1);
    f.opts.lo[EcmModel::IH] = Real(-1);  f.opts.hi[EcmModel::IH] = Real(1);
    f.opts.project = true;
}

void register_optimization(EstimatorList& list) {
    // Real-time iteration: exactly one Gauss-Newton step per sample, warm-started
    // by shifting the previous solution. Fixed, predictable cost per sample.
    add_cell_estimator<MheCell>(list, "MHE-RTI", "optimization",
        [](MheCell& f, const EstimatorConfig&) {
            configure_box(f);
            f.opts.max_iter = 1;
            f.opts.backtrack_max = 0;
        });
    // Converged MHE: up to 5 Gauss-Newton iterations, stopped early by a
    // relative step-norm tolerance.
    add_cell_estimator<MheCell>(list, "MHE", "optimization",
        [](MheCell& f, const EstimatorConfig&) {
            configure_box(f);
            f.opts.max_iter = 5;
            f.opts.tol = Real(1e-5);
            f.opts.backtrack_max = 0;   // pure Gauss-Newton; the cost-decrease
                                        // safeguard is available but not needed
                                        // on this problem (see the chapter)
        });
}

}  // namespace estkit
