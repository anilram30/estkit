// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  Deterministic-observer family — registered here (see include/estkit/observers/)
//
//  Luenberger / steady-state Kalman, proportional-integral, first- and
//  second-order sliding mode, high-gain, extended-state, unknown-input,
//  disturbance and adaptive observers, plus a cooperative interval observer.
//
//  All tuning below is expressed through each observer's public Options, so the
//  headers stay model-agnostic and every battery-specific choice is visible here.
// =============================================================================
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/observers/luenberger.hpp"
#include "estkit/observers/pi_observer.hpp"
#include "estkit/observers/smo.hpp"
#include "estkit/observers/super_twisting.hpp"
#include "estkit/observers/hgo.hpp"
#include "estkit/observers/eso.hpp"
#include "estkit/observers/uio.hpp"
#include "estkit/observers/dob.hpp"
#include "estkit/observers/adaptive_observer.hpp"
#include "estkit/observers/interval_observer.hpp"

namespace estkit {

namespace {
// z-plane pole for a closed-loop time constant tau at sample time dt
inline Real pole_tau(Real tau, Real dt) { return std::exp(-dt / tau); }
}  // namespace

void register_observers(EstimatorList& list) {
    using M = EcmModel;

    // --- 1. Luenberger observer, pole placement (Ackermann) -------------------
    //  Repeated pole set at tau = 60 s.  The 2-RC model sampled at 10 Hz has
    //  three eigenvalues within 2 % of each other, so a *spread* pole set costs
    //  gains of order 10^2-10^3 and amplifies the voltage noise by two orders of
    //  magnitude; a repeated set does not (see the chapter).
    //  max_gain is the "gain budget": near zero current the hysteresis eigenvalue
    //  a_h collides with the SOC integrator and exact pole assignment then asks
    //  for |L| of order 1-10 with sign-flipped RC entries, a design that is
    //  stable only for the frozen linearisation.  Rejecting it (and keeping the
    //  last validated gain) is what makes the scheduled observer robust.
    add_cell_estimator<LuenbergerObserver<M>>(list, "Luenberger", "observers",
        [](LuenbergerObserver<M>& f, const EstimatorConfig& c) {
            f.opts.design = obs::Design::PolePlacement;
            f.opts.set_poles_tau(Real(60), c.dt);
            f.opts.max_gain = Real(0.3);
            f.opts.q_extra[M::IZ] = Real(1e-4);      // used by the start-up LQE design
        });

    // --- 2. Steady-state Kalman / LQE gain (same class, Riccati design) -------
    //  The model's own Q has a very small SOC entry (q_floor = 1e-5 per step),
    //  which yields an SOC loop bandwidth of ~1/1300 s and therefore an
    //  impractically slow transient.  A practitioner adds a process-noise
    //  allowance for the unmodelled SOC-rate error (capacity fade, coulombic
    //  efficiency, current-sensor bias); 1e-4 per step corresponds to 1.4 % SOC
    //  of random walk over a 2000 s mission, which is the right order for a
    //  0.25 A bias on a 5 Ah cell.
    add_cell_estimator<LuenbergerObserver<M>>(list, "SSKF", "observers",
        [](LuenbergerObserver<M>& f, const EstimatorConfig& c) {
            (void)c;
            f.opts.design = obs::Design::Riccati;
            f.opts.q_extra[M::IZ] = Real(1e-4);
        });

    // --- 3. Proportional-integral observer ------------------------------------
    add_cell_estimator<PiObserver<M>>(list, "PI-Observer", "observers",
        [](PiObserver<M>& f, const EstimatorConfig& c) {
            f.opts.design = obs::Design::PolePlacement;
            for (int i = 0; i < M::NX; ++i) f.opts.pole[i] = pole_tau(Real(40), c.dt);
            f.opts.pole_i[0] = pole_tau(Real(100), c.dt);
            f.opts.max_gain = Real(0.3);
            //  Anti-windup on the integral state.  The bound is two-sided: too
            //  large and the integrator absorbs the cold-R0 output error into
            //  SOC (it blew up at 42 % before the gain certification was added);
            //  too small and it can no longer absorb the *real* capacity and
            //  current-bias errors, which costs 36 % on `cold` at d_max = 1e-6.
            //  6e-5 SOC/step (~10 A equivalent) is the measured optimum.
            f.opts.d_max = Real(6e-5);
        });

    // --- 4./5. First-order sliding-mode observer, fixed and adaptive gain -----
    add_cell_estimator<SlidingModeObserver<M>>(list, "SMO", "observers",
        [](SlidingModeObserver<M>& f, const EstimatorConfig& c) {
            f.opts.design = obs::Design::PolePlacement;
            for (int i = 0; i < M::NX; ++i) f.opts.pole[i] = pole_tau(Real(60), c.dt);
            f.opts.phi = Real(2.5) * c.sigma_v;     // boundary layer ~ 2.5 sigma_v
            f.opts.ks_scale = Real(2);
            f.opts.max_gain = Real(0.3);
            f.opts.ks_max = Real(3e-5);   // cap on the saturated injection (SOC per step)
            f.opts.q_extra[M::IZ] = Real(1e-4);
            f.opts.adaptive_gain = false;
        });
    add_cell_estimator<SlidingModeObserver<M>>(list, "SMO-Adaptive", "observers",
        [](SlidingModeObserver<M>& f, const EstimatorConfig& c) {
            f.opts.design = obs::Design::PolePlacement;
            for (int i = 0; i < M::NX; ++i) f.opts.pole[i] = pole_tau(Real(60), c.dt);
            f.opts.phi = Real(2.5) * c.sigma_v;
            f.opts.ks_scale = Real(2);
            f.opts.max_gain = Real(0.3);
            f.opts.ks_max = Real(3e-5);
            f.opts.q_extra[M::IZ] = Real(1e-4);
            f.opts.adaptive_gain = true;
            f.opts.adapt_rate = Real(2);
            f.opts.adapt_leak = Real(1e-3);
            f.opts.adapt_tau = Real(20) / c.dt;     // 20 s low-pass on |r|
            f.opts.kappa0 = Real(1);
            f.opts.kappa_min = Real(0.5);
            f.opts.kappa_max = Real(8);
        });

    // --- 6. Super-twisting (second-order sliding mode) observer ---------------
    add_cell_estimator<SuperTwistingObserver<M>>(list, "SuperTwisting-SMO", "observers",
        [](SuperTwistingObserver<M>& f, const EstimatorConfig& c) {
            f.opts.k1 = Real(6e-4);
            f.opts.k2 = Real(2e-8);
            f.opts.w_max = Real(2e-3);
            f.opts.phi = Real(c.sigma_v);
        });

    // --- 7. High-gain observer ------------------------------------------------
    add_cell_estimator<HighGainObserver<M>>(list, "HGO", "observers",
        [](HighGainObserver<M>& f, const EstimatorConfig& c) {
            f.opts.eps = Real(0.2);
            f.opts.mu = c.dt / Real(300);      // omega_0 = 1/300 s^-1 -> tau_cl = eps*300 s
            f.opts.corr_sigmas = Real(0.5);
            f.opts.max_gain = Real(0.3);
            f.opts.q_extra[M::IZ] = Real(1e-4);
        });

    // --- 8. Extended state observer (ADRC) ------------------------------------
    add_cell_estimator<ExtendedStateObserver<M>>(list, "ESO", "observers",
        [](ExtendedStateObserver<M>& f, const EstimatorConfig& c) {
            f.opts.pole_o = std::exp(-Real(0.02) * c.dt);   // omega_o = 0.02 rad/s
            f.opts.g_scale = c.dt;                          // d is an SOC rate [1/s]
            f.opts.max_gain = Real(0.3);
            f.opts.d_max = Real(6e-4);   // anti-windup on the extended state (~10 A equivalent)
            f.opts.q_d = Real(1e-5);        // start-up LQE design
        });

    // --- 9. Unknown-input observer -------------------------------------------
    //  d_max is the current sensor's specified bias range and doubles as the
    //  *admissibility* bound of the whole unknown-input hypothesis: the observer
    //  decouples the residual as an input bias only while the reconstructed bias
    //  dhat = r/(C E) stays inside it.  On the ECM plant dhat settles at
    //  0.25-0.4 A, the gate stays at 1 and the exact Chen-Patton-Zhang design is
    //  used; on the SPM plant the structured ECM-vs-SPM voltage error implies
    //  30-70 A, the gate closes and the observer degrades into an ordinary LQE
    //  observer instead of amplifying that error into a 10-58 % SOC error.
    //  h_max additionally caps the a-priori conditioning of the decoupling
    //  (||H||_inf = ||E||_inf/|C E| is 2.1 with the ECM parameters but 15-21 with
    //  the parameters identified against the single-particle plant).
    add_cell_estimator<UnknownInputObserver<M>>(list, "UIO", "observers",
        [](UnknownInputObserver<M>& f, const EstimatorConfig& c) {
            f.opts.u_index = M::UI;
            f.opts.design = obs::Design::Riccati;
            f.opts.q_extra[M::IZ] = Real(1e-4);
            f.opts.d_tau_steps = Real(60) / c.dt;
            f.opts.h_max = Real(4);
            f.opts.d_max = Real(5);                  // current-sensor bias bound [A]
            f.opts.d_mon_steps = Real(300) / c.dt;   // admissibility monitor horizon
        });

    // --- 10. Disturbance observer (augmented current-sensor bias) -------------
    add_cell_estimator<DisturbanceObserver<M>>(list, "DOB", "observers",
        [](DisturbanceObserver<M>& f, const EstimatorConfig& c) {
            (void)c;
            f.opts.u_index = M::UI;
            f.opts.design = obs::Design::Riccati;
            f.opts.q_extra[M::IZ] = Real(3e-4);
            f.opts.q_d = Real(1e-2);
            f.opts.d_max = Real(0.4);       // sensor-spec bound on the current bias [A]
            f.opts.compensate_output = true;
        });

    // --- 11. Adaptive observer (state + [Q_Ah, R0]) ---------------------------
    add_cell_estimator<AdaptiveObserver<M>>(list, "AdaptiveObserver", "observers",
        [](AdaptiveObserver<M>& f, const EstimatorConfig& c) {
            f.opts.design = obs::Design::PolePlacement;
            for (int i = 0; i < M::NX; ++i) f.opts.pole[i] = pole_tau(Real(60), c.dt);
            f.opts.max_gain = Real(0.3);
            f.opts.q_extra[M::IZ] = Real(1e-4);
            // Q_Ah is only weakly excited (it acts through the slow coulomb
            // integral) while R0 is strongly excited by every current step, so
            // the two adaptation gains differ by a factor of five.
            f.opts.gamma[0] = Real(0.1);   // Q_Ah (relative deviation)
            f.opts.gamma[1] = Real(0.5);   // R0   (relative deviation)
            f.opts.phi_min = Real(-0.45);
            f.opts.phi_max = Real(0.45);
        }).with_capacity([](const AdaptiveObserver<M>& f) { return f.params()[0]; });

    // --- 12. Interval observer -------------------------------------------------
    //  Uncertainty budget: the current channel must cover the sensor bias and
    //  gain error of the scenarios (0.25 A + 2 % of up to 22.5 A) plus 3 sigma of
    //  noise and the ADC quantisation; w_bar on the SOC channel covers the
    //  capacity error that the estimator is not told about (up to 15 %).
    add_cell_estimator<IntervalObserver<M>>(list, "IntervalObserver", "observers",
        [](IntervalObserver<M>& f, const EstimatorConfig& c) {
            f.opts.u_bar[M::UI] = Real(0.8) + Real(3) * c.sigma_i;
            f.opts.u_bar[M::UT] = Real(3) * c.sigma_T;
            f.opts.v_sigmas = Real(4);
            f.opts.v_extra[0] = Real(0.09);          // outliers (50 mV) + quantisation + R0 mismatch
            f.opts.w_bar[M::IZ] = Real(6e-6);        // capacity-error allowance per step
            f.opts.w_bar[M::IV1] = Real(1e-5);
            f.opts.w_bar[M::IV2] = Real(1e-5);
            f.opts.w_bar[M::IH] = Real(5e-5);
            f.opts.gain_frac = Real(0.5);
            f.opts.init_sigmas = Real(5);
            //  The reported point estimate is a certified LQE estimate projected
            //  onto the guaranteed interval.  The midpoint is a bad estimator
            //  here: its gain is limited by the worst OCV slope anywhere in the
            //  box, so it inherits the conservatism of the bounds (2.4 % SOC
            //  bias on the SPM plant, insensitive to gain_frac).
            f.opts.point_from_lqe = true;
            f.opts.point_q_extra[M::IZ] = Real(1e-4);
        }).with_bounds([](const IntervalObserver<M>& f) { return f.lower()[M::IZ]; },
                       [](const IntervalObserver<M>& f) { return f.upper()[M::IZ]; });
}

}  // namespace estkit
