// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/parameter/rls_ekf.hpp — online recursive-least-squares identification
//  of the equivalent-circuit impedance parameters, feeding an EKF for the state.
//  (Battery specific: implements estkit::Estimator directly.)
//
//  References
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs — Part 2. Modeling and
//      identification. J. Power Sources 134(2), 262-276.   (ECM, ARX form,
//      least-squares identification of the cell impedance)
//    * Plett, G. L. (2004). ... Part 3. State and parameter estimation.
//      J. Power Sources 134(2), 277-292.       (parameter/state co-estimation)
//    * Ljung, L. (1999). System Identification: Theory for the User, 2nd ed.,
//      Prentice Hall.  Ch. 11 (recursive estimation, exponential forgetting).
//    * Plett, G. L. (2016). Battery Management Systems, Volume II:
//      Equivalent-Circuit Methods. Artech House.
//
//  ---------------------------------------------------------------------------
//  ARX form of the 1-RC equivalent circuit
//
//  Define the (hysteresis-compensated) OVERVOLTAGE actually measured at step k
//
//      s_k = v_k - OCV(zhat_k, T_k) - M * hhat_k                             (1)
//
//  where zhat_k, hhat_k are the SOC and hysteresis states of the EKF.  For a
//  1-RC circuit with series resistance R0, polarisation resistance R1 and pole
//  a1 = exp(-dt/tau1),
//
//      s_k = -R0 i_k - v1_k,   v1_{k+1} = a1 v1_k + R1 (1-a1) i_k.           (2)
//
//  Eliminating v1 with the shift operator gives the ARX(1) model
//
//      s_k = a1 s_{k-1} + b0 i_k + b1 i_{k-1} + d0,                          (3)
//      b0 = -R0,   b1 = a1 R0 - R1 (1 - a1),
//
//  with an optional constant d0 that absorbs a slowly varying OCV/SOC bias.
//  With the regressor and parameter vectors
//
//      phi_k = [ s_{k-1}, i_k, i_{k-1}, 1 ]^T,   psi = [ a1, b0, b1, d0 ]^T,
//
//  (3) reads s_k = phi_k^T psi and exponentially-weighted RLS (Ljung 1999,
//  Alg. 11.1) applies:
//
//      K_k   = P_{k-1} phi_k / (lambda + phi_k^T P_{k-1} phi_k)              (4)
//      psi_k = psi_{k-1} + K_k ( s_k - phi_k^T psi_{k-1} )                   (5)
//      P_k   = ( P_{k-1} - K_k phi_k^T P_{k-1} ) / lambda                    (6)
//
//  The impedance is recovered from psi by
//
//      R0   = -b0,   tau1 = -dt / ln(a1),   R1 = (a1 R0 - b1) / (1 - a1).    (7)
//
//  ---------------------------------------------------------------------------
//  Practical safeguards
//   * DECIMATION.  The regression is run on block averages over `rls_decim`
//     samples (1 Hz by default).  At the 10 Hz raw rate the sample-to-sample
//     change of the current is dominated by the current-sensor noise, which is
//     an errors-in-variables problem and attenuates the identified R0 by ~15 %;
//     averaging ten samples reduces the noise by sqrt(10) while the true current
//     varies by much more over one second, so the bias becomes negligible.
//     Averaging both s and i is exact for the static feed-through term -R0 i and
//     approximate for the RC branch.
//   * EXCITATION GATE: the RLS is frozen while |i| < i_min.
//   * COVARIANCE BOUND: trace(P) is capped at p_trace_ratio times its initial
//     value (a standard wind-up guard for forgetting-factor RLS).
//   * VALIDITY WINDOWS: an identified parameter is only accepted if it lies in a
//     plausible band around the value the BMS was commissioned with; accepted
//     values are applied through a first-order relaxation so the EKF model never
//     jumps.  Because the truth cell has two RC branches, the single identified
//     pole frequently locks onto the slow branch; the tau1 window then rejects
//     it and only R0 (the direct feed-through, which is what dominates the
//     voltage error) is refreshed.
//   * The identified resistances are valid AT THE PRESENT TEMPERATURE; they are
//     divided by the Arrhenius factor before being written into CellParams,
//     which stores reference-temperature values.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../battery/ecm_model.hpp"
#include "../filters/ekf.hpp"

namespace estkit {

class RlsEkf final : public Estimator {
public:
    static constexpr int NR = 4;      // [a1, b0, b1, d0]
    using Phi = Vec<NR>;

    struct Options {
        Real lambda = Real(0.9995);     // exponential forgetting factor (~2000 s at 1 Hz)
        int  rls_decim = 10;            // raw samples per regression sample (10 -> 1 Hz)
        int  refresh_every = 10;        // regression samples between model refreshes (-> 10 s)
        int  warmup = 60;               // regression samples before the first refresh (-> 60 s)
        Real relax = Real(0.20);        // first-order relaxation of an accepted update
        Real i_min = Real(0.10);        // [A] excitation gate
        bool include_offset = true;     // estimate the constant d0
        // Refresh the polarisation branch as well as R0.  OFF by default: the
        // identifier is a 1-RC model while the observer model has two RC
        // branches, so writing the identified (R1, tau1) into the first branch
        // while the second one is still active double-counts the polarisation
        // (measured: SOC RMSE rises from 0.0025 to 0.054 on `nominal`).  When
        // enabled, the observer model is COLLAPSED to the identified 1-RC
        // structure (R2 relaxed to zero) so that it stays self-consistent.
        bool identify_r1_tau1 = false;
        // acceptance windows, relative to the commissioned (config) values
        Real r0_lo = Real(0.30), r0_hi = Real(3.0);
        Real r1_lo = Real(0.25), r1_hi = Real(4.0);
        Real tau_lo = Real(0.2), tau_hi = Real(6.0);
        // priors (standard deviations) used to build the initial RLS covariance
        Real p0_a = Real(0.2);          // on the pole a1
        Real p0_r_rel = Real(0.5);      // on b0, b1, relative to the nominal R0
        Real p0_offset = Real(0.02);    // [V] on d0
        Real p_trace_ratio = Real(10);  // wind-up bound on trace(P)
        Real sigma_s = Real(-1);        // [V] residual std of (1); < 0 -> derive from cfg
        bool joseph = true;
    } opts;

    const char* name() const override { return "RLS-EKF"; }
    const char* group() const override { return "soh"; }
    size_t state_bytes() const override { return sizeof(*this); }

    void reset(const EstimatorConfig& c) override {
        m_ = EcmModel(c);
        ekf_ = Ekf<EcmModel>(m_);
        ekf_.opts.joseph = opts.joseph;
        ekf_.init(m_.x0(c), m_.P0(c));
        dt_rls_ = c.dt * Real(opts.rls_decim > 0 ? opts.rls_decim : 1);
        R0_ref0_ = c.params.R0; R1_ref0_ = c.params.R1; tau1_ref0_ = c.params.tau1;
        // --- initial regression vector from the commissioned parameters ---
        const Real a1 = std::exp(-dt_rls_ / std::max(tau1_ref0_, Real(1e-3)));
        psi_ = Phi();
        psi_[0] = a1;
        psi_[1] = -R0_ref0_;
        psi_[2] = a1 * R0_ref0_ - R1_ref0_ * (Real(1) - a1);
        psi_[3] = Real(0);
        // --- initial covariance: prior variance divided by the residual variance ---
        Real ss = opts.sigma_s;
        if (!(ss > Real(0))) ss = c.sigma_v / std::sqrt(Real(opts.rls_decim > 0 ? opts.rls_decim : 1));
        const Real r = std::max(sq(ss), Real(1e-12));
        Prls_ = Mat<NR, NR>();
        Prls_(0, 0) = sq(opts.p0_a) / r;
        Prls_(1, 1) = sq(opts.p0_r_rel * R0_ref0_) / r;
        Prls_(2, 2) = sq(opts.p0_r_rel * R0_ref0_) / r;
        Prls_(3, 3) = opts.include_offset ? sq(opts.p0_offset) / r : Real(0);
        trace_max_ = opts.p_trace_ratio * Prls_.trace();
        // --- bookkeeping ---
        have_prev_ = false; have_prev_rls_ = false;
        acc_s_ = acc_i_ = acc_T_ = Real(0); acc_n_ = 0;
        s_prev_ = i_prev_ = Real(0);
        rls_n_ = 0; ypred_ = kNaN;
        r0_hat_ = R0_ref0_; r1_hat_ = R1_ref0_; tau1_hat_ = tau1_ref0_;
    }

    void step(Real i, Real v, Real T) override {
        const Vec<2> u = EcmModel::u_of(i, T);
        if (have_prev_) ekf_.predict(u_prev_);
        ypred_ = ekf_.predict_measurement(u)[0];
        Vec<1> y; y[0] = v;
        ekf_.update(y, u);
        // --- measured overvoltage, Eq. (1), from the posterior state ---
        const EcmModel& mm = ekf_.model();
        const Vec<4>& x = ekf_.x();
        const Real s = v - mm.ocv.ocv(x[EcmModel::IZ], T) - mm.p.M * x[EcmModel::IH];
        // --- block average (anti-aliasing before decimation) ---
        acc_s_ += s; acc_i_ += i; acc_T_ += T; ++acc_n_;
        if (acc_n_ >= (opts.rls_decim > 0 ? opts.rls_decim : 1)) {
            const Real inv = Real(1) / Real(acc_n_);
            rls_sample_(acc_s_ * inv, acc_i_ * inv, acc_T_ * inv);
            acc_s_ = acc_i_ = acc_T_ = Real(0); acc_n_ = 0;
        }
        u_prev_ = u; have_prev_ = true;
    }

    Real soc() const override { return ekf_.x()[EcmModel::IZ]; }
    Real soc_std() const override { const Real p = ekf_.P()(EcmModel::IZ, EcmModel::IZ); return p > Real(0) ? std::sqrt(p) : Real(-1); }
    Real voltage_pred() const override { return ypred_; }

    // --- identified impedance (reference-temperature values, as stored in CellParams) ---
    Real r0_hat() const { return r0_hat_; }
    Real r1_hat() const { return r1_hat_; }
    Real tau1_hat() const { return tau1_hat_; }
    const Phi& regression_params() const { return psi_; }
    const Ekf<EcmModel>& filter() const { return ekf_; }

private:
    void rls_sample_(Real s, Real i, Real T) {
        if (!have_prev_rls_) { s_prev_ = s; i_prev_ = i; have_prev_rls_ = true; return; }
        if (std::fabs(i) > opts.i_min) {
            Phi phi;
            phi[0] = s_prev_; phi[1] = i; phi[2] = i_prev_;
            phi[3] = opts.include_offset ? Real(1) : Real(0);
            const Vec<NR> Pphi = Prls_ * phi;
            const Real denom = opts.lambda + dot(phi, Pphi);
            if (denom > Real(1e-30)) {
                const Vec<NR> K = Pphi / denom;
                const Real err = s - dot(phi, psi_);
                const Phi psi_new = psi_ + K * err;
                Mat<NR, NR> Pn = (Prls_ - outer(K, Pphi)) * (Real(1) / opts.lambda);
                Pn.symmetrize();
                if (psi_new.is_finite() && Pn.is_finite()) {
                    psi_ = psi_new; Prls_ = Pn;
                    const Real tr = Prls_.trace();
                    if (tr > trace_max_ && tr > Real(0)) Prls_ *= trace_max_ / tr;
                }
            }
        }
        s_prev_ = s; i_prev_ = i; ++rls_n_;
        if (rls_n_ >= opts.warmup && opts.refresh_every > 0 && (rls_n_ % opts.refresh_every) == 0) apply_(T);
    }

    void apply_(Real T) {
        EcmModel& mm = ekf_.model();
        const Real a1 = psi_[0], b0 = psi_[1], b1 = psi_[2];
        // --- series resistance, Eq. (7) ---
        const Real arr_R0 = mm.p.arr(mm.p.Ea_R0, T);
        const Real R0T = -b0;
        const Real R0nomT = R0_ref0_ * arr_R0;
        if (R0T > opts.r0_lo * R0nomT && R0T < opts.r0_hi * R0nomT && arr_R0 > Real(0)) {
            mm.p.R0 += opts.relax * (R0T / arr_R0 - mm.p.R0);
            r0_hat_ = mm.p.R0;
        }
        // --- fast RC branch (accepted only inside a plausible window) ---
        if (opts.identify_r1_tau1 && a1 > Real(0.01) && a1 < Real(0.9999)) {
            const Real tau1T = -dt_rls_ / std::log(a1);
            const Real R1T = (a1 * R0T - b1) / (Real(1) - a1);
            const Real arr_R1 = mm.p.arr(mm.p.Ea_R1, T), arr_tau = mm.p.arr(mm.p.Ea_tau, T);
            const Real R1nomT = R1_ref0_ * arr_R1, tau1nomT = tau1_ref0_ * arr_tau;
            if (tau1T > opts.tau_lo * tau1nomT && tau1T < opts.tau_hi * tau1nomT &&
                R1T > opts.r1_lo * R1nomT && R1T < opts.r1_hi * R1nomT) {
                mm.p.R1   += opts.relax * (R1T / arr_R1 - mm.p.R1);
                mm.p.tau1 += opts.relax * (tau1T / arr_tau - mm.p.tau1);
                mm.p.R2   += opts.relax * (Real(0) - mm.p.R2);   // collapse to 1-RC
                r1_hat_ = mm.p.R1; tau1_hat_ = mm.p.tau1;
            }
        }
    }

    EcmModel m_;
    Ekf<EcmModel> ekf_{EcmModel()};
    Vec<2> u_prev_;
    bool have_prev_ = false, have_prev_rls_ = false;
    Phi psi_;
    Mat<NR, NR> Prls_;
    Real dt_rls_ = Real(1), trace_max_ = Real(0);
    Real R0_ref0_ = Real(0.012), R1_ref0_ = Real(0.006), tau1_ref0_ = Real(5);
    Real r0_hat_ = Real(0.012), r1_hat_ = Real(0.006), tau1_hat_ = Real(5);
    Real acc_s_ = 0, acc_i_ = 0, acc_T_ = 0; int acc_n_ = 0;
    Real s_prev_ = 0, i_prev_ = 0;
    int rls_n_ = 0;
    Real ypred_ = kNaN;
};

}  // namespace estkit
