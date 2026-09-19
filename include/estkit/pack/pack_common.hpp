// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/pack/pack_common.hpp — building blocks shared by the pack-level
//  ("common state + per-cell delta") estimators of this family.
//
//  References
//    * Plett, G. L. (2009). Efficient battery pack state estimation using
//      bar-delta filtering. Proc. 24th International Battery, Hybrid and Fuel
//      Cell Electric Vehicle Symposium (EVS24), Stavanger, Norway.
//    * Plett, G. L. (2016). Battery Management Systems, Volume II:
//      Equivalent-Circuit Methods. Artech House, Ch. 4 (pack SOC estimation:
//      the "bar" filter on the average cell and the low-rate "delta" filters).
//    * Carlson, N. A. (1990). Federated square root filter for decentralized
//      parallel processes. IEEE Trans. Aerospace and Electronic Systems 26(3),
//      517-525.                                     (information sharing)
//    * Olfati-Saber, R. (2007). Distributed Kalman filtering for sensor
//      networks. Proc. 46th IEEE CDC, 5492-5498.    (Kalman-consensus filter)
//    * Mutambara, A. G. O. (1998). Decentralized Estimation and Control for
//      Multisensor Systems. CRC Press.              (information-form fusion)
//
//  ---------------------------------------------------------------------------
//  The common-state + delta decomposition
//  ---------------------------------------------------------------------------
//  NC cells in series share one current i_k.  Write the SOC of cell j as
//
//      z_j = zbar + dz_j ,      zbar = (1/NC) sum_j z_j ,   sum_j dz_j = 0 ,
//
//  and likewise for the RC-branch and hysteresis states.  Because every cell
//  sees the same current, the *common* part xbar = [zbar, v1bar, v2bar, hbar]
//  carries all of the fast dynamics, while the deviations dz_j are small
//  (manufacturing spread, thermal gradient) and vary slowly (they only change
//  through capacity/efficiency differences, i.e. at rate |i| dz/dQ).  The
//  deviation of the measured cell voltage from the pack average is, to first
//  order in the deviations,
//
//      v_j - vbar = OCV'(zbar,T) dz_j - drho_j (R0 i + v1 + v2) + M dh_j + noise ,   (*)
//
//  where drho_j is the RELATIVE deviation of cell j's impedance.  Impedance
//  spread is multiplicative in practice (all of R0, R1, R2 of a given cell
//  scale together with electrode area / porosity), so a single dimensionless
//  state drho_j multiplying the total overpotential w = R0 i + v1 + v2
//  represents it; drho_j R0 is the classical delta-resistance dR0_j in ohms
//  (set `include_rc_in_regressor = false` to recover exactly that model).
//
//  (*) is a scalar linear measurement of the 2-vector d_j = [dz_j, drho_j]^T,
//  which is what `DeltaBank` below filters — one 2x2 Kalman filter per cell,
//  optionally at a reduced rate (Plett runs the delta filters at ~1 Hz while
//  the common filter runs at the sampling rate).
// =============================================================================
#pragma once
#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "pack_dataset.hpp"
#include "pack_estimator.hpp"

namespace estkit {

// ---- compile-time detection of a `opts.q_scale` tuning knob on a filter ------
template <class F, class = void> struct has_q_scale : std::false_type {};
template <class F> struct has_q_scale<F, std::void_t<decltype(std::declval<F&>().opts.q_scale)>> : std::true_type {};

// ---- std::array of N copies of a prototype (filters have no default ctor) ----
namespace detail {
template <class T, std::size_t... I>
inline std::array<T, sizeof...(I)> fill_array_impl(const T& v, std::index_sequence<I...>) {
    return {{ (static_cast<void>(I), v)... }};
}
}  // namespace detail
template <class T, std::size_t N>
inline std::array<T, N> fill_array(const T& v) { return detail::fill_array_impl<T>(v, std::make_index_sequence<N>{}); }

// -----------------------------------------------------------------------------
//  ECM-family helpers used by every pack estimator.  They only require the model
//  to expose the members of `EcmModel` / `EcmModelReduced` (`ocv`, `p.M`, `R0`,
//  `IZ`, `IH`, `UT`), i.e. any equivalent-circuit cell model.
// -----------------------------------------------------------------------------
//  Total overpotential  w = R0(T) i + v1 + v2 = OCV(z,T) + M h - v.
//  Obtained from h() so that it is independent of how many RC branches the model
//  keeps as states (works for the reduced-order model too).
template <class M>
inline Real ecm_overpotential(const M& m, const Vec<M::NX>& x, const Vec<M::NU>& u) {
    return m.ocv.ocv(x[M::IZ], u[M::UT]) + m.p.M * x[M::IH] - m.h(x, u)[0];
}
template <class M>
inline Real ecm_docv(const M& m, const Vec<M::NX>& x, Real T) { return m.ocv.docv_dz(x[M::IZ], T); }

// Reference signal the delta filters difference the cell voltage against.
enum class DeltaReference {
    MeanVoltage,       // v_j - mean_k(v_k)          (Plett's bar-delta; needs all voltages at one node)
    ModelPrediction    // v_j - h(xhat_common, u_j)  (purely local: node j only needs the common state)
};

// -----------------------------------------------------------------------------
//  DeltaBank — NC independent low-order Kalman filters, one per cell (Plett
//  2009, 2016 Vol. II Ch. 4: the delta-SOC, delta-resistance and delta-capacity
//  filters of the bar-delta scheme, merged into one small joint filter).
//
//  state      d_j = [ dz_j , drho_j , dq_j ]^T
//               dz_j    SOC deviation of cell j from the common state        [-]
//               drho_j  RELATIVE impedance deviation (drho_j R0 = dR0_j)     [-]
//               dq_j    inverse-capacity deviation, dq_j = (1/Q_j - 1/Qbar) Qnom  [-]
//                       (dq_j > 0 <=> cell j is weaker than the average)
//
//  dynamics   the delta-SOC dynamics are EXACT, not a random walk:
//               z_j(k+1)  = z_j(k)  - eta i dtu / (3600 Q_j)
//               zbar(k+1) = zbar(k) - eta i dtu / (3600 Qbar)
//             subtracting,
//               dz_j(k+1) = dz_j(k) + a_k dq_j ,   a_k = -eta i dtu/(3600 Qnom)   (D1)
//             so the deviations are driven by the shared current through the
//             unknown capacity deviation.  dq_j and drho_j are constants of the
//             cell (ageing time scale), modelled as slow random walks:
//               A = [[1,0,a_k],[0,1,0],[0,0,1]] ,  Qd = diag(qz, qr, qc) .        (D2)
//             With `estimate_dcap = false` the third state is switched off and
//             (D1) degenerates to the classical 2-state random-walk model
//               Qd(1,1) = (kappa_Q |i| dtu/(3600 Qnom))^2 + (s_dz dtu)^2 ,
//             i.e. the capacity deviation is replaced by its prior standard
//             deviation kappa_Q.  That form cannot follow a diverging weak cell
//             without excessive noise gain (see the BarDelta chapter).
//
//  measurement  y_j = v_j - ref_j
//                   = OCV(zbar + dz_j, T) - OCV(zbar, T) - drho_j w + e_j         (D3)
//               w is the impedance regressor: R0 i by default, so that drho_j R0
//               is Plett's delta-resistance dR0_j, or the TOTAL overpotential
//               R0 i + v1 + v2 when `include_rc_in_regressor = true` (impedance
//               spread is multiplicative when it comes from area/porosity spread,
//               but an aged cell scales its branches unequally, which is why the
//               simpler R0 i regressor is the default -- see the BarDelta
//               chapter).  The measurement noise is
//               var(e_j) = sigma_v^2 (1 - 1/NC)  [MeanVoltage: the pack average
//                                                 shares 1/NC of the noise]
//                        + (eps_R wtot)^2        [unmodelled impedance shape,
//                                                 scaled by the TOTAL overpotential]
//                        + sigma_h^2             [hysteresis spread]
//               The OCV difference in (D3) is kept EXACT (an EKF on the delta
//               state, H = [OCV'(zbar+dz_j,T), -w, 0]) instead of the first-order
//               form OCV'(zbar) dz_j of the original bar-delta derivation.  Near
//               the ends of the SOC window OCV is strongly curved, |OCV''| ~ 20
//               V, and with a weak cell 0.1 SOC below the average the tangent
//               model under-predicts the voltage deviation, which drives dz_j
//               away instead of correcting it (see the BarDelta chapter);
//               `nonlinear_ocv = false` restores the linear form.
//
//  rate         updated every `rate_div` samples (dtu = rate_div dt); Plett runs
//               the delta filters at ~1 Hz against a 10 Hz common filter.
//  constraint   sum_j dz_j = 0 and sum_j dq_j = 0 hold by the definition of the
//               decomposition and are restored by subtracting the ensemble mean
//               (orthogonal projection onto the constraint set).  This is a
//               pack-wide operation, so a purely local architecture (the
//               Kalman-consensus filter) switches it off.
//
//  Cost per sampling period: NC 3x3 scalar-measurement updates every `rate_div`
//  samples (~NC/rate_div x 90 flops), two orders of magnitude below one
//  4-state EKF per cell.  No heap: fixed arrays of size kPackCells.
// -----------------------------------------------------------------------------
class DeltaBank {
public:
    static constexpr int ND = 3;                 // delta states per cell
    static constexpr int IDZ = 0, IDR = 1, IDQ = 2;

    struct Options {
        int rate_div = 10;                       // delta update every rate_div samples (1 Hz at dt = 0.1 s)
        bool estimate_drho = true;               // impedance-deviation state
        bool estimate_dcap = true;               // capacity-deviation state (exact dz dynamics, eq. D1)
        bool include_rc_in_regressor = false;    // regressor R0 i (Plett's dR0) or R0 i + v1 + v2 (multiplicative spread)
        bool zero_mean = true;                   // project onto sum_j dz_j = 0 (a pack-wide operation)
        bool nonlinear_ocv = true;               // exact OCV difference in (D3) instead of the tangent form
        DeltaReference reference = DeltaReference::MeanVoltage;
        Real sigma_dz0 = Real(0.05);             // prior std of the initial SOC imbalance [-]
        Real sigma_rho0 = Real(0.15);            // prior std of the relative impedance deviation [-]
        Real sigma_dq0 = Real(0.12);             // prior std of the inverse-capacity deviation [-]
        Real kappa_Q = Real(0.10);               // prior capacity spread used when estimate_dcap = false
        Real sigma_dz_walk = Real(2e-5);         // SOC drift floor [1/sqrt(s)]
        Real sigma_rho_walk = Real(2e-4);        // impedance drift [1/sqrt(s)]
        Real sigma_dq_walk = Real(1e-5);         // capacity drift [1/sqrt(s)]
        Real eps_R = Real(0.05);                 // unmodelled fraction of the overpotential deviation [-]
        Real sigma_hyst = Real(0.0015);          // cell-to-cell hysteresis spread [V]
        Real dz_limit = Real(0.5);               // saturation of dz (physical bound)
        Real Q_Ah = Real(5);                     // nominal capacity Qnom [Ah]
        Real eta_c = Real(0.995);                // coulombic efficiency on charge
    } opts;

    void reset(Real dt) {
        dt_ = dt; count_ = 0;
        for (int c = 0; c < kPackCells; ++c) {
            d_[c] = Vec<ND>();
            P_[c] = Mat<ND, ND>();
            P_[c](IDZ, IDZ) = sq(opts.sigma_dz0);
            P_[c](IDR, IDR) = opts.estimate_drho ? sq(opts.sigma_rho0) : Real(0);
            P_[c](IDQ, IDQ) = opts.estimate_dcap ? sq(opts.sigma_dq0) : Real(0);
        }
    }

    // One sampling period.
    //   m        the cell model (only `ocv` is used)
    //   zc[c]    common-state SOC that cell c's estimate is referred to
    //   Tbar     temperature at which the OCV is evaluated [K]
    //   v[c]     measured cell voltage, ref[c] its reference (pack mean voltage
    //            or the common-state prediction for cell c)
    //   w        impedance regressor of the average cell [V] (R0 i, or R0 i+v1+v2)
    //   wtot     total overpotential of the average cell [V] (noise model only)
    // Returns true if a delta update was performed.
    template <class M>
    bool step(const M& m, const Real* zc, Real Tbar, Real i, const Real* v, const Real* ref,
              Real w, Real wtot, Real sigma_v) {
        if (++count_ < opts.rate_div) return false;
        const Real dtu = Real(count_) * dt_;
        count_ = 0;

        // --- time update, eqs. (D1)-(D2) ---
        const Real eta = (i >= Real(0)) ? Real(1) : opts.eta_c;
        const Real a = -eta * i * dtu / (Real(3600) * opts.Q_Ah);
        Mat<ND, ND> A = Mat<ND, ND>::identity();
        if (opts.estimate_dcap) A(IDZ, IDQ) = a;
        Mat<ND, ND> Qd;
        Qd(IDZ, IDZ) = sq(opts.sigma_dz_walk * dtu)
                     + (opts.estimate_dcap ? Real(0) : sq(opts.kappa_Q * a));
        Qd(IDR, IDR) = opts.estimate_drho ? sq(opts.sigma_rho_walk * dtu) : Real(0);
        Qd(IDQ, IDQ) = opts.estimate_dcap ? sq(opts.sigma_dq_walk * dtu) : Real(0);

        const Real share = (opts.reference == DeltaReference::MeanVoltage)
                             ? (Real(1) - Real(1) / Real(kPackCells)) : Real(1);
        const Real Rd = sq(sigma_v) * share + sq(opts.eps_R * wtot) + sq(opts.sigma_hyst);
        const Real wr = opts.estimate_drho ? w : Real(0);

        for (int c = 0; c < kPackCells; ++c) {
            Vec<ND> dp = A * d_[c];
            Mat<ND, ND> P = A * P_[c] * A.t() + Qd;
            P.symmetrize();
            // --- linearise (D3) about the predicted deviation ---
            const Real zb = clampr(zc[c], Real(-0.05), Real(1.05));
            const Real zj = clampr(zb + dp[IDZ], Real(-0.05), Real(1.05));
            Mat<1, ND> H;
            Real pred;
            if (opts.nonlinear_ocv) {
                H(0, IDZ) = m.ocv.docv_dz(zj, Tbar);
                pred = m.ocv.ocv(zj, Tbar) - m.ocv.ocv(zb, Tbar);
            } else {
                H(0, IDZ) = m.ocv.docv_dz(zb, Tbar);
                pred = H(0, IDZ) * dp[IDZ];
            }
            H(0, IDR) = -wr;
            pred -= wr * dp[IDR];
            const Real s = (H * P * H.t())(0, 0) + Rd;
            if (!(s > Real(0)) || !std::isfinite(s)) { d_[c] = dp; P_[c] = P; continue; }
            const Vec<ND> K = (P * H.t()) * (Real(1) / s);
            const Real innov = (v[c] - ref[c]) - pred;
            Vec<ND> dn = dp + K * innov;
            dn[IDZ] = clampr(dn[IDZ], -opts.dz_limit, opts.dz_limit);
            dn[IDR] = opts.estimate_drho ? clampr(dn[IDR], Real(-0.9), Real(3)) : Real(0);
            dn[IDQ] = opts.estimate_dcap ? clampr(dn[IDQ], Real(-0.6), Real(0.6)) : Real(0);
            if (!dn.is_finite()) dn = dp;
            const Mat<ND, ND> IKH = Mat<ND, ND>::identity() - K * H;
            Mat<ND, ND> Pn = IKH * P * IKH.t() + (K * K.t()) * Rd;
            Pn.symmetrize();
            if (!Pn.is_finite()) Pn = P;
            d_[c] = dn; P_[c] = Pn;
        }
        if (opts.zero_mean) project_zero_mean();
        return true;
    }

    // Voltage deviation of cell c predicted by the delta state, eq. (D3):
    // OCV(zc + dz_c, T) - OCV(zc, T) - drho_c w.  Used by the common-state
    // filters to remove the known part of the deviation from their measurement.
    template <class M>
    Real deviation_voltage(const M& m, int c, Real zc, Real T, Real w) const {
        const Real zb = clampr(zc, Real(-0.05), Real(1.05));
        const Real zj = clampr(zb + d_[c][IDZ], Real(-0.05), Real(1.05));
        const Real dv = opts.nonlinear_ocv ? (m.ocv.ocv(zj, T) - m.ocv.ocv(zb, T))
                                           : m.ocv.docv_dz(zb, T) * d_[c][IDZ];
        return dv - (opts.estimate_drho ? d_[c][IDR] * w : Real(0));
    }

    Real dz(int c) const { return d_[c][IDZ]; }
    Real rho(int c) const { return d_[c][IDR]; }
    Real dq(int c) const { return d_[c][IDQ]; }
    Real dz_var(int c) const { return P_[c](IDZ, IDZ); }
    // classical delta-resistance in ohms, given the nominal series resistance
    Real dR0(int c, Real R0_nom) const { return d_[c][IDR] * R0_nom; }
    // capacity of cell c implied by the delta-capacity state [Ah]
    Real cell_capacity_Ah(int c) const {
        const Real inv = Real(1) / opts.Q_Ah * (Real(1) + d_[c][IDQ]);
        return inv > Real(0) ? Real(1) / inv : opts.Q_Ah;
    }

private:
    // Orthogonal projection of dz and dq onto the constraint sum_j (.) = 0.
    void project_zero_mean() {
        Real mz = Real(0), mq = Real(0);
        for (int c = 0; c < kPackCells; ++c) { mz += d_[c][IDZ]; mq += d_[c][IDQ]; }
        mz /= Real(kPackCells); mq /= Real(kPackCells);
        for (int c = 0; c < kPackCells; ++c) { d_[c][IDZ] -= mz; d_[c][IDQ] -= mq; }
    }

    Vec<ND> d_[kPackCells];
    Mat<ND, ND> P_[kPackCells];
    Real dt_ = Real(0.1);
    int count_ = 0;
};

// -----------------------------------------------------------------------------
//  DecentralizedPackM — same as `DecentralizedPack<Filter>` (pack_estimator.hpp)
//  but with an explicit model type, so that filters written against a model
//  other than `EcmModel` (e.g. the reduced-order `EcmModelReduced`) can be used.
//  `DecentralizedPack<F>` hard-codes `CellEstimator<F>` whose default model is
//  `EcmModel`, which does not compile for `Ekf<EcmModelReduced>`.
// -----------------------------------------------------------------------------
template <class Filter, class Model>
class DecentralizedPackM final : public PackEstimator {
public:
    explicit DecentralizedPackM(std::string name, std::string group = "pack")
        : name_(std::move(name)), group_(std::move(group)) {
        for (int c = 0; c < kPackCells; ++c) cells_.emplace_back(std::make_unique<CellEstimator<Filter, Model>>(name_, group_));
    }
    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }
    void reset(const EstimatorConfig& cfg) override { for (auto& c : cells_) c->reset(cfg); }
    void step(Real i, const Real* v, const Real* T) override {
        for (int c = 0; c < kPackCells; ++c) cells_[c]->step(i, v[c], T[c]);
    }
    Real cell_soc(int c) const override { return cells_[c]->soc(); }
    size_t state_bytes() const override { return kPackCells * sizeof(Filter); }

private:
    std::string name_, group_;
    std::vector<std::unique_ptr<CellEstimator<Filter, Model>>> cells_;
};

}  // namespace estkit
