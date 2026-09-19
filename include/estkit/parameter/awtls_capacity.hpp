// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/parameter/awtls_capacity.hpp — recursive total-capacity (SOH) estimation
//  by approximate weighted total least squares (AWTLS), fed by SOC "anchor" pairs
//  produced by an inner EKF.  (Battery specific: implements estkit::Estimator.)
//
//  References
//    * Plett, G. L. (2011). Recursive approximate weighted total least squares
//      estimation of battery cell total capacity. J. Power Sources 196(4),
//      2319-2331.                              (WLS / TLS / WTLS / AWTLS)
//    * Plett, G. L. (2016). Battery Management Systems, Volume II:
//      Equivalent-Circuit Methods. Artech House, Ch. 5 (capacity estimation).
//    * Zhao, S., Duncan, S. R. & Howey, D. A. (2017). Observability analysis and
//      state estimation of lithium-ion batteries in the presence of sensor
//      biases. IEEE Trans. Control Syst. Technol. 25(1), 326-333.
//                                               (capacity/current-gain identifiability)
//    * Golub, G. H. & Van Loan, C. F. (2013). Matrix Computations, 4th ed.,
//      Johns Hopkins University Press, §6.3 (total least squares).
//
//  ---------------------------------------------------------------------------
//  Data model
//
//  Over the j-th "anchor interval" [t_j, t_j + T_anchor] the SOC drop reported by
//  an inner SOC filter and the integrated current are
//
//      x_j = zhat(t_j) - zhat(t_j + T),      y_j = (1/3600) int i dt   [Ah],
//
//  and the cell total capacity Q satisfies the errors-in-variables relation
//
//      y_j = Q x_j,     x_j = xbar_j + ex_j,   y_j = ybar_j + ey_j,          (1)
//
//  with var(ex_j) = sx_j^2 (from the filter covariance) and var(ey_j) = sy_j^2
//  (current-sensor noise, gain error and bias over the interval).  BOTH variables
//  are noisy, so ordinary least squares is biased: this is exactly Plett's
//  argument for a total-least-squares treatment.
//
//  Exact WTLS.  Minimising  sum_j [ (x_j-xh_j)^2/sx_j^2 + (y_j-Q xh_j)^2/sy_j^2 ]
//  over the latent xh_j gives xh_j = (x_j sy_j^2 + Q y_j sx_j^2)/(sy_j^2+Q^2 sx_j^2)
//  and, after substitution, the exact WTLS cost
//
//      J_WTLS(Q) = sum_j (y_j - Q x_j)^2 / ( sy_j^2 + Q^2 sx_j^2 ).          (2)
//
//  (2) cannot be written with a fixed number of accumulated sums because each
//  term carries its own Q-dependent denominator — hence Plett's APPROXIMATE
//  WTLS.
//
//  AWTLS.  Replace the exact (harmonic) combination of the two one-sided
//  residuals by their sum, i.e. project each datum onto the model both along y
//  (xh_j = x_j) and along x (yh_j = y_j) and weight each projection by its own
//  variance:
//
//      J(Q) = sum_j (y_j - Q x_j)^2 / sy_j^2  +  sum_j (x_j - y_j/Q)^2 / sx_j^2
//           = (a Q^2 - 2 b Q + c) + (d - 2 e/Q + f/Q^2),                     (3)
//
//      a = sum_j x_j^2/sy_j^2,  b = sum_j x_j y_j/sy_j^2,  c = sum_j y_j^2/sy_j^2,
//      d = sum_j x_j^2/sx_j^2,  e = sum_j x_j y_j/sx_j^2,  f = sum_j y_j^2/sx_j^2.
//
//  The six sums (plus a forgetting factor) are all the memory the estimator
//  needs, so the method is recursive and O(1) per anchor.  Setting dJ/dQ = 0 and
//  multiplying by Q^3/2 gives a QUARTIC in Q,
//
//      p(Q) = a Q^4 - b Q^3 + e Q - f = 0,                                   (4)
//
//  and the AWTLS estimate is the real positive root of (4) that minimises (3).
//  Properties: for noise-free data y_j = Q0 x_j, (4) factorises as
//  (Q-Q0)(Q^3 + Q0) sum_j x_j^2 = 0, so the estimator is consistent; as
//  sx -> infinity it degenerates to WLS (Q = b/a); as sy -> infinity it
//  degenerates to the inverse regression (Q = f/e).
//
//  A capacity PRIOR N(Q_nom, sigma_Q0^2) is a datum (x,y) = (1, Q_nom) with
//  sy = sigma_Q0 and sx = infinity, so it simply initialises
//  a = 1/sigma_Q0^2, b = Q_nom/sigma_Q0^2, c = Q_nom^2/sigma_Q0^2, d = e = f = 0,
//  which makes Q = b/a = Q_nom before any data arrive.
//
//  Root finding.  p(0) = -f <= 0 and p(Q) -> +infinity, so a positive root always
//  exists.  We scan p on a uniform grid over the physically admissible bracket
//  [Q_lo, Q_hi], bisect every sign change to machine precision and return the
//  root with the smallest J.  This is deterministic, derivative-free, allocation
//  free and costs a few hundred flops once per anchor (about six times per
//  flight) — a closed-form Ferrari solution would be faster but far less robust
//  in single precision.
//
//  The curvature of (3) at the optimum gives the capacity uncertainty,
//      var(Qhat) = 2 / J''(Qhat),  J'' = 2a - 4e/Q^3 + 6f/Q^4.
//
//  WLS and TLS (Plett's comparison methods) are available through Options:
//      WLS: Qhat = b/a  (only the y-errors are weighted);
//      TLS: B Q^2 + (C - A) Q - B = 0 with the UNWEIGHTED sums
//           A = sum y^2, B = sum x y, C = sum x^2 — note that TLS implicitly
//           assumes sx = sy = 1, which is dimensionally inconsistent here
//           (SOC vs Ah), which is precisely why Plett reports it to be inferior.
//  Only AWTLS is registered in the benchmark.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../battery/ecm_model.hpp"
#include "../filters/ekf.hpp"

namespace estkit {

class AwtlsCapacity final : public Estimator {
public:
    enum class Method { AWTLS, WLS, TLS };

    struct Options {
        Method method = Method::AWTLS;
        Real T_anchor = Real(300);      // [s] length of one anchor interval
        // [s] before the first anchor interval opens.  One anchor length of
        // settling: the SOC filter recovers from a 35 % initial error in under
        // 200 s on this mission, and an anchor taken across the convergence
        // transient reports a meaningless SOC difference.
        Real t_warmup = Real(300);
        Real min_dz = Real(0.02);       // reject anchors with too little SOC swing
        Real sigma_Q0 = Real(0.5);      // [Ah] prior std on the capacity
        // Fading memory applied to the six sums once per accepted anchor.  Total
        // capacity is a slowly TIME-VARYING quantity, so an estimator with
        // gamma = 1 converges to the average capacity over its whole history and
        // lags a fading cell without bound: measured over a 120-flight ageing
        // study (bench_soh --flights 120 --accel 3) gamma = 1 gives a capacity
        // RMSE of 374 mAh and a final error of +576 mAh, while gamma = 0.98 gives
        // 49 mAh and -35 mAh.  1/(1-gamma) = 50 anchors is about ten eVTOL
        // flights here; within a single flight (4-5 anchors) the fading is
        // negligible, so the single-mission results are unaffected.
        Real gamma = Real(0.98);           // forgetting factor applied per anchor
        // current-sensor error model used for sy_j
        Real i_gain_err = Real(0.010);  // relative gain uncertainty
        Real i_bias_std = Real(0.050);  // [A] bias uncertainty
        // SOC error model used for sx_j: the filter covariance is optimistic
        // (it ignores model error), so a floor is added in quadrature
        Real sigma_z_extra = Real(2e-3);
        Real Q_lo_rel = Real(0.40), Q_hi_rel = Real(1.40);   // search bracket / clamp
        // Reject an anchor whose implied capacity y_j/x_j is more than
        // outlier_sigma standard deviations away from the current estimate.  An
        // anchor taken while the SOC filter is still converging (or across a
        // sensor fault) reports a meaningless SOC difference; without this gate
        // a single bad anchor is fed back into the filter and the loop diverges.
        Real outlier_sigma = Real(3);
        int  grid = 64;                 // grid points for the sign scan of p(Q)
        bool feedback = true;           // write Qhat back into the inner EKF model
        Real feedback_relax = Real(1);  // 1 = apply immediately
        bool joseph = true;
    } opts;

    const char* name() const override { return "AWTLS"; }
    const char* group() const override { return "soh"; }
    size_t state_bytes() const override { return sizeof(*this); }

    void reset(const EstimatorConfig& c) override {
        m_ = EcmModel(c);
        ekf_ = Ekf<EcmModel>(m_);
        ekf_.opts.joseph = opts.joseph;
        ekf_.init(m_.x0(c), m_.P0(c));
        dt_ = c.dt; sigma_i_ = c.sigma_i;
        Q_nom_ = c.params.Q_Ah;
        Q_lo_ = opts.Q_lo_rel * Q_nom_; Q_hi_ = opts.Q_hi_rel * Q_nom_;
        // prior as a pseudo-datum (x,y) = (1, Q_nom) with sy = sigma_Q0, sx = inf
        const Real w0 = Real(1) / std::max(sq(opts.sigma_Q0), Real(1e-12));
        a_ = w0; b_ = Q_nom_ * w0; c_ = sq(Q_nom_) * w0;
        d_ = e_ = f_ = Real(0);
        uxx_ = uxy_ = uyy_ = Real(0);
        Qhat_ = Q_nom_; Qstd_ = opts.sigma_Q0;
        t_ = Real(0); ta_ = Real(0); ah_ = Real(0);
        anchor_open_ = false; n_anchor_ = 0;
        have_prev_ = false; ypred_ = kNaN;
        z_open_ = Real(0); p_open_ = Real(0);
    }

    void step(Real i, Real v, Real T) override {
        const Vec<2> u = EcmModel::u_of(i, T);
        if (have_prev_) ekf_.predict(u_prev_);
        ypred_ = ekf_.predict_measurement(u)[0];
        Vec<1> y; y[0] = v;
        ekf_.update(y, u);
        u_prev_ = u; have_prev_ = true;

        // ---- anchor bookkeeping -------------------------------------------------
        if (!anchor_open_ && t_ >= opts.t_warmup) open_anchor_();
        if (anchor_open_) {
            ah_ += i * dt_ / Real(3600);
            ta_ += dt_;
            if (ta_ >= opts.T_anchor) {
                close_anchor_();
                open_anchor_();
            }
        }
        t_ += dt_;
    }

    Real soc() const override { return ekf_.x()[EcmModel::IZ]; }
    Real soc_std() const override { const Real p = ekf_.P()(EcmModel::IZ, EcmModel::IZ); return p > Real(0) ? std::sqrt(p) : Real(-1); }
    Real voltage_pred() const override { return ypred_; }
    Real capacity_Ah() const override { return Qhat_; }

    Real capacity_std() const { return Qstd_; }
    int  anchors() const { return n_anchor_; }
    const Ekf<EcmModel>& filter() const { return ekf_; }

private:
    void open_anchor_() {
        z_open_ = ekf_.x()[EcmModel::IZ];
        p_open_ = std::max(ekf_.P()(EcmModel::IZ, EcmModel::IZ), Real(0));
        ah_ = Real(0); ta_ = Real(0); anchor_open_ = true;
    }

    void close_anchor_() {
        anchor_open_ = false;
        const Real x = z_open_ - ekf_.x()[EcmModel::IZ];      // SOC drop  [-]
        const Real y = ah_;                                   // charge out [Ah]
        if (!(x > opts.min_dz) || !(y > Real(0))) return;
        const Real pz = std::max(ekf_.P()(EcmModel::IZ, EcmModel::IZ), Real(0));
        const Real sx2 = p_open_ + pz + Real(2) * sq(opts.sigma_z_extra);
        // y-variance: white sensor noise + gain error + bias drift over the interval
        const Real sy2 = sq(sigma_i_) * dt_ * ta_ / Real(3600 * 3600)
                       + sq(opts.i_gain_err * y)
                       + sq(opts.i_bias_std * ta_ / Real(3600));
        // --- consistency gate: is this anchor compatible with what we know? ---
        if (opts.outlier_sigma > Real(0)) {
            const Real Qj = y / x;                       // capacity implied by this anchor alone
            const Real sQj2 = sq(Qj) * (sx2 / sq(x) + sy2 / sq(y));
            if (sq(Qj - Qhat_) > sq(opts.outlier_sigma) * (sQj2 + sq(Qstd_))) return;
        }
        add_point_(x, y, std::max(sx2, Real(1e-14)), std::max(sy2, Real(1e-14)));
        ++n_anchor_;
        solve_();
        if (opts.feedback) {
            EcmModel& mm = ekf_.model();
            mm.p.Q_Ah += opts.feedback_relax * (Qhat_ - mm.p.Q_Ah);
        }
    }

    void add_point_(Real x, Real y, Real sx2, Real sy2) {
        const Real g = opts.gamma;
        a_ *= g; b_ *= g; c_ *= g; d_ *= g; e_ *= g; f_ *= g;
        uxx_ *= g; uxy_ *= g; uyy_ *= g;
        const Real wy = Real(1) / sy2, wx = Real(1) / sx2;
        a_ += x * x * wy; b_ += x * y * wy; c_ += y * y * wy;
        d_ += x * x * wx; e_ += x * y * wx; f_ += y * y * wx;
        uxx_ += x * x;    uxy_ += x * y;    uyy_ += y * y;
    }

    Real cost_(Real Q) const {
        if (!(Q > Real(0))) return std::numeric_limits<Real>::max();
        const Real inv = Real(1) / Q;
        return (a_ * Q - Real(2) * b_) * Q + c_ + d_ + (f_ * inv - Real(2) * e_) * inv;
    }
    Real poly_(Real Q) const { return ((a_ * Q - b_) * Q) * Q * Q + e_ * Q - f_; }

    void solve_() {
        Real Q = Qhat_;
        switch (opts.method) {
            case Method::WLS:
                if (a_ > Real(0)) Q = b_ / a_;
                break;
            case Method::TLS: {
                const Real A = uyy_, B = uxy_, C = uxx_;
                if (std::fabs(B) > Real(1e-20)) {
                    const Real disc = sq(C - A) + Real(4) * B * B;
                    Q = ((A - C) + std::sqrt(std::max(disc, Real(0)))) / (Real(2) * B);
                }
                break;
            }
            case Method::AWTLS:
            default:
                Q = awtls_root_();
                break;
        }
        if (!std::isfinite(Q)) Q = Qhat_;
        Qhat_ = clampr(Q, Q_lo_, Q_hi_);
        const Real inv = Real(1) / Qhat_;
        const Real jpp = Real(2) * a_ - Real(4) * e_ * inv * inv * inv + Real(6) * f_ * inv * inv * inv * inv;
        Qstd_ = (jpp > Real(0)) ? std::sqrt(Real(2) / jpp) : opts.sigma_Q0;
    }

    // Grid scan + bisection for the minimising positive root of p(Q) = 0.
    Real awtls_root_() const {
        const int ng = (opts.grid > 3) ? opts.grid : 4;
        Real best = Qhat_, bestJ = cost_(Qhat_);
        Real qprev = Q_lo_, pprev = poly_(Q_lo_);
        const Real jlo = cost_(Q_lo_);
        if (jlo < bestJ) { bestJ = jlo; best = Q_lo_; }
        for (int g = 1; g <= ng; ++g) {
            const Real q = Q_lo_ + (Q_hi_ - Q_lo_) * Real(g) / Real(ng);
            const Real pv = poly_(q);
            if ((pprev < Real(0)) != (pv < Real(0))) {
                Real lo = qprev, hi = q;
                const bool loneg = (pprev < Real(0));
                for (int it = 0; it < 50; ++it) {
                    const Real mid = Real(0.5) * (lo + hi);
                    if ((poly_(mid) < Real(0)) == loneg) lo = mid; else hi = mid;
                }
                const Real root = Real(0.5) * (lo + hi);
                const Real jv = cost_(root);
                if (jv < bestJ) { bestJ = jv; best = root; }
            }
            qprev = q; pprev = pv;
        }
        const Real jhi = cost_(Q_hi_);
        if (jhi < bestJ) { bestJ = jhi; best = Q_hi_; }
        return best;
    }

    EcmModel m_;
    Ekf<EcmModel> ekf_{EcmModel()};
    Vec<2> u_prev_;
    bool have_prev_ = false;
    // recursive AWTLS sums (weighted) and unweighted sums used by the TLS variant
    Real a_ = 0, b_ = 0, c_ = 0, d_ = 0, e_ = 0, f_ = 0;
    Real uxx_ = 0, uxy_ = 0, uyy_ = 0;
    // anchor state
    Real t_ = 0, ta_ = 0, ah_ = 0, z_open_ = 0, p_open_ = 0;
    bool anchor_open_ = false;
    int  n_anchor_ = 0;
    // results and configuration
    Real Qhat_ = Real(5), Qstd_ = Real(0.5), Q_nom_ = Real(5), Q_lo_ = Real(2), Q_hi_ = Real(7);
    Real dt_ = Real(0.1), sigma_i_ = Real(0.05), ypred_ = kNaN;
};

}  // namespace estkit
