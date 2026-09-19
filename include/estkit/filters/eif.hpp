// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/eif.hpp — Extended information filter (EIF), the information
//  (inverse-covariance) form of the extended Kalman filter.
//
//  References
//    * Mutambara, A. G. O. (1998). Decentralized Estimation and Control for
//      Multisensor Systems. CRC Press, Boca Raton, Ch. 3-4.        (primary)
//    * Anderson, B. D. O. & Moore, J. B. (1979). Optimal Filtering. Prentice-Hall,
//      Englewood Cliffs, NJ, Section 6.3 ("information filter").
//    * Maybeck, P. S. (1979). Stochastic Models, Estimation, and Control, Vol. 1.
//      Academic Press, Section 5.7.
//    * Thrun, S., Burgard, W. & Fox, D. (2005). Probabilistic Robotics. MIT Press,
//      Ch. 3 (the "sparse extended information filter" view).
//
//  Parameterisation.  Instead of (xhat, P) the filter carries the information pair
//      Y = P^{-1}   (Fisher information matrix),      yv = Y xhat  (information vector).
//  The measurement update is then a simple ADDITION, which is the whole point:
//      Y^+  = Y^- + H^T R^{-1} H                                              (1)
//      yv^+ = yv^- + H^T R^{-1} [ y - h(x^-,u) + H x^- ]                      (2)
//  so N conditionally independent sensors contribute N additive terms that can be
//  computed in parallel and summed in any order (the basis of decentralised and
//  federated filters, Mutambara 1998), and a completely unknown initial state is
//  represented exactly by Y = 0 rather than by "P = large".
//
//  The price is the time update, which is additive in the covariance form.  Using
//  M = F^{-T} Y^+ F^{-1} and the Woodbury identity (Anderson & Moore 1979, §6.3)
//      Y^- = [ F (Y^+)^{-1} F^T + Q ]^{-1} = M - M (M + Q^{-1})^{-1} M        (3)
//  which needs F invertible (true for any discretised physical system) and Q
//  invertible.  The nonlinear mean is propagated as x^- = f(x^+, u) and
//  yv^- = Y^- x^-.  Equations (1)-(3) are algebraically identical to the EKF, so
//  the two filters produce the same estimate up to round-off.
//
//  xhat and P are recovered on demand (and cached) by one matrix inversion.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class Eif {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        bool information_time_update = true;   // (3); false -> invert the covariance form directly
        bool apply_constraints = true;         // project with Model::constrain (if defined)
        Real q_scale = Real(1);                // multiplicative tuning of Q
        Real r_scale = Real(1);                // multiplicative tuning of R
    } opts;

    explicit Eif(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        P_ = P0; x_ = x0;
        Y_ = inverse(P0);
        if (!Y_.is_finite()) Y_ = Mat<NX, NX>::identity() * Real(1e6);
        Y_.symmetrize();
        yv_ = Y_ * x_;
        dirty_ = false;
    }

    void predict(const U& u) {
        refresh_();
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        const Mat<NX, NX> Q = m_.Q(x_, u) * opts.q_scale;
        bool ok = false;
        if (opts.information_time_update) {
            const Mat<NX, NX> Finv = inverse(F);
            const Mat<NX, NX> Qinv = inverse(Q);
            if (Finv.is_finite() && Qinv.is_finite()) {
                Mat<NX, NX> M = Finv.t() * Y_ * Finv; M.symmetrize();
                Mat<NX, NX> Yn = M - M * solve_spd(M + Qinv, M);     // Y^- = M - M (M+Q^{-1})^{-1} M
                if (Yn.is_finite()) { Yn.symmetrize(); Y_ = Yn; ok = true; }
            }
        }
        if (!ok) {                                                    // covariance-form fallback
            Mat<NX, NX> Pm = F * P_ * F.t() + Q; Pm.symmetrize();
            Y_ = inverse(Pm);
            if (!Y_.is_finite()) Y_ = Mat<NX, NX>::identity() * Real(1e6);
            Y_.symmetrize();
        }
        yv_ = Y_ * x_;
        dirty_ = true;
    }

    Y predict_measurement(const U& u) const { refresh_(); return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        refresh_();
        const X xm = x_;                                   // x^- (linearisation point)
        const Mat<NY, NX> H = jac_H(m_, xm, u);
        const Mat<NY, NY> R = m_.R(xm, u) * opts.r_scale;
        const Mat<NY, NY> Rinv = inverse(R);
        if (!Rinv.is_finite()) return;
        innovation_ = y - m_.h(xm, u);
        const Mat<NX, NY> HtRi = H.t() * Rinv;
        S_ = H * P_ * H.t() + R;
        Y_ = Y_ + HtRi * H;                                // (1)
        yv_ = yv_ + HtRi * (innovation_ + H * xm);         // (2)
        Y_.symmetrize();
        dirty_ = true;
        refresh_();
        if (opts.apply_constraints) { x_ = constrain(m_, x_); yv_ = Y_ * x_; }
    }

    const X& x() const { refresh_(); return x_; }
    const Mat<NX, NX>& P() const { refresh_(); return P_; }
    const Mat<NX, NX>& information() const { return Y_; }        // Y = P^{-1}
    const X& information_vector() const { return yv_; }          // yv = Y x
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    void refresh_() const {                                       // (x,P) from (yv,Y), cached
        if (!dirty_) return;
        Mat<NX, NX> Pn = inverse(Y_);
        if (Pn.is_finite()) { Pn.symmetrize(); P_ = Pn; x_ = P_ * yv_; }
        dirty_ = false;
    }
    Model m_;
    Mat<NX, NX> Y_;            // information matrix
    X yv_;                     // information vector
    mutable Mat<NX, NX> P_;    // cached covariance
    mutable X x_;              // cached state estimate
    mutable bool dirty_ = true;
    Y innovation_;
    Mat<NY, NY> S_;
};

}  // namespace estkit
