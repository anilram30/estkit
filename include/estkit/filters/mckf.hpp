// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/mckf.hpp — Maximum-correntropy Kalman filter (extended)
//
//  References
//    * Chen, B., Liu, X., Zhao, H. & Principe, J. C. (2017). Maximum correntropy
//      Kalman filter. Automatica 76, 70-77.
//      (fixed-point form, Eqs. (20)-(28): augmented regression with the Cholesky
//       factor of blkdiag(P^-, R), Gaussian-kernel weights on prediction AND
//       measurement residuals, Joseph covariance with the nominal R)
//    * Liu, W., Pokharel, P. P. & Principe, J. C. (2007). Correntropy: properties
//      and applications in non-Gaussian signal processing. IEEE Trans. Signal
//      Processing 55(11), 5286-5298.                  (correntropy criterion)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, Ch. 8.                                (EKF linearisation)
//
//  Criterion --------------------------------------------------------------
//  Correntropy V(A,B) = E[G_sigma(A-B)] with the Gaussian kernel
//  G_sigma(e) = exp(-e^2 / (2 sigma^2)) is a *local* similarity measure: unlike
//  the mean-square error it saturates, so samples far from the mode contribute
//  almost nothing.  Writing the measurement update as the augmented linear
//  regression (Chen et al. 2017, Eq. (13))
//
//      D_k = W_k x_k + e_k,
//      D_k = B_k^{-1} [ x^- ; ytil ],   W_k = B_k^{-1} [ I ; H ],
//      B_k = blkdiag(B_p, B_r),  B_p B_p^T = P^-,  B_r B_r^T = R,
//      ytil = y - h(x^-,u) + H x^- ,
//
//  the maximum-correntropy criterion  max_x sum_i G_sigma(d_i - w_i^T x)  has the
//  stationary (fixed-point) condition  x = (sum_i c_i w_i w_i^T)^{-1} sum_i c_i w_i d_i
//  with c_i = G_sigma(d_i - w_i^T x).  Partitioning c into the n_x "prediction"
//  weights C_x and the n_y "measurement" weights C_y, Chen et al. show this is
//  equivalent to an ordinary Kalman update with REWEIGHTED covariances
//
//      Ptil^- = B_p C_x^{-1} B_p^T,        Rtil = B_r C_y^{-1} B_r^T,          (1)
//      Ktil   = Ptil^- H^T (H Ptil^- H^T + Rtil)^{-1},                          (2)
//      x^(t+1) = x^- + Ktil (y - h(x^-,u)),                                     (3)
//
//  iterated to convergence (a fixed-point iteration, contraction for sigma large
//  enough - Chen et al. 2017, Theorem 2).  The residuals that drive the weights are
//
//      e_x = B_p^{-1} (x^- - x^(t)),   e_y = B_r^{-1} (y - h(x^-,u) - H (x^(t)-x^-)),
//
//  i.e. they are already normalised by the prior/noise standard deviations, so
//  sigma is measured in "sigmas" (typical choice 2...5).  A measurement outlier
//  drives C_y -> 0 and hence Rtil -> infinity: the measurement is rejected
//  (a REDESCENDING influence function, in contrast to Huber's bounded one).
//  A floor c_min on the weights keeps the reweighting finite; it also guarantees
//  that a *persistently* large residual (e.g. a large initial state error, which
//  is not an outlier) is still able to correct the state, only more slowly.
//  The posterior covariance uses the Joseph form with the NOMINAL R
//  (Chen et al. 2017, Eq. (28)).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class MaxCorrentropyKf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real sigma    = Real(5);      // Gaussian kernel bandwidth, in normalised (sigma) units
        Real c_min    = Real(1e-2);   // floor on the kernel weights (numerical + convergence guard)
        int  max_iter = 10;           // fixed-point iterations
        Real tol      = Real(1e-6);   // relative convergence tolerance of the fixed point
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    explicit MaxCorrentropyKf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; iters_ = 0; }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        P_ = F * P_ * F.t() + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> R = m_.R(x_, u) * opts.r_scale;
        const X xprior = x_;
        innovation_ = y - m_.h(x_, u);

        const Mat<NX, NX> Bp = cholesky_safe(P_);          // P^- = Bp Bp^T
        const Mat<NY, NY> Br = cholesky_safe(R);           // R   = Br Br^T
        const Mat<NX, NX> Bpi = inverse_lower(Bp);
        const Mat<NY, NY> Bri = inverse_lower(Br);
        if (!Bpi.is_finite() || !Bri.is_finite()) return;

        X xit = xprior;
        Mat<NX, NY> K;
        Mat<NY, NY> S = H * P_ * H.t() + R;
        const Real inv2s2 = Real(1) / (Real(2) * sq(opts.sigma));
        int it = 0;
        for (; it < opts.max_iter; ++it) {
            const X dx = xit - xprior;
            const X ex = Bpi * (-dx);                       // standardised prediction residual
            const Y ey = Bri * (innovation_ - H * dx);      // standardised measurement residual
            Mat<NX, NX> Cxi;                                // C_x^{-1}
            for (int i = 0; i < NX; ++i) {
                Real c = safe_exp(-sq(ex[i]) * inv2s2);
                if (c < opts.c_min) c = opts.c_min;
                Cxi(i, i) = Real(1) / c;
            }
            Mat<NY, NY> Cyi;                                // C_y^{-1}
            for (int j = 0; j < NY; ++j) {
                Real c = safe_exp(-sq(ey[j]) * inv2s2);
                if (c < opts.c_min) c = opts.c_min;
                Cyi(j, j) = Real(1) / c;
                cy_[j] = c;
            }
            Mat<NX, NX> Pt = Bp * Cxi * Bp.t(); Pt.symmetrize();
            Mat<NY, NY> Rt = Br * Cyi * Br.t(); Rt.symmetrize();
            S = H * Pt * H.t() + Rt; S.symmetrize();
            const Mat<NY, NY> Sinv = inverse(S);
            if (!Sinv.is_finite()) return;
            K = Pt * H.t() * Sinv;
            const X xnew = xprior + K * innovation_;
            if (!xnew.is_finite()) return;
            const Real step = (xnew - xit).norm();
            xit = xnew;
            if (step <= opts.tol * (Real(1) + xit.norm())) { ++it; break; }
        }
        iters_ = it;
        x_ = xit; S_ = S; K_ = K;
        // Joseph form with the NOMINAL R (Chen et al. 2017, Eq. (28))
        const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
        P_ = IKH * P_ * IKH.t() + K * R * K.t();
        P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    Real kernel_weight(int j) const { return cy_[j]; }   // last measurement kernel weight
    int  iterations() const { return iters_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    Real cy_[NY] = {};
    int iters_ = 0;
};

}  // namespace estkit
