// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/soekf.hpp — Second-order (Gaussian second-order) extended Kalman
//  filter.
//
//  References
//    * Athans, M., Wishner, R. P. & Bertolini, A. (1968). Suboptimal state
//      estimation for continuous-time nonlinear systems from discrete noisy
//      measurements. IEEE Transactions on Automatic Control 13(5), 504-514.
//      (primary; the "modified truncated second-order filter")
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory. Academic
//      Press, Ch. 9 (truncated and Gaussian second-order filters).
//    * Bass, R. W., Norum, V. D. & Schwartz, L. (1966). Optimal multichannel
//      nonlinear filtering. Journal of Mathematical Analysis and Applications
//      16(1), 152-164.                       (the second-order correction terms)
//    * Maybeck, P. S. (1982). Stochastic Models, Estimation, and Control, Vol. 2.
//      Academic Press, Section 12.3.
//    * Gelb, A. (ed., 1974). Applied Optimal Estimation. MIT Press, Section 6.1.
//
//  Idea.  Expanding f and h to SECOND order about the current estimate and taking
//  expectations under x ~ N(xhat, P) gives, for the i-th component,
//
//      E[f_i(x)] = f_i(xhat) + 1/2 tr( D^2 f_i(xhat) P ) + O(P^2)              (1)
//      Cov[f(x)]_{ij} = (F P F^T)_{ij} + 1/2 tr( D^2f_i P D^2f_j P ) + O(P^3)  (2)
//
//  where D^2 f_i is the n x n Hessian of the i-th component.  The EKF keeps only
//  the first term of (1) and of (2); the second-order filter adds the trace terms,
//  which removes the leading bias caused by curvature - exactly the error the EKF
//  makes on a curved OCV characteristic when the SOC uncertainty is large.  The
//  same correction is applied to the predicted measurement, and the extra term in
//  (2) inflates the innovation covariance so that the filter does not become
//  over-confident while the curvature bias is significant.
//
//  Algorithm (one cycle)
//    time update:   x^- = f(x^+,u) + 1/2 sum_i e_i tr(D^2f_i P^+)
//                   P^- = F P^+ F^T + Q + [1/2 tr(D^2f_i P^+ D^2f_j P^+)]_{ij}
//    measurement:   yhat = h(x^-,u) + 1/2 sum_i e_i tr(D^2h_i P^-)
//                   Rt   = R + [1/2 tr(D^2h_i P^- D^2h_j P^-)]_{ij}
//                   S = H P^- H^T + Rt,  K = P^- H^T S^{-1}
//                   x^+ = x^- + K (y - yhat)
//                   P^+ = (I-KH) P^- (I-KH)^T + K Rt K^T   (= P^- - K S K^T)
//
//  The Hessians are obtained by central differences (no analytic second derivatives
//  are required from the model).  Because the estimator-side OCV is stored as a
//  piecewise-linear table, the differencing step must span several table segments,
//  otherwise the curvature estimate alternates between 0 and a spike; see
//  Options::hess_step.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

namespace kalman_detail {
// Central-difference Hessians of all M components of g: R^N -> R^M at x.
//   d2 g_c / dx_a dx_b ~ [g(x+ha e_a+hb e_b) - g(x+ha e_a-hb e_b)
//                         - g(x-ha e_a+hb e_b) + g(x-ha e_a-hb e_b)] / (4 ha hb)
// with h_a = rel (1+|x_a|).  For a = b this degenerates to the usual
// [g(x+2h) - 2 g(x) + g(x-2h)] / (4h^2).  Cost: 2 N (N+1) evaluations of g.
template <int N, int M, class G>
inline void numeric_hessian(const G& g, const Vec<N>& x, Real rel, Mat<N, N> (&Hs)[M]) {
    for (int c = 0; c < M; ++c) Hs[c] = Mat<N, N>();
    for (int a = 0; a < N; ++a) {
        const Real ha = rel * (Real(1) + std::fabs(x[a]));
        for (int b = a; b < N; ++b) {
            const Real hb = rel * (Real(1) + std::fabs(x[b]));
            Vec<N> xpp = x, xpm = x, xmp = x, xmm = x;
            xpp[a] += ha; xpp[b] += hb;
            xpm[a] += ha; xpm[b] -= hb;
            xmp[a] -= ha; xmp[b] += hb;
            xmm[a] -= ha; xmm[b] -= hb;
            const Vec<M> gpp = g(xpp), gpm = g(xpm), gmp = g(xmp), gmm = g(xmm);
            const Real den = Real(4) * ha * hb;
            for (int c = 0; c < M; ++c) {
                const Real v = (gpp[c] - gpm[c] - gmp[c] + gmm[c]) / den;
                Hs[c](a, b) = v; Hs[c](b, a) = v;
            }
        }
    }
}
}  // namespace kalman_detail

template <class Model>
class SecondOrderEkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    struct Options {
        // Relative step of the central-difference Hessian.  The theoretical optimum
        // for a smooth function in double precision is eps^(1/4) ~ 1e-4, but the
        // estimator-side OCV is a 241-point linear table (spacing 0.005 in SOC), so
        // the stencil must be wider than one table segment to see the curvature.
        Real hess_step = Real(1e-2);
        bool hessian_f = true;             // include the second-order terms of f
        bool hessian_h = true;             // include the second-order terms of h
        bool second_order_cov = true;      // include the trace terms in (2)
        bool joseph = true;                // Joseph stabilised covariance update
        bool apply_constraints = true;     // project with Model::constrain (if defined)
        Real q_scale = Real(1);            // multiplicative tuning of Q
        Real r_scale = Real(1);            // multiplicative tuning of R
    } opts;

    explicit SecondOrderEkf(const Model& m) : m_(m) {}
    void init(const X& x0, const Mat<NX, NX>& P0) { x_ = x0; P_ = P0; }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        const Mat<NX, NX> Pp = P_;
        X xn = m_.f(x_, u);
        Mat<NX, NX> Pn = F * Pp * F.t();
        if (opts.hessian_f) {
            Mat<NX, NX> Hf[NX];
            const Model& mm = m_;
            kalman_detail::numeric_hessian<NX, NX>([&mm, &u](const X& xx) { return mm.f(xx, u); }, x_, opts.hess_step, Hf);
            for (int i = 0; i < NX; ++i) xn[i] += Real(0.5) * (Hf[i] * Pp).trace();
            if (opts.second_order_cov) {
                Mat<NX, NX> HP[NX];
                for (int i = 0; i < NX; ++i) HP[i] = Hf[i] * Pp;
                for (int i = 0; i < NX; ++i)
                    for (int j = 0; j < NX; ++j) Pn(i, j) += Real(0.5) * (HP[i] * HP[j]).trace();
            }
        }
        x_ = xn;
        P_ = Pn + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const {
        Y ym = m_.h(x_, u);
        if (opts.hessian_h) {
            ensure_h_hessian_(u);
            for (int i = 0; i < NY; ++i) ym[i] += Real(0.5) * (hh_[i] * P_).trace();
        }
        return ym;
    }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        Mat<NY, NY> Rt = m_.R(x_, u) * opts.r_scale;
        Y ym = m_.h(x_, u);
        if (opts.hessian_h) {
            ensure_h_hessian_(u);           // reuses the Hessian built by predict_measurement()
            for (int i = 0; i < NY; ++i) ym[i] += Real(0.5) * (hh_[i] * P_).trace();
            if (opts.second_order_cov) {
                Mat<NX, NX> HP[NY];
                for (int i = 0; i < NY; ++i) HP[i] = hh_[i] * P_;
                for (int i = 0; i < NY; ++i)
                    for (int j = 0; j < NY; ++j) Rt(i, j) += Real(0.5) * (HP[i] * HP[j]).trace();
            }
        }
        const Mat<NY, NY> S = H * P_ * H.t() + Rt;
        const Mat<NX, NY> K = P_ * H.t() * inverse(S);
        if (!K.is_finite()) return;                   // keep the prior rather than produce NaN
        innovation_ = y - ym; S_ = S;
        x_ = x_ + K * innovation_;
        if (opts.joseph) {
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_ = IKH * P_ * IKH.t() + K * Rt * K.t();
        } else {
            P_ = P_ - K * S * K.t();
        }
        P_.symmetrize();
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    // Hessians of h at (x_, u), cached so that predict_measurement() and update()
    // - which the benchmark adapter calls back to back with the same arguments -
    // share one set of 2n(n+1) model evaluations.
    void ensure_h_hessian_(const U& u) const {
        if (hh_valid_ && (hh_x_ - x_).norm_inf() == Real(0) && (hh_u_ - u).norm_inf() == Real(0)) return;
        const Model& mm = m_;
        kalman_detail::numeric_hessian<NX, NY>([&mm, &u](const X& xx) { return mm.h(xx, u); }, x_, opts.hess_step, hh_);
        hh_x_ = x_; hh_u_ = u; hh_valid_ = true;
    }

    Model m_;
    X x_;
    Mat<NX, NX> P_;
    mutable Mat<NX, NX> hh_[NY];
    mutable X hh_x_;
    mutable U hh_u_;
    mutable bool hh_valid_ = false;
    Y innovation_;
    Mat<NY, NY> S_;
};

}  // namespace estkit
