// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/vb_akf.hpp — Variational Bayesian adaptive Kalman filter with
//  inverse-Gamma priors on the (diagonal) measurement-noise variances.
//
//  References
//    * Sarkka, S. & Nummenmaa, A. (2009). Recursive noise adaptive Kalman
//      filtering by variational Bayesian approximations. IEEE Trans. Autom.
//      Control 54(3), 596-600.                                    (the method)
//    * Sarkka, S. (2013). Bayesian Filtering and Smoothing. Cambridge
//      University Press, Ch. 12.                     (VB filtering background)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, New York, Ch. 8.                  (EKF linearisation)
//
//  Model.  y_k = h(x_k,u_k) + v_k,  v_k ~ N(0, Sigma_k),
//          Sigma_k = diag(sigma_{k,1}^2, ..., sigma_{k,n_y}^2) unknown and slowly
//          time-varying.  The prior on each variance is inverse-Gamma,
//              p(sigma_{k,i}^2) = Inv-Gamma(alpha_{k,i}, beta_{k,i}),
//              E[1/sigma^2] = alpha/beta,  E[sigma^2] = beta/(alpha-1).
//
//  Variational approximation.  The joint filtering density is approximated by a
//  factorised one,  p(x_k, Sigma_k | y_{1:k}) ~= Q_x(x_k) Q_S(Sigma_k), and the
//  factors are found by minimising the Kullback-Leibler divergence
//      KL[ Q_x Q_S || p ] = - int Q_x Q_S log( p / (Q_x Q_S) ) dx dSigma,
//  i.e. by maximising the variational free energy.  The stationary conditions
//      log Q_x  = E_{Q_S}[ log p(y_k, x_k, Sigma_k | y_{1:k-1}) ] + const
//      log Q_S  = E_{Q_x}[ log p(y_k, x_k, Sigma_k | y_{1:k-1}) ] + const
//  are conjugate: Q_x stays Gaussian N(m_k, P_k) and Q_S stays inverse-Gamma,
//  giving the coupled fixed-point iteration (N_vb sweeps)
//
//      alpha_{k,i} = rho alpha_{k-1,i} + 1/2                               (1)
//      Sigma^{(j)} = diag( beta_i^{(j)} / alpha_{k,i} )   ( = E_Q[Sigma^{-1}]^{-1} )
//      S^{(j)}     = H P^- H^T + Sigma^{(j)}                               (2)
//      K^{(j)}     = P^- H^T (S^{(j)})^{-1}                                (3)
//      m^{(j+1)}   = x^- + K^{(j)} ( y_k - h(x^-, u_k) )                   (4)
//      P^{(j+1)}   = (I - K H) P^- (I - K H)^T + K Sigma^{(j)} K^T         (5)
//      beta_i^{(j+1)} = rho beta_{k-1,i}
//                       + 1/2 [ ( y - h(x^-) - H (m^{(j+1)} - x^-) )_i^2
//                               + ( H P^{(j+1)} H^T )_{ii} ]               (6)
//
//  The heuristic dynamic model of Sarkka & Nummenmaa (Sec. III), alpha^- = rho
//  alpha, beta^- = rho beta with rho in (0,1], leaves the mean beta/alpha
//  unchanged while inflating the spread, so the filter can track a changing
//  noise level; the effective memory is 1/(1-rho) samples and the stationary
//  shape parameter is alpha* = 1/(2(1-rho)).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class VbAkf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        int  n_vb = 3;                // fixed-point sweeps per measurement
        Real rho = Real(0.995);       // forgetting: alpha,beta <- rho*(.)  (memory 1/(1-rho))
        Real alpha0 = Real(-1);       // initial shape; <0 -> stationary value 1/(2(1-rho))
        Real r_min_scale = Real(0.05);   // beta/alpha kept in [r_min_scale, r_max_scale] * R_ii(model)
        Real r_max_scale = Real(5e3);
        bool joseph = true;
        bool apply_constraints = true;
        Real q_scale = Real(1);
        Real r_scale = Real(1);
    } opts;

    explicit VbAkf(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0; initialised_ = false;
        for (int i = 0; i < NY; ++i) { alpha_[i] = Real(0); beta_[i] = Real(0); }
        innovation_ = Y(); S_ = Mat<NY, NY>(); K_ = Mat<NX, NY>(); R_hat_ = Mat<NY, NY>();
    }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        P_ = F * P_ * F.t() + m_.Q(x_, u) * opts.q_scale;
        P_.symmetrize();
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> R_model = m_.R(x_, u) * opts.r_scale;
        const Real rho = clampr(opts.rho, Real(1e-3), Real(1));
        if (!initialised_) {
            const Real a0 = (opts.alpha0 > Real(0)) ? opts.alpha0
                          : ((rho < Real(1)) ? Real(1) / (Real(2) * (Real(1) - rho)) : Real(1e4));
            for (int i = 0; i < NY; ++i) { alpha_[i] = a0; beta_[i] = a0 * pos(R_model(i, i)); }
            initialised_ = true;
        }

        const Mat<NX, NX> Pminus = P_;
        const X x_prior = x_;
        const Y resid0 = y - m_.h(x_prior, u);     // y - h(x^-, u_k), fixed over the sweeps

        // ---- (1) time update of the inverse-Gamma parameters ---------------
        Real a[NY], bpred[NY], b[NY];
        for (int i = 0; i < NY; ++i) {
            a[i] = rho * alpha_[i] + Real(0.5);
            bpred[i] = rho * beta_[i];
            b[i] = bpred[i];                       // beta^{(0)} = beta^-
        }

        // ---- (2)-(6) variational fixed-point iteration ---------------------
        X m_it = x_prior;
        Mat<NX, NX> P_it = Pminus;
        Mat<NY, NY> Sig, S, R_lim;
        Mat<NX, NY> K;
        const int nsweep = (opts.n_vb > 0) ? opts.n_vb : 1;
        for (int j = 0; j < nsweep; ++j) {
            Sig = Mat<NY, NY>();
            for (int i = 0; i < NY; ++i) {
                const Real rref = pos(R_model(i, i));
                Sig(i, i) = clampr(b[i] / a[i], opts.r_min_scale * rref, opts.r_max_scale * rref);
            }
            S = H * Pminus * H.t() + Sig;
            K = Pminus * H.t() * inverse(S);
            if (!K.is_finite()) return;            // keep the previous estimate
            m_it = x_prior + K * resid0;
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_it = opts.joseph ? (IKH * Pminus * IKH.t() + K * Sig * K.t()) : (IKH * Pminus);
            P_it.symmetrize();
            const Mat<NY, NY> HPH = H * P_it * H.t();
            const Y lin = resid0 - H * (m_it - x_prior);     // linearised a-posteriori residual
            for (int i = 0; i < NY; ++i)
                b[i] = bpred[i] + Real(0.5) * (lin[i] * lin[i] + pos(HPH(i, i)));
        }
        R_lim = Sig;

        // ---- commit -------------------------------------------------------
        for (int i = 0; i < NY; ++i) {
            const Real rref = pos(R_model(i, i));
            alpha_[i] = a[i];
            // keep beta consistent with the clamped variance so the recursion cannot run away
            const Real var = clampr(b[i] / a[i], opts.r_min_scale * rref, opts.r_max_scale * rref);
            beta_[i] = var * a[i];
        }
        x_ = m_it; P_ = P_it; P_.symmetrize();
        innovation_ = resid0; S_ = S; K_ = K; R_hat_ = R_lim;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }
    X& x_mut() { return x_; }
    Mat<NX, NX>& P_mut() { return P_; }
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    const Mat<NY, NY>& R_hat() const { return R_hat_; }
    Real sigma_v_hat() const { return std::sqrt(std::fabs(R_hat_(0, 0))); }
    Real alpha(int i) const { return alpha_[i]; }
    Real beta(int i) const { return beta_[i]; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    static Real pos(Real v) { return v > Real(0) ? v : Real(1e-12); }

    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Real alpha_[NY], beta_[NY];
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    Mat<NY, NY> R_hat_;
    bool initialised_ = false;
};

}  // namespace estkit
