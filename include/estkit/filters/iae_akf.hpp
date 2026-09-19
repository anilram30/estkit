// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/iae_akf.hpp — Innovation-based Adaptive Estimation (IAE) of
//  the measurement-noise covariance inside an extended Kalman filter.
//
//  References
//    * Mehra, R. K. (1970). On the identification of variances and adaptive
//      Kalman filtering. IEEE Trans. Autom. Control 15(2), 175-184.
//    * Mohamed, A. H. & Schwarz, K. P. (1999). Adaptive Kalman filtering for
//      INS/GPS. Journal of Geodesy 73(4), 193-203.       (windowed IAE and RAE)
//    * Maybeck, P. S. (1979). Stochastic Models, Estimation, and Control,
//      Volume 1, Academic Press, New York, Ch. 10.
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, New York, Ch. 8.                    (EKF this builds on)
//
//  Idea.  For a consistent filter the innovation e_k = y_k - h(x^-_k, u_k) is
//  zero-mean white with covariance  S_k = H_k P^-_k H_k^T + R_k.  Estimating the
//  innovation covariance from the last N samples,
//
//      C_v,k = (1/N) sum_{j=k-N+1}^{k} e_j e_j^T                            (1)
//
//  and matching it to the theoretical value gives Mohamed & Schwarz's IAE
//
//      R_hat_k = C_v,k - H_k P^-_k H_k^T .                                  (2)
//
//  (2) is an unbiased but not necessarily positive-definite estimator, so the
//  diagonal is floored at a fraction of the nominal model R (and capped), which
//  is the standard safeguard.  The process-noise counterpart
//  Q_hat = C_dx + P^+ - F P^+ F^T with C_dx the windowed covariance of K e is
//  much noisier and is NOT enabled here (see the chapter).
//
//  The window is a fixed-size ring buffer of NWIN innovations with an
//  incrementally maintained sum of outer products: O(n_y^2) per step, no heap.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model, int NWIN = 80>
class IaeAkf {
public:
    static_assert(NWIN >= 2, "IAE window must contain at least two innovations");
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int WIN = NWIN;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        bool adapt_R = true;
        int  min_samples = NWIN;      // start adapting only once this many innovations are buffered
        Real r_min_scale = Real(0.05);   // R_hat_ii >= r_min_scale * R_ii(model)   (PD safeguard)
        Real r_max_scale = Real(5e3);    // R_hat_ii <= r_max_scale * R_ii(model)
        Real smooth = Real(0);           // optional 1st-order smoothing of R_hat in [0,1); 0 = none
        bool joseph = true;
        bool apply_constraints = true;
        Real q_scale = Real(1);
        Real r_scale = Real(1);
    } opts;

    explicit IaeAkf(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        x_ = x0; P_ = P0;
        head_ = 0; count_ = 0; sum_ = Mat<NY, NY>();
        for (int j = 0; j < NWIN; ++j) buf_[j] = Y();
        initialised_ = false; R_hat_ = Mat<NY, NY>();
        innovation_ = Y(); S_ = Mat<NY, NY>(); K_ = Mat<NX, NY>();
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
        if (!initialised_) { R_hat_ = R_model; initialised_ = true; }

        const Mat<NX, NX> Pminus = P_;
        const Y e = y - m_.h(x_, u);              // a-priori innovation

        // ---- ring buffer of innovations, incremental sum of e e^T ----------
        if (count_ == NWIN) sum_ -= outer(buf_[head_], buf_[head_]);
        buf_[head_] = e;
        sum_ += outer(e, e);
        head_ = (head_ + 1) % NWIN;
        if (count_ < NWIN) ++count_;

        // ---- IAE estimate of R  (eq. 1-2) ----------------------------------
        if (opts.adapt_R && count_ >= opts.min_samples) {
            const Mat<NY, NY> Cv = sum_ / Real(count_);
            Mat<NY, NY> Rn = Cv - H * Pminus * H.t();
            Rn.symmetrize();
            floor_diag(Rn, R_model, opts.r_min_scale, opts.r_max_scale);
            if (Rn.is_finite()) {
                const Real a = clampr(opts.smooth, Real(0), Real(0.999));
                R_hat_ = R_hat_ * a + Rn * (Real(1) - a);
            }
        }
        const Mat<NY, NY> R_use = opts.adapt_R ? R_hat_ : R_model;

        const Mat<NY, NY> S = H * Pminus * H.t() + R_use;
        const Mat<NX, NY> K = Pminus * H.t() * inverse(S);
        if (!K.is_finite()) return;               // keep the previous estimate
        innovation_ = e; S_ = S; K_ = K;
        x_ = x_ + K * e;
        if (opts.joseph) {
            const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
            P_ = IKH * Pminus * IKH.t() + K * R_use * K.t();
        } else {
            P_ = (Mat<NX, NX>::identity() - K * H) * Pminus;
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
    const Mat<NX, NY>& K() const { return K_; }
    const Mat<NY, NY>& R_hat() const { return R_hat_; }
    Real sigma_v_hat() const { return std::sqrt(std::fabs(R_hat_(0, 0))); }
    int window_fill() const { return count_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    template <int N>
    static void floor_diag(Mat<N, N>& A, const Mat<N, N>& ref, Real lo, Real hi) {
        for (int i = 0; i < N; ++i) {
            const Real r = ref(i, i) > Real(0) ? ref(i, i) : Real(1e-12);
            A(i, i) = clampr(A(i, i), lo * r, hi * r);
        }
        // discard off-diagonal entries that would break definiteness
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (i != j) {
                    const Real lim = Real(0.99) * std::sqrt(A(i, i) * A(j, j));
                    A(i, j) = clampr(A(i, j), -lim, lim);
                }
    }

    Model m_;
    X x_;
    Mat<NX, NX> P_;
    Y buf_[NWIN];
    Mat<NY, NY> sum_;
    Mat<NY, NY> R_hat_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
    int head_ = 0, count_ = 0;
    bool initialised_ = false;
};

}  // namespace estkit
