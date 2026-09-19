// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/smoothers/batch_nls.hpp — full-information (batch) nonlinear
//  least-squares state estimation by sparse Gauss-Newton.
//
//  References
//    * Rawlings, J. B., Mayne, D. Q. & Diehl, M. M. (2017). Model Predictive
//      Control: Theory, Computation, and Design, 2nd ed., Nob Hill Publishing,
//      Ch. 4 (full-information estimation and moving-horizon estimation).
//    * Nocedal, J. & Wright, S. J. (2006). Numerical Optimization, 2nd ed.,
//      Springer, Ch. 10 (Gauss-Newton for nonlinear least squares).
//    * Bell, B. M. (1994). The iterated Kalman smoother as a Gauss-Newton
//      method. SIAM J. Optimization 4(3), 626-636.
//    * Rao, C. V., Rawlings, J. B. & Mayne, D. Q. (2003). Constrained state
//      estimation for nonlinear discrete-time systems: stability and moving
//      horizon approximations. IEEE Trans. Autom. Control 48(2), 246-258.
//    * Golub, G. H. & Van Loan, C. F. (2013). Matrix Computations, 4th ed.,
//      §4.5 (block-tridiagonal LDL^T / Thomas algorithm).
//
//  Problem.  Over the whole record k = 0..N-1 minimise the maximum-a-posteriori
//  cost of the trajectory  X = (x_0, ..., x_{N-1}):
//
//   Phi(X) = ||x_0 - xhat_0||^2_{P_0^{-1}}
//          + sum_{k=0}^{N-2} ||x_{k+1} - f(x_k,u_k)||^2_{Q_k^{-1}}
//          + sum_{k=0}^{N-1} ||y_k - h(x_k,u_k)||^2_{R_k^{-1}}.              (1)
//
//  Unlike the EKF, (1) never discards information: the estimate of x_k uses the
//  measurements before AND after k, and the linearisation point is re-chosen at
//  every iteration, so the first-order bias of the EKF largely disappears.
//
//  Gauss-Newton.  With g = (1/2) dPhi/dX and the Gauss-Newton Hessian
//  A = (1/2) d^2Phi/dX^2 (second derivatives of f and h dropped), the step
//  solves A dX = -g.  Writing F_k = df/dx|_{x_k,u_k}, H_k = dh/dx|_{x_k,u_k},
//  W^q_k = Q_k^{-1}, W^r_k = R_k^{-1}, e_k = x_{k+1} - f(x_k,u_k),
//  r_k = y_k - h(x_k,u_k), the contributions are
//
//      A_{00}   += P_0^{-1},                       g_0    += P_0^{-1}(x_0-xhat_0)
//      A_{kk}   += F_k^T W^q_k F_k + H_k^T W^r_k H_k,
//      A_{k+1,k+1} += W^q_k,       A_{k,k+1} = -F_k^T W^q_k,
//      g_k      += -F_k^T W^q_k e_k - H_k^T W^r_k r_k,   g_{k+1} += W^q_k e_k.
//
//  A is therefore BLOCK TRIDIAGONAL with n_x x n_x blocks, and the step costs
//  O(N n_x^3) by a block LDL^T (Thomas) sweep -- the sparse Gauss-Newton of
//  Rawlings et al. (2017) Ch. 4, which is the batch counterpart of the Riccati
//  recursion:
//
//      Lambda_0 = A_{00};   M_k = Lambda_k^{-1} A_{k,k+1}
//      Lambda_{k+1} = A_{k+1,k+1} - A_{k,k+1}^T M_k              (forward)
//      z_0 = Lambda_0^{-1} b_0;  z_{k+1} = Lambda_{k+1}^{-1}(b_{k+1} - A_{k,k+1}^T z_k)
//      dX_{N-1} = z_{N-1};  dX_k = z_k - M_k dX_{k+1}            (backward)
//
//  with b = -g.  Each Lambda_k is a Schur complement of an SPD matrix and hence
//  SPD, so the sweep is performed with Cholesky solves.
//
//  Initialisation is the EKF forward pass (a good starting trajectory keeps the
//  number of Gauss-Newton iterations to three or four); the weights W^q, W^r are
//  frozen at the linearisation point so the backtracking line search minimises a
//  genuine quadratic model of (1).  The states are projected with
//  Model::constrain after every accepted step, which keeps the SOC inside the
//  range of the OCV table (a projected Gauss-Newton, cf. Rao et al. 2003).
//
//  The record is processed at the FULL 10 Hz rate: the whole solve costs about
//  20 flops per state per iteration, which measures out at a few tens of
//  microseconds per sample on this machine, so the 1 Hz sub-sampling that would
//  otherwise be required is unnecessary and no interpolation error is incurred.
//
//  OFFLINE method (estkit::BatchEstimator); std::vector is permitted (§1 of
//  docs/ESTIMATOR_API.md) because it never runs on the vehicle.
// =============================================================================
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "../core/linalg.hpp"
#include "../core/model.hpp"
#include "../battery/ecm_model.hpp"
#include "../filters/ekf.hpp"

namespace estkit {

template <class Model>
class BatchNls final : public BatchEstimator {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        int  iterations = 4;            // Gauss-Newton iterations
        int  max_backtrack = 4;         // halvings of the step before giving up
        bool apply_constraints = true;  // project with Model::constrain after each step
        Real q_scale = Real(1), r_scale = Real(1);
        Real rel_tol = Real(1e-10);     // stop when the cost stops decreasing
    } opts;

    BatchNls(std::string nm, std::string grp) : name_(std::move(nm)), group_(std::move(grp)) {}

    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }
    size_t state_bytes() const override {
        return sizeof(*this)
             + (x_.capacity() + xt_.capacity() + g_.capacity() + b_.capacity() + z_.capacity() + d_.capacity()) * sizeof(X)
             + (D_.capacity() + B_.capacity() + M_.capacity() + Dinv_.capacity() + Wq_.capacity()) * sizeof(Mat<NX, NX>)
             + Wr_.capacity() * sizeof(Mat<NY, NY>) + u_.capacity() * sizeof(U) + y_.capacity() * sizeof(Y);
    }
    void reset(const EstimatorConfig& c) override { cfg_ = c; m_ = Model(c); }

    void run(const Real* i, const Real* v, const Real* T, int n, Real* soc_out, Real* v_pred_out) override {
        if (n <= 0) return;
        const size_t N = static_cast<size_t>(n);
        x_.assign(N, X()); xt_.assign(N, X()); g_.assign(N, X()); b_.assign(N, X());
        z_.assign(N, X()); d_.assign(N, X()); u_.assign(N, U()); y_.assign(N, Y());
        D_.assign(N, Mat<NX, NX>()); Dinv_.assign(N, Mat<NX, NX>()); Wq_.assign(N, Mat<NX, NX>());
        B_.assign(N, Mat<NX, NX>()); M_.assign(N, Mat<NX, NX>());
        Wr_.assign(N, Mat<NY, NY>());

        x0hat_ = m_.x0(cfg_);
        W0_ = inverse(m_.P0(cfg_));

        // ---------------- initial trajectory: EKF forward pass ----------------
        {
            Ekf<Model> f(m_);
            f.opts.q_scale = opts.q_scale; f.opts.r_scale = opts.r_scale;
            f.opts.apply_constraints = opts.apply_constraints;
            f.init(x0hat_, m_.P0(cfg_));
            U u_prev;
            for (int k = 0; k < n; ++k) {
                const size_t sk = static_cast<size_t>(k);
                const U u = Model::u_of(i[k], T[k]);
                if (k > 0) f.predict(u_prev);
                Y yk; yk[0] = v[k];
                f.update(yk, u);
                x_[sk] = f.x(); u_[sk] = u; y_[sk] = yk;
                u_prev = u;
            }
        }

        // ---------------- Gauss-Newton iterations ----------------
        for (int it = 0; it < opts.iterations; ++it) {
            assemble_(n);
            const Real J0 = objective_(x_, n);
            if (!solve_tridiagonal_(n)) break;
            bool accepted = false;
            Real alpha = Real(1);
            for (int bt = 0; bt <= opts.max_backtrack; ++bt) {
                for (int k = 0; k < n; ++k) {
                    const size_t sk = static_cast<size_t>(k);
                    X xn = x_[sk] + d_[sk] * alpha;
                    if (opts.apply_constraints) xn = constrain(m_, xn);
                    xt_[sk] = xn;
                }
                const Real J1 = objective_(xt_, n);
                if (std::isfinite(J1) && J1 < J0) { x_.swap(xt_); accepted = true; if (J0 - J1 < opts.rel_tol * (Real(1) + std::fabs(J0))) it = opts.iterations; break; }
                alpha *= Real(0.5);
            }
            if (!accepted) break;
        }

        // ---------------- outputs ----------------
        for (int k = 0; k < n; ++k) {
            const size_t sk = static_cast<size_t>(k);
            soc_out[k] = x_[sk][Model::IZ];
            v_pred_out[k] = m_.h(x_[sk], u_[sk])[0];
        }
    }

    const std::vector<X>& trajectory() const { return x_; }

private:
    // Frozen-weight objective (1).  The weights are those of the current
    // linearisation point, so this is exactly the function the Gauss-Newton step
    // models quadratically.
    Real objective_(const std::vector<X>& xs, int n) const {
        Real J = quad_form(W0_, xs[0] - x0hat_);
        for (int k = 0; k < n - 1; ++k) {
            const size_t sk = static_cast<size_t>(k);
            const X e = xs[sk + 1] - m_.f(xs[sk], u_[sk]);
            J += quad_form(Wq_[sk], e);
        }
        for (int k = 0; k < n; ++k) {
            const size_t sk = static_cast<size_t>(k);
            const Y r = y_[sk] - m_.h(xs[sk], u_[sk]);
            J += quad_form(Wr_[sk], r);
        }
        return J;
    }

    void assemble_(int n) {
        for (int k = 0; k < n; ++k) {
            const size_t sk = static_cast<size_t>(k);
            D_[sk] = Mat<NX, NX>(); g_[sk] = X(); B_[sk] = Mat<NX, NX>();
            Wr_[sk] = inverse(m_.R(x_[sk], u_[sk]) * opts.r_scale);
            if (k < n - 1) Wq_[sk] = inverse(m_.Q(x_[sk + 1], u_[sk]) * opts.q_scale);
        }
        // prior
        D_[0] += W0_;
        g_[0] += W0_ * (x_[0] - x0hat_);
        // dynamics
        for (int k = 0; k < n - 1; ++k) {
            const size_t sk = static_cast<size_t>(k);
            const Mat<NX, NX> F = jac_F(m_, x_[sk], u_[sk]);
            const X e = x_[sk + 1] - m_.f(x_[sk], u_[sk]);
            const Mat<NX, NX> FtW = F.t() * Wq_[sk];
            D_[sk]     += FtW * F;
            D_[sk + 1] += Wq_[sk];
            B_[sk]      = -(FtW);
            g_[sk]     -= FtW * e;
            g_[sk + 1] += Wq_[sk] * e;
        }
        // measurements
        for (int k = 0; k < n; ++k) {
            const size_t sk = static_cast<size_t>(k);
            const Mat<NY, NX> H = jac_H(m_, x_[sk], u_[sk]);
            const Y r = y_[sk] - m_.h(x_[sk], u_[sk]);
            const Mat<NX, NY> HtW = H.t() * Wr_[sk];
            D_[sk] += HtW * H;
            g_[sk] -= HtW * r;
        }
        for (int k = 0; k < n; ++k) { const size_t sk = static_cast<size_t>(k); D_[sk].symmetrize(); b_[sk] = -g_[sk]; }
    }

    // Block-tridiagonal LDL^T sweep; fills d_ with the Gauss-Newton step.
    bool solve_tridiagonal_(int n) {
        const Mat<NX, NX> I = Mat<NX, NX>::identity();
        Dinv_[0] = solve_spd(D_[0], I);
        if (!Dinv_[0].is_finite()) return false;
        z_[0] = Dinv_[0] * b_[0];
        for (int k = 0; k < n - 1; ++k) {
            const size_t sk = static_cast<size_t>(k);
            M_[sk] = Dinv_[sk] * B_[sk];
            const Mat<NX, NX> Lam = D_[sk + 1] - B_[sk].t() * M_[sk];
            Dinv_[sk + 1] = solve_spd(Lam, I);
            if (!Dinv_[sk + 1].is_finite()) return false;
            z_[sk + 1] = Dinv_[sk + 1] * (b_[sk + 1] - B_[sk].t() * z_[sk]);
        }
        const size_t last = static_cast<size_t>(n - 1);
        d_[last] = z_[last];
        for (int k = n - 2; k >= 0; --k) {
            const size_t sk = static_cast<size_t>(k);
            d_[sk] = z_[sk] - M_[sk] * d_[sk + 1];
        }
        for (int k = 0; k < n; ++k) if (!d_[static_cast<size_t>(k)].is_finite()) return false;
        return true;
    }

    std::string name_, group_;
    EstimatorConfig cfg_;
    Model m_;
    X x0hat_;
    Mat<NX, NX> W0_;
    std::vector<X> x_, xt_, g_, b_, z_, d_;
    std::vector<U> u_;
    std::vector<Y> y_;
    std::vector<Mat<NX, NX>> D_, B_, M_, Dinv_, Wq_;
    std::vector<Mat<NY, NY>> Wr_;
};

}  // namespace estkit
