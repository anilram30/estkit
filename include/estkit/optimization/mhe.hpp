// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/optimization/mhe.hpp — Moving-horizon estimation (MHE) with a
//  filtering-based arrival cost, box constraints, a block-tridiagonal
//  Gauss-Newton solver, and an optional real-time-iteration (RTI) mode.
//
//  References
//    * Rao, C. V., Rawlings, J. B. & Mayne, D. Q. (2003). Constrained state
//      estimation for nonlinear discrete-time systems: Stability and moving
//      horizon approximations. IEEE Transactions on Automatic Control 48(2),
//      246-258.                       (formulation, arrival cost, stability)
//    * Rawlings, J. B., Mayne, D. Q. & Diehl, M. M. (2017). Model Predictive
//      Control: Theory, Computation, and Design, 2nd ed. Nob Hill Publishing,
//      Ch. 4 ("State Estimation").      (full-information vs moving-horizon,
//                                        smoothing/filtering update, eq. 4.46)
//    * Robertson, D. G., Lee, J. H. & Rawlings, J. B. (1996). A moving
//      horizon-based approach for least-squares estimation. AIChE Journal 42(8),
//      2209-2224.                                (the least-squares MHE problem)
//    * Diehl, M., Bock, H. G. & Schloeder, J. P. (2005). A real-time iteration
//      scheme for nonlinear optimization in optimal feedback control. SIAM
//      Journal on Control and Optimization 43(5), 1714-1736.       (RTI scheme)
//    * Kuehl, P., Diehl, M., Kraus, T., Schloeder, J. P. & Bock, H. G. (2011).
//      A real-time algorithm for moving horizon state and parameter estimation.
//      Computers & Chemical Engineering 35(1), 71-83.       (RTI applied to MHE)
//    * Bertsekas, D. P. (1982). Projected Newton methods for optimization
//      problems with simple constraints. SIAM Journal on Control and
//      Optimization 20(2), 221-246.               (projection onto a box)
//    * Golub, G. H. & Van Loan, C. F. (2013). Matrix Computations, 4th ed.,
//      Johns Hopkins University Press, §4.3.        (block tridiagonal LDL^T)
//    * Nocedal, J. & Wright, S. J. (2006). Numerical Optimization, 2nd ed.,
//      Springer, §10.3.                          (Gauss-Newton, normal equations)
//
//  ---------------------------------------------------------------------------
//  Problem.  At time k, with the N+1 most recent samples (window nodes
//  j = 0..n, node j holding time k-n+j, n = min(k,N)), MHE solves
//
//    min   1/2 || x_0 - xbar ||^2_{Pi^{-1}}
//    x_0..x_n   + 1/2 sum_{j=0}^{n-1} || w_j ||^2_{Q_j^{-1}}
//                + 1/2 sum_{j=0}^{n}   || v_j ||^2_{R_j^{-1}}                (1)
//    s.t.  w_j = x_{j+1} - f(x_j, u_j)      (multiple shooting: the dynamics
//                                            enter as penalised residuals, not
//                                            as eliminated equalities)
//          v_j = y_j - h(x_j, u_j)
//          x^lo <= x_j <= x^hi                                            (2)
//
//  The first term is the ARRIVAL COST: it summarises everything the data before
//  the window said about x_{k-N}. The exact arrival cost of the nonlinear
//  full-information problem is not computable; Rao et al. (2003) propose the
//  "filtering" approximation in which Pi is propagated by the EKF covariance
//  recursion along the optimal trajectory and xbar is the MHE estimate of the
//  same instant from the previous window,
//
//    Pi_{+} = Q + F Pi F^T - F Pi H^T ( R + H Pi H^T )^{-1} H Pi F^T,      (3)
//    xbar_{+} = f( xhat_{k-N|.}, u_{k-N} ),                                (3b)
//
//  with F = df/dx, H = dh/dx evaluated at the node that leaves the window.
//  Equation (3) is the prior-to-prior Riccati recursion: it takes the prior at
//  the leaving instant, folds in that instant's measurement and predicts one
//  step, so it carries exactly the information in y_0..y_{k-N}.  Whether the
//  mean (3b) is built from the leaving node's FILTERED value (statistically
//  consistent with (3)) or from its present SMOOTHED value (inconsistent, but
//  refreshed by the newest data) is a genuine design choice with a large
//  measured effect; see Options::arrival_mean.
//  Rao et al. (2003, Thm. 4) show that with a bounded arrival-cost
//  approximation of this type the MHE is an asymptotically stable observer for
//  detectable systems; the same result requires the window to be at least as
//  long as the observability index, which for the cell model is 1 (the SOC is
//  instantaneously observable through OCV whenever dOCV/dz != 0), so N is a
//  noise-averaging and constraint-handling choice rather than a feasibility one.
//
//  ---------------------------------------------------------------------------
//  Gauss-Newton and the KKT structure.  Writing (1) as 1/2||r(X)||^2 with
//  X = (x_0..x_n) and linearising r about the current iterate gives the normal
//  equations  M dX = -g,  M = J^T W J,  g = J^T W r, with
//
//    M_{jj} = [j=0] Pi^{-1} + [j>0] Q_{j-1}^{-1}
//             + [j<n] F_j^T Q_j^{-1} F_j + H_j^T R_j^{-1} H_j,             (4)
//    M_{j,j+1} = - F_j^T Q_j^{-1} =: B_j,     M_{j+1,j} = B_j^T,           (5)
//    g_j    = [j=0] Pi^{-1}(x_0 - xbar) + [j>0] Q_{j-1}^{-1} w_{j-1}
//             - [j<n] F_j^T Q_j^{-1} w_j - H_j^T R_j^{-1} v_j.             (6)
//
//  M is BLOCK TRIDIAGONAL: the only coupling is between consecutive nodes.
//  It is factored in place by a block LDL^T sweep (GVL §4.3), never assembled
//  as a dense (n+1)n_x square matrix:
//
//    D_0 = M_{00};   L_j = B_{j-1}^T D_{j-1}^{-1},  D_j = M_{jj} - L_j B_{j-1}
//    z_0 = -g_0;     z_j = -g_j - L_j z_{j-1}                              (7)
//    dx_n = D_n^{-1} z_n;   dx_j = D_j^{-1}( z_j - B_j dx_{j+1} )
//
//  which costs O(N n_x^3) — linear in the horizon — against O(N^3 n_x^3) for a
//  dense solve.  A useful by-product: after the forward sweep D_n is exactly the
//  Schur complement of the whole system onto the newest node, i.e. the
//  information matrix of x_k marginalised over the rest of the window, so
//  P = D_n^{-1} is the Laplace (Gauss-Newton) posterior covariance of the
//  current estimate and is reported through P().
//
//  Constraints.  The box (2) is enforced by a PROJECTED Gauss-Newton step
//  (Bertsekas 1982): the unconstrained step is computed from (7) and the
//  iterate is then clipped onto the box,
//
//      x_j <- P_[lo,hi]( x_j + alpha dx_j ).                               (8)
//
//  This is exact whenever the solution is interior (the projection is inactive)
//  and otherwise returns a feasible, generally suboptimal point: it is a
//  gradient/Newton-projection step, not an active-set QP solve.  For the cell
//  problem the constraints z in [0,1], h in [-1,1] are almost always inactive
//  and only matter during large transients (a 35 % initial SOC error, a stuck
//  voltage sensor), which is exactly where clipping earns its keep and where
//  the loss of optimality is irrelevant.
//
//  Real-time iteration (Diehl et al. 2005; Kuehl et al. 2011).  `max_iter = 1`
//  performs ONE Gauss-Newton iteration per sample, initialised by SHIFTING the
//  previous window's solution one sample and appending f(x_n, u_n).  Because the
//  window moves by one sample the shifted solution is O(dt) away from the new
//  optimum, the Newton contraction takes over, and the iterates track the
//  solution manifold instead of converging to it at every sample.  The cost per
//  sample is then fixed and known in advance - the property that makes the
//  scheme usable in a hard-real-time task.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model, int N = 20>
class Mhe {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int HORIZON = N;
    static constexpr int CAP = N + 1;            // number of window nodes
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    static_assert(N >= 1, "the horizon must contain at least one interval");
    enum : int { ArrivalFiltered = 0, ArrivalSmoothed = 1 };

    struct Options {
        int  max_iter = 5;             // 1 = real-time iteration (RTI)
        Real tol      = Real(1e-5);    // relative step norm that stops the iteration
        Real alpha    = Real(1);       // step damping in eq. (8)
        int  backtrack_max = 0;        // cost-decrease safeguard (0 = pure Gauss-Newton)
        Real q_scale  = Real(1);
        Real r_scale  = Real(1);
        Real arrival_scale = Real(1);  // inflate/deflate Pi (trust in the arrival cost)
        bool use_arrival = true;
        // How the arrival MEAN is carried over when the window slides, eq. (3b).
        //   ArrivalSmoothed (default): xbar <- f( xhat_{t|t+N}, u_t ), the leaving
        //       node's CURRENT value, i.e. the SMOOTHING update of Rawlings,
        //       Mayne & Diehl (2017, Sec. 4.3.2).  That value was fitted to
        //       y_{t+1..t+N}, which the next window uses again, so the arrival
        //       mean is not statistically consistent with the covariance
        //       recursion (3) - but it is continuously refreshed by the newest
        //       data, which gives the estimator a shorter effective memory.
        //   ArrivalFiltered: xbar <- f( xhat_{t|t}, u_t ), the estimate of the
        //       leaving node made WHEN IT WAS THE NEWEST NODE.  It summarises
        //       y_0..y_t exactly once - exactly the information the recursion (3)
        //       carries - so mean and covariance ARE consistent (the FILTERING
        //       update).
        //   Measured on the eVTOL benchmark (rmse_ss, MHE-RTI, N = 20): the
        //   consistent filtering update is the better estimator when the model is
        //   right and the data are noisy (noise_step 0.0013 vs 0.0133, cold
        //   0.0019 vs 0.0093, sensor_dropout 0.0011 vs 0.0053) and the worse one
        //   when the model is WRONG (aged_cell 0.0708 vs 0.0070, combined 0.0484
        //   vs 0.0049), because a consistent arrival cost keeps shrinking and
        //   then refuses to let the window explain a persistent modelling error.
        //   Since every flight-representative scenario in this benchmark carries
        //   some model error, the smoothing update is registered; a deployment on
        //   a well-identified cell should prefer ArrivalFiltered, or
        //   ArrivalFiltered with arrival_scale ~ 3 as a fading-memory compromise.
        int  arrival_mean = ArrivalSmoothed;
        bool project = true;           // enforce the box (2)
        bool use_model_constrain = false;  // additionally apply Model::constrain
        Real reg = Real(1e-9);         // relative Levenberg regularisation of D_j
        Real pi_max = Real(1e4);       // cap on ||Pi||_inf (arrival-cost windup guard)
        Vec<NX> lo, hi;                // box bounds; defaults are inactive
    } opts;

    explicit Mhe(const Model& m) : m_(m) {
        for (int i = 0; i < NX; ++i) { opts.lo[i] = Real(-1e30); opts.hi[i] = Real(1e30); }
    }

    void init(const X& x0, const Mat<NX, NX>& P0) {
        n_ = 0;
        for (int j = 0; j < CAP; ++j) { xs_[j] = x0; xfilt_[j] = x0; us_[j] = U(); ys_[j] = Y(); }
        xbar_ = x0; Pi_ = P0; Pmarg_ = P0; x_ = x0;
        iters_ = 0;
    }

    // Time update: slide the window one sample and warm-start the new node.
    void predict(const U& u) {
        if (n_ < N) {
            ++n_;
            us_[n_ - 1] = u;
            xs_[n_] = m_.f(xs_[n_ - 1], u);
        } else {
            advance_arrival_();                       // eq. (3), before the shift
            for (int j = 0; j < N; ++j) {
                xs_[j] = xs_[j + 1]; xfilt_[j] = xfilt_[j + 1];
                us_[j] = us_[j + 1]; ys_[j] = ys_[j + 1];
            }
            us_[N - 1] = u;
            xs_[N] = m_.f(xs_[N - 1], u);             // shift-and-append warm start (RTI)
        }
        x_ = xs_[n_];
    }

    Y predict_measurement(const U& u) const { return m_.h(xs_[n_], u); }

    // Measurement update: append (y_k, u_k) and run the Gauss-Newton iteration.
    void update(const Y& y, const U& u) {
        us_[n_] = u; ys_[n_] = y;
        // Q^-1, R^-1 are frozen at the entry iterate: they depend on the state only
        // through the process-noise mapping and re-evaluating them inside the
        // iteration would change the cost being minimised.
        for (int j = 0; j <= n_; ++j) {
            if (j < n_) Qi_[j] = inv_spd_(m_.Q(xs_[j], us_[j]) * opts.q_scale);
            Ri_[j] = inverse(m_.R(xs_[j], us_[j]) * opts.r_scale);
        }
        Mat<NX, NX> Piinv;
        if (opts.use_arrival) Piinv = inv_spd_(Pi_ * opts.arrival_scale);

        Real cost = (opts.backtrack_max > 0) ? cost_(Piinv) : Real(0);
        iters_ = 0;
        for (int it = 0; it < opts.max_iter; ++it) {
            build_(Piinv);
            if (!solve_()) break;
            ++iters_;
            Real step = opts.alpha;
            Real sn = Real(0);
            if (opts.backtrack_max > 0) {
                for (int j = 0; j <= n_; ++j) xsave_[j] = xs_[j];
                bool ok = false;
                for (int bt = 0; bt <= opts.backtrack_max; ++bt) {
                    sn = apply_step_(step);
                    const Real c2 = cost_(Piinv);
                    if (c2 <= cost) { cost = c2; ok = true; break; }
                    for (int j = 0; j <= n_; ++j) xs_[j] = xsave_[j];
                    step *= Real(0.5);
                }
                if (!ok) { sn = apply_step_(step); cost = cost_(Piinv); }
            } else {
                sn = apply_step_(step);
            }
            if (sn < opts.tol) break;
        }
        Pmarg_ = Dinv_[n_];
        Pmarg_.symmetrize();
        x_ = xs_[n_];
        xfilt_[n_] = x_;      // the filtered estimate of this instant, eq. (3b)
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return Pmarg_; }          // Laplace covariance, = D_n^{-1}
    const Mat<NX, NX>& arrival_covariance() const { return Pi_; }
    const X& arrival_mean() const { return xbar_; }
    int window_length() const { return n_; }
    int last_iterations() const { return iters_; }
    // smoothed estimate of the node `j` samples back from the current time
    const X& node(int j) const { return xs_[clampr(n_ - j, 0, n_)]; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    static Mat<NX, NX> inv_spd_(const Mat<NX, NX>& A) { return solve_spd(A, Mat<NX, NX>::identity()); }
    void add_reg_(Mat<NX, NX>& D) const {
        const Real s = opts.reg * (Real(1) + D.norm_inf());
        for (int i = 0; i < NX; ++i) D(i, i) += s;
    }

    // eqs. (4)-(6)
    void build_(const Mat<NX, NX>& Piinv) {
        for (int j = 0; j < n_; ++j) {
            Fj_[j] = jac_F(m_, xs_[j], us_[j]);
            rw_[j] = xs_[j + 1] - m_.f(xs_[j], us_[j]);
        }
        for (int j = 0; j <= n_; ++j) {
            Hj_[j] = jac_H(m_, xs_[j], us_[j]);
            rv_[j] = ys_[j] - m_.h(xs_[j], us_[j]);
        }
        for (int j = 0; j <= n_; ++j) {
            Mat<NX, NX> A;
            X g;
            if (j == 0) {
                if (opts.use_arrival) { A = Piinv; g = Piinv * (xs_[0] - xbar_); }
            } else {
                A = Qi_[j - 1];
                g = Qi_[j - 1] * rw_[j - 1];
            }
            if (j < n_) {
                const Mat<NX, NX> FtQ = Fj_[j].t() * Qi_[j];
                A += FtQ * Fj_[j];
                g -= FtQ * rw_[j];
                B_[j] = -FtQ;                                   // eq. (5)
            }
            const Mat<NX, NY> HtR = Hj_[j].t() * Ri_[j];
            A += HtR * Hj_[j];
            g -= HtR * rv_[j];
            A.symmetrize();
            A_[j] = A; g_[j] = g;
        }
    }

    // eq. (7): block LDL^T forward sweep + back substitution
    bool solve_() {
        Mat<NX, NX> D = A_[0];
        add_reg_(D);
        Dinv_[0] = inv_spd_(D);
        if (!Dinv_[0].is_finite()) return false;
        zf_[0] = -g_[0];
        for (int j = 1; j <= n_; ++j) {
            L_[j] = B_[j - 1].t() * Dinv_[j - 1];
            D = A_[j] - L_[j] * B_[j - 1];
            D.symmetrize();
            add_reg_(D);
            Dinv_[j] = inv_spd_(D);
            if (!Dinv_[j].is_finite()) return false;
            zf_[j] = -g_[j] - L_[j] * zf_[j - 1];
        }
        dx_[n_] = Dinv_[n_] * zf_[n_];
        for (int j = n_ - 1; j >= 0; --j) dx_[j] = Dinv_[j] * (zf_[j] - B_[j] * dx_[j + 1]);
        for (int j = 0; j <= n_; ++j) if (!dx_[j].is_finite()) return false;
        return true;
    }

    // eq. (8): projected step; returns the relative step norm
    Real apply_step_(Real step) {
        Real sn = Real(0);
        for (int j = 0; j <= n_; ++j) {
            X xn = xs_[j] + dx_[j] * step;
            if (opts.project) for (int i = 0; i < NX; ++i) xn[i] = clampr(xn[i], opts.lo[i], opts.hi[i]);
            if (opts.use_model_constrain) xn = constrain(m_, xn);
            for (int i = 0; i < NX; ++i)
                sn = std::max(sn, std::fabs(xn[i] - xs_[j][i]) / (Real(1) + std::fabs(xs_[j][i])));
            xs_[j] = xn;
        }
        return sn;
    }

    // eq. (1) evaluated at the current iterate
    Real cost_(const Mat<NX, NX>& Piinv) const {
        Real c = Real(0);
        if (opts.use_arrival) { const X e = xs_[0] - xbar_; c += dot(e, Piinv * e); }
        for (int j = 0; j < n_; ++j) {
            const X w = xs_[j + 1] - m_.f(xs_[j], us_[j]);
            c += dot(w, Qi_[j] * w);
        }
        for (int j = 0; j <= n_; ++j) {
            const Y v = ys_[j] - m_.h(xs_[j], us_[j]);
            c += dot(v, Ri_[j] * v);
        }
        return Real(0.5) * c;
    }

    // eq. (3): one EKF covariance step past the node that leaves the window
    void advance_arrival_() {
        const X xnew = m_.f((opts.arrival_mean == ArrivalFiltered) ? xfilt_[0] : xs_[0], us_[0]);
        if (!opts.use_arrival) { xbar_ = xnew; return; }
        const Mat<NX, NX> F = jac_F(m_, xs_[0], us_[0]);
        const Mat<NY, NX> H = jac_H(m_, xs_[0], us_[0]);
        const Mat<NX, NX> Q = m_.Q(xs_[0], us_[0]) * opts.q_scale;
        const Mat<NY, NY> R = m_.R(xs_[0], us_[0]) * opts.r_scale;
        const Mat<NY, NY> S = H * Pi_ * H.t() + R;
        const Mat<NX, NY> K = F * Pi_ * H.t() * inverse(S);
        Mat<NX, NX> Pn = F * Pi_ * F.t() + Q - K * S * K.t();
        Pn.symmetrize();
        if (Pn.is_finite()) {
            for (int i = 0; i < NX; ++i) if (!(Pn(i, i) > Real(0))) Pn(i, i) = Real(1e-12);
            const Real nrm = Pn.norm_inf();
            if (nrm > opts.pi_max) Pn *= opts.pi_max / nrm;
            Pi_ = Pn;
        }
        xbar_ = xnew;
    }

    Model m_;
    // --- window data ---
    X xs_[CAP], xsave_[CAP], xfilt_[CAP];
    U us_[CAP];
    Y ys_[CAP];
    // --- cached weights and linearisation ---
    Mat<NX, NX> Qi_[CAP], Fj_[CAP];
    Mat<NY, NY> Ri_[CAP];
    Mat<NY, NX> Hj_[CAP];
    X rw_[CAP];
    Y rv_[CAP];
    // --- normal equations and factorisation ---
    Mat<NX, NX> A_[CAP], B_[CAP], Dinv_[CAP], L_[CAP];
    X g_[CAP], zf_[CAP], dx_[CAP];
    // --- arrival cost and output ---
    X xbar_, x_;
    Mat<NX, NX> Pi_, Pmarg_;
    int n_ = 0, iters_ = 0;
};

}  // namespace estkit
