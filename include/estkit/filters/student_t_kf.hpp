// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/student_t_kf.hpp — Student's t filter (heavy-tailed process
//  and measurement noise), extended (EKF-linearised) form.
//
//  References
//    * Roth, M., Ozkan, E. & Gustafsson, F. (2013). A Student's t filter for
//      heavy tailed process and measurement noise. Proc. IEEE ICASSP, 5770-5774.
//      (exact conditional of the multivariate t, moment matching of the degrees
//       of freedom in the time update)
//    * Agamennoni, G., Nieto, J. I. & Nebot, E. M. (2012). Approximate inference
//      in state-space models with heavy-tailed noise. IEEE Trans. Signal
//      Processing 60(10), 5024-5037.     (independent-scale / variational form)
//    * Huang, Y., Zhang, Y., Li, N. & Chambers, J. A. (2016). A robust Student's
//      t based cubature filter. Proc. 19th Int. Conf. Information Fusion, 9-16.
//    * Kotz, S. & Nadarajah, S. (2004). Multivariate t Distributions and Their
//      Applications. Cambridge University Press, Ch. 1.   (conditional/marginal t)
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, Ch. 8.                                (EKF linearisation)
//
//  Distribution -----------------------------------------------------------
//  x ~ St(mu, Sigma, nu) has density proportional to
//      [1 + (x-mu)^T Sigma^{-1} (x-mu)/nu]^{-(nu+n)/2},
//  mean mu (nu > 1) and covariance  Cov[x] = nu/(nu-2) Sigma  (nu > 2).  Sigma is
//  the SCALE matrix, not the covariance - the filter therefore converts the
//  model's Q and R (which are covariances) with  Sigma = (nu-2)/nu * Cov, and
//  converts back for the reported P().
//
//  (A) Exact conditional, shared latent scale (Roth et al. 2013) -----------
//  If  x|y_{1:k-1} ~ St(x^-, P^-, nu)  and  v_k ~ St(0, R, nu)  are driven by the
//  SAME latent scale, [x;y] is jointly t and the conditional is again t:
//
//      S      = H P^- H^T + R,     e = y - h(x^-,u),                        (1)
//      Delta2 = e^T S^{-1} e,      K = P^- H^T S^{-1},                      (2)
//      x^+    = x^- + K e,                                                  (3)
//      P^+    = (P^- - K S K^T) * (nu + Delta2)/(nu + n_y),                 (4)
//      nu^+   = nu + n_y.                                                   (5)
//
//  (B) Time update by moment matching of the degrees of freedom -----------
//  The sum F x + w of two independent t vectors with different degrees of
//  freedom is not t.  Roth et al. approximate it by a t with
//  nu* = min(nu_1, nu_2) (the heavier tail) after rescaling both scale matrices
//  so that the COVARIANCES are preserved:
//
//      Sigma~ = Sigma * [ nu_1 (nu*-2) ] / [ nu* (nu_1-2) ],                (6)
//      P^-_{k+1} = F Sigma~ F^T + Q~,     nu <- nu*.                        (7)
//
//  The same rescaling is applied before (1) so that the state and the
//  measurement noise share one nu.  With nu_w = nu_v + n_y the degrees of
//  freedom settle into the two-step cycle nu^- = nu_v, nu^+ = nu_v + n_y.
//
//  Why (A) is NOT used as the measurement update here ----------------------
//  Two properties of (1)-(5) are fatal on a long-horizon, weakly informative
//  problem such as single-output SOC estimation:
//    (i)  the conditional MEAN (3) is exactly the Kalman mean - with one shared
//         latent scale the gain is scale invariant, so (A) has NO outlier
//         rejection at all; its only robustness is the covariance inflation (4);
//    (ii) combining (4) and (6) over a cycle gives, in covariance terms,
//             Cov^+ = Cov^+_KF * (nu - 2 + delta^2)/(nu - 1),                (8)
//         delta^2 = e^T S_cov^{-1} e ~ chi^2(n_y) under the nominal hypothesis.
//         Hence E[factor] = 1 but E[log factor] < 0 (chi^2 is right skewed,
//         median 0.455 for n_y = 1): the product of many such factors drifts
//         geometrically to zero.  The only restoring mechanism is the feedback
//         through delta^2, which needs H P H^T to dominate S = H P H^T + R; for
//         the cell model H P H^T is ~3 % of S, so there is none.  Measured on
//         the nominal eVTOL mission the scale matrix falls five decades within
//         500 samples, the filter goes blind, and the subsequent huge delta^2
//         values make it oscillate over decades until it breaks down.  Set
//         opts.shared_scale = true to reproduce this.
//
//  (C) Independent latent scales - the form implemented --------------------
//  The realistic heavy-tailed measurement model gives the measurement noise its
//  OWN latent scale, v_k | u_k ~ N(0, R/u_k), u_k ~ Gamma(nu_v/2, nu_v/2), which
//  marginalises to v_k ~ St(0, R, nu_v).  The posterior of u_k given y_k is again
//  Gamma and its mean is available in closed form, giving the one-step
//  EM / variational fixed point (Agamennoni et al. 2012; Huang et al. 2016)
//
//      R_eff = R / u,                                                       (9)
//      S     = H Sigma H^T + R_eff,   Delta2 = e^T S^{-1} e,               (10)
//      u     = (nu_v + n_y) / (nu_v + Delta2),   u <- min(u, 1),           (11)
//
//  iterated (with geometric relaxation, which removes the oscillation of the
//  raw iteration) until u converges, followed by the ordinary Kalman update
//  with R_eff:
//
//      K = Sigma H^T S^{-1},  x^+ = x^- + K e,
//      Sigma^+ = (I-KH) Sigma (I-KH)^T + K R_eff K^T.                      (12)
//
//  u is exactly the weight function of a Student's t likelihood, so (9)-(12) is
//  the maximum-likelihood measurement update for t-distributed measurement
//  noise; u = 1 recovers (1)-(3).  Because u <= 1 implies R_eff >= R, (12) is a
//  genuine Kalman posterior for a LARGER noise covariance and therefore
//  contracts: the recursion is unconditionally stable, unlike (4).  Note that
//  Delta2 in (10)-(11) is normalised by the FULL innovation covariance, so a
//  large but legitimate residual during a transient (e.g. a wrong initial SOC,
//  where H Sigma H^T is huge) is NOT rejected - a decisive advantage over
//  weight functions that normalise by R alone.
//  The degrees-of-freedom bookkeeping (6)-(7) is retained in both branches.
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class StudentTKf {
public:
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;

    struct Options {
        Real nu_v = Real(3);        // degrees of freedom of the measurement noise
        Real nu_w = Real(4);        // degrees of freedom of the process noise
        Real nu_0 = Real(4);        // degrees of freedom of the initial state
        // Interpret Model::Q, Model::R and P0 as COVARIANCES and convert them to
        // scale matrices with the factor (nu-2)/nu.  Set false to pass scale
        // matrices directly.
        bool cov_to_scale = true;
        Real nu_min = Real(2.5);    // lower clamp so that nu/(nu-2) stays finite
        int  n_iter = 5;            // fixed-point iterations for u, Eq. (11)
        Real u_min = Real(1e-3);    // floor on the latent scale (gating)
        bool shared_scale = false;  // true -> literal recursion (1)-(5); see the note above
        Real max_inflation = Real(50);   // cap on (nu + Delta2)/(nu + n_y) in branch (A)
        bool apply_constraints = true;
        Real q_scale = Real(1), r_scale = Real(1);
    } opts;

    explicit StudentTKf(const Model& m) : m_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        nu_ = std::max(opts.nu_0, opts.nu_min);
        x_ = x0;
        Sig_ = opts.cov_to_scale ? P0 * ((nu_ - Real(2)) / nu_) : P0;
        u_ = Real(1);
        sync_();
    }

    void predict(const U& u) {
        const Mat<NX, NX> F = jac_F(m_, x_, u);
        x_ = m_.f(x_, u);
        const Mat<NX, NX> Qc = m_.Q(x_, u) * opts.q_scale;
        const Real nu2 = std::max(opts.nu_w, opts.nu_min);
        const Mat<NX, NX> Qs = opts.cov_to_scale ? Qc * ((nu2 - Real(2)) / nu2) : Qc;
        const Real nu1 = nu_;
        const Real nus = std::max(std::min(nu1, nu2), opts.nu_min);
        Sig_ = F * (Sig_ * rescale_(nu1, nus)) * F.t() + Qs * rescale_(nu2, nus);
        Sig_.symmetrize();
        nu_ = nus;
        sync_();
    }

    Y predict_measurement(const U& u) const { return m_.h(x_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NX> H = jac_H(m_, x_, u);
        const Mat<NY, NY> Rc = m_.R(x_, u) * opts.r_scale;
        const Real nu2 = std::max(opts.nu_v, opts.nu_min);
        const Mat<NY, NY> Rs = opts.cov_to_scale ? Rc * ((nu2 - Real(2)) / nu2) : Rc;
        const Real nu1 = nu_;
        const Real nus = std::max(std::min(nu1, nu2), opts.nu_min);
        Mat<NX, NX> Sig = Sig_ * rescale_(nu1, nus); Sig.symmetrize();
        const Mat<NY, NY> Rr = Rs * rescale_(nu2, nus);
        innovation_ = y - m_.h(x_, u);

        // --- fixed point (9)-(11) for the latent measurement scale ------------
        Real w = Real(1), delta2 = Real(0);
        const int nit = opts.shared_scale ? 1 : (opts.n_iter < 1 ? 1 : opts.n_iter);
        for (int it = 0; it < nit; ++it) {
            Mat<NY, NY> S = H * Sig * H.t() + Rr / w; S.symmetrize();
            const Mat<NY, NY> Si = inverse(S);
            if (!Si.is_finite()) return;
            delta2 = std::max(dot(innovation_, Si * innovation_), Real(0));
            if (opts.shared_scale) break;
            Real wn = (nus + Real(NY)) / (nus + delta2);
            if (wn > Real(1)) wn = Real(1);                        // never up-weight
            if (wn < opts.u_min) wn = opts.u_min;
            const Real wg = std::sqrt(w * wn);                     // geometric relaxation
            const bool done = std::fabs(wg - w) <= Real(1e-6) * w;
            w = wg;
            if (done) break;
        }
        u_ = w; delta2_ = delta2;
        const Mat<NY, NY> Reff = Rr / w;
        Mat<NY, NY> S = H * Sig * H.t() + Reff; S.symmetrize(); S_ = S;
        const Mat<NY, NY> Sinv = inverse(S);
        if (!Sinv.is_finite()) return;
        const Mat<NX, NY> K = Sig * H.t() * Sinv;
        if (!K.is_finite()) return;
        K_ = K;
        x_ = x_ + K * innovation_;
        const Mat<NX, NX> IKH = Mat<NX, NX>::identity() - K * H;
        Mat<NX, NX> Sp = IKH * Sig * IKH.t() + K * Reff * K.t();   // (12), Joseph
        Sp.symmetrize();
        if (opts.shared_scale) {                                   // literal branch (4)
            Real infl = (nus + delta2) / (nus + Real(NY));
            if (infl > opts.max_inflation) infl = opts.max_inflation;
            if (!(infl > Real(0))) infl = Real(1);
            inflation_ = infl;
            Sp *= infl;
        } else {
            inflation_ = Real(1) / w;
        }
        Sig_ = Sp;
        // Degrees of freedom of the posterior.  Branch (A) gains n_y from the
        // exact conditional (5); branch (C) is a moment-matched t with the SAME
        // nu, because the scale of (12) already carries the whole posterior
        // covariance - incrementing nu there would deflate Cov = nu/(nu-2) Sigma
        // by the constant factor [nu*(nu*+n_y-2)]/[(nu*-2)(nu*+n_y)] < 1 at every
        // cycle and make the filter collapse geometrically.
        nu_ = opts.shared_scale ? (nus + Real(NY)) : nus;
        if (opts.apply_constraints) x_ = constrain(m_, x_);
        sync_();
    }

    const X& x() const { return x_; }
    const Mat<NX, NX>& P() const { return P_; }          // COVARIANCE nu/(nu-2) * Sigma
    const Mat<NX, NX>& scale() const { return Sig_; }    // scale matrix Sigma
    Real nu() const { return nu_; }
    Real delta2() const { return delta2_; }              // squared Mahalanobis innovation
    Real latent_scale() const { return u_; }             // u of Eq. (11): 1 = Gaussian regime
    Real inflation() const { return inflation_; }        // R_eff / R  (= 1/u)
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NX, NY>& K() const { return K_; }
    Model& model() { return m_; }
    const Model& model() const { return m_; }

private:
    // Scale factor that re-expresses a t(nu_from) scale matrix as a t(nu_to) one
    // with the SAME covariance:  Sigma_to = Sigma_from * nu_from(nu_to-2)/(nu_to(nu_from-2)).
    static Real rescale_(Real nu_from, Real nu_to) {
        return (nu_from * (nu_to - Real(2))) / (nu_to * (nu_from - Real(2)));
    }
    void sync_() { P_ = Sig_ * (nu_ / (nu_ - Real(2))); }

    Model m_;
    X x_;
    Mat<NX, NX> Sig_;
    Mat<NX, NX> P_;
    Real nu_ = Real(5);
    Real delta2_ = Real(0), inflation_ = Real(1), u_ = Real(1);
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NX, NY> K_;
};

}  // namespace estkit
