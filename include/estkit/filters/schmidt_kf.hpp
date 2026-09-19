// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/filters/schmidt_kf.hpp — Schmidt-Kalman ("consider") filter
//
//  References
//    * Schmidt, S. F. (1966). Application of state-space methods to navigation
//      problems. In: Leondes, C. T. (ed.), Advances in Control Systems, Vol. 3,
//      Academic Press, 293-340.                        (original consider filter)
//    * Tapley, B. D., Schutz, B. E. & Born, G. H. (2004). Statistical Orbit
//      Determination. Elsevier Academic Press, Burlington, MA, Ch. 6
//      ("consider covariance analysis", partitioned covariance recursion).
//    * Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory,
//      Academic Press, Ch. 8.                                (EKF linearisation)
//    * Bucy, R. S. & Joseph, P. D. (1968). Filtering for Stochastic Processes
//      with Applications to Guidance. Wiley.        (Joseph-form covariance update)
//    * Plett, G. L. (2004). Extended Kalman filtering for battery management
//      systems of LiPB-based HEV battery packs - Part 3. J. Power Sources 134,
//      277-292.                                             (EKF for SOC, timing)
//
//  Idea -------------------------------------------------------------------
//  The model parameters theta = Model::params() (for the cell ECM
//  theta = [Q_Ah, R0]) are uncertain but are NOT estimated: they are
//  "considered".  The filter carries the augmented covariance
//
//      P_a = [ P_xx  P_xc ]                      (x: solve-for states,
//            [ P_cx  P_cc ]                       c: consider parameters)
//
//  and propagates ALL blocks through the augmented Jacobians
//  F_a = [F_xx F_xc; 0 I], H_a = [H_x H_c] obtained from AugmentedModel<Model>,
//  but the gain is constrained to have a zero consider block,
//
//      K_a = [K_x; 0],   K_x = P_xy S^{-1},
//      P_xy = P_xx H_x^T + P_xc H_c^T,
//      S    = H_x P_xx H_x^T + H_x P_xc H_c^T + H_c P_cx H_x^T
//             + H_c P_cc H_c^T + R                    (consider contribution).
//
//  Because K_a is sub-optimal the covariance update MUST use the Joseph form
//
//      P_a^+ = (I - K_a H_a) P_a^- (I - K_a H_a)^T + K_a R K_a^T,
//
//  which automatically leaves P_cc unchanged (bottom block row of I - K_a H_a
//  is [0 I]) and yields the Schmidt cross-covariance recursion
//      P_xc^+ = P_xc^- - K_x (H_x P_xc^- + H_c P_cc).
//  The parameters themselves are never corrected, so a consider filter never
//  "learns" the parameters: it only produces an estimate whose covariance
//  honestly accounts for the parameter uncertainty (consistent, conservative).
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/model.hpp"

namespace estkit {

template <class Model>
class SchmidtKf {
public:
    static_assert(has_params<Model>::value,
                  "SchmidtKf requires Model::NP, Model::params() and Model::set_params()");
    static constexpr int NX = Model::NX, NU = Model::NU, NY = Model::NY;
    static constexpr int NC = Model::NP;        // number of consider parameters
    static constexpr int NA = NX + NC;          // augmented dimension
    using X = Vec<NX>; using U = Vec<NU>; using Y = Vec<NY>;
    using Aug = AugmentedModel<Model>;

    struct Options {
        // Prior standard deviation of each consider parameter.  If abs_std[k] > 0
        // it is used directly, otherwise rel_std[k] * |theta_k(0)| (relative form,
        // which is how parameter tolerances are quoted on a datasheet).
        Real rel_std[NC];
        Real abs_std[NC];
        // Per-step random-walk variance of the consider parameters.  0 keeps them
        // strictly constant (pure "consider" analysis, Tapley et al. 2004 Ch. 6).
        Real q_theta[NC];
        bool apply_constraints = true;
        Real q_scale = Real(1);     // multiplicative tuning of Q
        Real r_scale = Real(1);     // multiplicative tuning of R
        Options() {
            for (int k = 0; k < NC; ++k) { rel_std[k] = Real(0.10); abs_std[k] = Real(0); q_theta[k] = Real(0); }
        }
    } opts;

    explicit SchmidtKf(const Model& m) : am_(m) {}

    void init(const X& x0, const Mat<NX, NX>& P0) {
        const Vec<NC> th = am_.base.params();
        xa_ = Vec<NA>();
        xa_.set_block(0, 0, x0);
        xa_.set_block(NX, 0, th);
        Pa_ = Mat<NA, NA>();
        Pa_.set_block(0, 0, P0);
        for (int k = 0; k < NC; ++k) {
            const Real s = (opts.abs_std[k] > Real(0)) ? opts.abs_std[k]
                                                       : opts.rel_std[k] * std::fabs(th[k]);
            Pa_(NX + k, NX + k) = sq(s);
            am_.q_theta[k] = opts.q_theta[k];
        }
        sync_();
    }

    void predict(const U& u) {
        const Mat<NA, NA> Fa = am_.F(xa_, u);
        xa_ = am_.f(xa_, u);
        Pa_ = Fa * Pa_ * Fa.t() + am_.Q(xa_, u) * opts.q_scale;
        Pa_.symmetrize();
        sync_();
    }

    Y predict_measurement(const U& u) const { return am_.h(xa_, u); }

    void update(const Y& y, const U& u) {
        const Mat<NY, NA> Ha = am_.H(xa_, u);
        const Mat<NY, NY> R  = am_.R(xa_, u) * opts.r_scale;
        const Mat<NA, NY> Pay = Pa_ * Ha.t();              // [P_xy ; P_cy]
        Mat<NY, NY> S = Ha * Pay + R;                      // consider-inflated innovation covariance
        S.symmetrize();
        const Mat<NY, NY> Sinv = inverse(S);
        if (!Sinv.is_finite()) return;                     // keep the prior rather than produce NaN
        const Mat<NA, NY> Kfull = Pay * Sinv;
        Mat<NA, NY> Ka;                                    // consider gain: zero rows on the parameter block
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NY; ++j) Ka(i, j) = Kfull(i, j);
        innovation_ = y - am_.h(xa_, u);
        S_ = S; Ka_ = Ka;
        xa_ = xa_ + Ka * innovation_;                      // theta block untouched (rows of Ka are zero)
        // Joseph form is mandatory: K_a is not the optimal gain of the augmented system.
        const Mat<NA, NA> IKH = Mat<NA, NA>::identity() - Ka * Ha;
        Pa_ = IKH * Pa_ * IKH.t() + Ka * R * Ka.t();
        Pa_.symmetrize();
        if (opts.apply_constraints) xa_ = am_.constrain(xa_);
        sync_();
    }

    // --- accessors -----------------------------------------------------------
    const X& x() const { return x_; }                      // solve-for states only
    const Mat<NX, NX>& P() const { return P_; }            // its consider covariance
    const Y& innovation() const { return innovation_; }
    const Mat<NY, NY>& S() const { return S_; }
    const Mat<NA, NY>& K() const { return Ka_; }
    Vec<NC> theta() const { return xa_.template block<NC, 1>(NX, 0); }
    Mat<NC, NC> Pcc() const { return Pa_.template block<NC, NC>(NX, NX); }
    Mat<NX, NC> Pxc() const { return Pa_.template block<NX, NC>(0, NX); }
    const Vec<NA>& x_aug() const { return xa_; }
    const Mat<NA, NA>& P_aug() const { return Pa_; }
    Model& model() { return am_.base; }
    const Model& model() const { return am_.base; }

private:
    void sync_() {
        x_ = xa_.template block<NX, 1>(0, 0);
        P_ = Pa_.template block<NX, NX>(0, 0);
    }

    Aug am_;
    Vec<NA> xa_;
    Mat<NA, NA> Pa_;
    X x_;
    Mat<NX, NX> P_;
    Y innovation_;
    Mat<NY, NY> S_;
    Mat<NA, NY> Ka_;
};

}  // namespace estkit
