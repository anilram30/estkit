// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/learning/nn_direct.hpp — purely data-driven SOC regression:
//  a feed-forward network maps the instantaneous and time-averaged sensor
//  signals directly onto the state of charge, with NO cell model in the loop.
//
//  References
//    * Chemali, E., Kollmeyer, P. J., Preindl, M. & Emadi, A. (2018).
//      State-of-charge estimation of Li-ion batteries using deep neural networks:
//      A machine learning approach. J. Power Sources 400, 242-250.
//      (voltage, current, temperature and their moving averages as inputs;
//       direct regression of SOC; no filter, no OCV table, no Coulomb counter)
//    * Hornik, K., Stinchcombe, M. & White, H. (1989). Multilayer feedforward
//      networks are universal approximators. Neural Networks 2(5), 359-366.
//    * Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization.
//      Proc. 3rd International Conference on Learning Representations (ICLR).
//    * Ng, K. S., Moo, C.-S., Chen, Y.-P. & Hsieh, Y.-C. (2009). Enhanced
//      coulomb counting method for estimating state-of-charge and state-of-health
//      of lithium-ion batteries. Applied Energy 86(9), 1506-1511.
//      (the Coulomb-count feature)
//
//  Feature vector (n_in = 8), all formed from measured quantities only:
//
//      phi_k = [ v_k,  i_k,  T_k,
//                <v>_10, <v>_60,  <i>_10, <i>_60,
//                Delta z^{CC}_k ]^T                                      (1)
//
//  where  <x>_W = (1/N_W) sum_{j=k-N_W+1..k} x_j,  N_W = round(W / dt)   (2)
//  are boxcar moving averages over W = 10 s and W = 60 s, and
//
//      Delta z^{CC}_k = - sum_{j<k} eta(i_j) i_j dt / (3600 Q_nom)       (3)
//
//  is the Coulomb-counted charge increment since the estimator was reset
//  (dimensionless, negative on discharge).  The regression target is the true
//  state of charge,
//
//      zhat_k = N(phi_k; theta),    theta = argmin sum_k (N(phi_k)-z_k)^2. (4)
//
//  The network is trained ONCE (lazily, on the first reset()) on ECM-plant data
//  from the fixed-wing and hover-hold missions across the nominal, cold, hot and
//  aged operating scenarios; the eVTOL mission used by the benchmark is never
//  seen during training (see train_data.hpp).  Inference is a fixed sequence of
//  multiply-accumulates and tanh evaluations: deterministic, heap-free and
//  constant-time, which is what a DO-178C argument for an airborne implementation
//  requires of the *inference* path (the training path is an off-line tool).
// =============================================================================
#pragma once
#include "../battery/ecm_model.hpp"
#include "mlp.hpp"
#include "train_data.hpp"

namespace estkit {

// -----------------------------------------------------------------------------
//  Boxcar (sliding-window) mean with O(1) update and a fixed-capacity buffer.
// -----------------------------------------------------------------------------
template <int NBUF>
class BoxcarMean {
public:
    void configure(int window) {
        win_ = clampr(window, 1, NBUF);
        reset();
    }
    void reset() {
        head_ = 0; cnt_ = 0; sum_ = Real(0);
        for (int k = 0; k < NBUF; ++k) buf_[k] = Real(0);
    }
    Real push(Real x) {
        if (cnt_ == win_) sum_ -= buf_[head_];
        else ++cnt_;
        buf_[head_] = x;
        sum_ += x;
        head_ = (head_ + 1) % win_;
        return sum_ / Real(cnt_);
    }
    int window() const { return win_; }

private:
    Real buf_[NBUF] = {};
    int  head_ = 0, cnt_ = 0, win_ = 1;
    Real sum_ = Real(0);
};

// -----------------------------------------------------------------------------
//  Feature extractor, eqs. (1)-(3). Used identically by the offline training-set
//  builder and by the on-line estimator so that there is no train/serve skew.
// -----------------------------------------------------------------------------
class SocFeatures {
public:
    static constexpr int NF = 8;
    static constexpr int NB10 = 128;     // >= 10 s / dt at dt = 0.1 s
    static constexpr int NB60 = 640;     // >= 60 s / dt at dt = 0.1 s

    void reset(const EstimatorConfig& c) {
        dt_ = c.dt;
        Q_  = std::max(c.params.Q_Ah, Real(0.5));
        eta_c_ = c.params.eta_c;
        dz_ = Real(0);
        const int n10 = int(std::lround(Real(10) / dt_));
        const int n60 = int(std::lround(Real(60) / dt_));
        v10_.configure(n10); v60_.configure(n60);
        i10_.configure(n10); i60_.configure(n60);
    }
    // Feed one sample; returns the feature vector for that sample.
    Vec<NF> push(Real i, Real v, Real T) {
        const Real va = v10_.push(v), vb = v60_.push(v);
        const Real ia = i10_.push(i), ib = i60_.push(i);
        Vec<NF> f;
        f[0] = v;
        f[1] = i;
        f[2] = T - kT0;            // degC (standardisation makes the offset irrelevant)
        f[3] = va; f[4] = vb;
        f[5] = ia; f[6] = ib;
        f[7] = dz_;                // Coulomb count BEFORE this sample, eq. (3)
        const Real eta = (i >= Real(0)) ? Real(1) : eta_c_;
        dz_ -= eta * i * dt_ / (Real(3600) * Q_);
        return f;
    }
    Real coulomb_increment() const { return dz_; }

private:
    BoxcarMean<NB10> v10_, i10_;
    BoxcarMean<NB60> v60_, i60_;
    Real dt_ = Real(0.1), Q_ = Real(5), eta_c_ = Real(1), dz_ = Real(0);
};

// -----------------------------------------------------------------------------
//  Direct SOC regressor. Implements the benchmark `Estimator` interface directly
//  (it is not a filter: it has no state-space model, no covariance and no
//  voltage prediction).
// -----------------------------------------------------------------------------
class NnDirectSoc final : public Estimator {
public:
    static constexpr int NF   = SocFeatures::NF;
    static constexpr int NHID = 32;
    using Net = Mlp<NF, NHID, 1>;

    // --- offline training configuration (all knobs of the method) ---
    struct Options {
        TrainingSetSpec data;          // which missions/scenarios/seeds to simulate
        MlpTrainOptions train;         // epochs / batch size / learning rate
        uint64_t init_seed = 0x5EED1234ull;   // deterministic weight initialisation
        Real z_lo = Real(-0.05), z_hi = Real(1.05);   // output saturation
    };
    static Options default_options() {
        Options o;
        o.data.profiles  = {"fixed_wing", "hover_hold"};
        o.data.scenarios = {"nominal", "cold", "hot", "aged_cell"};
        o.data.n_seeds   = 3;
        o.data.seed0     = 101;
        o.data.preisach  = 2;          // half the training cells carry Preisach hysteresis
        o.data.subsample = 10;         // 1 Hz training samples
        o.train.epochs = 20;           // the training loss is still falling at 10 epochs;
        o.train.batch  = 32;           // 20 epochs cost ~1.2 s on a 2-core x86 host
        o.train.lr     = Real(6e-3);
        o.train.l2     = Real(1e-6);
        return o;
    }

    const char* name()  const override { return "NN-Direct"; }
    const char* group() const override { return "learning"; }
    // the moving-average ring buffers dominate; the (shared) weights are counted once
    size_t state_bytes() const override { return sizeof(*this) + sizeof(Net); }

    void reset(const EstimatorConfig& c) override {
        net_ = &trained_net();          // trains once, on the first reset (shared)
        feat_.reset(c);
        z_ = c.z0;
    }
    void step(Real i, Real v, Real T) override {
        const Vec<NF> f = feat_.push(i, v, T);
        const Real z = (*net_)(f)[0];
        z_ = clampr(z, Real(-0.05), Real(1.05));
    }
    Real soc() const override { return z_; }
    Real voltage_pred() const override { return kNaN; }   // no measurement model

    // The shared, lazily trained network (one instance per process).
    static const Net& trained_net() {
        static const Net net = train();
        return net;
    }

private:
    static Net train() {
        const Options o = default_options();
        std::vector<Vec<NF>> X;
        std::vector<Vec<1>>  Y;
        X.reserve(60000); Y.reserve(60000);
        for_each_training_dataset(o.data, [&](const Dataset& d) {
            SocFeatures fe;
            fe.reset(d.est_cfg);
            for (int k = 0; k < d.n; ++k) {
                const Vec<NF> f = fe.push(d.i_meas[k], d.v_meas[k], d.T_meas[k]);
                if (k % o.data.subsample == 0) {
                    X.push_back(f);
                    Vec<1> y; y[0] = d.soc_true[k];
                    Y.push_back(y);
                }
            }
        });
        Net net;
        Rng rng(o.init_seed);
        net.init_glorot(rng);
        train_mlp(net, X.data(), Y.data(), int(X.size()), o.train, rng);
        return net;
    }

    const Net*  net_ = nullptr;
    SocFeatures feat_;
    Real z_ = Real(0);
};

}  // namespace estkit
