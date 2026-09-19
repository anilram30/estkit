// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/learning/mlp.hpp — fixed-size multilayer perceptron with one tanh
//  hidden layer and a linear output, backpropagation, and the Adam optimiser.
//
//  The inference object (`Mlp`) is heap-free, exception-free and has compile-time
//  dimensions, so a trained network can be executed on an MCU in bounded time.
//  Training (`train_mlp`) is an OFFLINE, host-side routine: it is still heap-free
//  — the caller owns the (contiguous) training arrays.
//
//  References
//    * Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). Learning
//      representations by back-propagating errors. Nature 323(6088), 533-536.
//                                                             (backpropagation)
//    * Hornik, K., Stinchcombe, M. & White, H. (1989). Multilayer feedforward
//      networks are universal approximators. Neural Networks 2(5), 359-366.
//                                                     (why one hidden layer suffices)
//    * Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization.
//      Proc. 3rd International Conference on Learning Representations (ICLR).
//    * Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training
//      deep feedforward neural networks. Proc. 13th International Conference on
//      Artificial Intelligence and Statistics (AISTATS), PMLR 9, 249-256.
//                                                       (weight initialisation)
//    * Bishop, C. M. (1995). Neural Networks for Pattern Recognition. Oxford
//      University Press, Ch. 4 & 8.                     (input standardisation)
//
//  Network (n_in -> n_h -> n_out), with input/output standardisation folded in:
//
//      s_i  = (x_i - mu_i) / sigma_i,                 i = 1..n_in        (1)
//      a_j  = tanh( sum_i W1_{ji} s_i + b1_j ),       j = 1..n_h         (2)
//      g_o  = sum_j W2_{oj} a_j + b2_o,               o = 1..n_out       (3)
//      y_o  = mu^y_o + sigma^y_o g_o                                     (4)
//
//  Training minimises the standardised half mean-squared error
//
//      L(theta) = (1/2N) sum_{n=1..N} || g(x^{(n)}) - t^{(n)} ||^2,      (5)
//      t^{(n)}_o = (y^{(n)}_o - mu^y_o)/sigma^y_o,
//
//  whose gradients follow from (2)-(3) by the chain rule
//
//      e_o      = g_o - t_o,
//      dL/dW2   = e a^T,      dL/db2 = e,
//      delta_j  = (1 - a_j^2) sum_o W2_{oj} e_o,                         (6)
//      dL/dW1   = delta s^T,  dL/db1 = delta.
//
//  Adam (Kingma & Ba 2015, Alg. 1) with bias correction:
//      m <- b1 m + (1-b1) g,   v <- b2 v + (1-b2) g.^2,
//      mhat = m/(1-b1^t),      vhat = v/(1-b2^t),
//      theta <- theta - lr * mhat ./ (sqrt(vhat) + eps).                 (7)
// =============================================================================
#pragma once
#include "../core/linalg.hpp"
#include "../core/rng.hpp"

namespace estkit {

// -----------------------------------------------------------------------------
//  Parameter block (weights + biases) of a one-hidden-layer MLP.
//  Provides a flat index so that optimisers can treat it as a plain vector.
// -----------------------------------------------------------------------------
template <int NIN, int NHID, int NOUT>
struct MlpParams {
    static constexpr int N1 = NHID * NIN;          // end of W1
    static constexpr int N2 = N1 + NHID;           // end of b1
    static constexpr int N3 = N2 + NOUT * NHID;    // end of W2
    static constexpr int NPAR = N3 + NOUT;         // total number of parameters

    Mat<NHID, NIN> W1;
    Vec<NHID>      b1;
    Mat<NOUT, NHID> W2;
    Vec<NOUT>      b2;

    void zero() { W1 = Mat<NHID, NIN>(); b1 = Vec<NHID>(); W2 = Mat<NOUT, NHID>(); b2 = Vec<NOUT>(); }
    Real& at(int k) {
        if (k < N1) return W1.d[k];
        if (k < N2) return b1.d[k - N1];
        if (k < N3) return W2.d[k - N2];
        return b2.d[k - N3];
    }
    Real at(int k) const {
        if (k < N1) return W1.d[k];
        if (k < N2) return b1.d[k - N1];
        if (k < N3) return W2.d[k - N2];
        return b2.d[k - N3];
    }
};

// -----------------------------------------------------------------------------
//  The network itself: parameters + the standardisation constants, so that the
//  deployed object is self-contained (no external scaler needed at run time).
// -----------------------------------------------------------------------------
template <int NIN, int NHID, int NOUT>
struct Mlp {
    using Params = MlpParams<NIN, NHID, NOUT>;
    static constexpr int n_in = NIN, n_hidden = NHID, n_out = NOUT;

    Params    th;                     // theta = {W1, b1, W2, b2}
    Vec<NIN>  mu_in,  sd_in;          // input standardisation, eq. (1)
    Vec<NOUT> mu_out, sd_out;         // output de-standardisation, eq. (4)

    Mlp() {
        for (int i = 0; i < NIN; ++i)  { mu_in[i]  = Real(0); sd_in[i]  = Real(1); }
        for (int o = 0; o < NOUT; ++o) { mu_out[o] = Real(0); sd_out[o] = Real(1); }
    }

    // Deterministic Glorot-uniform initialisation (Glorot & Bengio 2010):
    //   W ~ U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))),  b = 0.
    void init_glorot(Rng& rng) {
        const Real l1 = std::sqrt(Real(6) / Real(NIN + NHID));
        for (int k = 0; k < NHID * NIN; ++k) th.W1.d[k] = rng.uniform(-l1, l1);
        for (int j = 0; j < NHID; ++j) th.b1[j] = Real(0);
        const Real l2 = std::sqrt(Real(6) / Real(NHID + NOUT));
        for (int k = 0; k < NOUT * NHID; ++k) th.W2.d[k] = rng.uniform(-l2, l2);
        for (int o = 0; o < NOUT; ++o) th.b2[o] = Real(0);
    }

    // eq. (1): standardise one input vector
    Vec<NIN> standardise(const Vec<NIN>& x) const {
        Vec<NIN> s;
        for (int i = 0; i < NIN; ++i) s[i] = (x[i] - mu_in[i]) / sd_in[i];
        return s;
    }
    // eqs. (2)-(3): forward pass in standardised coordinates, keeping the hidden
    // activation `a` for backpropagation.
    void forward_std(const Vec<NIN>& x, Vec<NHID>& a, Vec<NOUT>& g) const {
        const Vec<NIN> s = standardise(x);
        for (int j = 0; j < NHID; ++j) {
            Real acc = th.b1[j];
            for (int i = 0; i < NIN; ++i) acc += th.W1(j, i) * s[i];
            a[j] = std::tanh(acc);
        }
        for (int o = 0; o < NOUT; ++o) {
            Real acc = th.b2[o];
            for (int j = 0; j < NHID; ++j) acc += th.W2(o, j) * a[j];
            g[o] = acc;
        }
    }
    // eqs. (1)-(4): the deployed inference call. No heap, no branches on data.
    Vec<NOUT> operator()(const Vec<NIN>& x) const {
        Vec<NHID> a; Vec<NOUT> g;
        forward_std(x, a, g);
        Vec<NOUT> y;
        for (int o = 0; o < NOUT; ++o) y[o] = mu_out[o] + sd_out[o] * g[o];
        return y;
    }

    // eq. (6): accumulate dL/dtheta of 1/2||g - t||^2 for one sample into `grad`;
    // returns the sample's squared error in standardised output units.
    Real accumulate_grad(const Vec<NIN>& x, const Vec<NOUT>& y, Params& grad) const {
        const Vec<NIN> s = standardise(x);
        Vec<NHID> a; Vec<NOUT> g;
        forward_std(x, a, g);
        Vec<NOUT> e;
        Real sse = Real(0);
        for (int o = 0; o < NOUT; ++o) {
            const Real t = (y[o] - mu_out[o]) / sd_out[o];
            e[o] = g[o] - t;
            sse += e[o] * e[o];
        }
        for (int o = 0; o < NOUT; ++o) {
            grad.b2[o] += e[o];
            for (int j = 0; j < NHID; ++j) grad.W2(o, j) += e[o] * a[j];
        }
        for (int j = 0; j < NHID; ++j) {
            Real d = Real(0);
            for (int o = 0; o < NOUT; ++o) d += th.W2(o, j) * e[o];
            d *= (Real(1) - a[j] * a[j]);                 // tanh'(z) = 1 - tanh^2(z)
            grad.b1[j] += d;
            for (int i = 0; i < NIN; ++i) grad.W1(j, i) += d * s[i];
        }
        return sse;
    }

    // Estimate mu/sigma of the inputs and outputs from the training set
    // (Bishop 1995 §8.2). A floor keeps constant features from dividing by zero.
    void fit_standardisation(const Vec<NIN>* X, const Vec<NOUT>* Y, int n, Real floor_sd = Real(1e-6)) {
        if (n <= 0) return;
        for (int i = 0; i < NIN; ++i) { mu_in[i] = Real(0); sd_in[i] = Real(0); }
        for (int o = 0; o < NOUT; ++o) { mu_out[o] = Real(0); sd_out[o] = Real(0); }
        const Real inv = Real(1) / Real(n);
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < NIN; ++i) mu_in[i] += X[k][i] * inv;
            for (int o = 0; o < NOUT; ++o) mu_out[o] += Y[k][o] * inv;
        }
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < NIN; ++i) sd_in[i] += sq(X[k][i] - mu_in[i]) * inv;
            for (int o = 0; o < NOUT; ++o) sd_out[o] += sq(Y[k][o] - mu_out[o]) * inv;
        }
        for (int i = 0; i < NIN; ++i) sd_in[i] = std::max(std::sqrt(sd_in[i]), floor_sd);
        for (int o = 0; o < NOUT; ++o) sd_out[o] = std::max(std::sqrt(sd_out[o]), floor_sd);
    }
};

// -----------------------------------------------------------------------------
//  Adam (Kingma & Ba 2015), eq. (7). First and second moment estimates are kept
//  in two parameter blocks, so the optimiser state is also fixed-size.
// -----------------------------------------------------------------------------
template <int NIN, int NHID, int NOUT>
class Adam {
public:
    using Params = MlpParams<NIN, NHID, NOUT>;
    struct Options {
        Real lr    = Real(3e-3);    // step size alpha
        Real beta1 = Real(0.9);     // exponential decay of the 1st moment
        Real beta2 = Real(0.999);   // exponential decay of the 2nd moment
        Real eps   = Real(1e-8);    // numerical floor
        Real l2    = Real(0);       // decoupled L2 penalty on the weights
    } opts;

    Adam() { reset(); }
    void reset() { m_.zero(); v_.zero(); t_ = 0; }

    void step(Params& theta, const Params& g) {
        ++t_;
        const Real c1 = Real(1) - std::pow(opts.beta1, Real(t_));
        const Real c2 = Real(1) - std::pow(opts.beta2, Real(t_));
        for (int k = 0; k < Params::NPAR; ++k) {
            const Real gk = g.at(k) + opts.l2 * theta.at(k);
            Real& mk = m_.at(k);
            Real& vk = v_.at(k);
            mk = opts.beta1 * mk + (Real(1) - opts.beta1) * gk;
            vk = opts.beta2 * vk + (Real(1) - opts.beta2) * gk * gk;
            const Real mh = mk / c1, vh = vk / c2;
            theta.at(k) -= opts.lr * mh / (std::sqrt(vh) + opts.eps);
        }
    }
    int iterations() const { return t_; }

private:
    Params m_, v_;
    int t_ = 0;
};

// -----------------------------------------------------------------------------
//  Offline mini-batch training driver.
//
//  Sample order: instead of a heap-allocated permutation, each epoch visits the
//  samples along the cyclic sequence  idx_t = (o + t*s) mod n  with a random
//  offset o and a random stride s coprime to n.  This is a bijection of
//  {0..n-1} (a "cyclic shuffle"), so every sample is seen exactly once per
//  epoch, the order differs per epoch, and no dynamic memory is needed.
// -----------------------------------------------------------------------------
struct MlpTrainOptions {
    int  epochs = 10;
    int  batch  = 32;
    Real lr     = Real(3e-3);
    Real l2     = Real(0);
    // Geometric learning-rate decay: lr_e = lr * lr_final_frac^(e/(E-1)).  A decaying
    // step size is what makes the LAST iterate reproducible: with a constant Adam step
    // the parameters keep bouncing inside the minimum and the network one happens to
    // stop at depends on the epoch count.  Determinism of the delivered weights is a
    // requirement, not a nicety, when the artefact has to be qualified.
    Real lr_final_frac = Real(0.05);
    bool fit_standardisation = true;
};

inline int mlp_gcd(int a, int b) {
    while (b != 0) { const int t = a % b; a = b; b = t; }
    return a < 0 ? -a : a;
}

// Returns the mean squared error (standardised output units) of the final epoch.
template <int NIN, int NHID, int NOUT>
inline Real train_mlp(Mlp<NIN, NHID, NOUT>& net, const Vec<NIN>* X, const Vec<NOUT>* Y, int n,
                      const MlpTrainOptions& o, Rng& rng) {
    using Params = MlpParams<NIN, NHID, NOUT>;
    if (n <= 0) return Real(0);
    if (o.fit_standardisation) net.fit_standardisation(X, Y, n);
    Adam<NIN, NHID, NOUT> adam;
    adam.opts.lr = o.lr;
    adam.opts.l2 = o.l2;
    Params grad;
    Real mse = Real(0);
    const int batch = (o.batch < 1) ? 1 : (o.batch > n ? n : o.batch);
    for (int ep = 0; ep < o.epochs; ++ep) {
        adam.opts.lr = (o.epochs > 1)
            ? o.lr * std::pow(o.lr_final_frac, Real(ep) / Real(o.epochs - 1))
            : o.lr;
        int stride = 1, offset = 0;
        if (n > 2) {
            offset = int(rng.next_u64() % uint64_t(n));
            stride = 1 + int(rng.next_u64() % uint64_t(n - 1));
            while (mlp_gcd(stride, n) != 1) stride = (stride % (n - 1)) + 1;
        }
        Real sse = Real(0);
        int t = 0;
        while (t < n) {
            const int bs = (n - t < batch) ? (n - t) : batch;
            grad.zero();
            for (int b = 0; b < bs; ++b) {
                const int64_t pos = int64_t(offset) + int64_t(t + b) * int64_t(stride);
                const int idx = int(pos % int64_t(n));
                sse += net.accumulate_grad(X[idx], Y[idx], grad);
            }
            const Real inv = Real(1) / Real(bs);
            for (int k = 0; k < Params::NPAR; ++k) grad.at(k) *= inv;
            adam.step(net.th, grad);
            t += bs;
        }
        mse = sse / Real(n);
    }
    return mse;
}

}  // namespace estkit
