// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/core/rng.hpp — deterministic pseudo-random numbers (xoshiro256**, Blackman &
//  Vigna 2018) with splitmix64 seeding, plus Gaussian sampling by the
//  Marsaglia polar method. Identical streams on every platform for a given seed,
//  so every benchmark run is exactly reproducible.
// =============================================================================
#pragma once
#include <cstdint>
#include <cmath>
#include "linalg.hpp"

namespace estkit {

class Rng {
public:
    explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) { reseed(seed); }
    void reseed(uint64_t seed) {
        uint64_t x = seed;
        for (int i = 0; i < 4; ++i) s_[i] = splitmix64(x);
        has_spare_ = false;
    }
    uint64_t next_u64() {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t; s_[3] = rotl(s_[3], 45);
        return result;
    }
    // uniform in [0,1)
    Real uniform() { return Real((next_u64() >> 11) * (1.0 / 9007199254740992.0)); }
    Real uniform(Real lo, Real hi) { return lo + (hi - lo) * uniform(); }
    // standard normal N(0,1)
    Real normal() {
        if (has_spare_) { has_spare_ = false; return spare_; }
        Real u, v, s;
        do { u = uniform(-1, 1); v = uniform(-1, 1); s = u * u + v * v; } while (s >= Real(1) || s == Real(0));
        const Real m = std::sqrt(Real(-2) * std::log(s) / s);
        spare_ = v * m; has_spare_ = true;
        return u * m;
    }
    Real normal(Real mean, Real stddev) { return mean + stddev * normal(); }
    template <int N> Vec<N> normal_vec() { Vec<N> v; for (int i = 0; i < N; ++i) v[i] = normal(); return v; }
    // sample x ~ N(mu, L L^T) given the lower Cholesky factor L
    template <int N> Vec<N> mvn(const Vec<N>& mu, const Mat<N, N>& L) { return mu + L * normal_vec<N>(); }

private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    static uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint64_t s_[4];
    bool has_spare_ = false;
    Real spare_ = 0;
};

}  // namespace estkit
