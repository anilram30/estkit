// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  tests/package/use_estkit.cpp — the consumer smoke test for the installed
//  CMake package.  It is built by tests/package/CMakeLists.txt, which is a
//  *separate* project: it finds estkit with find_package() and links
//  estkit::estkit, exactly as a downstream user would.  Nothing here may refer
//  to the estkit source tree.
//
//  A 1-D constant-velocity target with a position measurement: enough to prove
//  that the headers were installed, that the include path and the C++17
//  requirement come through the imported target, and that a filter converges.
// =============================================================================
#include <cmath>
#include <cstdio>
#include "estkit/core/model.hpp"
#include "estkit/core/rng.hpp"
#include "estkit/filters/ekf.hpp"
#include "estkit/filters/ukf.hpp"
using namespace estkit;

struct ConstantVelocity {
    static constexpr int NX = 2, NU = 1, NY = 1;   // x = [position, velocity], u = [acceleration], y = [position]
    Real dt = Real(0.05);
    Real sigma_a = Real(0.20);                      // unmodelled acceleration
    Real sigma_p = Real(0.50);                      // position-sensor noise
    Vec<NX> f(const Vec<NX>& x, const Vec<NU>& u) const {
        Vec<NX> xn;
        xn[0] = x[0] + dt * x[1] + Real(0.5) * dt * dt * u[0];
        xn[1] = x[1] + dt * u[0];
        return xn;
    }
    Mat<NX, NX> F(const Vec<NX>&, const Vec<NU>&) const {
        Mat<NX, NX> Fm = Mat<NX, NX>::identity(); Fm(0, 1) = dt; return Fm;
    }
    Vec<NY> h(const Vec<NX>& x, const Vec<NU>&) const { Vec<NY> y; y[0] = x[0]; return y; }
    Mat<NX, NX> Q(const Vec<NX>&, const Vec<NU>&) const {
        Mat<NX, 1> G; G(0, 0) = Real(0.5) * dt * dt; G(1, 0) = dt;
        return G * G.t() * sq(sigma_a);
    }
    Mat<NY, NY> R(const Vec<NX>&, const Vec<NU>&) const { Mat<NY, NY> Rm; Rm(0, 0) = sq(sigma_p); return Rm; }
};

template <class Filter>
static Real rmse(Filter& flt, const ConstantVelocity& m, unsigned seed) {
    Rng rng(seed);
    Vec<2> x0; x0[0] = 0; x0[1] = 0;
    Mat<2, 2> P0; P0(0, 0) = Real(4); P0(1, 1) = Real(1);
    flt.init(x0, P0);
    Vec<2> truth; truth[0] = 0; truth[1] = Real(1.5);
    Vec<1> u; u[0] = 0;
    Real se = 0; const int n = 400;
    for (int k = 0; k < n; ++k) {
        truth = m.f(truth, u);
        Vec<1> y; y[0] = truth[0] + m.sigma_p * rng.normal();
        flt.predict(u);
        flt.update(y, u);
        se += sq(flt.x()[0] - truth[0]);
    }
    return std::sqrt(se / Real(n));
}

int main() {
    const ConstantVelocity m;
    Ekf<ConstantVelocity> ekf{m};
    Ukf<ConstantVelocity> ukf{m};
    const Real e = rmse(ekf, m, 11), u = rmse(ukf, m, 11);
    std::printf("estkit package smoke test: EKF position RMSE = %.4f m, UKF = %.4f m\n",
                double(e), double(u));
    // The filters must beat the raw sensor (0.50 m) by a clear margin and stay finite.
    const bool ok = std::isfinite(e) && std::isfinite(u) && e < Real(0.30) && u < Real(0.30);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
