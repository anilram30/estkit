// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  examples/drone_range_ekf_ukf.cpp — estkit on a model that has nothing to do
//  with batteries: a 2-D constant-velocity drone with a commanded acceleration
//  and a single range measurement to a beacon.  The same Ekf<> / Ukf<> /
//  SqrtUkf<> / BootstrapPf<> templates that produce the battery results are used
//  unchanged; the model provides f, h, Q, R (and F; H is supplied by central
//  differences).  Builds in double and float.
//
//  Run:  ./build/example_drone   (prints position RMSE of four estimators)
// =============================================================================
#include <cmath>
#include <cstdio>
#include <vector>
#include "estkit/core/model.hpp"
#include "estkit/core/rng.hpp"
#include "estkit/filters/ekf.hpp"
#include "estkit/filters/ukf.hpp"
#include "estkit/filters/srukf.hpp"
#include "estkit/particle/pf_bootstrap.hpp"
using namespace estkit;

struct RangeCV {
    static constexpr int NX = 4, NU = 2, NY = 1;   // x=[px,py,vx,vy], u=[ax,ay], y=range to beacon
    Real dt = Real(0.1);
    Real bx = Real(120), by = Real(-40);            // beacon position [m]
    Real sigma_a = Real(0.30);                      // unmodelled acceleration [m/s^2]
    Real sigma_r = Real(0.50);                      // range noise [m]
    Vec<NX> f(const Vec<NX>& x, const Vec<NU>& u) const {
        Vec<NX> xn;
        xn[0] = x[0] + dt * x[2] + Real(0.5) * dt * dt * u[0];
        xn[1] = x[1] + dt * x[3] + Real(0.5) * dt * dt * u[1];
        xn[2] = x[2] + dt * u[0];
        xn[3] = x[3] + dt * u[1];
        return xn;
    }
    Mat<NX, NX> F(const Vec<NX>&, const Vec<NU>&) const {
        Mat<NX, NX> Fm = Mat<NX, NX>::identity(); Fm(0, 2) = dt; Fm(1, 3) = dt; return Fm;
    }
    Vec<NY> h(const Vec<NX>& x, const Vec<NU>&) const {
        Vec<NY> y; y[0] = std::sqrt(sq(x[0] - bx) + sq(x[1] - by) + Real(1e-9)); return y;
    }
    Mat<NX, NX> Q(const Vec<NX>&, const Vec<NU>&) const {
        Mat<NX, 2> G; G(0, 0) = Real(0.5) * dt * dt; G(1, 1) = Real(0.5) * dt * dt; G(2, 0) = dt; G(3, 1) = dt;
        return G * G.t() * sq(sigma_a);
    }
    Mat<NY, NY> R(const Vec<NX>&, const Vec<NU>&) const { Mat<NY, NY> Rm; Rm(0, 0) = sq(sigma_r); return Rm; }
};

// A 60 s flight: cruise, a banked turn, cruise.  Truth is simulated with the same model plus noise.
template <class Filter>
static Real run(Filter& flt, const RangeCV& m, const std::vector<Vec<4>>& truth, const std::vector<Vec<2>>& cmd, const std::vector<Real>& range) {
    Vec<4> x0; x0[0] = 0; x0[1] = 0; x0[2] = 12; x0[3] = 0;
    Mat<4, 4> P0; P0(0, 0) = P0(1, 1) = Real(25); P0(2, 2) = P0(3, 3) = Real(4);
    flt.init(x0, P0);
    Real se = 0; bool have_prev = false; Vec<2> u_prev;
    for (size_t k = 0; k < truth.size(); ++k) {
        if (have_prev) flt.predict(u_prev);
        Vec<1> y; y[0] = range[k];
        flt.update(y, cmd[k]);
        se += sq(flt.x()[0] - truth[k][0]) + sq(flt.x()[1] - truth[k][1]);
        u_prev = cmd[k]; have_prev = true;
    }
    (void)m;
    return std::sqrt(se / Real(truth.size()));
}

int main() {
    RangeCV m; Rng rng(7);
    const int n = 600;
    std::vector<Vec<4>> truth(n); std::vector<Vec<2>> cmd(n); std::vector<Real> range(n);
    Vec<4> x; x[0] = 0; x[1] = 0; x[2] = 12; x[3] = 0;
    for (int k = 0; k < n; ++k) {
        const Real t = k * m.dt;
        Vec<2> u; u[0] = (t > 20 && t < 40) ? Real(-2) * std::sin(Real(0.3) * (t - 20)) : Real(0);
        u[1] = (t > 20 && t < 40) ? Real(2) * std::cos(Real(0.3) * (t - 20)) : Real(0);
        truth[k] = x; cmd[k] = u;
        range[k] = m.h(x, u)[0] + m.sigma_r * rng.normal();
        Vec<2> eta; eta[0] = u[0] + m.sigma_a * rng.normal(); eta[1] = u[1] + m.sigma_a * rng.normal();
        x = m.f(x, eta);
    }
    Ekf<RangeCV> ekf(m);
    Ukf<RangeCV> ukf(m); ukf.opts.alpha = Real(0.5);
    SqrtUkf<RangeCV> srukf(m); srukf.opts.alpha = Real(0.5);
    BootstrapPf<RangeCV, 500> pf(m);
    std::printf("2-D drone, beacon range only, %d samples at %.0f Hz\n", n, 1 / m.dt);
    std::printf("  EKF          position RMSE = %6.2f m   (%zu bytes)\n", run(ekf, m, truth, cmd, range), sizeof(ekf));
    std::printf("  UKF          position RMSE = %6.2f m   (%zu bytes)\n", run(ukf, m, truth, cmd, range), sizeof(ukf));
    std::printf("  SR-UKF       position RMSE = %6.2f m   (%zu bytes)\n", run(srukf, m, truth, cmd, range), sizeof(srukf));
    std::printf("  PF-Bootstrap position RMSE = %6.2f m   (%zu bytes)\n", run(pf, m, truth, cmd, range), sizeof(pf));
    return 0;
}
