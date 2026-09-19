// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// Minimal self-contained unit tests (no framework dependency).
#include <cstdio>
#include <cmath>
#include <vector>
#include "estkit/core/linalg.hpp"
#include "estkit/core/rng.hpp"
#include "estkit/core/model.hpp"
#include "estkit/battery/ecm_model.hpp"
#include "estkit/battery/plant_ecm.hpp"
#include "estkit/battery/plant_spm.hpp"
#include "estkit/battery/datasets.hpp"
#include "estkit/battery/cell_estimator.hpp"
#include "estkit/battery/metrics.hpp"

using namespace estkit;
static int g_fail = 0, g_pass = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } else ++g_pass; } while (0)
// tolerances are relaxed in the single-precision build
static double TOL(double t) { return (sizeof(estkit::Real) == 4) ? std::max(t, 2e-4) : t; }
#define CHECK_NEAR(a, b, tol) do { const double _a = double(a), _b = double(b); if (!(std::fabs(_a - _b) <= TOL(tol))) { std::printf("  FAIL %s:%d  %s=%g vs %s=%g (tol %g)\n", __FILE__, __LINE__, #a, _a, #b, _b, double(tol)); ++g_fail; } else ++g_pass; } while (0)

static void test_linalg() {
    std::printf("[linalg]\n");
    Mat<3, 3> A = { 4, 1, 2,  1, 5, 3,  2, 3, 6 };
    Mat<3, 3> Ai = inverse(A);
    Mat<3, 3> I = A * Ai;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) CHECK_NEAR(I(i, j), (i == j) ? 1.0 : 0.0, 1e-10);
    Mat<3, 3> L; CHECK(cholesky(A, L));
    Mat<3, 3> LLt = L * L.t();
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) CHECK_NEAR(LLt(i, j), A(i, j), 1e-10);
    // QR: R^T R = A^T A
    Mat<5, 3> B; Rng rng(3); for (int k = 0; k < 15; ++k) B.d[k] = rng.normal();
    Mat<3, 3> R = qr_r(B); Mat<3, 3> RtR = R.t() * R, BtB = B.t() * B;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) CHECK_NEAR(RtR(i, j), BtB(i, j), 1e-9);
    // rank-1 Cholesky update / downdate
    Vec<3> x = { 0.3, -0.2, 0.5 };
    Mat<3, 3> L2 = L; CHECK(chol_update(L2, x, Real(1)));
    Mat<3, 3> Aup = A + outer(x, x), L2L2 = L2 * L2.t();
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) CHECK_NEAR(L2L2(i, j), Aup(i, j), 1e-9);
    CHECK(chol_update(L2, x, Real(-1)));
    L2L2 = L2 * L2.t();
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) CHECK_NEAR(L2L2(i, j), A(i, j), 1e-9);
    // solve
    Vec<3> b = { 1, 2, 3 }; Vec<3> s = solve(A, b); Vec<3> r = A * s;
    for (int i = 0; i < 3; ++i) CHECK_NEAR(r[i], b[i], 1e-10);
    Vec<3> s2 = solve_spd(A, b); for (int i = 0; i < 3; ++i) CHECK_NEAR(s2[i], s[i], 1e-10);
    // Ackermann: place poles of (A - L C) at 0.5, 0.6, 0.7
    Mat<3, 3> Ad = { 1, 0.1, 0,  0, 1, 0.1,  0, 0, 1 }; RowVec<3> C = { 1, 0, 0 };
    Real poles[3] = { 0.5, 0.6, 0.7 }; Real coef[3]; poly_from_roots(poles, coef);
    CHECK_NEAR(coef[2], -(0.5 + 0.6 + 0.7), 1e-12); CHECK_NEAR(coef[0], -(0.5 * 0.6 * 0.7), 1e-12);
    Vec<3> Lg; CHECK(ackermann_observer(Ad, C, coef, Lg));
    Mat<3, 3> Acl = Ad - Lg * C;   // characteristic polynomial check via Cayley-Hamilton: phi(Acl) = 0
    Mat<3, 3> phi = mat_pow(Acl, 3) + Acl * Acl * coef[2] + Acl * coef[1] + Mat<3, 3>::identity() * coef[0];
    CHECK(phi.norm_inf() < TOL(1e-9));
    // conditioned pole placement on a 10 Hz battery-like pair (poles near 1)
    Mat<4, 4> Ab; Ab(0, 0) = 1; Ab(1, 1) = 0.98; Ab(2, 2) = 0.9992; Ab(3, 3) = 0.9997;
    RowVec<4> Cb = { 0.8, -1, -1, 0.005 }; Real pb[4] = { 0.990, 0.992, 0.994, 0.996 }; Vec<4> Lb;
    CHECK(ackermann_observer_poles(Ab, Cb, pb, Lb));
    Real cb[4]; poly_from_roots(pb, cb); Mat<4, 4> Aclb = Ab - Lb * Cb;
    Mat<4, 4> phib = mat_pow(Aclb, 4) + mat_pow(Aclb, 3) * cb[3] + mat_pow(Aclb, 2) * cb[2] + Aclb * cb[1] + Mat<4, 4>::identity() * cb[0];
    CHECK(phib.norm_inf() < TOL(1e-8));
    // DARE: doubling solution satisfies the Riccati equation and matches the fixed-point iteration
    Mat<3, 3> Qd = Mat<3, 3>::identity() * 1e-4; Mat<1, 1> Rd; Rd(0, 0) = 1e-2; Mat<3, 1> K;
    Mat<3, 3> P = dare_doubling(Ad, Mat<1, 3>(C), Qd, Rd, &K);
    Mat<3, 3> Pn = Ad * (P - K * Mat<1, 3>(C) * P) * Ad.t() + Qd;
    CHECK((Pn - P).norm_inf() < TOL(1e-8));
    Mat<3, 3> P2 = dare_iterate(Ad, Mat<1, 3>(C), Qd, Rd);
    CHECK((P2 - P).norm_inf() < TOL(1e-7));
    // battery-like pair with an integrator mode
    Mat<4, 4> Qb = Mat<4, 4>::identity() * 1e-8; Mat<1, 1> Rb; Rb(0, 0) = 4e-6; Mat<4, 1> Kb;
    Mat<4, 4> Pb = dare_doubling(Ab, Cb, Qb, Rb, &Kb);
    Mat<4, 4> Pbn = Ab * (Pb - Kb * Cb * Pb) * Ab.t() + Qb;
    CHECK(Pb.is_finite() && (Pbn - Pb).norm_inf() < TOL(1e-10) * (1 + Pb.norm_inf()) * 100);
}

static void test_model_jacobians() {
    std::printf("[ecm model]\n");
    EstimatorConfig c; EcmModel m(c);
    Vec<4> x = { 0.6325, 0.01, -0.005, 0.2 }; Vec<2> u = EcmModel::u_of(7.5, 300.0);
    if (sizeof(Real) == 8) {   // finite-difference checks are only meaningful in double precision
        Mat<4, 4> Fa = m.F(x, u), Fn = numeric_F(m, x, u);
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) CHECK_NEAR(Fa(i, j), Fn(i, j), 1e-6);
        Mat<1, 4> Ha = m.H(x, u), Hn = numeric_H(m, x, u);
        for (int j = 0; j < 4; ++j) CHECK_NEAR(Ha(0, j), Hn(0, j), 1e-5);
        Mat<4, 2> Ba = m.B(x, u), Bn = numeric_B(m, x, u);
        for (int i = 0; i < 4; ++i) CHECK_NEAR(Ba(i, 0), Bn(i, 0), 1e-6);
    }
    // OCV monotone and within physical range; closed-form vs table
    Real prev = -1; bool mono = true;
    for (int k = 0; k <= 100; ++k) { Real z = k / 100.0; Real v = Chen2020::ocv(z); if (v <= prev) mono = false; prev = v; CHECK_NEAR(m.ocv.ocv(z), v, 3e-3); }
    CHECK(mono);
    CHECK(Chen2020::ocv(0) > 2.7 && Chen2020::ocv(0) < 3.1);
    CHECK(Chen2020::ocv(1) > 4.1 && Chen2020::ocv(1) < 4.25);
    // analytic OCV slope
    if (sizeof(Real) == 8) for (int k = 1; k < 100; ++k) { Real z = k / 100.0; Real dn = (Chen2020::ocv(z + 1e-6) - Chen2020::ocv(z - 1e-6)) / 2e-6; CHECK_NEAR(Chen2020::docv_dz(z), dn, 1e-4 * std::fabs(dn) + 1e-6); }
    // lithium balance: SPM stoichiometry map consistent with Q_nom
    CHECK_NEAR(Chen2020::dx() * Chen2020::q_n() / 3600.0, Chen2020::Q_nom_Ah, 1e-9);
}

static void test_plants() {
    std::printf("[plants]\n");
    // ECM plant: 1C discharge for 30 min removes 0.5 of SOC (from Coulomb counting)
    EcmPlant p; p.enable_aging = false; p.reset(1.0, kTref, kTref);
    for (int k = 0; k < 18000; ++k) p.step(5.0, 0.1);
    CHECK_NEAR(p.soc(), 0.5, 1e-6);
    CHECK(p.temperature() > kTref && p.temperature() < kTref + 15);
    // rest: RC voltages decay, voltage returns to OCV
    for (int k = 0; k < 6000; ++k) p.step(0.0, 0.1);
    CHECK_NEAR(p.voltage(0.0), p.ocv_now() + p.hysteresis_voltage(), 1e-3);
    // SPM plant: 1C discharge for 30 min also removes 0.5 SOC; voltage stays in range
    SpmPlant<20> s; s.reset(1.0, kTref, kTref);
    Real vmin = 10, vmax = 0;
    for (int k = 0; k < 18000; ++k) { s.step(5.0, 0.1); Real v = s.voltage(5.0); vmin = std::min(vmin, v); vmax = std::max(vmax, v); }
    CHECK_NEAR(s.soc(), 0.5, 1e-4);
    CHECK(vmin > 3.0 && vmax < 4.25);
    CHECK(std::fabs(s.voltage(5.0) - Chen2020::ocv(0.5)) < 0.25);
    // relaxation towards the equilibrium OCV
    for (int k = 0; k < 36000; ++k) s.step(0.0, 0.1);
    CHECK_NEAR(s.voltage(0.0), Chen2020::ocv(0.5) + (s.temperature() - kTref) * entropic_dUdT(0.5), 5e-3);
    // Preisach: major loop symmetric, output bounded
    Preisach<32> pr; pr.init(0.02); pr.set_saturated(0.0, false);
    Real y1 = pr.update(1.0), y0 = pr.update(0.0);
    CHECK_NEAR(y1, 0.02, 1e-9); CHECK_NEAR(y0, -0.02, 1e-9);
    Real ymid_up = 0; pr.set_saturated(0.0, false); for (int k = 1; k <= 50; ++k) ymid_up = pr.update(k / 100.0);
    Real ymid_dn = 0; pr.set_saturated(1.0, true); for (int k = 99; k >= 50; --k) ymid_dn = pr.update(k / 100.0);
    CHECK(ymid_dn > ymid_up);   // raw relay output: descending branch above ascending (magnetic orientation)
    // in the plant the sign is flipped so that the CHARGING branch is above the discharging one (battery convention)
    EcmPlant pp; pp.enable_aging = false; pp.enable_thermal = false; pp.use_preisach = true; pp.preisach_M = 0.02;
    pp.reset(0.2, kTref, kTref);                                   // arrived from a discharge
    for (int k = 0; k < 10800; ++k) pp.step(-5.0, 0.1);            // charge 0.2 -> 0.5
    const Real v_charge_branch = pp.hysteresis_voltage();
    pp.reset(0.8, kTref, kTref);                                   // arrived from a charge
    for (int k = 0; k < 10800; ++k) pp.step(5.0, 0.1);             // discharge 0.8 -> 0.5
    CHECK(v_charge_branch > pp.hysteresis_voltage());
}

static void test_ekf_convergence() {
    std::printf("[EKF/UKF sanity]\n");
    Scenario sc; sc.name = "init_error"; sc.z0_est_offset = -0.35;
    Dataset d = make_dataset("ecm", mission_evtol(), sc, 1);
    EstimatorList ests = make_all_estimators();
    for (auto& e : ests) {
        if (std::string(e->name()) != "EKF" && std::string(e->name()) != "UKF") continue;
        e->reset(d.est_cfg);
        std::vector<Real> soc(d.n), vp(d.n);
        for (int k = 0; k < d.n; ++k) { e->step(d.i_meas[k], d.v_meas[k], d.T_meas[k]); soc[k] = e->soc(); vp[k] = e->voltage_pred(); }
        RunMetrics m = compute_metrics(d.soc_true, soc, d.v_true, vp, d.dt);
        std::printf("  %s rmse=%.4f ss=%.4f conv=%.1f\n", e->name(), m.rmse, m.rmse_ss, m.conv_time);
        CHECK(!m.diverged); CHECK(m.rmse_ss < 0.02); CHECK(m.conv_time >= 0 && m.conv_time < 900);
    }
}

int main() {
    test_linalg(); test_model_jacobians(); test_plants(); test_ekf_convergence();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
