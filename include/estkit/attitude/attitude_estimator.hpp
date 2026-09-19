// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/attitude/attitude_estimator.hpp — interface for attitude (AHRS)
//  estimators and the registry.  Implementations: complementary filter,
//  Madgwick, Mahony, multiplicative EKF, invariant EKF, TRIAD/QUEST (see
//  include/estkit/attitude/*.hpp).
// =============================================================================
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "../core/quaternion.hpp"
#include "imu_dataset.hpp"

namespace estkit {

struct AttitudeConfig {
    Real dt = Real(0.01);
    Quat q0;                        // initial attitude estimate (body -> nav)
    Vec<3> mag_ref_nav;             // reference magnetic field (nav frame, uT)
    Real g = Real(9.80665);
    Real gyro_noise = Real(0.1 * kPi / 180);
    Real accel_noise = Real(0.05);
    Real mag_noise = Real(0.5);
    Real gyro_bias_rw = Real(0.002 * kPi / 180);
    uint64_t seed = 1;
};

class AttitudeEstimator {
public:
    virtual ~AttitudeEstimator() = default;
    virtual const char* name() const = 0;
    virtual const char* group() const = 0;
    virtual void reset(const AttitudeConfig& cfg) = 0;
    virtual void step(const Vec<3>& gyro, const Vec<3>& accel, const Vec<3>& mag) = 0;
    virtual Quat quaternion() const = 0;                       // body -> nav
    virtual Vec<3> gyro_bias() const { return Vec<3>(); }      // estimated gyro bias (if any)
    virtual size_t state_bytes() const = 0;
};

using AttitudeList = std::vector<std::unique_ptr<AttitudeEstimator>>;
void register_attitude(AttitudeList&);
inline AttitudeList make_all_attitude_estimators() { AttitudeList l; register_attitude(l); return l; }

}  // namespace estkit
