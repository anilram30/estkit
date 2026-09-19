// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/pack/pack_estimator.hpp — interface for pack-level (multi-cell) SOC
//  estimators plus the reference "decentralised" implementation (one EKF per
//  cell, no information exchange — Plett 2016 Vol. II §4.2).
// =============================================================================
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "pack_dataset.hpp"
#include "../battery/cell_estimator.hpp"
#include "../filters/ekf.hpp"

namespace estkit {

class PackEstimator {
public:
    virtual ~PackEstimator() = default;
    virtual const char* name() const = 0;
    virtual const char* group() const = 0;
    virtual void reset(const EstimatorConfig& cfg) = 0;                       // same config for every cell
    virtual void step(Real i, const Real* v, const Real* T) = 0;              // v[kPackCells], T[kPackCells]
    virtual Real cell_soc(int c) const = 0;
    virtual Real pack_soc_min() const { Real m = 2; for (int c = 0; c < kPackCells; ++c) m = std::min(m, cell_soc(c)); return m; }
    virtual Real pack_soc_max() const { Real m = -1; for (int c = 0; c < kPackCells; ++c) m = std::max(m, cell_soc(c)); return m; }
    virtual Real pack_soc_mean() const { Real s = 0; for (int c = 0; c < kPackCells; ++c) s += cell_soc(c); return s / kPackCells; }
    virtual size_t state_bytes() const = 0;
    virtual int messages_per_step() const { return 0; }   // communication cost (numbers exchanged between nodes)
};

// One independent generic filter per cell (baseline).
template <class Filter, class Model = EcmModel>
class DecentralizedPack final : public PackEstimator {
public:
    explicit DecentralizedPack(std::string name, std::string group = "pack") : name_(std::move(name)), group_(std::move(group)) {
        for (int c = 0; c < kPackCells; ++c) cells_.emplace_back(std::make_unique<CellEstimator<Filter, Model>>(name_, group_));
    }
    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }
    void reset(const EstimatorConfig& cfg) override { for (auto& c : cells_) c->reset(cfg); }
    void step(Real i, const Real* v, const Real* T) override { for (int c = 0; c < kPackCells; ++c) cells_[c]->step(i, v[c], T[c]); }
    Real cell_soc(int c) const override { return cells_[c]->soc(); }
    size_t state_bytes() const override { return kPackCells * sizeof(Filter); }
private:
    std::string name_, group_;
    std::vector<std::unique_ptr<CellEstimator<Filter, Model>>> cells_;
};

using PackList = std::vector<std::unique_ptr<PackEstimator>>;
void register_pack(PackList&);

inline PackList make_all_pack_estimators() {
    PackList list;
    list.push_back(std::make_unique<DecentralizedPack<Ekf<EcmModel>>>("Decentralized-EKF"));
    register_pack(list);
    return list;
}

}  // namespace estkit
