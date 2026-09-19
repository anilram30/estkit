// Copyright 2026 Sreeram Anil
// SPDX-License-Identifier: Apache-2.0
// =============================================================================
//  estkit/battery/cell_estimator.hpp — adapter that turns any generic estkit
//  filter/observer (written against the model concept) into a benchmark
//  `Estimator` for the cell-level SOC problem, plus the estimator registry.
//
//  Generic filter interface expected (see docs/ESTIMATOR_API.md):
//     explicit Filter(const Model& m);          // tuning through public member `opts`
//     void init(const X& x0, const Mat<NX,NX>& P0);
//     void predict(const U& u);                 // time update with u_{k-1}
//     void update(const Y& y, const U& u);      // measurement update with u_k
//     Y    predict_measurement(const U& u) const;   // h(x^-, u), called before update()
//     const X& x() const;                       // state estimate
//     const Mat<NX,NX>& P() const;              // (optional) covariance
// =============================================================================
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "ecm_model.hpp"

namespace estkit {

template <class F, class = void> struct has_P : std::false_type {};
template <class F> struct has_P<F, std::void_t<decltype(std::declval<const F&>().P())>> : std::true_type {};

template <class Filter, class Model = EcmModel>
class CellEstimator final : public Estimator {
public:
    using Configure = std::function<void(Filter&, const EstimatorConfig&)>;
    using Hook = std::function<Real(const Filter&)>;

    CellEstimator(std::string name, std::string group, Configure configure = {})
        : name_(std::move(name)), group_(std::move(group)), configure_(std::move(configure)), model_(), filter_(model_) {}

    const char* name() const override { return name_.c_str(); }
    const char* group() const override { return group_.c_str(); }
    size_t state_bytes() const override { return sizeof(Filter); }

    void reset(const EstimatorConfig& c) override {
        model_ = Model(c);
        filter_ = Filter(model_);
        if (configure_) configure_(filter_, c);
        filter_.init(model_.x0(c), model_.P0(c));
        have_prev_ = false; y_pred_ = kNaN;
    }
    void step(Real i, Real v, Real T) override {
        const Vec<Model::NU> u = Model::u_of(i, T);
        if (have_prev_) filter_.predict(u_prev_);
        y_pred_ = filter_.predict_measurement(u)[0];
        Vec<Model::NY> y; y[0] = v;
        filter_.update(y, u);
        u_prev_ = u; have_prev_ = true;
    }
    Real soc() const override { return filter_.x()[Model::IZ]; }
    Real soc_std() const override {
        if constexpr (has_P<Filter>::value) { const Real p = filter_.P()(Model::IZ, Model::IZ); return p >= 0 ? std::sqrt(p) : Real(-1); }
        else return Real(-1);
    }
    Real voltage_pred() const override { return y_pred_; }
    Real capacity_Ah() const override { return capacity_hook_ ? capacity_hook_(filter_) : kNaN; }
    Real soc_lower() const override { return lower_hook_ ? lower_hook_(filter_) : kNaN; }
    Real soc_upper() const override { return upper_hook_ ? upper_hook_(filter_) : kNaN; }

    CellEstimator& with_capacity(Hook h) { capacity_hook_ = std::move(h); return *this; }
    CellEstimator& with_bounds(Hook lo, Hook hi) { lower_hook_ = std::move(lo); upper_hook_ = std::move(hi); return *this; }
    Filter& filter() { return filter_; }
    const Filter& filter() const { return filter_; }

private:
    std::string name_, group_;
    Configure configure_;
    Hook capacity_hook_, lower_hook_, upper_hook_;
    Model model_;
    Filter filter_;
    Vec<Model::NU> u_prev_;
    bool have_prev_ = false;
    Real y_pred_ = kNaN;
};

// -----------------------------------------------------------------------------
//  Registry: every estimator family registers itself in its own translation unit
//  (benchmarks/registry_*.cpp) through one of these functions; the benchmark collects all.
// -----------------------------------------------------------------------------
using EstimatorList = std::vector<std::unique_ptr<Estimator>>;

template <class Filter, class Model = EcmModel>
inline CellEstimator<Filter, Model>& add_cell_estimator(EstimatorList& list, std::string name, std::string group,
                                                        typename CellEstimator<Filter, Model>::Configure cfg = {}) {
    auto e = std::make_unique<CellEstimator<Filter, Model>>(std::move(name), std::move(group), std::move(cfg));
    auto& ref = *e; list.push_back(std::move(e)); return ref;
}

void register_baselines(EstimatorList&);
void register_kalman(EstimatorList&);
void register_adaptive_kalman(EstimatorList&);
void register_robust_kalman(EstimatorList&);
void register_observers(EstimatorList&);
void register_soh(EstimatorList&);
void register_particle(EstimatorList&);
void register_learning(EstimatorList&);
void register_optimization(EstimatorList&);
void register_smoothers(EstimatorList&);

inline EstimatorList make_all_estimators() {
    EstimatorList list;
    register_baselines(list);
    register_kalman(list);
    register_adaptive_kalman(list);
    register_robust_kalman(list);
    register_observers(list);
    register_soh(list);
    register_particle(list);
    register_learning(list);
    register_optimization(list);
    register_smoothers(list);
    return list;
}

}  // namespace estkit
