# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(estkit LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(ESTKIT_FLOAT "Build with single-precision Real (float) to study MCU numerics" OFF)
option(ESTKIT_WERROR "Treat warnings as errors" ON)
option(ESTKIT_NATIVE "Use -march=native" OFF)

# ---- header-only library target ----
add_library(estkit INTERFACE)
add_library(estkit::estkit ALIAS estkit)
target_include_directories(estkit INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(estkit INTERFACE cxx_std_17)
if(ESTKIT_FLOAT)
  target_compile_definitions(estkit INTERFACE ESTKIT_REAL=float)
endif()

if(MSVC)
  set(ESTKIT_WARN /W4 /permissive- /Zc:__cplusplus /EHsc)
  if(ESTKIT_WERROR)
    list(APPEND ESTKIT_WARN /WX)
  endif()
  set(ESTKIT_OPT /O2)
else()
  set(ESTKIT_WARN -Wall -Wextra -Wpedantic -Wno-unused-parameter)
  if(ESTKIT_WERROR)
    list(APPEND ESTKIT_WARN -Werror)
  endif()
  if(ESTKIT_NATIVE)
    list(APPEND ESTKIT_WARN -march=native)
  endif()
  set(ESTKIT_OPT -O2)
endif()
option(ESTKIT_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(ESTKIT_SANITIZE AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- registries (one translation unit per estimator family) ----
set(ESTKIT_REGISTRY_SOURCES
  benchmarks/registry_baselines.cpp benchmarks/registry_kalman.cpp benchmarks/registry_adaptive_kalman.cpp benchmarks/registry_robust_kalman.cpp
  benchmarks/registry_observers.cpp benchmarks/registry_soh.cpp benchmarks/registry_particle.cpp
  benchmarks/registry_learning.cpp benchmarks/registry_optimization.cpp benchmarks/registry_smoothers.cpp
  benchmarks/registry_pack.cpp benchmarks/registry_attitude.cpp benchmarks/cell_params_fit.cpp)
add_library(estkit_registry STATIC ${ESTKIT_REGISTRY_SOURCES})
target_link_libraries(estkit_registry PUBLIC estkit)
target_compile_options(estkit_registry PRIVATE ${ESTKIT_WARN} ${ESTKIT_OPT})

# ---- benchmark executables ----
function(estkit_exe name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE estkit estkit_registry)
  target_compile_options(${name} PRIVATE ${ESTKIT_WARN} ${ESTKIT_OPT})
endfunction()
estkit_exe(bench_cell benchmarks/bench_cell.cpp)
estkit_exe(bench_pack benchmarks/bench_pack.cpp)
estkit_exe(bench_imu benchmarks/bench_imu.cpp)
estkit_exe(bench_soh benchmarks/bench_soh.cpp)

# ---- tests ----
enable_testing()
add_executable(unit_tests tests/unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE estkit estkit_registry)
target_compile_options(unit_tests PRIVATE ${ESTKIT_WARN} ${ESTKIT_OPT})
add_test(NAME unit_tests COMMAND unit_tests)

# ---- examples (built by default; also run as tests) ----
option(ESTKIT_BUILD_EXAMPLES "Build the examples" ON)
if(ESTKIT_BUILD_EXAMPLES)
  add_executable(example_drone examples/drone_range_ekf_ukf.cpp)
  target_link_libraries(example_drone PRIVATE estkit)
  target_compile_options(example_drone PRIVATE ${ESTKIT_WARN} ${ESTKIT_OPT})
  add_test(NAME example_drone COMMAND example_drone)
  add_executable(example_battery examples/battery_soc_quickstart.cpp)
  target_link_libraries(example_battery PRIVATE estkit estkit_registry)
  target_compile_options(example_battery PRIVATE ${ESTKIT_WARN} ${ESTKIT_OPT})
  add_test(NAME example_battery COMMAND example_battery combined)
  add_executable(example_embedded examples/embedded_cortex_m7.cpp)
  target_link_libraries(example_embedded PRIVATE estkit)
  target_compile_options(example_embedded PRIVATE ${ESTKIT_WARN} ${ESTKIT_OPT})
  add_test(NAME example_embedded COMMAND example_embedded)
endif()

# ---- tools ----
estkit_exe(ident_spm_ecm tools/ident_spm_ecm.cpp)
estkit_exe(export_validation tools/export_validation.cpp)
estkit_exe(export_model_curves tools/export_model_curves.cpp)
