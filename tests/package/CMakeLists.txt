# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# A standalone downstream project: it consumes estkit through find_package()
# only, so configuring it against an install prefix proves that the exported
# CMake package is complete and usable.  CI builds and runs it; see
# .github/workflows/ci.yml, job `package`.
#
#   cmake -S . -B build --install-prefix /abs/prefix   (estkit installed there)
#   cmake -S tests/package -B pkg -DCMAKE_PREFIX_PATH=/abs/prefix
#   cmake --build pkg && ctest --test-dir pkg
cmake_minimum_required(VERSION 3.16)
project(estkit_package_test LANGUAGES CXX)

find_package(estkit 1.0 CONFIG REQUIRED)

add_executable(use_estkit use_estkit.cpp)
target_link_libraries(use_estkit PRIVATE estkit::estkit)   # brings the include path and cxx_std_17

enable_testing()
add_test(NAME use_estkit COMMAND use_estkit)

message(STATUS "estkit found: version ${estkit_VERSION}, config ${estkit_CONFIG}")
