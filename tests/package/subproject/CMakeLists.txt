# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# The second downstream path: estkit pulled into another build with
# add_subdirectory() or FetchContent_MakeAvailable().  In that mode estkit must
# define the library target and nothing else — no benchmarks, no tests, no
# examples in the consumer's build.  CI builds this; see the `package` job.
#
#   cmake -S tests/package/subproject -B sub && cmake --build sub && ctest --test-dir sub
cmake_minimum_required(VERSION 3.16)
project(estkit_subproject_test LANGUAGES CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../.. estkit_build)

add_executable(use_estkit_subproject ${CMAKE_CURRENT_SOURCE_DIR}/../use_estkit.cpp)
target_link_libraries(use_estkit_subproject PRIVATE estkit::estkit)

enable_testing()
add_test(NAME use_estkit_subproject COMMAND use_estkit_subproject)
