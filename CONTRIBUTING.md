<!-- Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0 -->
# Contributing

Contributions are welcome — new estimators, new models, ports to other targets, corrections.

1. Read `docs/ESTIMATOR_API.md`: it is the contract every estimator in the library follows
   (model concept, filter interface, numerical rules, registration, benchmark, chapter).
2. One header per estimator under the family directory, templated on the model, with the
   primary references in the header comment. No heap allocation, no exceptions, no I/O in the
   step, `-Wall -Wextra -Wpedantic -Werror` clean on GCC and Clang, in `double` and in `float`.
3. Register it in `benchmarks/registry_<family>.cpp`, add a unit test, run
   `./build/bench_cell --only <Name> --scenario all` and include the numbers in the pull request.
4. Add the licence header (`scripts/add_license_headers.py` does it) and the references to
   `report/refs_additions_<family>.bib`.

By contributing you agree that your contribution is licensed under the Apache License 2.0
(code) or CC BY 4.0 (report material), like the rest of the project.
