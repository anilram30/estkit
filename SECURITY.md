<!-- Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0 -->
# Security

estkit is a header-only numerical library with no network, file or process interface of its own;
the benchmark executables read only the files named on their command line. If you find a
memory-safety or numerical-safety problem (an out-of-bounds access, an unbounded recursion, a
NaN that escapes the safeguards described in the report), please open an issue or contact the
author through the repository. There is no bug bounty.
