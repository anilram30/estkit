<!-- Copyright 2026 Sreeram Anil — SPDX-License-Identifier: CC-BY-4.0 -->
# Third-party notices

estkit itself has **no runtime dependency** beyond the C++17 standard library.
The software listed here is used by the benchmark tooling and by the validation
scripts as *independent reference implementations*; none of it is incorporated,
vendored, modified or redistributed with estkit. Each package is obtained by the
user from its own distribution under its own licence.

| Package | Used for | Licence | Source |
|---|---|---|---|
| FilterPy 1.4.x | independent cross-validation of the EKF, UKF and RTS smoother (`scripts/validate_filterpy.py`) | MIT | https://github.com/rlabbe/filterpy |
| PyBaMM 26.x | independent cross-validation of the single-particle plant (`scripts/validate_pybamm.py`) | BSD-3-Clause | https://github.com/pybamm-team/PyBaMM |
| ahrs 0.4.x | independent cross-validation of the Madgwick and Mahony attitude filters (`scripts/validate_ahrs.py`) | MIT | https://github.com/Mayitzin/ahrs |
| NumPy, pandas, matplotlib, Pillow | figure and table generation, validation scripts | BSD-3-Clause / BSD-3-Clause / PSF-based / MIT-CMU | https://numpy.org, https://pandas.pydata.org, https://matplotlib.org, https://python-pillow.org |
| CMake ≥ 3.16, GCC/Clang/MSVC | build | their own licences | — |
| TeX Live (pdflatex, bibtex, `lmodern`, `siunitx`, …) | building the report | LPPL and package licences | https://tug.org/texlive |

## Scientific data and models

The electrode open-circuit-potential functions and the electrode parameters of
the LG M50 cell used by the single-particle plant are the published values of
Chen et al. (2020), *Development of experimental techniques for parameterization
of multi-scale lithium-ion battery models*, J. Electrochem. Soc. 167, 080534,
re-implemented from the paper and cited. No parameter file from any other
software package is redistributed. Every other model parameter is either
derived from those values or a representative value chosen for this benchmark,
as labelled in the report's parameter appendix and in docs/PROVENANCE.md.

All algorithms are implemented from the publications cited in the source
headers and in docs/REFERENCES.md. Citing a publication does not place any
obligation on, or imply endorsement by, its authors.
