## Changelog

### [v1.0.0] — 2026-04-08

Modernization release. The library now runs on Python ≥ 3.11 with Cython 3.x.

**Breaking changes:**
 - Dropped Python 2.7 and 3.4 support
 - Data file format changed from pickle to `.npz` (regenerate with `python -m pydgq.utils.precalc` if using custom settings)
 - Removed `setup.py`; now uses meson-python as build backend

**New:**
 - Build system: meson-python + PDM dev workflow
 - CI: GitHub Actions with Python 3.11–3.14 × Linux/macOS/Windows
 - Wheel publishing: cibuildwheel with trusted publisher to PyPI
 - Test suite: 79 pytest tests covering all integrators, kernel types, built-in kernels, convergence ordering, and cimport verification
 - Example Cython kernel compiled as part of the package (`pydgq.examples.example_kernel`)
 - Data file loading via `importlib.resources` (replaces deprecated `pkg_resources`)
 - Version: single source of truth in `pydgq/VERSION`
 - Dependabot for GitHub Actions version updates

**Fixed:**
 - Bug in `Linear2ndOrderKernel.compute()`: wrong index in matrix-vector product inner loop (`w_in[2*j]` → `w_in[2*k]`). Was invisible for diagonal matrices.

**Cython 3.x migration:**
 - `noexcept` added to all 63 `cdef nogil` method signatures (prevents silent GIL acquisition for exception checking)
 - `DEF` compile-time constants replaced with `cdef enum`
 - Language level set globally to 3 via meson
 - `_USE_MATH_DEFINES` for Windows `M_PI` compatibility

**Removed:**
 - Python 2.7 / 3.4 support and all compatibility code
 - `pydgq_data_27.bin` and `pydgq_data_34.bin` (replaced by `pydgq_data.npz`)
 - `pkg_resources` dependency
 - `pickle` dependency (data file is now `.npz`)

### [v0.1.2]
 - support both Python 3.4 and 2.7

### [v0.1.1]
 - setup.py is now Python 3 compatible
 - general tidiness:
   - added installation instructions to [README.md](README.md)
   - removed unused module `pydgq.utils.ptrwrap`
   - removed unused dependencies (oops)
   - use correct compile flags for all modules
   - updated [TODO.md](TODO.md) to reflect the current status of the project


### [v0.1.0]
 - initial release as separate project (cleaned up and generalized)
