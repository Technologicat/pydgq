# pydgq Modernization Brief

*For Claude Code. April 2026.*

## 1. Overview

pydgq is a Cython-accelerated ODE solver library using the time-discontinuous Galerkin method with Lobatto basis. It was written in 2016–2017 for Python 2.7/3.4. The goal is to modernize it to run on Python ≥3.11, with modern build tooling, CI, and a PyPI release as v1.0.0.

**Repository:** `~/Documents/koodit/pydgq`
**Current version:** 0.1.2 (April 2017)
**License:** BSD-2-Clause

### 1.1 Dependencies

- **PyLU** — Cython LU solver, used via `cimport` in the Galerkin integrator. **Already modernized** (v1.0.0, April 2026). Available on PyPI (`pip install pylu`). Source at `~/Documents/koodit/pylu` if needed for reference.
- **NumPy** — array operations, typed memoryviews
- **SymPy + mpmath** — only for precalculating Lobatto basis function data (dev dependencies, not runtime). The precalc module currently imports `sympy.mpmath`; update to `import mpmath` (standalone package, works on Python 3.11+).
- **Matplotlib** — only for example scripts (dev dependency)

### 1.2 What this project is NOT like PyLU

Read this section carefully. pydgq is substantially more complex than PyLU.

1. **Multi-package structure.** Three packages: `pydgq`, `pydgq.solver`, `pydgq.utils`. Each subpackage needs its own `meson.build`. PyLU had one package with one extension.

2. **Many Cython extensions (~10).** `types`, `compsum`, `kernel_interface`, `builtin_kernels`, `integrator_interface`, `explicit`, `implicit`, `galerkin`, `odesolve` (in `pydgq.solver`), and `discontify` (in `pydgq.utils`). They have inter-module dependencies (e.g. `galerkin` cimports `types`, `integrator_interface`, and `pylu.dgesv`). Build order matters.

3. **cdef classes (extension types).** PyLU only had `cdef inline` functions. pydgq uses `cdef class` with inheritance hierarchies: IntegratorBase → ExplicitIntegrator / ImplicitIntegrator → GalerkinIntegrator, and KernelBase → CythonKernel / PythonKernel. These interact with Cython 3.x differently than plain functions — see section 4.

4. **Binary data file.** `pydgq_data.bin` contains precalculated Lobatto basis function values. Currently pickle format, loaded via `pkg_resources` (deprecated). Must be migrated — see section 5.

5. **Cross-package cimport.** `galerkin.pyx` does `cimport pylu.dgesv`. This works because PyLU 1.0.0 installs its `.pxd` files. PyLU must be listed as both a build and runtime dependency.

6. **Existing test structure.** Tests are standalone scripts in `test/`, not pytest. Includes a Cython example kernel with its own `setup.py` for compilation. Needs careful handling — see section 7.

7. **Precision type aliases.** `pydgq.solver.types` defines `RTYPE`, `ZTYPE`, `DTYPE` with `_t` suffixed Cython variants, allowing compile-time precision switching. Don't alter this system.

---

## 2. Build System Migration

### 2.1 From setup.py to meson-python + PDM

Replace `setup.py` with meson-python as build backend and PDM for dev workflow (same pattern as PyLU — see `~/Documents/koodit/pylu/pyproject.toml` and its README for reference):

- `pyproject.toml` — package metadata, build system declaration, tool config, PDM dev deps
- `meson.build` (top-level) — project declaration, subdir calls
- `pydgq/meson.build` — `__init__.py`, data files, subdir calls
- `pydgq/solver/meson.build` — all solver `.pyx` extensions, `.pxd` installs
- `pydgq/utils/meson.build` — utils `.pyx` extensions (if any), `.pxd` installs

Dev workflow:
```bash
pdm install                              # creates venv, installs dev deps
pip install --no-build-isolation -e .     # editable install via meson-python
```

The `--no-build-isolation` flag is required for editable installs with meson-python — the on-import rebuild mechanism needs build dependencies to remain available in the environment.

Follow the PyLU pattern:

```toml
[build-system]
requires = ["meson-python>=0.17", "Cython>=3.0", "numpy>=1.25", "pylu>=1.0"]
build-backend = "mesonpy"
```

Note `pylu` in build-system requires — needed for the `cimport` during compilation.

### 2.2 Version: single source of truth

Use the VERSION file pattern from PyLU:
- `pydgq/VERSION` — single plain-text file containing the version string
- `meson.build` — `version: files('pydgq/VERSION')`
- `pyproject.toml` — `dynamic = ["version"]`
- `__init__.py` — reads VERSION at import time, same as PyLU:

```python
from pathlib import Path as _Path
__version__ = (_Path(__file__).parent / "VERSION").read_text().strip()
```

### 2.3 Compiler flags

The current `setup.py` uses `-march=native -O2 -msse -msse2 -mfma -mfpmath=sse` for math-heavy extensions, and `-O2` for non-math extensions. It does NOT use `-ffast-math` (correctly — `-ffast-math` causes precision issues in numerical code).

**For wheels (CI builds):** Do NOT use `-march=native`, `-mfma`, or other arch-specific flags. Use meson's `buildtype=release` (gives `-O2`).

**For source builds:** Document how users can add `-march=native -mfma` via `CFLAGS`, as in PyLU's README.

In `meson.build`, the non-portable flags are omitted — just use meson's default release optimization. The `libm` linkage (`libraries=["m"]`) is handled by meson's `cc.find_library('m', required: false)`.

### 2.4 pyproject.toml

```toml
[project]
name = "pydgq"
dynamic = ["version"]
description = "ODE system solver using dG(q), time-discontinuous Galerkin with Lobatto basis"
license = "BSD-2-Clause"
requires-python = ">=3.11"
authors = [{name = "Juha Jeronen", email = "juha.jeronen@jamk.fi"}]
dependencies = [
    "numpy>=1.25",
    "pylu>=1.0",
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Programming Language :: Cython",
    "Operating System :: OS Independent",
]

[project.optional-dependencies]
test = ["pytest>=8.0"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "Cython>=3.0",
    "cython-lint",
    "pip",
    "sympy",
    "mpmath",
    "matplotlib",
    "flake8",
    "autopep8",
    "importmagic",
    "epc",
    "jedi>=0.19.2",
]
```

Note: Do NOT use the SPDX `license` field together with a `License ::` classifier. Use SPDX only.

---

## 3. Python 2 Removal

Mechanical changes:

- Remove all `from __future__ import division, print_function, absolute_import`
- Remove `cPickle` / `pickle` conditionals — just `import pickle`
- Remove Python 2/3 data file symlink dance in `setup.py` (setup.py itself is being deleted)
- Remove `pydgq_data_27.bin` — only the Python 3 version is needed
- Rename `pydgq_data_34.bin` → `pydgq_data.npz` (see section 5)
- Update any `sys.version_info` checks
- Remove Python 2 classifiers from metadata

---

## 4. Cython 3.x Migration

### 4.1 Known required changes (apply to all `.pyx` / `.pxd` files)

For reference, the PyLU modernization brief is at `~/Documents/koodit/pylu/briefs/pylu-modernization-brief.md`. The same class of issues applies here, plus cdef-class-specific items documented in `briefs/cython3-cdef-class-report.md`.

**Language level directive.** Add `# cython: language_level=3` to the top of every `.pyx` and `.pxd` file, OR set it globally in `meson.build` via Cython compiler args. Prefer the global setting to avoid missing a file.

**`noexcept` on cdef functions/methods.** Every `cdef` function and `cdef` method that does not raise Python exceptions needs `noexcept`. In Cython 3, the default changed — without `noexcept`, Cython assumes the function *can* raise, adding overhead for exception checking in `nogil` blocks.

Audit every `cdef` function signature in all `.pxd` and `.pyx` files. The pattern:
- If the function is `nogil` and doesn't raise → add `noexcept`
- If the function can raise (calls Python, does allocation) → leave as-is or add `except *`

**`DEF` / `IF` directives.** If pydgq uses `DEF` for compile-time constants, replace with module-level `cdef` constants or `enum`. Check all files.

**`from __future__` in `.pyx`.** Remove `from __future__ import` lines. Cython 3 with `language_level=3` handles this.

### 4.2 cdef class specific issues

See `briefs/cython3-cdef-class-report.md` for the full investigation with test cases (reusable for python-wlsqm).

**CRITICAL — noexcept on nogil methods.** Without `noexcept`, every cdef method call in a `nogil` block silently acquires the GIL for exception checking. This is a **silent performance regression** — code compiles and runs correctly, just slower. Cython 3 emits a hint: *"Exception check after calling X will always require the GIL to be acquired."* Audit the entire solver call chain from `odesolve.ivp()` downward. Every `cdef` method in `nogil` paths needs `noexcept nogil`.

**noexcept must match in inheritance.** If a base class method has `noexcept`, all subclass overrides must also have `noexcept` (compile error otherwise). The reverse is allowed — child can add `noexcept` that base doesn't have. Work top-down: add `noexcept` to IntegratorBase/KernelBase first, then to all subclasses.

**`__cinit__` signatures.** Cython calls ALL `__cinit__` methods up the MRO with the SAME arguments. If `Base.__cinit__` doesn't accept `*args, **kwargs` and a subclass constructor adds parameters, you get a runtime TypeError. Check IntegratorBase and KernelBase `__cinit__` signatures.

**Old-style properties.** Convert Cython-specific `property x: def __get__(self):` syntax to standard Python `@property` decorators. Both work, but the decorator syntax is standard Python — same in both languages.

**cpdef methods.** Work fine in Cython 3, no changes needed.

### 4.3 Things to NOT change

- **Do not rewrite numerical algorithms.** The integrators, kernels, and Banach/Picard iteration are correct and tested. Modernization is infrastructure only.
- **Do not change the type alias system** (`RTYPE`, `DTYPE`, etc.). It exists for a reason (future complex-valued support).
- **Do not convert raw pointer APIs to memoryviews.** The pointer-based interfaces exist for performance in `nogil` blocks.
- **Do not change the `.pxd` / `.pyx` split.** Implementations in `.pxd` (as `cdef inline`) is intentional — same pattern as PyLU.
- **Do not touch the precalc algorithm.** It uses extended precision (via SymPy/mpmath) to work around catastrophic cancellation in the Lobatto basis function evaluation. The approach is intentionally "easy but needs extra precision" rather than "numerically stable but complex."

---

## 5. Data File Migration

### 5.1 Format change: pickle → npz

The current `pydgq_data.bin` is a pickled dict of NumPy arrays. Change to `pydgq_data.npz` (NumPy's native compressed archive format).

**Why:** Pickle files are Python-version-sensitive (hence the `_27` / `_34` split). `.npz` is stable across Python versions and architectures.

**Context on the precalculated data:** The Lobatto basis function evaluation uses a mathematically straightforward but numerically fragile approach that suffers from catastrophic cancellation. Rather than rewriting with a numerically stable algorithm, the basis functions are evaluated once using extended-precision arithmetic (via mpmath), producing double-precision results with full significance. The data file stores these precomputed values. Don't alter the precalc algorithm — the "easy approach + extra precision" tradeoff is intentional.

**Migration steps:**

1. Try loading the existing `pydgq_data_34.bin` pickle on the current Python — if it loads, convert directly to `.npz`. If it doesn't (pickle compatibility issue), regenerate from `precalc.py` first.
2. Update `precalc.py` to save as `.npz` instead of pickle
3. Update the data loading code (in `galerkin.pyx` or wherever `pkg_resources` is used) to load `.npz`
4. Ship `pydgq_data.npz` in the source tree
5. Delete `pydgq_data_27.bin` and `pydgq_data_34.bin`

### 5.2 Loading path: pkg_resources → importlib.resources

Replace `pkg_resources.resource_filename(...)` with `importlib.resources` (available since Python 3.9, stable API since 3.11).

Note: `importlib.resources.files()` returns a `Traversable` object — an abstract interface for "things that might be files" (could be in a zip, could be on disk). To get a real filesystem path for `np.load()`, use `as_file()` as a context manager:

```python
from importlib.resources import files, as_file

with as_file(files("pydgq") / "pydgq_data.npz") as path:
    data = np.load(path)
```

The three-tier search path (local override, user config dir, package) can be preserved:

```python
import os
from pathlib import Path
from importlib.resources import files, as_file

def find_data_file():
    """Find pydgq_data.npz, checking local, user config, and package locations."""
    # 1. Local override
    local = Path("pydgq_data.npz")
    if local.exists():
        return local

    # 2. User config override
    user = Path.home() / ".config" / "pydgq" / "pydgq_data.npz"
    if user.exists():
        return user

    # 3. Installed package data (returns a Traversable; use as_file at call site)
    return files("pydgq") / "pydgq_data.npz"
```

**Note:** The data loading currently happens inside Cython code (`galerkin.pyx`). Since `importlib.resources` is pure Python, the loading wrapper should be in a `.py` file (e.g., `pydgq/data.py`), called from the Cython code via a Python function call during `galerkin.init()`.

### 5.3 meson.build: install the data file

```meson
# In pydgq/meson.build
py.install_sources(
    'pydgq_data.npz',
    subdir: 'pydgq',
)
```

---

## 6. CI and Publishing

### 6.1 Follow the PyLU pattern exactly

See `~/Documents/koodit/pylu/.github/workflows/ci.yml` for the reference implementation.

- **Lint job:** cython-lint on all `.pyx` / `.pxd` files, with appropriate ignores for numerical code style
- **Test job:** Python 3.11–3.14 × Linux/macOS/Windows
- **Build wheels:** cibuildwheel ≥3.4 (for Python 3.14 support)
- **Publish job:** trusted publisher via GitHub, triggers on `v*` tags

### 6.2 cibuildwheel config

```toml
[tool.cibuildwheel]
build = "cp311-* cp312-* cp313-* cp314-*"
test-command = "pytest {project}/tests -v"
test-requires = ["pytest", "numpy", "pylu"]
```

### 6.3 Trusted publisher

Set up on PyPI for `pydgq` package, workflow `ci.yml`, environment `pypi`. Same procedure as PyLU — see `~/.claude/CI-SETUP-NOTES.md` (if available; may need updating on this machine).

---

## 7. Test Migration

### 7.1 Convert to pytest

The existing `test/` directory contains standalone scripts:
- `python_kernel_test.py` — tests using PythonKernel (pure Python RHS)
- `cython_kernel_test.py` — tests using CythonKernel (needs compilation)
- `lorenz_example.py` — Lorenz attractor demo
- Other example scripts

**Strategy:**
- Move to `tests/` (pytest convention)
- Convert the Python kernel tests to proper pytest with assertions
- The Lorenz example and other visualization scripts → `examples/` (not run in CI)
- The Cython kernel test → see below

### 7.2 Cython test kernel

The `test/` directory has its own `setup.py` for compiling a Cython example kernel. This is a user-facing example of how to write custom Cython kernels for pydgq.

**The Cython example kernel must be tested in CI** — otherwise it will silently rot. The approach: compile it as a proper Cython extension target in the meson build, installed into a subpackage (e.g., `pydgq.examples`). This way it builds alongside everything else, with no separate compilation step.

In `pydgq/examples/meson.build`:
```meson
py.extension_module(
    'example_kernel',
    'example_kernel.pyx',
    dependencies: [numpy_dep],
    install: true,
    subdir: 'pydgq/examples',
)
```

The test then just `from pydgq.examples.example_kernel import ...` and exercises it through `odesolve.ivp()`.

Move the Lorenz example and other visualization scripts to `examples/` as well (as standalone `.py` scripts, not run in CI — they need matplotlib and a display).

### 7.3 What to test

- All integrators (RK4, RK3, RK2, FE, SE, IMR, BE, DG, CG) on a simple ODE
- PythonKernel interface
- Built-in linear kernels (1st order, 2nd order, with/without mass matrix)
- Data file loading (init/load cycle)
- Solver utility functions (`n_saved_timesteps`, `result_len`, `timestep_boundaries`)
- Convergence: verify that higher-order methods give smaller errors on a smooth problem
- Import test: `import pydgq` succeeds, version is correct
- cimport test (as in PyLU): verify `.pxd` files are installed

---

## 8. Documentation Updates

- **README.md:** Rewrite installation section (pip install, no more setup.py). Add badges (CI, PyPI, license). Add performance build tip (`CFLAGS=-march=native`). Add semver policy. Add AI contributions link to substrate-independent.
- **CHANGELOG.md:** Add v1.0.0 entry.
- **LICENSE.md:** Update year range and affiliation to JAMK.
- **CLAUDE.md:** Create, following the pattern from PyLU (`~/Documents/koodit/pylu/CLAUDE.md`).
- **Data file documentation:** Update the README section on precalculation to reflect `.npz` format and `importlib.resources` loading.

---

## 9. Existing Issues

The GitHub issue tracker has 5 open issues. Three are tagged for milestone 1.0:

- **#1** `pydgq_data.bin not found after pip3 install` — should be fixed by the `importlib.resources` migration. Verify and close.
- **#2** `Drop support for Python 2.7` — will be done as part of this modernization.
- **#3** `Update contact email` — update to JAMK address in pyproject.toml and LICENSE.md.

Two are enhancement requests, out of scope for modernization:
- **#4** Add convergence tolerance setting
- **#5** Add Newton-Raphson iteration

Note these in `TODO_DEFERRED.md` and leave them for a future session.

---

## 10. Migration Sequence

1. **Read all source files first.** Understand the cdef class hierarchy, inter-module dependencies, and data flow before touching anything.
2. **Create pyproject.toml and meson.build files.** Get the build working with meson-python. Set up PDM dev workflow (`pdm install`).
3. **Remove Python 2 artifacts.** `__future__` imports, pickle conditionals, data file symlink dance.
4. **Cython 3.x fixes.** Language level, noexcept, DEF replacements. Consult `briefs/cython3-cdef-class-report.md` for cdef class guidance. Add `noexcept` to all nogil methods (critical for performance), working top-down through the class hierarchy. Work file by file, starting from leaf modules (types) toward the root (odesolve).
5. **Data file migration.** pickle → npz, pkg_resources → importlib.resources. Regenerate data file.
6. **Test migration.** Convert to pytest. Compile Cython example kernel as meson extension target in `pydgq.examples`.
7. **CI setup.** Lint, test matrix, cibuildwheel, trusted publisher.
8. **Documentation.** README, CHANGELOG, LICENSE, CLAUDE.md.
9. **Verify.** All tests pass, cython-lint clean, CI green.
10. **Tag and publish.** v1.0.0, push to PyPI via trusted publisher.

---

## 11. Open Questions for Juha

Before CC begins, confirm:

1. ~~Email~~ → JAMK ✓
2. ~~Version~~ → 1.0.0 ✓
3. ~~Data file format~~ → npz ✓
4. ~~Precalc defaults~~ → keep `q=10, nx=101` ✓
5. **sympy.mpmath → mpmath:** The precalc module imports `sympy.mpmath`. Modern SymPy uses mpmath as a standalone top-level package. `import mpmath` works on Python 3.11+ (verified on mpmath 1.3.0). Update the import.
6. **User manual PDF** (`doc/pydgq_user_manual.pdf`): Keep as-is for now. Source is `doc/*.lyx` (LyX format). A content update pass is deferred.

---

## 12. Post-Session Reminders

- Bump version to `1.0.1-dev` in VERSION file after release
- Set up Dependabot (copy config from PyLU)
- Push the brief to `briefs/` in the repo
- Push the Cython 3 cdef class report (`briefs/cython3-cdef-class-report.md`) — already written, reusable for python-wlsqm
- Update `~/.claude/CLAUDE.md` project list to include pydgq
