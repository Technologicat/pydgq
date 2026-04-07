# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is pydgq

Cython-accelerated ODE solver library using the time-discontinuous Galerkin method with Lobatto basis. Solves first-order IVPs `u'(t) = f(u, t)` with arbitrary nonlinear RHS. Also provides classical integrators (RK4, RK3, RK2, FE, SE, IMR, BE) for convenience.

Runtime dependencies: NumPy (arrays, typed memoryviews) and PyLU (nogil-compatible LU solver, used via `cimport` in the Galerkin integrator). This constraint is intentional — keep it that way.

## Build and Development

Uses meson-python as build backend, PDM for dependency management. Python ≥ 3.11.

```bash
pdm install                              # creates venv, installs dev deps
pip install --no-build-isolation -e .    # editable install (needs venv activated)
```

The `--no-build-isolation` flag is required for editable installs with meson-python — the on-import rebuild mechanism needs build dependencies to remain available in the environment.

**Version:** single source of truth is `pydgq/VERSION`. Read by `meson.build` (build-time), `pyproject.toml` (dynamic), and `__init__.py` (runtime). Only edit `pydgq/VERSION` when bumping.

## Running Tests

```bash
pdm run pytest tests/ -v
```

79 tests covering all 9 integrators, PythonKernel and CythonKernel interfaces, all 4 built-in linear kernels (including gyroscopic/skew-symmetric systems), convergence ordering (RK and dG polynomial degree), Galerkin interpolation, utility functions, data file loading, and cimport verification.

## Architecture

### Package structure

Three subpackages: `pydgq.solver` (core), `pydgq.utils` (precalculation, plotting helpers), `pydgq.examples` (example Cython kernel). Each has its own `meson.build`.

### Class hierarchies (cdef classes)

**Integrators:** IntegratorBase → ExplicitIntegrator (RK4, RK3, RK2, FE, SE) / ImplicitIntegrator (IMR, BE) → GalerkinIntegrator (DG, CG)

**Kernels:** KernelBase → CythonKernel (builtin kernels, user kernels) / PythonKernel (pure Python RHS)

### The .pxd / .pyx split

Like PyLU, implementations in `.pxd` as `cdef inline` is intentional for helper functions (compsum, fputils, cminmax) — they get compiled into each downstream module that `cimport`s them, avoiding cross-module function call overhead. **Do not move implementations from `.pxd` to `.pyx`.**

The class method implementations live in `.pyx` files. The `.pxd` files declare the class layout and method signatures.

### Precalculated data file

`pydgq_data.npz` contains Lobatto basis function values at quadrature and visualization points, precomputed using extended-precision arithmetic (mpmath) to avoid catastrophic cancellation. Loaded by `pydgq.data.load_data()` via `importlib.resources`, with local and user config overrides.

### Type alias system

`pydgq.solver.types` defines `RTYPE` (real), `ZTYPE` (complex), `DTYPE` (problem data — currently aliased to RTYPE). The `_t` suffixed variants are Cython-level. This exists for future complex-valued support. **Do not alter this system.**

### Cross-package cimport

`galerkin.pyx` and `builtin_kernels.pyx` do `cimport pylu.dgesv`. This works because PyLU installs its `.pxd` files. PyLU must be listed as both a build and runtime dependency.

## Critical constraint: .pxd installation

All `.pxd` files in `pydgq/solver/` must be installed for downstream `cimport`. This is handled by `py.install_sources(...)` in `pydgq/solver/meson.build`. Includes inline-only `.pxd` files (fputils, cminmax) that have no `.pyx` counterpart. Verify after build changes:

```bash
pdm run python -c "import pydgq; from pathlib import Path; print(list((Path(pydgq.__file__).parent / 'solver').glob('*.pxd')))"
```

## Linting

**Python files** (flake8):

```bash
pdm run flake8 tests/ pydgq/__init__.py pydgq/data.py --select=E9,F63,F7,F82 --show-source
pdm run flake8 tests/ pydgq/__init__.py pydgq/data.py --exit-zero --max-line-length=200
```

**Cython files** (cython-lint; config in `pyproject.toml` under `[tool.cython-lint]`):

```bash
pdm run cython-lint pydgq/solver/*.pyx pydgq/solver/*.pxd pydgq/utils/discontify.pyx
```

Many existing style warnings in the numerical code (alignment spacing, `l` as loop var, commented-out debug code) are pre-existing and intentional. The `|| true` in CI reflects this.

## Code Conventions

- **Line width:** ~200 characters (numerical code with long expressions).
- **Docstrings:** existing custom format in `.pyx` files — leave as-is.
- **Dependencies:** NumPy + PyLU are the only runtime dependencies. Do not add others. mpmath/sympy are dev-only (precalculation).

## Key Rules

- **Do not refactor numerical algorithms.** The integrators, kernels, and Banach/Picard iteration are correct and tested. Modernization is infrastructure only.
- **Do not change the type alias system** (`RTYPE`, `DTYPE`, etc.). It exists for a reason (future complex-valued support).
- **Do not convert raw pointer APIs to memoryviews.** The pointer-based interfaces exist for performance in `nogil` blocks.
- **Do not hardcode `-march=native` or similar arch-specific compiler flags.** Meson's `buildtype=release` gives `-O2`. Users building from source can set `CFLAGS` if desired.
- **Do not touch the precalc algorithm.** It uses extended precision (via mpmath) to work around catastrophic cancellation. The approach is intentionally "easy but needs extra precision" rather than "numerically stable but complex."

## User Manual

`doc/pydgq_user_manual.pdf` contains the mathematical background. Source is `doc/*.lyx` (LyX format). A content update pass is deferred to a future session.
