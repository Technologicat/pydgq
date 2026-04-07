# Lessons Learned: Cython Numerics Library Modernization

*From the pydgq modernization, April 2026. Intended as reference for python-wlsqm and similar projects.*

---

## 1. The `noexcept` Audit Is the Critical Path

In Cython 3, a `cdef` method declared `nogil` but without `noexcept` silently acquires the GIL for exception checking on every call. The code compiles, runs correctly, and produces correct results — it's just slower than it should be. This is the most dangerous class of regression in numerics code because it's invisible without benchmarking.

**Work top-down through inheritance.** If a base class method has `noexcept`, all subclass overrides must also have `noexcept` (compile error otherwise). The reverse is allowed — a child can add `noexcept` that the base doesn't have. Start from the root of each class hierarchy and work down.

**Audit both code and documentation.** Signatures in `.pxd` and `.pyx` must match exactly. But also check docstrings and comments that show method signatures — if the code says `noexcept nogil` but the docstring says `nogil`, someone copying from the docs will miss it.

**Audit technique:** grep for `cdef.*nogil`, verify every hit has `noexcept`. Exclude `cdef extern` (C declarations) from the fix, but review comments/docstrings for accuracy. Run the audit twice — once to make changes, once to verify.

## 2. Windows `M_PI`

Neither MSVC nor MinGW define `M_PI` in `<math.h>` without `_USE_MATH_DEFINES`. This affects both `cdef extern from "math.h": double M_PI` and `from libc.math cimport M_PI` (since both generate code that references the C-level `M_PI`).

Fix globally in `meson.build` — it's a no-op on Linux/macOS:

```meson
add_project_arguments('-D_USE_MATH_DEFINES', language: 'c')
```

Add this from the start to avoid burning CI cycles discovering it.

## 3. Test Migration Finds Real Bugs

Writing proper tests for "working" code uncovered a real bug in pydgq's `Linear2ndOrderKernel.compute()` — the inner loop used the row index instead of the column index in the matrix-vector product. The bug was invisible for:
- Diagonal matrices (off-diagonal elements are zero)
- Identical initial conditions across DOFs (wrong index gives same value)
- The harmonic oscillator test case (`M0 = -I`, which is diagonal)

Only testing with dense non-diagonal matrices and non-trivial ICs (a gyroscopic system with skew-symmetric damping) exposed it.

**Lesson:** Don't skip test migration because the code "already works." Test with non-trivial inputs: dense matrices, skew-symmetric coupling, mixed initial conditions. The existing tests may have been designed around simple cases that don't exercise the full code paths.

## 4. The Sequence Matters

This order worked well and each step depends on the previous ones:

1. **Audit** — read everything before touching anything. Write a source audit report.
2. **Build system** — get meson-python compiling all extensions. This proves the code can build on modern tooling before making any code changes.
3. **Python 2 removal** — mechanical, low risk. `__future__` imports, `cPickle`, `sympy.mpmath`, `pkg_resources`.
4. **Cython 3.x fixes** — `noexcept` (high stakes, do carefully), `DEF` → `cdef enum`, language level.
5. **Data file migration** — functional change (pickle → npz, pkg_resources → importlib.resources). Test immediately.
6. **Tests** — catches integration issues from steps 2–5. Write tests that exercise the actual solver end-to-end with known analytical solutions.
7. **CI** — proves it works on all platforms (the Windows `M_PI` issue was caught here).
8. **Documentation** — last, because everything else might change during the migration.
9. **Verify + tag + publish.**

Don't reorder. In particular, don't try to write tests (step 6) before the code changes (steps 3–5) are done — the tests won't be able to import the package until `pkg_resources` and other Py2 artifacts are cleaned up.

## 5. Reusable Patterns

### pickle → npz

For nested dicts of NumPy arrays, flatten with path-like keys:

```python
arrays["integ/11/x"] = data["integ"][11]["x"]
np.savez_compressed("data.npz", **arrays)
```

Reconstruct in the loader by parsing key prefixes. Benefits: architecture-independent, ~30% smaller (compressed), no pickle security concerns, stable across Python versions.

### pkg_resources → importlib.resources

```python
from importlib.resources import files, as_file

ref = files("mypkg") / "data.npz"
with as_file(ref) as path:
    data = np.load(path)
```

The three-tier search pattern (local override → user config dir → package data) is clean in a dedicated helper module (e.g., `mypkg/data.py`).

### meson.build for multi-package Cython

One `meson.build` per subpackage. Define `pydgq_inc = include_directories('.')` at the top level and pass it to all `extension_module` calls for cross-package `cimport` resolution.

### DEF → cdef enum

Mechanical replacement. Move constants from inside functions to module level:

```cython
# Before (inside function):
DEF MODE_NAN = 1

# After (module level):
cdef enum:
    MODE_NAN = 1
```

### Editable install PATH trick

meson-python's editable loader needs `meson` and `ninja` on `PATH` for on-import rebuilds. Document this in README and CLAUDE.md:

```bash
export PATH="$(pwd)/.venv/bin:$PATH"
```

## 6. Brief + Audit Pattern

Writing a detailed modernization brief before starting, then doing a thorough source audit as step 1, paid off significantly. The brief serves as a contract for what will and won't change, and the audit catches structural issues (class hierarchies, cimport dependencies, build order) before they become surprises mid-migration.

For wlsqm: write the brief first (referencing this lessons-learned doc and the Cython 3 cdef class report at `briefs/cython3-cdef-class-report.md`), do the source audit, then execute.
