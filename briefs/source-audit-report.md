# pydgq Source Audit Report

*Generated April 2026, prior to modernization.*

---

## 1. Cdef Class Hierarchy

For `noexcept` top-down work order.

```
KernelBase                          [kernel_interface.pxd]
├── CythonKernel                    [kernel_interface.pxd]
│   ├── Linear1stOrderKernel        [builtin_kernels.pxd]
│   │   └── Linear1stOrderKernelWithMassMatrix
│   ├── Linear2ndOrderKernel        [builtin_kernels.pxd]
│   │   └── Linear2ndOrderKernelWithMassMatrix
│   └── MyKernel                    [test/cython_kernel.pxd]
└── PythonKernel                    [kernel_interface.pxd]

IntegratorBase                      [integrator_interface.pxd]
├── ExplicitIntegrator              [integrator_interface.pxd]
│   ├── RK4, RK3, RK2, FE, SE      [explicit.pxd]
└── ImplicitIntegrator              [integrator_interface.pxd]
    ├── IMR, BE                     [implicit.pxd]
    └── GalerkinIntegrator          [galerkin.pxd]
        ├── DG                      [galerkin.pxd]
        └── CG                      [galerkin.pxd]
```

---

## 2. `noexcept` Audit — Every `cdef` Method That Needs It

All of these run in `nogil` blocks via `odesolve.ivp()`:

| Method | File | Currently | Needs |
|--------|------|-----------|-------|
| `KernelBase.begin_timestep` | kernel_interface.pxd | `nogil` | `noexcept nogil` |
| `KernelBase.begin_iteration` | kernel_interface.pxd | `nogil` | `noexcept nogil` |
| `KernelBase.call` | kernel_interface.pxd | `nogil` | `noexcept nogil` |
| `CythonKernel.call` | kernel_interface.pxd | `nogil` | `noexcept nogil` |
| `CythonKernel.callback` | kernel_interface.pxd | `nogil` | `noexcept nogil` |
| `PythonKernel.call` | kernel_interface.pxd | `nogil` | `noexcept nogil` |
| `IntegratorBase.call` | integrator_interface.pxd | `nogil` | `noexcept nogil` |
| `RK4.call` through `SE.call` | explicit.pxd | `nogil` | `noexcept nogil` |
| `IMR.call`, `BE.call` | implicit.pxd | `nogil` | `noexcept nogil` |
| `GalerkinIntegrator.assemble` | galerkin.pxd | `nogil` | `noexcept nogil` |
| `GalerkinIntegrator.final_value` | galerkin.pxd | `nogil` | `noexcept nogil` |
| `GalerkinIntegrator.do_quadrature` | galerkin.pxd | `nogil` | `noexcept nogil` |
| `DG.call`, `CG.call` | galerkin.pxd | `nogil` | `noexcept nogil` |
| All `builtin_kernels` `callback`/`compute` | builtin_kernels.pxd | `nogil` | `noexcept nogil` |
| `MyKernel.callback` | test/cython_kernel.pxd | `nogil` | `noexcept nogil` |
| `odesolve.store` | odesolve.pyx | `nogil` | `noexcept nogil` |
| All inline functions in `compsum.pxd`, `fputils.pxd`, `cminmax.pxd` | — | `nogil` | `noexcept nogil` |

**PythonKernel.call** is `nogil` but acquires GIL internally via `with gil:`. It should still get `noexcept nogil` because the method *interface* is called from `nogil` blocks; the GIL acquisition inside is explicit and intentional.

---

## 3. `__cinit__` Check

No classes use `__cinit__` — they all use `__init__` with explicit `SuperClass.__init__(self, ...)` calls. No `__cinit__` chain issues.

---

## 4. DEF Directives

Only in `discontify.pyx`:
```cython
DEF MODE_NAN  = 1
DEF MODE_PREV = 2
```
Replace with module-level constants.

---

## 5. Python 2 Artifacts to Remove

- `from __future__ import` — **every single `.py` and `.pyx` file**
- `cPickle` fallback — `galerkin.pyx`, `precalc.py`
- `sympy.mpmath` fallback — `precalc.py`
- `pkg_resources` — `galerkin.pyx`
- Version checks and symlink dance — `setup.py`, `test/setup.py`
- `pydgq_data_27.bin` — delete

---

## 6. Data Loading Path

`galerkin.pyx` → `DataManager.__load_data()` → three-tier search (local, `~/.config/pydgq/`, `pkg_resources`). Needs full rewrite to `importlib.resources` + `np.load()` for `.npz`.

---

## 7. Inter-Module cimport/import Graph

| Module | cimports | imports |
|--------|----------|--------|
| `types.pyx` | numpy (float64, complex128) | — |
| `compsum.pxd` | types | — |
| `compsum.pyx` | types, (accumulate from compsum.pxd) | numpy |
| `fputils.pxd` | types, libc.math (fpclassify) | — |
| `cminmax.pxd` | types | — |
| `kernel_interface.pxd` | types | — |
| `kernel_interface.pyx` | types | — |
| `integrator_interface.pxd` | types, kernel_interface | — |
| `integrator_interface.pyx` | types, kernel_interface | — |
| `explicit.pxd` | types, kernel_interface, integrator_interface | — |
| `explicit.pyx` | types, kernel_interface, integrator_interface, compsum | numpy |
| `implicit.pxd` | types, kernel_interface, integrator_interface | — |
| `implicit.pyx` | types, kernel_interface, integrator_interface | numpy |
| `galerkin.pxd` | types, kernel_interface, integrator_interface | — |
| `galerkin.pyx` | types, kernel_interface, integrator_interface, compsum; **pylu.dgesv** | pickle, pkg_resources, numpy, pylu.dgesv |
| `builtin_kernels.pxd` | types, kernel_interface | — |
| `builtin_kernels.pyx` | types, kernel_interface; **pylu.dgesv** | numpy, pylu.dgesv |
| `odesolve.pyx` | types, kernel_interface, integrator_interface, fputils, cminmax; explicit, implicit, galerkin | numpy, explicit, implicit, galerkin |
| `discontify.pyx` | types | numpy |

**Build order (leaf → root):** types → compsum → kernel_interface → integrator_interface → explicit, implicit → builtin_kernels → galerkin → odesolve; discontify (independent leaf)

---

## 8. Example Kernel cimports

`test/cython_kernel.pyx` cimports:
- `pydgq.solver.types` (RTYPE_t, DTYPE_t)
- `pydgq.solver.kernel_interface` (CythonKernel)
- `libc.math` (sin, cos, etc.)

When moved to `pydgq/examples/`, it needs the `.pxd` files from `pydgq.solver` on the include path — handled automatically by meson within the same project.

---

## 9. Inline-Only `.pxd` Files (No Corresponding `.pyx`)

These contain only `cdef inline` functions and are consumed via `cimport` — they do NOT generate extension modules:

- `fputils.pxd` — floating-point classification (nan/inf checks)
- `cminmax.pxd` — min/max helpers

These must be installed for downstream `cimport` but are NOT listed as extension modules in the build.

---

## 10. Extension Modules to Build

| Extension | Package | Math-heavy | External deps |
|-----------|---------|------------|---------------|
| `types` | `pydgq.solver` | no | — |
| `compsum` | `pydgq.solver` | yes | — |
| `kernel_interface` | `pydgq.solver` | yes | — |
| `integrator_interface` | `pydgq.solver` | yes | — |
| `explicit` | `pydgq.solver` | yes | — |
| `implicit` | `pydgq.solver` | yes | — |
| `galerkin` | `pydgq.solver` | yes | pylu |
| `builtin_kernels` | `pydgq.solver` | yes | pylu |
| `odesolve` | `pydgq.solver` | yes | — |
| `discontify` | `pydgq.utils` | yes | — |
| `example_kernel` | `pydgq.examples` | yes | — |
