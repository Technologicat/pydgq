# Cython 3.x cdef Class Migration Report

*Investigated April 2026, using Cython 3.2.4 on Python 3.12.*

This report documents Cython 3.x behavior changes relevant to `cdef class` (extension types). Produced for the pydgq modernization; reusable for python-wlsqm.

---

## 1. noexcept Annotation Matching in Inheritance

**Rule:** If a base class declares a `cdef` method as `noexcept`, all subclass overrides MUST also be `noexcept`. Mismatch is a compile-time error: *"Signature not compatible with previous declaration."*

**The reverse is allowed:** A child can add `noexcept` to a method that the base didn't declare as `noexcept` (the child is making a stricter promise).

```cython
# FAILS: base has noexcept, child doesn't
cdef class Base:
    cdef int compute(self) noexcept:
        return 0

cdef class Child(Base):
    cdef int compute(self):  # ERROR: Signature not compatible
        return 1

# OK: base doesn't have noexcept, child adds it
cdef class Base:
    cdef int compute(self):
        return 0

cdef class Child(Base):
    cdef int compute(self) noexcept:  # OK: stricter promise
        return 1
```

**Implication for pydgq:** When adding `noexcept` to base class methods (IntegratorBase, KernelBase), ALL overrides in ALL subclasses must get it too. Work top-down through the hierarchy.

## 2. noexcept and nogil Interaction

**Rule:** `nogil` does NOT imply `noexcept`. A `cdef` method declared `nogil` but without `noexcept` will compile, but calling it inside a `with nogil:` block triggers GIL acquisition for exception checking on every call.

Cython 3 emits a performance hint:
> *"Exception check after calling 'advance' will always require the GIL to be acquired."*

**This is a silent performance regression if missed.** The code compiles and runs correctly — it's just slower than it should be, because the GIL is acquired and released on every method call in the inner loop.

```cython
# Performance problem: nogil method without noexcept
cdef class Solver:
    cdef void advance(self, double* data, int n) nogil:
        # Every call to this acquires the GIL for exception checking!
        pass

# Fix: add noexcept
cdef class Solver:
    cdef void advance(self, double* data, int n) noexcept nogil:
        pass
```

**Same matching rule applies:** If base has `noexcept nogil`, child must also have `noexcept nogil`.

**Implication for pydgq:** Every `cdef` method in the solver hierarchy that runs in `nogil` blocks needs `noexcept nogil`. This is the most performance-critical change in the entire migration. Audit the call chains from `odesolve.ivp()` downward.

## 3. `__cinit__` Signature Strictness

**Rule:** Cython calls ALL `__cinit__` methods up the MRO with the SAME arguments. If `Child(10, 20)` is called, `Base.__cinit__` also receives `(10, 20)`. If `Base.__cinit__` only accepts `(self)`, you get a runtime TypeError.

This is not new in Cython 3 — it has always worked this way. But it's a common source of bugs when adding arguments to subclass constructors.

```cython
# Runtime error: Base.__cinit__ gets args it doesn't expect
cdef class Base:
    def __cinit__(self, int x_val):
        pass

cdef class Child(Base):
    def __cinit__(self, int x_val, int y_val):
        pass

Child(10, 20)  # TypeError: Base.__cinit__() takes 1 arg, got 2

# Fix: Base.__cinit__ absorbs extras
cdef class Base:
    def __cinit__(self, *args, **kwargs):
        pass
```

**Implication for pydgq:** Check the IntegratorBase/KernelBase `__cinit__` signatures. If they don't use `*args, **kwargs`, and subclasses add parameters, this will crash at runtime. Note: `__cinit__` without `*args/**kwargs` but with `__init__` that takes different args is fine — it's the `__cinit__` chain that gets the constructor arguments, not `__init__`.

## 4. Old-Style Property Syntax

**Result:** Still compiles in Cython 3.2.4 with no warnings. However, the old-style syntax is Cython-specific, while the `@property` decorator is standard Python — same syntax in both languages. Since we're already touching every file for `noexcept` and `DEF`, modernize properties during this pass.

```cython
# Old style (Cython-specific):
cdef class Foo:
    cdef int _x
    property x:
        def __get__(self):
            return self._x
        def __set__(self, int value):
            self._x = value

# New style (standard Python):
cdef class Foo:
    cdef int _x
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, int value):
        self._x = value
```

**Recommendation:** Convert all old-style properties to `@property` decorator syntax during this modernization pass.

## 5. cpdef Methods with Typed Arguments

**Result:** Work correctly in Cython 3.2.4, including with memoryview types and inheritance with overrides.

```cython
cdef class Integrator:
    cpdef int step(self, double dt, double[:] state):
        pass

cdef class ChildIntegrator(Integrator):
    cpdef int step(self, double dt, double[:] state):
        pass
```

**Recommendation:** No changes needed for existing `cpdef` methods.

## 6. DEF / IF Compile-Time Directives

`DEF` is deprecated in Cython 3. Replace with a C-level `#define` via verbatim C injection. This preserves the named constant, works in `nogil`, and has zero runtime cost:

```cython
# Old (deprecated):
DEF MY_CONSTANT = 1e-15

# New:
cdef extern from *:
    """
    #define MY_CONSTANT 1e-15
    """
    double MY_CONSTANT
```

The call sites do not change — `MY_CONSTANT` is still used as a bare name.

`IF`/`ELIF`/`ELSE` compile-time conditionals (e.g., `IF PY_MAJOR_VERSION >= 3:`) are also deprecated. For Python-2-vs-3 conditionals, simply delete the Python 2 branch. For other compile-time branching, use build-system-level configuration or runtime checks.

Search all `.pyx` and `.pxd` files for `DEF` and `IF` directives.

---

## Summary: What to Audit in pydgq

| Issue | Priority | Action |
|-------|----------|--------|
| `noexcept` on all `cdef` methods in `nogil` paths | **Critical** — silent perf regression | Audit entire solver hierarchy, add `noexcept` to all nogil methods. Match across base/child. |
| `noexcept` matching in inheritance | **High** — compile error | When adding `noexcept` to base, must add to all subclass overrides. Work top-down. |
| `__cinit__` signatures | **Medium** — runtime error | Check if base `__cinit__` accepts `*args, **kwargs`. If not, verify no subclass adds extra constructor args. |
| `DEF` / `IF` directives | **Medium** — deprecated | Replace `DEF` with `cdef extern from *` + `#define`. Delete Python 2 `IF` branches. |
| Old-style properties | **Low** — works but non-standard | Convert to `@property` decorator syntax (standard Python). |
| `cpdef` methods | **None** — works fine | No changes needed. |

---

*This report was tested on Cython 3.2.4, Python 3.12. Results should apply to Cython ≥3.0.*
