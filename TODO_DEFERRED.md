# Deferred TODOs

Items noticed during modernization that are out of scope for the current task.

## Sanity-check numerics on the first MSVC-built Windows wheels

Windows wheels up to and including the current release were built with MinGW-w64 gcc — meson
picked it up off the runner PATH, and nothing flagged it because the extensions link no
OpenMP and so imported cleanly. CI now activates MSVC, so the *next* release ships wheels
from a different compiler than every previous one.

The C API side is fine; the part worth a look is floating point, where gcc and MSVC differ
in default contraction and optimization of expressions. Time integrators accumulate, so a
last-bit difference per step is not necessarily last-bit by the end of a run. Before
publishing, run the suite on a Windows wheel and compare against the Linux build — the
compensated-summation paths in `compsum` are the most sensitive place to look.

Discovered while adding MSVC activation to CI (2026-08-14).

## NEW-MACHINE-SETUP.md: deadsnakes Python `-dev` package note

For each deadsnakes Python installed (e.g. `python3.13`, `python3.14`), the corresponding `-dev` package must also be installed (e.g. `sudo apt install python3.14-dev`). Without it, meson-python cannot find `Python.h` and extension module builds fail. Check which machine (maia/electra) has the latest version before updating.

## Add convergence tolerance setting (GitHub #4)

Enhancement. Needs changes in `implicit.pyx` and `galerkin.pyx` wherever `maxit` is used.

## Add Newton-Raphson iteration (GitHub #5)

Enhancement. Would allow faster convergence for stiff problems compared to Banach/Picard.
