# Deferred TODOs

Items noticed during modernization that are out of scope for the current task.

## NEW-MACHINE-SETUP.md: deadsnakes Python `-dev` package note

For each deadsnakes Python installed (e.g. `python3.13`, `python3.14`), the corresponding `-dev` package must also be installed (e.g. `sudo apt install python3.14-dev`). Without it, meson-python cannot find `Python.h` and extension module builds fail. Check which machine (maia/electra) has the latest version before updating.

## Add convergence tolerance setting (GitHub #4)

Enhancement. Needs changes in `implicit.pyx` and `galerkin.pyx` wherever `maxit` is used.

## Add Newton-Raphson iteration (GitHub #5)

Enhancement. Would allow faster convergence for stiff problems compared to Banach/Picard.
