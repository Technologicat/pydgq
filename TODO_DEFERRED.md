# Deferred TODOs

Items noticed during modernization that are out of scope for the current task.

- **NEW-MACHINE-SETUP.md**: Add note that for each deadsnakes Python installed (e.g. `python3.13`, `python3.14`), the corresponding `-dev` package must also be installed (e.g. `sudo apt install python3.14-dev`). Without it, meson-python cannot find `Python.h` and extension module builds fail. Check which machine (maia/electra) has the latest version before updating.

- **GitHub issue #4**: Add convergence tolerance setting (enhancement). Needs changes in `implicit.pyx` and `galerkin.pyx` wherever `maxit` is used.

- **GitHub issue #5**: Add Newton-Raphson iteration (enhancement). Would allow faster convergence for stiff problems compared to Banach/Picard.

