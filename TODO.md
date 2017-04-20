## TODO

### High priority

 - rename "nx" to relate to t, not x (but the name "nt" is already used for number of timesteps to take; "nt_vis"?)
 - write some usage examples
 - write some unit tests
   - linear problem? Known solution... (and good for testing the provided example kernels)
   - Lorenz problem?
   - and/or compare solutions to the same problem computed by different algorithms
 - update user manual:
   - change the title
   - add a short introductory paragraph about what the software is for
   - note Python and Cython kernel support
   - note that the user manual explains the algorithms
   - note that for usage of the code itself, see tests and examples
 - write README.md
   - mention financial support from Jenny and Antti Wihuri foundation
 - figure out how to distribute pydgq_data.bin (and make the installation find it), this file is rather large
   - maybe make a small version of the file available with the distribution (will do for now?)
 - publish on GitHub
 - tag v0.1.0
 - release v0.1.0 on GitHub
 - release v0.1.0 on PyPI

### Maybe later

 - clean up `doc/legtest*.py`
 - add some mechanism to report metadata to caller (e.g. number of iterations taken at each timestep)?
 - add support for convergence tolerance (requires small changes to [`implicit.pyx`](pydgq/solver/implicit.pyx) and [`galerkin.pyx`](pydgq/solver/galerkin.pyx); see anything that uses `maxit`)
 - finish unifying the integrator interface; let the user pass in an IntegratorBase reference
   - implication: switchable user-defined integrators, no need to modify odesolve.pyx to add a new algorithm
   - may need to abstract `pydgq.solver.odesolve.store()` for this. It cannot be simply moved to the integrator side, because the solution arrays are (and should be) known only to odesolve, and store() needs access to them. Maybe add another interface class and export a default implementation?
 - separate the nonlinear iteration algorithm from the individual implicit solvers (should be implemented OnceAndOnlyOnce in `pydgq.solver.integrator_interface.ImplicitIntegrator`)
 - add support for simultaneous processing with different settings 
 - relax the technical limitation on matching `interp` to the last initialized `nx`
 - `odesolve.make_tt()` should have access to `vis_x` from `galerkin.DataManager`; this array gives the actually used visualization points (Once and Only Once)
 - finish increasing object-orientedness (a separate function `galerkin.init()` is not very convenient)
 - change data file to `.mat` format to make it more self-documenting (currently arbitrary binary data, incorrectly detected by `file`)
 - add support for Newton-Raphson (requires a user-provided Jacobian or a numerical approximation)
 - add support for variable timestep length (the algorithms already allow it; this is just an infrastructure limitation)
 - add support for saving only every `n`th timestep (complicates the result index computation logic)

