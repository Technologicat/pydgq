## TODO

### High priority

 - (blank for now)

### Maybe later

 - add the implicit algorithm from Matculevich, Neittaanmäki and Repin (2013). Guaranteed Error Bounds for a Class of Picard-Lindelöf Iteration Methods. S. Repin et al. (eds.), Numerical Methods for Differential Equations, Optimization, and Technological Problems, Computational Methods in Applied Sciences 27, Springer. [doi:10.1007/978-94-007-5288-7_10](http://dx.doi.org/10.1007/978-94-007-5288-7_10).
   - this would allow integrating an ODE system up to a specified tolerance, without knowledge of the exact solution. (The price is that the RHS must be Lipschitz, and an estimate for the Lipschitz constant is needed. Could be either specified by the user, or probed for by the solver.)
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

