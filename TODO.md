## TODO

### High priority

 - provide a mechanism for Python-level kernels `f()`
   - see the discussion [Passing functions to Cython](https://groups.google.com/forum/#!topic/cython-users/nuMpfVeAUA0)
 - remove prints in normal operation
 - write some usage examples
 - write some unit tests
   - linear problem? Known solution... (and good for testing the provided example kernels)
   - Lorenz problem?
   - and/or compare solutions to the same problem computed by different algorithms
 - clean up `doc/legtest*.py`
 - update user manual (mainly title and a new introductory paragraph)
 - odesolve.pyx could as well use np.empty and buffer interface instead of `malloc`/`free`
 - figure out how to distribute pydgq_data.bin (and make the installation find it), this file is rather large
 - fix keywords in `setup.py`
 - publish on GitHub
 - tag v0.1.0
 - release v0.1.0 on GitHub
 - release v0.1.0 on PyPI

### Maybe later

 - add support for convergence tolerance (requires small changes to [`implicit.pyx`](pydgq/solver/implicit.pyx) and [`galerkin.pyx`](pydgq/solver/galerkin.pyx); see the loops that use `maxit`)
 - add support for simultaneous processing with different settings 
 - relax the technical limitation on matching `interp` to the last initialized `nx`
 - `odesolve.make_tt()` should have access to `vis_x` from `galerkin.Helper`; this array gives the actually used visualization points (Once and Only Once)
 - increase object-orientedness (a separate function `galerkin.init()` is not very convenient)
 - change data file to `.mat` format to make it more self-documenting (currently arbitrary binary data, incorrectly detected by `file`)
 - add support for Newton-Raphson (requires a user-provided Jacobian or a numerical approximation)
 - add support for variable timestep length (the algorithms already allow it; this is just an infrastructure limitation)
 - add support for saving only every `n`th timestep (complicates the result index computation logic)
