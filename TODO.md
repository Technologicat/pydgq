## TODO

### High priority

 - provide a mechanism for Python-level kernels `f()`
 - write some usage examples
 - write some unit tests
 - clean up `doc/legtest*.py`
 - update user manual (mainly title and a new introductory paragraph)
 - odesolve.pyx could as well use np.empty and buffer interface instead of `malloc`/`free`
 - figure out how to distribute pydgq_data.bin (and make the installation find it), this file is rather large

### Maybe later

 - add support for simultaneous processing with different settings 
 - relax the technical limitation on matching nx and interp
 - `odesolve.make_tt()` has no access to `vis_x` from integrator; should use this array to get the interpolated points (Once and Only Once)
 - increase object-orientedness (a separate function `galerkin.init()` is not very convenient)
 - change data file to `.mat` format to make it more self-documenting (currently arbitrary binary data, incorrectly detected by `file`)

