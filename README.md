# pydgq

Python dG(q): Solve ordinary differential equation (ODE) systems using the time-discontinuous Galerkin method.

![Lorenz attractor, dG(2)](example.png)


## Introduction

This is a Cython-accelerated library that integrates initial value problems (IVPs) of first-order ordinary differential equation (ODE) systems of the form `u'(t) = f(u, t)`.

The main feature of the library is dG(q), i.e. the time-discontinuous Galerkin method using a Lobatto basis (a.k.a. hierarchical polynomial basis). Some classical explicit and implicit integrators (`RK4`, `RK3`, `RK2`, `FE`, `SE`, `IMR`, `BE`) are also provided, mainly for convenience.

Time-discontinuous Galerkin is a very accurate implicit method that often allows using a rather large timestep. Due to its Galerkin nature, it also models the behavior of the solution inside the timestep (although best accuracy is obtained at the endpoint of the timestep).

Arbitrary polynomial degrees `q` are supported, but often best results are obtained for `q = 1` (dG(1)) or `q = 2` (dG(2)). (The example image above has been computed using dG(2). Note the discontinuities at timestep boundaries.)

The focus of this library is on arbitrary nonlinear problems. All implicit methods are implemented using fixed-point (Banach/Picard) iteration, relying on the Picard-Lindelöf theorem (which itself relies on the Banach fixed point theorem).

For supplying the user code implementing the right-hand side (RHS) `f(u, t)` for a given problem, both Python and Cython interfaces are provided.

For material on the algorithms used, see the [user manual](doc/pydgq_user_manual.pdf).


## Usage summary

The user is expected to provide a custom kernel, which computes the RHS `f(u, t)` for the specific problem to be solved.

The problem is solved by instantiating this custom kernel, and passing the instance to the `ivp()` function of the [`pydgq.solver.odesolve`](pydgq/solver/odesolve.pyx) module (along with solver options).

Code examples are provided the [`test`](test/) subdirectory. For compiling the Cython example, the test subdirectory contains its own [`setup.py`](test/setup.py) that is only used for this purpose.


## Installation

### From PyPI

Install as user:

```bash
pip install pydgq --user
```

Install as admin:

```bash
sudo pip install pydgq
```

### From GitHub

As user:

```bash
git clone https://github.com/Technologicat/pydgq.git
cd pydgq
python setup.py install --user
```

As admin, change the last command to

```bash
sudo python setup.py install
```


## Software architecture

The design of pydgq is based on two main class hierarchies, consisting of Cython extension types (cdef classes):

 - [**IntegratorBase**](pydgq/solver/integrator_interface.pyx): interface class for integrator algorithms
   - [**ExplicitIntegrator**](pydgq/solver/integrator_interface.pyx): base class for explicit methods, which are implemented in [`pydgq.solver.explicit`](pydgq/solver/explicit.pyx):
     - _RK4_: fourth-order Runge-Kutta
     - _RK3_: Kutta's third-order method
     - _RK2_: parametric second-order Runge-Kutta
     - _FE_: forward Euler (explicit Euler)
     - _SE_: symplectic Euler, for 2nd-order systems reduced to a twice larger 1st-order system
   - [**ImplicitIntegrator**](pydgq/solver/integrator_interface.pyx): base class for implicit methods, which are implemented in [`pydgq.solver.implicit`](pydgq/solver/implicit.pyx):
     - _IMR_: implicit midpoint rule
     - _BE_: backward Euler (implicit Euler)
     - [**GalerkinIntegrator**](pydgq/solver/galerkin.pyx): base class for Galerkin methods using a Lobatto basis, which are implemented in [`pydgq.solver.galerkin`](pydgq/solver/galerkin.pyx):
       - _DG_: discontinuous Galerkin
       - _CG_: continuous Galerkin
 - [**KernelBase**](pydgq/solver/kernel_interface.pyx): interface class for RHS kernels
   - [**CythonKernel**](pydgq/solver/kernel_interface.pyx): base class for kernels implemented in Cython, see [`pydgq.solver.builtin_kernels`](pydgq/solver/builtin_kernels.pyx) for examples (and the [`test`](test/) subdirectory for their usage):
     - _Linear1stOrderKernel_: `w' = M w`
     - _Linear1stOrderKernelWithMassMatrix_: `A w' = M w`
     - _Linear2ndOrderKernel_: `u'' = M0 u + M1 u'`
       - solved as a twice larger 1st-order system, by defining `v := u'`, thus obtaining `u' = v` and `v' = M0 u + M1 v`
       - the DOF vector is defined as `w := (u1, v1, u2, v2, ..., um, vm)`, where `m` is the number of DOFs of the original 2nd-order system.
     - _Linear2ndOrderKernelWithMassMatrix_: `M2 u'' = M0 u + M1 u'`
       - solved as `u' = v` and `M2 v' = M0 u + M1 v` similarly as above
     - CythonKernel acts as a base class for your own Cython-based kernels
   - [**PythonKernel**](pydgq/solver/kernel_interface.pyx): base class for kernels implemented in Python
     - [minimal example](test/python_kernel_test.py)
     - [Lorenz system](test/lorenz_example.py)
     - PythonKernel acts as a base class for your own Python-based kernels

The `ivp()` function of [`pydgq.solver.odesolve`](pydgq/solver/odesolve.pyx) understands the `IntegratorBase` and `KernelBase` interfaces, and acts as the driver routine.

Aliases to primitive data types (to allow precision switching at compile time) are defined in `pydgq.solver.types`: [Python (import)](pydgq/solver/types.pyx), [Cython (cimport)](pydgq/solver/types.pxd). The Python-accessible names point to the appropriate NumPy symbols. `RTYPE` is real, `ZTYPE` is complex, and `DTYPE` is an alias representing the problem data type. The corresponding Cython-accessible datatypes have the `_t` suffix (`RTYPE_t`, `ZTYPE_t`, `DTYPE_t`).

Currently, `DTYPE` is real, but it is kept conceptually separate from `RTYPE` so that complex-valued problems can be later supported, if necessary (this requires some changes in the code, especially any calls to `dgesv`).


## Linear and nonlinear problems

The built-in kernels concentrate on linear problems, because for this specific class of problems, it is possible to provide generic pre-built kernels. The provided set of four kernels attempts to cover the most common use cases, especially the case of small-amplitude vibration problems in mechanics, which are of the second order in time, and typically have all three matrices `M0`, `M1` and `M2`.

The main focus of the library, however, is on solving arbitrary nonlinear problems, and thus no algorithms specific to linear problems have been used. This makes the solver severely suboptimal for linear problems, but significantly increases flexibility.

Be aware that due to this choice of approach, the usual stability results for integrators for linear problems do not apply. For example, BE is no longer stable at any timestep size, because for large enough `dt`, contractivity in the Banach/Picard iteration is lost. (The standard results are based on classical von Neumann stability analysis; without modifications, these arguments are not applicable to the algorithm based on Banach/Picard iteration.)


## Lobatto basis functions / precalculation

The numerical evaluation of the Lobatto basis functions is numerically highly sensitive to catastrophic cancellation (see the [user manual](doc/pydgq_user_manual.pdf)); IEEE-754 double precision is insufficient to compute the intermediate results. To work around this issue, the basis functions are pre-evaluated once, by the module [`pydgq.utils.precalc`](pydgq/utils/precalc.py), using higher precision in software (via `sympy.mpmath`).

The precalculation can be run again by running the module as the main program, e.g. `python -m pydgq.utils.precalc`. The module has some command-line options; use the standard `--help` option to see them.

The precalc module stores the values of the basis functions (at quadrature points and at visualization points, both on the reference element `[-1,1]`) into a binary data file called `pydgq_data.bin`, by pickling the computed `np.ndarray` objects. A data file with default settings, that works at least with the `x86_64` architecture, is built into the package as `pydgq/pydgq_data.bin`.

The setuptools-based `setup.py` expects to find, in the source tree, the data file to be rolled into the package as `pydgq/pydgq_data.bin`. The data file provided with the sources has been generated using the default settings of `pydgq.utils.precalc` (`q = 10`, `nx = 101`).

When running the solver, to use Galerkin methods, [`pydgq.solver.galerkin.init()`](pydgq/solver/galerkin.pyx) must be called before calling [`pydgq.solver.odesolve.ivp()`](pydgq/solver/odesolve.pyx).


## Data file location

During `init()`, the solver attempts to load `pydgq_data.bin` from these locations, in this order:

 1. `./pydgq_data.bin` (local override),
 2. `~/.config/pydgq/pydgq_data.bin` (user override),
 3. `pydgq` package, via `pkg_resources` (likely installed in `/usr/local/lib/python2.7/dist-packages/pydgq...` or similar).

If all three steps fail, `IOError` is raised.


## Dependencies

- [NumPy](http://www.numpy.org)
- [Cython](http://www.cython.org)
- [PyLU](https://github.com/Technologicat/pylu) or `pip install pylu`
- [SymPy](http://www.sympy.org) (for precalculating the data file)
- [Matplotlib](http://www.matplotlib.org) (for examples in the [`test`](test/) subdirectory)


## License

[BSD](LICENSE.md). Copyright 2016-2017 Juha Jeronen and University of Jyväskylä.


#### Acknowledgement

This work was financially supported by the Jenny and Antti Wihuri Foundation.

