# -*- coding: utf-8 -*-
#
# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True
#
"""Interface for evaluating the f() in  w' = f(w, t).

These base classes connect the IVP solver with the user-provided custom code
for computing the RHS. Cython and Python interfaces are available.

For a specific, arbitrary nonlinear problem, a custom kernel must be provided.
This can be done in either of two ways:

    - a) implement a cdef class
        - inherit from pydgq.solver.kernel_interface.CythonKernel
        - override __init__(), add any needed parameters, call CythonKernel.__init__(self, n)
        - override cdef callback(...) to compute w' for your RHS

    - b) implement a regular Python class
        - inherit from pydgq.solver.kernel_interface.PythonKernel
        - override __init__(), add any needed parameters, call PythonKernel.__init__(self, n)
        - override def callback(...) to compute w' for your RHS

Cython kernels run in nogil mode, allowing parallel processing of several independent problems.
Useful in use cases with many independent small problems, and for OpenMP based parallel processing
inside callback() (using cython.parallel.prange()).

Python kernels acquire the GIL for each call of callback(). Useful for quick prototyping
of new ODE systems, and for large problems where the Python function call overhead is not significant.

See pydgq.solver.builtin_kernels for some Cython-based example kernels.
"""

from __future__ import division, print_function, absolute_import

# use fast math functions from <math.h>, available via Cython
#from libc.math cimport sin, cos, log, exp, sqrt


###############
# Base classes
###############

cdef class KernelBase:
#    cdef double* w    # old state vector (memory owned by caller)
#    cdef double* out  # new state vector (memory owned by caller)
#    cdef int n        # n_space_dofs
#    cdef int timestep
#    cdef int iteration

    def __init__(self, int n):
        """def __init__(self, int n):

Base class for all kernels.

Do not inherit directly from this; instead, see CythonKernel and PythonKernel
depending on which language you wish to implement your kernel in.

Parameters:
    n : int
        Number of DOFs of the ODE system.

Data attributes:
    w : double*
        input, old state vector (memory owned by caller)
    out : double*
        output, new state vector (memory owned by caller)
    n : int
        number of DOFs (taken from __init__)
    timestep : int
        number of current timestep
    iteration : int
        number of current nonlinear (Banach/Picard) iteration
"""
        self.n = n

    # The solver (odesolve.ivp()) calls begin_timestep() when it begins a new timestep.
    #
    # Each integrator implementation calls begin_iteration() when it begins a new Banach/Picard iteration.
    # (Explicit integrators also call it once, for "iteration 0".)
    #
    # This metadata is intended for the actual computational kernel; the kernel base classes do not need it.
    #
    # timestep  : 0-based, 0 = initial condition, 1 = first timestep, 2 = second timestep, ...
    # iteration : 0-based. Special value -1 = evaluation of final result from this timestep. (Used if saving also w' in the results arrays.)
    #
    cdef void begin_timestep(self, int timestep) nogil:
        self.timestep  = timestep
    cdef void begin_iteration(self, int iteration) nogil:
        self.iteration = iteration

    # The call interface. The solver calls this when it wants to evaluate w'.
    #
    # Implemented in derived classes.
    #
    cdef void call(self, double* w, double* out, double t) nogil:
        pass


cdef class CythonKernel(KernelBase):
    # we override __init__ only in order to provide a docstring.
    def __init__(self, int n):
        """def __init__(self, int n):

Base class for kernels implemented in Cython.

Cython kernels will run in nogil mode.

No further docstrings are provided, because the rest of this class
is not visible from the Python level. See the source code in
pydgq.solvers.kernel_interface.pyx for details.

Basically, in your cdef class, override the method

    cdef void callback(self, double t) nogil:

The necessary arrays can be accessed as self.w and self.out.
Both arrays have self.n elements.

If callback() is not overridden, this class implements a no-op kernel: w' = 0.
"""
        KernelBase.__init__(self, n)

    # Implementation of call() for Cython kernels.
    #
    cdef void call(self, double* w, double* out, double t) nogil:
        self.w   = w
        self.out = out
        self.callback(t)

    # Hook for custom code.
    #
    # Default no-op kernel: w' = 0
    #
    # Override this method in derived classes to provide your computational kernel.
    #
    cdef void callback(self, double t) nogil:
        cdef int j
        for j in range(self.n):
            self.out[j] = 0.0


cdef class PythonKernel(KernelBase):
#    cdef double[::1] w_arr
#    cdef double[::1] out_arr

    # we override __init__ only in order to provide a docstring.
    def __init__(self, int n):
        """def __init__(self, int n):

Base class for kernels implemented in pure Python.

Python kernels will acquire the GIL for calling callback(),
which is a regular Python-level method (in Cython parlance,
"def method").

See callback().

Data attributes added by PythonKernel:
    w_arr : double[::1]
        Python-accessible view of self.w
    out_arr : double[::1]
        Python-accessible view of self.out
"""
        KernelBase.__init__(self, n)

    # Implementation of call() for Python kernels.
    #
    cdef void call(self, double* w, double* out, double t) nogil:
        self.w   = w
        self.out = out
        with gil:
            self.w_arr   = <double[:self.n:1]>w
            self.out_arr = <double[:self.n:1]>out
            self.callback(t)

    def callback(self, double t):
        """def callback(self, double t):

Python-based callback (hook for custom code).

Override this method in derived classes to provide your computational kernel.

Use self.w_arr and self.out_arr to access self.w and self.out;
these are Python-accessible views to the same memory.

Both arrays have self.n elements.

If callback() is not overridden, this class implements a no-op kernel: w' = 0.
"""
        cdef int j
        for j in range(self.n):
            self.out[j] = 0.0

