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
"""Interface for integration algorithms.

These base classes provide the interface that connects the IVP solver
with the implementations of the specific algorithms (RK4, dG, etc.).

Beyond instantiation of the objects, this interface is only available
at the Cython level.

See the source code in pydgq.solvers.integrator_interface.pyx for details.
"""

from __future__ import absolute_import

from pydgq.solver.types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase

cdef class IntegratorBase:
    def __init__(self, str name, KernelBase rhs):
        """def __init__(self, str name, KernelBase rhs):

Base class for all integration algorithms.

Do not inherit directly from this! Inherit from ExplicitIntegrator or ImplicitIntegrator depending on your algorithm.

Parameters:
    name : str
        Human-readable algorithm name such as "RK4", "dG", etc.
        This is intended to be filled in by a concrete derived class
        that implements a specific algorithm.

        See pydgq.solver.explicit, pydgq.solver.implicit and
        pydgq.solver.galerkin for examples.

    rhs : KernelBase instance (in practice, an instance of a derived class)
        The RHS computational kernel. This is intended to be
        chosen by the user at problem solve time.

Data attributes of IntegratorBase:
    name : str
        saved from __init__
    rhs : KernelBase instance
        saved from __init
    wrk_arr : DTYPE_t[::1]
        work space for integration algorithm (to be allocated by derived classes)
    wrk : DTYPE_t*
        raw C pointer to wrk_arr (to be filled in by derived classes)

No further docstrings are provided, because the rest of this class
is not visible from the Python level. See the source code in
pydgq.solvers.integrator_interface.pyx for details.

Basically, when implementing a new algorithm, the derived cdef class
should override the method

    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil:

where
    w  : in/out
        old state vector -> new state vector
    t  : in
        time at the beginning of the timestep (passed through to self.rhs.call())
    dt : in
        size of timestep to take
"""
        self.name = name
        self.rhs  = rhs

        # work array not created by default
        self.wrk_arr = None
        self.wrk = <DTYPE_t*>0

    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil:
        return 0  # no-op, no iterations taken


cdef class ExplicitIntegrator(IntegratorBase):
    def __init__(self, str name, KernelBase rhs):
        """def __init__(self, str name, KernelBase rhs):

Base class for explicit integrators (such as RK4).

Inheriting from this class simply marks the algorithm
as an explicit one.

No new parameters or data attributes are added here.
"""
        # super
        IntegratorBase.__init__(self, name, rhs)


# TODO/FIXME: currently each integrator needs to implement the iteration loop separately.
cdef class ImplicitIntegrator(IntegratorBase):
    def __init__(self, str name, KernelBase rhs, int maxit):
        """def __init__(self, str name, KernelBase rhs, int maxit):

Base class for implicit integrators (such as IMR and dG).

Note that inheriting from this class simply marks the algorithm
as an implicit one; each integrator needs to implement the
nonlinear iteration loop separately.

New parameters:
    maxit : int
        Maximum number of Banach/Picard iterations to take
        in the nonlinear iteration loop.

Data attributes added by ImplicitIntegrator:
    maxit : int
        saved from __init__
"""
        # super
        IntegratorBase.__init__(self, name, rhs)

        self.maxit = maxit

