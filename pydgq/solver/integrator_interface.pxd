# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.pydgq_types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase

# base class for all integrators
#
# (Do not inherit directly from this! Inherit from ExplicitIntegrator or ImplicitIntegrator depending on your algorithm.)
#
cdef class IntegratorBase:
    cdef str name              # human-readable name of algorithm (set by derived classes): "RK4", "dG", etc.
    cdef KernelBase rhs        # RHS computational kernel (problem-specific, passed in by user)

    cdef DTYPE_t[::1] wrk_arr  # work space (to be allocated by derived classes)
    cdef DTYPE_t* wrk          # raw C pointer to wrk_arr (to be filled in by derived classes)

    # call interface for solver
    #
    # w  : in/out: old state vector -> new state vector
    # t  : in: time at the beginning of the timestep (passed through to self.rhs.call())
    # dt : in: size of timestep to take
    #
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

# base class for explicit methods
#
cdef class ExplicitIntegrator(IntegratorBase):
    pass  # no new data attributes or cdef methods

# base class for implicit methods
#
cdef class ImplicitIntegrator(IntegratorBase):
    cdef int maxit  # maximum number of Banach/Picard iterations in implicit solve

