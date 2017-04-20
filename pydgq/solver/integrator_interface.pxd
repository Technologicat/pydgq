# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.types cimport DTYPE_t, RTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase

# Base class for all integrators.
#
# Do not inherit directly from this!
#
# Inherit from ExplicitIntegrator or ImplicitIntegrator depending on your algorithm.
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
    # return value: number of implicit solve iterations taken for this timestep (explicit integrators must always return 1)
    #
    cdef int call(self, DTYPE_t* w, RTYPE_t t, RTYPE_t dt) nogil

# Base class for explicit integrators.
#
cdef class ExplicitIntegrator(IntegratorBase):
    pass  # no new data attributes or cdef methods

# Base class for implicit integrators.
#
cdef class ImplicitIntegrator(IntegratorBase):
    cdef int maxit  # maximum number of Banach/Picard iterations in implicit solve

