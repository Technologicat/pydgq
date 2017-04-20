# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase
from pydgq.solver.integrator_interface cimport ExplicitIntegrator

# Fourth-order Runge-Kutta
cdef class RK4(ExplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

# Kutta's third-order method
cdef class RK3(ExplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

# Parametric second-order Runge-Kutta
cdef class RK2(ExplicitIntegrator):
    cdef DTYPE_t beta  # saved from __init__ (def method, not declared here)
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

# Forward Euler (not recommended!)
cdef class FE(ExplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

# Symplectic Euler (for 2nd-order problems reduced to a twice larger 1st-order system)
cdef class SE(ExplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

