# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase
from pydgq.solver.integrator_interface cimport ImplicitIntegrator

# Implicit midpoint rule
cdef class IMR(ImplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

# Backward Euler (implicit Euler)
cdef class BE(ImplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

