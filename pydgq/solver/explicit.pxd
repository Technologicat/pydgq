# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.pydgq_types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase
from pydgq.solver.integrator_interface cimport ExplicitIntegrator

cdef class RK4(ExplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

cdef class RK3(ExplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

cdef class RK2(ExplicitIntegrator):
    cdef DTYPE_t beta
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

cdef class FE(ExplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

cdef class SE(ExplicitIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

