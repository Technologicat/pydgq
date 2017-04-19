# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.pydgq_types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase

cdef class IntegratorBase:
    def __init__(self, str name, KernelBase rhs):
        self.name = name
        self.rhs  = rhs

        # work array not created by default
        self.wrk_arr = None
        self.wrk = <DTYPE_t*>0

    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil:
        return 0  # no-op, no iterations taken

cdef class ExplicitIntegrator(IntegratorBase):
    def __init__(self, str name, KernelBase rhs):
        # super
        IntegratorBase.__init__(self, name, rhs)

cdef class ImplicitIntegrator(IntegratorBase):
    def __init__(self, str name, KernelBase rhs, int maxit):
        # super
        IntegratorBase.__init__(self, name, rhs)

        self.maxit = maxit

