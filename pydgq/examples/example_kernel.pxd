# -*- coding: utf-8 -*-
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True

from pydgq.solver.kernel_interface cimport CythonKernel
from pydgq.solver.types cimport RTYPE_t

cdef class MyKernel(CythonKernel):
    cdef RTYPE_t omega
    cdef void callback(self, RTYPE_t t) noexcept nogil
