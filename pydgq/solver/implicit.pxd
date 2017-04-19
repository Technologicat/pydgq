# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.pydgq_types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase

# typeedef for easy algorithm switching in calling code
ctypedef int (*integrator_ptr)( KernelBase, DTYPE_t*, DTYPE_t, DTYPE_t, DTYPE_t*, int ) nogil

cdef int IMR( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk, int maxit ) nogil
cdef int  BE( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk, int maxit ) nogil

