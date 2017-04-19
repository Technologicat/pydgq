# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.pydgq_types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase

# typeedef for easy algorithm switching in calling code
#
# all of our classical explicit integrators except RK2 come in this format:
# TODO: make this OO, beta is an instantiation-time parameter
#
ctypedef int (*integrator_ptr)( KernelBase, DTYPE_t*, DTYPE_t, DTYPE_t, DTYPE_t* ) nogil

# The t parameter is supplied to support also RHS that may explicitly depend on t (non-autonomous ODEs).
#
cdef int RK4( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil
cdef int RK3( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil
cdef int RK2( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk, DTYPE_t beta ) nogil
cdef int  FE( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil
cdef int  SE( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil

