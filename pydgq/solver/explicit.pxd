# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.pydgq_types cimport DTYPE_t
from pydgq.solver.kernels cimport kernelfuncptr

# The t parameter is supplied to support also RHS f() that may explicitly depend on t (non-autonomous ODEs).
#
cdef int RK4( kernelfuncptr f, DTYPE_t* w, void* data, int n_space_dofs, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil
cdef int RK3( kernelfuncptr f, DTYPE_t* w, void* data, int n_space_dofs, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil
cdef int RK2( kernelfuncptr f, DTYPE_t* w, void* data, int n_space_dofs, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk, DTYPE_t beta ) nogil
cdef int  FE( kernelfuncptr f, DTYPE_t* w, void* data, int n_space_dofs, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil
cdef int  SE( kernelfuncptr f, DTYPE_t* w, void* data, int n_space_dofs, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil

