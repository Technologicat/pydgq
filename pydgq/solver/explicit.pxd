# -*- coding: utf-8 -*-

from . cimport types
from . cimport kernels

# The t parameter is supplied to support also RHS f() that may explicitly depend on t (non-autonomous ODEs).
#
cdef int RK4( kernels.kernelfuncptr f, types.DTYPE_t* w, void* data, int n_space_dofs, types.DTYPE_t t, types.DTYPE_t dt ) nogil
cdef int RK3( kernels.kernelfuncptr f, types.DTYPE_t* w, void* data, int n_space_dofs, types.DTYPE_t t, types.DTYPE_t dt ) nogil
cdef int RK2( kernels.kernelfuncptr f, types.DTYPE_t* w, void* data, int n_space_dofs, types.DTYPE_t t, types.DTYPE_t dt, types.DTYPE_t beta ) nogil
cdef int  FE( kernels.kernelfuncptr f, types.DTYPE_t* w, void* data, int n_space_dofs, types.DTYPE_t t, types.DTYPE_t dt ) nogil
cdef int  SE( kernels.kernelfuncptr f, types.DTYPE_t* w, void* data, int n_space_dofs, types.DTYPE_t t, types.DTYPE_t dt ) nogil

