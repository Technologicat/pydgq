# -*- coding: utf-8 -*-

from . cimport types
from . cimport kernels

cdef int IMR( kernels.kernelfuncptr f, types.DTYPE_t* w, void* data, int n_space_dofs, types.DTYPE_t t, types.DTYPE_t dt, int maxit ) nogil
cdef int  BE( kernels.kernelfuncptr f, types.DTYPE_t* w, void* data, int n_space_dofs, types.DTYPE_t t, types.DTYPE_t dt, int maxit ) nogil

