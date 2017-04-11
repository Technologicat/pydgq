# -*- coding: utf-8 -*-

cimport pydgq_types
cimport kernels

cdef int IMR( kernels.kernelfuncptr f, pydgq_types.DTYPE_t* w, void* data, int n_space_dofs, pydgq_types.DTYPE_t t, pydgq_types.DTYPE_t dt, pydgq_types.DTYPE_t* wrk, int maxit ) nogil
cdef int  BE( kernels.kernelfuncptr f, pydgq_types.DTYPE_t* w, void* data, int n_space_dofs, pydgq_types.DTYPE_t t, pydgq_types.DTYPE_t dt, pydgq_types.DTYPE_t* wrk, int maxit ) nogil

