# -*- coding: utf-8 -*-

cimport pydgq_types
cimport kernels

# The t parameter is supplied to support also RHS f() that may explicitly depend on t (non-autonomous ODEs).
#
cdef int RK4( kernels.kernelfuncptr f, pydgq_types.DTYPE_t* w, void* data, int n_space_dofs, pydgq_types.DTYPE_t t, pydgq_types.DTYPE_t dt, pydgq_types.DTYPE_t* wrk ) nogil
cdef int RK3( kernels.kernelfuncptr f, pydgq_types.DTYPE_t* w, void* data, int n_space_dofs, pydgq_types.DTYPE_t t, pydgq_types.DTYPE_t dt, pydgq_types.DTYPE_t* wrk ) nogil
cdef int RK2( kernels.kernelfuncptr f, pydgq_types.DTYPE_t* w, void* data, int n_space_dofs, pydgq_types.DTYPE_t t, pydgq_types.DTYPE_t dt, pydgq_types.DTYPE_t* wrk, pydgq_types.DTYPE_t beta ) nogil
cdef int  FE( kernels.kernelfuncptr f, pydgq_types.DTYPE_t* w, void* data, int n_space_dofs, pydgq_types.DTYPE_t t, pydgq_types.DTYPE_t dt, pydgq_types.DTYPE_t* wrk ) nogil
cdef int  SE( kernels.kernelfuncptr f, pydgq_types.DTYPE_t* w, void* data, int n_space_dofs, pydgq_types.DTYPE_t t, pydgq_types.DTYPE_t dt, pydgq_types.DTYPE_t* wrk ) nogil

