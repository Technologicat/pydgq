# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.pydgq_types cimport DTYPE_t
from pydgq.solver.kernels cimport kernelfuncptr

cdef int IMR( kernelfuncptr f, DTYPE_t* w, void* data, int n_space_dofs, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk, int maxit ) nogil
cdef int  BE( kernelfuncptr f, DTYPE_t* w, void* data, int n_space_dofs, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk, int maxit ) nogil

