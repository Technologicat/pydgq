# -*- coding: utf-8 -*-
#
# Interface for Cython and Python based computational kernels for evaluating the RHS f() in  w' = f(w).

# cpdef methods cannot be nogil, so we must work around as follows:
#
# KernelBase --> cdef void call(...) nogil, abstract method for interface (NOTE: this class **does not** define callback())
#   CythonKernel --> implement call(), provide hook for Cython cdef callback(...) nogil
#   PythonKernel --> implement call(), provide hook for Python def callback(...), which our call() invokes in a "with gil" block

from __future__ import absolute_import

from pydgq.solver.types cimport DTYPE_t, RTYPE_t

cdef class KernelBase:
    cdef DTYPE_t* w    # old state vector (memory owned by caller)
    cdef DTYPE_t* out  # new state vector (memory owned by caller)
    cdef int n         # n_space_dofs
    cdef int timestep
    cdef int iteration

    # interface for solver
    #
    cdef void begin_timestep(self, int timestep) nogil
    cdef void begin_iteration(self, int iteration) nogil
    cdef void call(self, DTYPE_t* w, DTYPE_t* out, RTYPE_t t) nogil  # interface for solver (abstract method)

# Base class for kernels implemented in Cython.
#
# Cython kernels will run in nogil mode.
#
cdef class CythonKernel(KernelBase):
    cdef void call(self, DTYPE_t* w, DTYPE_t* out, RTYPE_t t) nogil  # interface for solver (implemented here)
    cdef void callback(self, RTYPE_t t) nogil  # hook for user code

# Base class for kernels implemented in pure Python.
#
# Python kernels will acquire the gil for calling callback().
#
cdef class PythonKernel(KernelBase):
    cdef DTYPE_t[::1] w_arr    # Python-accessible view into w   (provided by __getattr__ as "w")
    cdef DTYPE_t[::1] out_arr  # Python-accessible view into out (provided by __getattr__ as "out")

    cdef void call(self, DTYPE_t* w, DTYPE_t* out, RTYPE_t t) nogil  # interface for solver (implemented here)
    # here callback() is a Python (regular def) function  # hook for user code

