# -*- coding: utf-8 -*-
#
# Example Cython-based computational kernels for evaluating the f() in  w' = f(w, t).
#
# This module provides generic kernels for autonomous linear problems of 1st and 2nd order,
# with and without mass matrices on the LHS.
#
# (Autonomous: f = f(w), i.e. f does not explicitly depend on t.)
#
# All matrices in this module are assumed to have C memory layout.

from __future__ import division, print_function, absolute_import

from pydgq.solver.types cimport DTYPE_t, RTYPE_t
from pydgq.solver.kernel_interface cimport CythonKernel


# w' = M w
#
# The matrix M is supplied to __init__ (def method, not declared here).
# but callback() is free to do whatever it wants with it.
#
# The data attributes self.timestep and self.iteration (in ancestor) are provided
# for user callback()s to keep track of whether they need to update something.
#
# This class assumes M is constant in time.
#
cdef class Linear1stOrderKernel(CythonKernel):
    cdef DTYPE_t* M
    cdef DTYPE_t[:,::1] M_arr

    cdef void callback(self, RTYPE_t t) nogil
    cdef void compute(self, DTYPE_t* w_in, DTYPE_t* wp_out) nogil


# A w' = M w
#
# with the mass matrix "A" constant with respect to time.
#
# Same note about "M" as above.
#
cdef class Linear1stOrderKernelWithMassMatrix(Linear1stOrderKernel):
    cdef DTYPE_t* LU
    cdef int* p
    cdef int* mincols
    cdef int* maxcols
    cdef DTYPE_t* wrk
    cdef DTYPE_t[:,::1] LU_arr
    cdef int[::1] p_arr
    cdef int[::1] mincols_arr
    cdef int[::1] maxcols_arr
    cdef DTYPE_t[::1] wrk_arr

    cdef void callback(self, RTYPE_t t) nogil


# u'' = M0 u + M1 u'
#
# reduced to the twice larger 1st-order system
#
#  u' = v
#  v' = M0 u + M1 v
#
# The callback() method computes w',
# where w = (u1, v1, u2, v2, ..., um, vm).
#
# The above note about "M" applies to "M0" and "M1".
#
# Note that  n = 2*m  still denotes the size of the **1st-order** system.
#
cdef class Linear2ndOrderKernel(CythonKernel):
    cdef int m  # size of original 2nd-order system, m = n/2
    cdef DTYPE_t* M0
    cdef DTYPE_t* M1
    cdef DTYPE_t[:,::1] M0_arr
    cdef DTYPE_t[:,::1] M1_arr

    cdef void callback(self, RTYPE_t t) nogil
    cdef void compute(self, DTYPE_t* w_in, DTYPE_t* wp_out) nogil


# M2 u'' = M0 u + M1 u'
#
# reduced to
#
#     u' = v
#  M2 v' = M0 u + M1 u'
#
# The callback() method computes w',
# where w = (u1, v1, u2, v2, ..., um, vm).
#
# with the mass matrix "M2" constant with respect to time.
#
# Same note about "M0" and "M1" as above.
#
# Note that  n = 2*m  still denotes the size of the **1st-order** system.
#
cdef class Linear2ndOrderKernelWithMassMatrix(Linear2ndOrderKernel):
    # cdef classes are single inheritance only, so we have some duplication here
    # (since this is both a "linear 2nd-order kernel" as well as a "kernel with mass matrix").
    cdef DTYPE_t* LU
    cdef int* p
    cdef int* mincols
    cdef int* maxcols
    cdef DTYPE_t* wrk1  # n elements
    cdef DTYPE_t* wrk2  # m elements
    cdef DTYPE_t* wrk3  # m elements
    cdef DTYPE_t[:,::1] LU_arr
    cdef int[::1] p_arr
    cdef int[::1] mincols_arr
    cdef int[::1] maxcols_arr
    cdef DTYPE_t[::1] wrk_arr  # wrk1, wrk2, wrk3 all stored here

    cdef void callback(self, RTYPE_t t) nogil

