# -*- coding: utf-8 -*-
#
# Examples for Cython-based computational kernels for evaluating the f() in  w' = f(w).

from __future__ import division, print_function, absolute_import

from pydgq.solver.kernel_interface cimport CythonKernel

cdef class Linear1stOrderKernel(CythonKernel):
    cdef double* M
    cdef double[:,::1] M_arr

    cdef void callback(self, double t) nogil
    cdef void compute(self, double* w_in, double* wp_out) nogil

cdef class Linear1stOrderKernelWithMassMatrix(Linear1stOrderKernel):
    cdef double* LU
    cdef int* p
    cdef double* wrk
    cdef double[:,::1] LU_arr
    cdef int[::1] p_arr
    cdef double[::1] wrk_arr

    cdef void callback(self, double t) nogil

cdef class Linear2ndOrderKernel(CythonKernel):
    cdef int m  # size of original 2nd-order system, m = n/2
    cdef double* M0
    cdef double* M1
    cdef double[:,::1] M0_arr
    cdef double[:,::1] M1_arr

    cdef void callback(self, double t) nogil
    cdef void compute(self, double* w_in, double* wp_out) nogil

# cdef classes are single inheritance only, so we have some duplication here
# (since this is both a "linear 2nd-order kernel" as well as a "kernel with mass matrix").
#
cdef class Linear2ndOrderKernelWithMassMatrix(Linear2ndOrderKernel):
    cdef double* LU
    cdef int* p
    cdef double* wrk1  # n elements
    cdef double* wrk2  # m elements
    cdef double* wrk3  # m elements
    cdef double[:,::1] LU_arr
    cdef int[::1] p_arr
    cdef double[::1] wrk_arr

    cdef void callback(self, double t) nogil

