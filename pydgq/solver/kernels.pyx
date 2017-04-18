# -*- coding: utf-8 -*-
#
# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True
#
"""Example computational kernels for evaluating the f() in  w' = f(w, t).

Generic kernels for autonomous linear problems of 1st and 2nd order
are provided, with and without mass matrices on the LHS.

(Autonomous: f = f(w), i.e. f does not explicitly depend on t.)

If the problem is nonlinear, the options are to:

  - implement a custom kernel
  - linearize at each timestep, and then use a linear kernel (slow)
"""

from __future__ import division, print_function, absolute_import

import numpy as np

import pylu.dgesv as dgesv
cimport pylu.dgesv as dgesv_c

# use fast math functions from <math.h>, available via Cython
#from libc.math cimport sin, cos, log, exp, sqrt


###############
# Base classes
###############

cdef class KernelBase:
#    cdef double* w    # old state vector (memory owned by caller)
#    cdef double* out  # new state vector (memory owned by caller)
#    cdef int n        # n_space_dofs
#    cdef int timestep
#    cdef int iteration

    def __init__(self, int n):
        self.n = n

    # TODO: The solver calls this when it begins a new timestep or a new Banach/Picard iteration.
    #
    # This metadata is provided for the actual computational kernel; the kernel classes themselves do not need it.
    #
    # timestep : 0-based, 0 = initial condition, 1 = first timestep, 2 = second timestep, ...
    # iteration : 0-based
    #
    cdef void update_metadata(self, int timestep, int iteration) nogil:
        self.timestep  = timestep
        self.iteration = iteration

    # The call interface. TODO: The solver calls this when it wants to evaluate w'.
    #
    # Implemented in derived classes.
    #
    cdef void call(self, double* w, double* out, double t) nogil:
        pass


# Base class for kernels implemented in Cython.
#
# Cython kernels will run in nogil mode.
#
cdef class CythonKernel(KernelBase):

    # Implementation of call() for Cython kernels.
    #
    cdef void call(self, double* w, double* out, double t) nogil:
        self.w   = w
        self.out = out
        self.callback(t)

    # Hook for custom code.
    #
    # Default no-op kernel: w' = 0
    #
    # Override this method in derived classes to provide your computational kernel.
    #
    cdef void callback(self, double t) nogil:
        cdef int j
        for j in range(self.n):
            self.out[j] = 0.0


# Base class for kernels implemented in pure Python.
#
# Python kernels will acquire the gil for calling f().
#
cdef class PythonKernel(KernelBase):
#    cdef double[::1] w_arr
#    cdef double[::1] out_arr

    # Implementation of call() for Python kernels.
    #
    cdef void call(self, double* w, double* out, double t) nogil:
        self.w   = w
        self.out = out
        with gil:
            self.w_arr   = <double[:self.n:1]>w
            self.out_arr = <double[:self.n:1]>out
            self.callback(t)

    # Hook for custom code.
    #
    # Default no-op kernel: w' = 0
    #
    # Override this method in derived classes to provide your computational kernel.
    #
    # Python classes can use self.w_arr and self.out_arr to access self.w and self.out;
    # they are Python-accessible views to the same arrays.
    #
    def callback(self, double t):
        cdef int j
        for j in range(self.n):
            self.out[j] = 0.0


##############################
# Specific kernels (examples)
##############################

# A generic kernel for a linear 1st-order problem.
# (This is basically just a matrix-vector product.)
#
# The problem reads
#
#   w' = M w
#
# The matrix M must use C memory layout.
#
# To use this kernel with NumPy arrays, do something like this in your Python code
# (but using the initial values and matrix from your actual problem):
#
# n  = 4
# w0 = np.ones( (n,), dtype=np.float64, order="C" )
# M  = np.eye( n, dtype=np.float64, order="C" )
# k  = Linear1stOrderKernel(n, M)
# pydgq.odesolve.ivp( ..., kernel=k, w0=w0 )
#
cdef class Linear1stOrderKernel(CythonKernel):
#    cdef double* M
#    cdef double[:,::1] M_arr

    def __init__(self, int n, double[:,::1] M):
        # super
        CythonKernel.__init__(self, n)

        self.M_arr = M  # keepalive (FIXME: do we need to .copy() to be sure?)
        self.M = &(self.M_arr[0,0])  # get raw pointer

    cdef void callback(self, double t) nogil:  # t unused in this example
        self.compute(self.w, self.out)

    # Derived classes need this, outputting to a temporary output array,
    # so we have this general version with parametrized input/output.
    #
    # In most problem-specific kernels, we wouldn't need this, and could instead
    # perform the computation in callback(), saving one function call per RHS evaluation.
    #
    cdef void compute(self, double* w_in, double* wp_out) nogil:
        cdef int j, k
        for j in range(self.n):  # row
            # w' = M w
            for k in range(self.n):  # column
                wp_out[j] = self.M[j*self.n + k] * w_in[k]


# First-order problem that has a nontrivial (but constant-in-time) mass matrix on the LHS:
#
#   A w' = M w
#
# Mainly this is a code demonstration. If A is small enough to invert, just write
# w' = inv(A) M w  and use a Linear1stOrderKernel with  inv(A) M  as the matrix.
#
# Here we obtain w' by first computing A w', and then solving for w':
#
#   g := M w
#   A w' = g
#
# To use this kernel, example:
#
#   n  = 4
#   w0 = np.ones( (n,), dtype=np.float64, order="C" )
#   A  = np.eye( n, dtype=np.float64, order="C" )
#   M  = np.eye( n, dtype=np.float64, order="C" )
#   k  = Linear1stOrderKernelWithMassMatrix(n, M, A)
#   pydgq.odesolve.ivp( ..., kernel=k, w0=w0 )
#
cdef class Linear1stOrderKernelWithMassMatrix(Linear1stOrderKernel):
#    cdef double* LU
#    cdef int* p
#    cdef double* wrk
#    cdef double[:,::1] LU_arr
#    cdef int[::1] p_arr
#    cdef double[::1] wrk_arr

    def __init__(self, int n, double[:,::1] M, double[:,::1] A):
        # super
        Linear1stOrderKernel.__init__(self, n, M)

        # LU decompose mass matrix
        self.LU_arr, self.p_arr = dgesv.lup_packed(A)
        self.wrk_arr = np.empty( (n,), dtype=np.float64, order="C" )

        # get raw pointers for C access
        self.LU  = &(self.LU_arr[0,0])
        self.p   = &(self.p_arr[0])
        self.wrk = &(self.wrk_arr[0])

    cdef void callback(self, double t) nogil:  # t unused in this example
        # compute g = M w, save result in self.wrk
        self.compute(self.w, self.wrk)

        # solve linear equation system A w' = g  (g stored in self.wrk, result stored in self.out)
        dgesv_c.solve_decomposed_c( self.LU, self.p, self.wrk, self.out, self.n )


# A generic kernel for a linear 2nd-order problem, as commonly encountered in mechanics.
#
# The problem reads
#
#   u'' = M0 u + M1 u'
#
# Following the companion method, we define
#
#   v := u'
#
# obtaining a 1st-order problem
#
#   v' = M0 u + M1 v
#   u' = v
#
# We now define
#
#   w := (u1, v1, u2, v2, ..., um, vm)
#
# where m is the number of DOFs of the original 2nd-order system.
#
# Given w, M0 and M1, this routine computes w'.
#
# The parameter n specifies the size of the *1st-order* system; n is always even.
#
cdef class Linear2ndOrderKernel(CythonKernel):
#    cdef int m
#    cdef double* M0
#    cdef double* M1
#    cdef double[:,::1] M0_arr
#    cdef double[:,::1] M1_arr

    def __init__(self, int n, double[:,::1] M0, double[:,::1] M1):
        if n % 2 != 0:
            raise ValueError("For a 2nd-order problem reduced to a 1st-order one, n must be even; got %d" % (n))

        # super
        CythonKernel.__init__(self, n)

        self.m = n // 2

        # keepalive (FIXME: .copy() to be sure?)
        self.M0_arr = M0
        self.M1_arr = M1

        # get raw pointers for C access
        self.M0 = &(self.M0_arr[0,0])
        self.M1 = &(self.M1_arr[0,0])

    cdef void callback(self, double t) nogil:  # t unused in this example
        self.compute(self.w, self.out)

    # Derived classes need this, outputting to a temporary output array,
    # so we have this general version with parametrized input/output.
    #
    # In most problem-specific kernels, we wouldn't need this, and could instead
    # perform the computation in callback(), saving one function call per RHS evaluation.
    #
    cdef void compute(self, double* w_in, double* wp_out) nogil:
        cdef int j, k
        for j in range(self.m):  # row
            # u' = v
            wp_out[2*j] = w_in[2*j + 1]

            # v' = M0 u + M1 v
            for k in range(self.m):  # column
                wp_out[2*j + 1] = self.M0[j*self.m + k] * w_in[2*j]  +  self.M1[j*self.m + k] * w_in[2*j + 1]


# Second-order problem with a nontrivial (but constant-in-time) mass matrix:
#
#   M2 u'' = M0 u + M1 u'
#
# This is also commonly encountered in mechanics.
#
# Companion form:
#
#   M2 v' = M0 u + M1 u'
#      u' = v
#
# The parameter n specifies the size of the *1st-order* system; n is always even.
#
# Here we gain an advantage by considering the companion form; we need to solve only an  m x m  linear system
# for the DOFs representing v'; the other m DOFs (u') are obtained directly.
#
#
# cdef classes are single inheritance only, so we have some duplication here
# (since this is both a "linear 2nd-order kernel" as well as a "kernel with mass matrix").
#
cdef class Linear2ndOrderKernelWithMassMatrix(Linear2ndOrderKernel):
#    cdef double* LU
#    cdef int* p
#    cdef double* wrk1
#    cdef double* wrk2
#    cdef double* wrk3
#    cdef double[:,::1] LU_arr
#    cdef int[::1] p_arr
#    cdef double[::1] wrk_arr

    def __init__(self, int n, double[:,::1] M0, double[:,::1] M1, double[:,::1] M2):
        # super
        Linear2ndOrderKernel.__init__(self, n, M0, M1)

        # LU decompose mass matrix
        self.LU_arr, self.p_arr = dgesv.lup_packed(M2)
        self.wrk_arr = np.empty( (2*n,), dtype=np.float64, order="C" )  # to avoid unnecessary memory fragmentation, allocate both work arrays as one block

        # get raw pointers for C access
        self.LU   = &(self.LU_arr[0,0])
        self.p    = &(self.p_arr[0])
        self.wrk1 = &(self.wrk_arr[0])         # first n elements of work space
        self.wrk2 = &(self.wrk_arr[n])         # next m elements of work space
        self.wrk3 = &(self.wrk_arr[n+self.m])  # last m elements of work space

    cdef void callback(self, double t) nogil:  # t unused in this example
        # compute RHS, store result in wrk1
        #
        # Note that the RHS is exactly the same as in Linear2ndOrderKernel,
        # only the interpretation of the result differs.
        #
        # Thus we can use super's compute() to evaluate the RHS.
        #
        self.compute(self.w, self.wrk1)

        # reorder DOFs, store result in wrk2
        #
        # (we must undo the interleaving to use pylu.dgesv)
        #
        cdef int j
        for j in range(self.m):
            self.wrk2[j] = self.wrk1[2*j+1]  # DOFs corresponding to M2 v'

        # solve  M2 v' = wrk2  for v', store result in wrk3
        dgesv_c.solve_decomposed_c( self.LU, self.p, self.wrk2, self.wrk3, self.m )

        # write output
        for j in range(self.m):
            self.out[2*j]   = self.wrk1[2*j]  # u'
            self.out[2*j+1] = self.wrk3[j]    # v'

