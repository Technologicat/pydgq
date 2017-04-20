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
"""Example Cython-based computational kernels for evaluating the f() in  w' = f(w, t).

This module provides generic kernels for autonomous linear problems of 1st and 2nd order,
with and without mass matrices on the LHS.

(Autonomous: f = f(w), i.e. f does not explicitly depend on t.)

All matrices in this module are assumed to have C memory layout.

See the code and comments in pydgq/solver/builtin_kernels.pyx for details.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

import pylu.dgesv as dgesv
cimport pylu.dgesv as dgesv_c

from pydgq.solver.types cimport DTYPE_t, RTYPE_t
from pydgq.solver.types import DTYPE


# TODO: add an update hook for updating the matrices (if not constant in time)?
#
# The problem is that in a nonlinear problem, f() is called in the innermost loop:
#
#    timestep (base value of t changes)
#      nonlinear iteration (w changes)
#        sub-evaluation at some t inside the timestep (w changes; actual t used for f() evaluation changes)
#
# so when should the solver call the matrix update methods? (assuming "at every call" is too expensive)


cdef class Linear1stOrderKernel(CythonKernel):
#    cdef DTYPE_t* M
#    cdef DTYPE_t[:,::1] M_arr

    def __init__(self, int n, DTYPE_t[:,::1] M):
        """def __init__(self, int n, DTYPE_t[:,::1] M):

A generic kernel for a linear 1st-order problem.
This is basically just a matrix-vector product.

The problem reads

    w' = M w

Trivial example to invoke this kernel:

    from pydgq.solver.types import DTYPE
    from pydgq.solver.builtin_kernels import Linear1stOrderKernel
    from pydgq.solver.odesolve import ivp

    n   = 3  # number of DOFs in your 1st-order system
    w0  = np.ones( (n,), dtype=DTYPE, order="C" )  # your IC here
    M   = np.eye( n, dtype=DTYPE, order="C" )  # your "M" matrix here
    rhs = Linear1stOrderKernel(n, M)
    ivp( ..., rhs=rhs, w0=w0 )
"""
        # super
        CythonKernel.__init__(self, n)

        self.M_arr = M  # keepalive (FIXME: do we need to .copy() to be sure?)
        self.M = &(self.M_arr[0,0])  # get raw pointer

    cdef void callback(self, RTYPE_t t) nogil:  # t unused in this example
        self.compute(self.w, self.out)

    # Derived classes need this, outputting to a temporary output array,
    # so we have this general version with parametrized input/output.
    #
    # In most problem-specific kernels, we wouldn't need this, and could instead
    # perform the computation in callback(), saving one function call per RHS evaluation.
    #
    cdef void compute(self, DTYPE_t* w_in, DTYPE_t* wp_out) nogil:
        cdef int j, k
        for j in range(self.n):  # row
            # w' = M w
            wp_out[j] = 0.
            for k in range(self.n):  # column
                wp_out[j] += self.M[j*self.n + k] * w_in[k]


cdef class Linear1stOrderKernelWithMassMatrix(Linear1stOrderKernel):
#    cdef DTYPE_t* LU
#    cdef int* p
#    cdef DTYPE_t* wrk
#    cdef DTYPE_t[:,::1] LU_arr
#    cdef int[::1] p_arr
#    cdef DTYPE_t[::1] wrk_arr

    def __init__(self, int n, DTYPE_t[:,::1] M, DTYPE_t[:,::1] A):
        """def __init__(self, int n, DTYPE_t[:,::1] M, DTYPE_t[:,::1] A):

First-order problem that has a nontrivial (but constant-in-time) mass matrix on the LHS:

    A w' = M w

Note that if "A" is small enough to invert, one can instead write

    w' = inv(A) M w

and just use a Linear1stOrderKernel with  inv(A) M  as the matrix.

This class obtains w' by first computing A w', and then solving a linear equation system for w':

    g := M w
    A w' = g

Trivial example to invoke this kernel:

    from pydgq.solver.types import DTYPE
    from pydgq.solver.builtin_kernels import Linear1stOrderKernelWithMassMatrix
    from pydgq.solver.odesolve import ivp

    n   = 3  # number of DOFs in your 1st-order system
    w0  = np.ones( (n,), dtype=DTYPE, order="C" )  # your IC here
    A   = np.eye( n, dtype=DTYPE, order="C" )  # your "A" matrix here
    M   = np.eye( n, dtype=DTYPE, order="C" )  # your "M" matrix here
    rhs = Linear1stOrderKernelWithMassMatrix(n, M, A)
    ivp( ..., rhs=rhs, w0=w0 )
"""
        # super
        Linear1stOrderKernel.__init__(self, n, M)

        # LU decompose mass matrix
        #
        # TODO: need some variant of zgesv if DTYPE is ZTYPE
        #
        self.LU_arr, self.p_arr = dgesv.lup_packed(A)
        self.mincols_arr,self.maxcols_arr = dgesv.find_bands(self.LU_arr, tol=1e-15)
        self.wrk_arr = np.empty( (n,), dtype=DTYPE, order="C" )

        # get raw pointers for C access
        self.LU      = &(self.LU_arr[0,0])
        self.p       = &(self.p_arr[0])
        self.mincols = &(self.mincols_arr[0])
        self.maxcols = &(self.maxcols_arr[0])
        self.wrk     = &(self.wrk_arr[0])

    cdef void callback(self, RTYPE_t t) nogil:  # t unused in this example
        # compute g = M w, save result in self.wrk
        self.compute(self.w, self.wrk)

        # solve linear equation system A w' = g  (g stored in self.wrk, result stored in self.out)
        #
        # TODO: need some variant of zgesv if DTYPE is ZTYPE
        #
        dgesv_c.solve_decomposed_banded_c( self.LU, self.p, self.mincols, self.maxcols, self.wrk, self.out, self.n )


cdef class Linear2ndOrderKernel(CythonKernel):
#    cdef int m
#    cdef DTYPE_t* M0
#    cdef DTYPE_t* M1
#    cdef DTYPE_t[:,::1] M0_arr
#    cdef DTYPE_t[:,::1] M1_arr

    def __init__(self, int n, DTYPE_t[:,::1] M0, DTYPE_t[:,::1] M1):
        """def __init__(self, int n, DTYPE_t[:,::1] M0, DTYPE_t[:,::1] M1):

A generic kernel for the linear 2nd-order problem:

    u'' = M0 u + M1 u'

Following the companion method, we define

    v := u'

obtaining a twice larger 1st-order problem

    v' = M0 u + M1 v
    u' = v

We now define

    w := (u1, v1, u2, v2, ..., um, vm)

where m is the number of DOFs of the original 2nd-order system.

Given w, M0 and M1, this class computes w'.

The parameter n specifies the size of the *1st-order* system; n is always even.

Trivial example to invoke this kernel:

    from pydgq.solver.types import DTYPE
    from pydgq.solver.builtin_kernels import Linear2ndOrderKernel
    from pydgq.solver.odesolve import ivp

    m   = 3        # number of DOFs in your 2nd-order system here
    n   = 2*m      # corresponding number of DOFs in the reduced 1st-order system (always 2*m)
    w0  = np.empty( (n,), dtype=DTYPE, order="C" )
    w0[0::2] = 0.  # your IC on u  here
    w0[1::2] = 1.  # your IC on u' here
    M0  = np.eye( m, dtype=DTYPE, order="C" )  # your "M0" matrix here
    M1  = np.eye( m, dtype=DTYPE, order="C" )  # your "M1" matrix here
    rhs = Linear1stOrderKernel(n, M0, M1)  # NOTE: size parameter is n, not m
    ivp( ..., rhs=rhs, w0=w0 )
"""
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

    cdef void callback(self, RTYPE_t t) nogil:  # t unused in this example
        self.compute(self.w, self.out)

    # Derived classes need this, outputting to a temporary output array,
    # so we have this general version with parametrized input/output.
    #
    # In most problem-specific kernels, we wouldn't need this, and could instead
    # perform the computation in callback(), saving one function call per RHS evaluation.
    #
    cdef void compute(self, DTYPE_t* w_in, DTYPE_t* wp_out) nogil:
        cdef int j, k
        for j in range(self.m):  # row
            # u' = v
            wp_out[2*j] = w_in[2*j + 1]

            # v' = M0 u + M1 v
            wp_out[2*j + 1] = 0.
            for k in range(self.m):  # column
                wp_out[2*j + 1] += self.M0[j*self.m + k] * w_in[2*j]  +  self.M1[j*self.m + k] * w_in[2*j + 1]


# cdef classes are single inheritance only, so we have some duplication here
# (since this is both a "linear 2nd-order kernel" as well as a "kernel with mass matrix").
#
cdef class Linear2ndOrderKernelWithMassMatrix(Linear2ndOrderKernel):
#    cdef DTYPE_t* LU
#    cdef int* p
#    cdef DTYPE_t* wrk1
#    cdef DTYPE_t* wrk2
#    cdef DTYPE_t* wrk3
#    cdef DTYPE_t[:,::1] LU_arr
#    cdef int[::1] p_arr
#    cdef DTYPE_t[::1] wrk_arr

    def __init__(self, int n, DTYPE_t[:,::1] M0, DTYPE_t[:,::1] M1, DTYPE_t[:,::1] M2):
        """def __init__(self, int n, DTYPE_t[:,::1] M0, DTYPE_t[:,::1] M1, DTYPE_t[:,::1] M2):

A generic kernel for the linear 2nd-order problem with a nontrivial (but constant-in-time) mass matrix:

    M2 u'' = M0 u + M1 u'

This problem is commonly encountered in mechanics.

Companion form:

    M2 v' = M0 u + M1 u'
       u' = v

The parameter n specifies the size of the *1st-order* system; n is always even.

Note that the kernel needs to solve only an  m x m  linear system for the DOFs representing v';
the other m DOFs (u') are obtained directly.

Trivial example to invoke this kernel:

    from pydgq.solver.types import DTYPE
    from pydgq.solver.builtin_kernels import Linear2ndOrderKernelWithMassMatrix
    from pydgq.solver.odesolve import ivp

    m   = 3        # number of DOFs in your 2nd-order system here
    n   = 2*m      # corresponding number of DOFs in the reduced 1st-order system (always 2*m)
    w0  = np.empty( (n,), dtype=DTYPE, order="C" )
    w0[0::2] = 0.  # your IC on u  here
    w0[1::2] = 1.  # your IC on u' here
    M0  = np.eye( m, dtype=DTYPE, order="C" )  # your "M0" matrix here
    M1  = np.eye( m, dtype=DTYPE, order="C" )  # your "M1" matrix here
    M2  = np.eye( m, dtype=DTYPE, order="C" )  # your "M2" matrix here
    rhs = Linear2ndOrderKernelWithMassMatrix(n, M0, M1, M2)  # NOTE: size parameter is n, not m
    ivp( ..., rhs=rhs, w0=w0 )
"""
        # super
        Linear2ndOrderKernel.__init__(self, n, M0, M1)

        # LU decompose mass matrix
        #
        # TODO: need some variant of zgesv if DTYPE is ZTYPE
        #
        self.LU_arr, self.p_arr = dgesv.lup_packed(M2)
        self.mincols_arr,self.maxcols_arr = dgesv.find_bands(self.LU_arr, tol=1e-15)
        self.wrk_arr = np.empty( (2*n,), dtype=DTYPE, order="C" )  # to avoid unnecessary memory fragmentation, allocate both work arrays as one block

        # get raw pointers for C access
        self.LU      = &(self.LU_arr[0,0])
        self.p       = &(self.p_arr[0])
        self.mincols = &(self.mincols_arr[0])
        self.maxcols = &(self.maxcols_arr[0])
        self.wrk1    = &(self.wrk_arr[0])         # first n elements of work space
        self.wrk2    = &(self.wrk_arr[n])         # next m elements of work space
        self.wrk3    = &(self.wrk_arr[n+self.m])  # last m elements of work space

    cdef void callback(self, RTYPE_t t) nogil:  # t unused in this example
        # compute RHS (i.e. w'), store result in wrk1
        #
        # Note that the RHS is exactly the same as in Linear2ndOrderKernel,
        # only the interpretation of the result differs.
        #
        # Thus we can use super's compute() to evaluate the RHS.
        #
        self.compute(self.w, self.wrk1)

        # reorder DOFs, store result in wrk2
        #
        # We must undo the interleaving  w' = ( u'1, M2 v'1, u'2, M2 v'2,..., u'm, M2 v'm )
        # to use pylu.dgesv on the  M2 v'  DOFs only.
        #
        cdef int j
        for j in range(self.m):
            self.wrk2[j] = self.wrk1[2*j+1]  # DOFs corresponding to M2 v'

        # solve  M2 v' = wrk2  for v', store result in wrk3
        #
        # TODO: need some variant of zgesv if DTYPE is ZTYPE
        #
        dgesv_c.solve_decomposed_banded_c( self.LU, self.p, self.mincols, self.maxcols, self.wrk2, self.wrk3, self.m )

        # write output
        for j in range(self.m):
            self.out[2*j]   = self.wrk1[2*j]  # u'
            self.out[2*j+1] = self.wrk3[j]    # v'

