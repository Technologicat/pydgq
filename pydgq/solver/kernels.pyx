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

cimport pylu.dgesv as dgesv_c

# use fast math functions from <math.h>, available via Cython
#from libc.math cimport sin, cos, log, exp, sqrt


# A generic kernel for a linear 1st-order problem.
# (This is basically just a matrix-vector product.)
#
# The problem reads
#
#   w' = M w
#
# The matrix M must use C memory layout.
#
# To use this kernel with NumPy arrays, do something like this in a Cython module
# (but using the initial values and matrix from your actual problem):
#
#   cdef int n = 4
#   cdef double[::1] w   = np.ones( (n,), dtype=np.float64, order="C" )
#   cdef double[:,::1] M = np.eye( n, dtype=np.float64, order="C" )
#
# and pass these w and M (M as "data", cast to void*) to odesolve.ivp().
#
cdef void f_lin_1st(double* w, double* out, int n, double t, void* data) nogil:
    # This kernel uses the user data pointer to store the matrix M.
    #
    # We could update the matrix here (by writing to M) if needed,
    # but this simple linear example needs only a constant matrix.
    #
    cdef double* M = <double*>data

    cdef int j, k
    for j in range(n):  # row
        # w' = M w
        for k in range(n):  # column
            out[j] = M[j*n + k] * w[k]


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
# The matrices M0 and M1 are stored in the data pointer.
# The pointer is first cast to type double*.
# The matrix M0 is stored in the first m*m elements (C layout),
# then M1 in the next m*m elements (C layout).
#
# To do this, use e.g.
#
#    cdef double[:,::1] data = np.empty( (n,m), dtype=np.float64, order="C" )
#
# where n and m are defined as above (n = 2*m). The first m rows store M0,
# and the final m rows store M1.
#
# The parameter n specifies the size of the *1st-order* system; n is always even.
#
cdef void f_lin_2nd(double* w, double* out, int n, double t, void* data) nogil:
    cdef int m = n // 2  # size of the 2nd-order system

    # This kernel uses the user data pointer to store two matrices.
    #
    cdef double* ddata = <double*>data
    cdef double* M0 = &ddata[0]
    cdef double* M1 = &ddata[m*m]

    cdef int j, k
    for j in range(m):  # row
        # u' = v
        out[2*j] = w[2*j + 1]

        # v' = M0 u + M1 v
        for k in range(m):  # column
            out[2*j + 1] = M0[j*m + k] * w[2*j]  +  M1[j*m + k] * w[2*j + 1]


# First-order problem that has a nontrivial mass matrix on the LHS:
#
#   A w' = M w
#
# Mainly this is a code demonstration, but this can be convenient if the problem
# is small enough to be solvable using dense matrices and direct methods,
# but still too large to invert A explicitly.
#
# (If A is small enough to invert, just write  w' = inv(A) M w  and use f_lin_1st()
#  with  inv(A) M  as the matrix.)
#
# We obtain w' by first computing A w', and then solving for w':
#
#   g := M w
#   A w' = g
#
# This routine assumes that A has been LU decomposed using PyLU.
# To use this kernel, example:
#
#   cdef int n = 4
#   cdef double[::1] w    = np.ones( (n,), dtype=np.float64, order="C" )
#   cdef double[::1] work = np.empty( (n,), dtype=np.float64, order="C" )
#   cdef double[:,::1] A  = np.eye( n, dtype=np.float64, order="C" )
#   cdef double[:,::1] M  = np.eye( n, dtype=np.float64, order="C" )
#
#   cdef double[:,::1] LU
#   cdef int[::1] p
#   LU,p = pylu.dgesv.lup_packed(A)
#
#   cdef pydgq.solver.kernels.lin_mass_matrix_data data
#   data.LU   = &LU[0,0]
#   data.p    = &p[0]
#   data.M    = &M[0,0]
#   data.work = &work[0]
#
# and pass these w and data to odesolve.ivp().
#
cdef void f_lin_1st_with_mass(double* w, double* out, int n, double t, void* data) nogil:
    cdef lin_mass_matrix_data* pdata = <lin_mass_matrix_data*>data

    # compute g = M w, save result in pdata.work
    f_lin_1st( w, pdata.work, n, t, pdata.M )

    # solve linear equation system A w' = g
    dgesv_c.solve_decomposed_c( pdata.LU, pdata.p, pdata.work, out, n )


# Second-order problem with a nontrivial mass matrix:
#
#   M2 u'' = M0 u + M1 u'  (commonly encountered in mechanics)
#
# Companion form:
#
#   M2 v' = M0 u + M1 u'
#      u' = v
#
# Notes on usage:
#
#  - data.LU, data.p must contain the LU decomposition of M2 (note: m by m matrix!)
#  - data.M must contain M0, M1 in format accepted by f_lin_2nd()
#  - data.work must have space for 2*n ( = 4*m ) doubles
#  - input n = 2*m = number of DOFs in the corresponding 1st-order problem
#
cdef void f_lin_2nd_with_mass(double* w, double* out, int n, double t, void* data) nogil:
    cdef lin_mass_matrix_data* pdata = <lin_mass_matrix_data*>data

    # compute RHS
    #
    # Note that the RHS is the same as in the original 2nd-order case,
    # only the interpretation of the result differs. Thus we can use f_lin_2nd()
    # to do the actual computation.
    #
    # M must be in the format accepted by f_lin_2nd() (containing M0 and M1).
    #
    f_lin_2nd( w, pdata.work, n, t, pdata.M )

    # use the unused half of work as scratch space for reordering DOFs
    #
    # (we must undo the interleaving to use dgesv)
    #
    cdef int m = n // 2
    cdef double* work_in  = &pdata.work[n]    # first m elements of scratch space
    cdef double* work_out = &pdata.work[n+m]  # last  m elements of scratch space
    cdef int j
    for j in range(m):
        work_in[j] = pdata.work[2*j+1]  # DOFs corresponding to M2 v'

    # solve  M2 v' = work_in  for v'
    dgesv_c.solve_decomposed_c( pdata.LU, pdata.p, work_in, work_out, m )

    # write output
    for j in range(m):
        out[2*j]   = pdata.work[2*j]  # u'
        out[2*j+1] = work_out[j]      # v'

