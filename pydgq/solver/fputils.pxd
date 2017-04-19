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
"""Floating-point flags detection."""

from __future__ import division, print_function, absolute_import

from pydgq.solver.types cimport DTYPE_t

cdef extern from "math.h":
    # fpclassify is actually a macro, so it does not have a specific input type.
    # Here we tell Cython we would like to use it for whatever type DTYPE_t is a typedef for.
    int fpclassify(DTYPE_t x) nogil
    int FP_INFINITE
    int FP_NAN
    int FP_NORMAL
    int FP_SUBNORMAL
    int FP_ZERO

# Check whether all components of w are denormal
#
# n : number of components in w
#
cdef inline int all_denormal( DTYPE_t* w, int n ) nogil:
    cdef unsigned int n_denormal = 0
    cdef unsigned int j
    for j in range(n):
        if fpclassify(w[j]) == FP_SUBNORMAL:
            n_denormal += 1

    # In practice this seems good enough, although it does run the theoretical risk
    # of triggering (with a nonzero probability) when the solution just passes through zero.
    #
    # The problem is that denormal numbers are so small that we cannot square them;
    # checking that the vector norm of the solution has been decreasing for N timesteps
    # (for some N) before triggering this is thus difficult.
    #
    if n_denormal == n:
        return 1
    else:
        return 0

# Check whether any component of w is nan or inf
#
# n : number of components in w
#
cdef inline int any_naninf( DTYPE_t* w, int n ) nogil:
    cdef unsigned int j
    cdef int c
    for j in range(n):
        c = fpclassify(w[j])
        if c == FP_NAN  or  c == FP_INFINITE:
            return 1
    return 0

