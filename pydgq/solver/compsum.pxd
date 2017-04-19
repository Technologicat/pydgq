# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from pydgq.solver.types cimport DTYPE_t, DTYPEZ_t

# Compensated summation (Kahan summation), as accumulation to a sum variable.
#
# For operands of wildly varying magnitudes, this helps accuracy a lot,
# at the cost of four times the arithmetic.
#
# Note that this only has any effect on accuracy when there are at least three terms to be summed.
# For two terms, the correction c will be computed, but will not be used to correct the sum.
#
# See:
#     http://en.wikipedia.org/wiki/Kahan_summation_algorithm
#
# Parameters:
#     s: accumulated sum, DTYPE_t* pointing to a single DTYPE_t
#     c: accumulated correction, DTYPE_t* pointing to a single DTYPE_t
#     operand: the new term to add to the sum, DTYPE_t
#
# We use pointers only for the ability to use "in/out" parameters.
#
# Usage:
#
#     # To initialize, set s and c to zero.
#     cdef DTYPE_t s = 0.0
#     cdef DTYPE_t c = 0.0
#
#     # Loop over the terms of the sum.
#     cdef int k
#     for k in range(10):
#       accumulate( &s, &c, my_data[k] )  # here my_data is an array of DTYPE_t containing the numbers to be summed.
#     # now s is the result of the sum
#
# Or, slightly optimized (may matter with small arrays):
#
#     cdef DTYPE_t s = my_data[0]  # Each term in the sum is DTYPE_t, so any one of them is exactly representable.
#     cdef DTYPE_t c = 0.0         # Correction is zero at the beginning.
#
#     # Loop over the terms of the sum, starting from the second one.
#     cdef int k
#     for k in range(1,10):
#       accumulate( &s, &c, my_data[k] )
#     # now s is the result of the sum
#
cdef inline void accumulate( DTYPE_t* s, DTYPE_t* c, DTYPE_t operand ) nogil:
    # In compensated summation (Kahan summation),
    #
    #   s += x
    #
    # becomes:
    #
    #   y = x - c          # Subtract the old correction. This works because typically |x| << |s|.
    #   t = s + y          # Accumulate y, but don't overwrite yet, we still need the old value of s to compute the new correction.
    #   c = (t - s) - y    # Update the correction: (new - old) values of the sum vs. what was actually added.
    #   s = t              # Update s.
    #
    # where s and c are initially zero (before the accumulation of the first term of the sum).
    #
    cdef DTYPE_t y = operand - c[0]
    cdef DTYPE_t t = s[0] + y
    c[0] = (t - s[0]) - y
    s[0] = t


# Version for complex numbers. Provided for completeness.
#
cdef inline void accumulatez( DTYPEZ_t* s, DTYPEZ_t* c, DTYPEZ_t operand ) nogil:
    cdef DTYPEZ_t y = operand - c[0]
    cdef DTYPEZ_t t = s[0] + y
    c[0] = (t - s[0]) - y
    s[0] = t


# Cumulative summation for 1D data (like np.cumsum, but with compensated summation).
#
cdef void cs1dr( DTYPE_t* data, DTYPE_t* out, unsigned int n ) nogil
cdef void cs1dz( DTYPEZ_t* data, DTYPEZ_t* out, unsigned int n ) nogil

