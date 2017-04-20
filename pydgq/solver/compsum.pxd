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

from __future__ import division, print_function, absolute_import

from pydgq.solver.types cimport RTYPE_t, ZTYPE_t

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
#     s :       in/out, accumulated sum, RTYPE_t* pointing to a single RTYPE_t
#     c :       in/out, accumulated correction, RTYPE_t* pointing to a single RTYPE_t
#     operand : in,     the new term to add to the sum, RTYPE_t
#
# We use pointers only for the ability to use "in/out" parameters.
#
# Usage:
#
#     # To initialize, set s and c to zero.
#     cdef RTYPE_t s = 0.0
#     cdef RTYPE_t c = 0.0
#
#     # Loop over the terms of the sum.
#     cdef int k
#     for k in range(10):
#       accumulate( &s, &c, my_data[k] )  # here my_data is an array of RTYPE_t containing the numbers to be summed.
#     # now:
#     #  - s is the result of the sum
#     #  - c contains the final correction term (that is too small to be representable in s)
#
# Or, slightly optimized (may matter with small arrays):
#
#     cdef RTYPE_t s = my_data[0]  # Each term in the sum is RTYPE_t, so any one of them is exactly representable.
#     cdef RTYPE_t c = 0.0         # Correction is zero at the beginning.
#
#     # Loop over the terms of the sum, starting from the second one.
#     cdef int k
#     for k in range(1,10):
#       accumulate( &s, &c, my_data[k] )
#     # now:
#     #  - s is the result of the sum
#     #  - c contains the final correction term (that is too small to be representable in s)
#
cdef inline void accumulate( RTYPE_t* s, RTYPE_t* c, RTYPE_t operand ) nogil:
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
    cdef RTYPE_t y = operand - c[0]
    cdef RTYPE_t t = s[0] + y
    c[0] = (t - s[0]) - y
    s[0] = t


# Version for complex numbers.
#
cdef inline void accumulatez( ZTYPE_t* s, ZTYPE_t* c, ZTYPE_t operand ) nogil:
    cdef ZTYPE_t y = operand - c[0]
    cdef ZTYPE_t t = s[0] + y
    c[0] = (t - s[0]) - y
    s[0] = t


# Cumulative summation for 1D data (like np.cumsum, but with compensated summation).
#
# See compsum.pyx for an explanation of the parameters.
#
cdef void cs1dr( RTYPE_t* data, RTYPE_t* out, unsigned int n ) nogil
cdef void cs1dz( ZTYPE_t* data, ZTYPE_t* out, unsigned int n ) nogil

