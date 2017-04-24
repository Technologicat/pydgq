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
"""Compensated summation (Kahan algorithm), Python/Cython.

This Python interface exposes compensated cumulative summation of rank-1 arrays.
"""

from __future__ import division, print_function, absolute_import

from pydgq.solver.types cimport RTYPE_t, ZTYPE_t
from pydgq.solver.types import RTYPE, ZTYPE

import numpy as np


#########################################################################################
# Python interface
#########################################################################################

# These are provided to produce accurate cumulative sums.
# Not currently used by pydgq.

def cumsum1d_compensated( data ):
    """def cumsum1d_compensated( data ):

Like np.cumsum(), but using compensated summation (Kahan algorithm).

Implemented only for rank-1 np.arrays of dtypes double (np.float64) and double complex (np.complex128).
"""
    cdef RTYPE_t[::1] inr, outr
    cdef ZTYPE_t[::1] inz, outz

    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("This function supports only rank-1 arrays, got an array of rank %d" % (data.ndim))

        if data.dtype == ZTYPE:
            inz  = data
            outz = np.empty_like(data)
            cs1dz( &inz[0], &outz[0], data.shape[0] )  # modifies outz in-place
            return np.asanyarray(outz)
        elif data.dtype == RTYPE:
            inr  = data
            outr = np.empty_like(data)
            cs1dr( &inr[0], &outr[0], data.shape[0] )  # modifies outr in-place
            return np.asanyarray(outr)
        else:
            raise TypeError("Unsupported dtype '%s' for cumsum1d_compensated() with np.ndarray; valid: %s, %s" % (data.dtype, ZTYPE, RTYPE))
    else:
        raise TypeError("Unsupported argument type '%s' for cumsum1d_compensated(); %s" % (type(data), np.ndarray))


#########################################################################################
# Cython interface
#########################################################################################

# data : in, values to sum (length n)
# out  : out, cumulative sum (length n-1)
# n    : length of input data
#
cdef void cs1dr( RTYPE_t* data, RTYPE_t* out, unsigned int n ) nogil:
    cdef RTYPE_t s = data[0]
    cdef RTYPE_t c = 0.0
    cdef unsigned int j
    out[0] = s
    for j in range(1,n):
        accumulate( &s, &c, data[j] )
        out[j] = s

# data : in, values to sum (length n)
# out  : out, cumulative sum (length n-1)
# n    : length of input data
#
cdef void cs1dz( ZTYPE_t* data, ZTYPE_t* out, unsigned int n ) nogil:
    cdef ZTYPE_t s = data[0]
    cdef ZTYPE_t c = 0.0
    cdef unsigned int j
    out[0] = s
    for j in range(1,n):
        accumulatez( &s, &c, data[j] )
        out[j] = s

