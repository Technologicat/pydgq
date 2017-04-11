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
"""Compensated summation (Kahan algorithm), Python/Cython."""

from __future__ import division, print_function, absolute_import

from . cimport types
from . cimport compsum

import numpy as np

#########################################################################################
# Cumulative sum using compensated summation
#########################################################################################

# These are provided to produce accurate cumulative sums.
# Not currently used by pydgq.

def cumsum1d_compensated( data ):
    """Like np.cumsum(), but using compensated summation (Kahan algorithm).

    Implemented only for rank-1 np.arrays of dtypes double (np.float64) and double complex (np.complex128).

    """
    cdef types.DTYPE_t[::1]  inr, outr
    cdef types.DTYPEZ_t[::1] inz, outz

    if isinstance(data, np.ndarray):
        if data.dtype == types.DTYPEZ:
            inz  = data
            outz = np.empty_like(data)
            cs1dz( &inz[0], &outz[0], np.size(data) )  # modifies outz in-place
            return np.asanyarray(outz)
        elif data.dtype == types.DTYPE:
            inr  = data
            outr = np.empty_like(data)
            cs1dr( &inr[0], &outr[0], np.size(data) )  # modifies outr in-place
            return np.asanyarray(outr)
        else:
            raise TypeError("Unsupported dtype '%s' for cumsum1d_compensated() with np.ndarray; valid: %s, %s" % (data.dtype, DTYPEZ, DTYPE))
    else:
        raise TypeError("Unsupported argument type '%s' for cumsum1d_compensated(); %s" % (type(data), np.ndarray))


cdef void cs1dr( types.DTYPE_t* data, types.DTYPE_t* out, unsigned int n ) nogil:
    cdef types.DTYPE_t s = data[0]
    cdef types.DTYPE_t c = 0.0
    cdef unsigned int j
    out[0] = s
    for j in range(1,n):
        accumulate( &s, &c, data[j] )
        out[j] = s

cdef void cs1dz( types.DTYPEZ_t* data, types.DTYPEZ_t* out, unsigned int n ) nogil:
    cdef types.DTYPEZ_t s = data[0]
    cdef types.DTYPEZ_t c = 0.0
    cdef unsigned int j
    out[0] = s
    for j in range(1,n):
        accumulatez( &s, &c, data[j] )
        out[j] = s

