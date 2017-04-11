# -*- coding: utf-8 -*-
#
# min/max functions for C code in Cython.

from . cimport types

# fast inline min/max for C code
#
cdef inline types.DTYPE_t cfmin(types.DTYPE_t a, types.DTYPE_t b) nogil:
    return a if a < b else b
cdef inline types.DTYPE_t cfmax(types.DTYPE_t a, types.DTYPE_t b) nogil:
    return a if a > b else b

cdef inline int cimin(int a, int b) nogil:
    return a if a < b else b
cdef inline int cimax(int a, int b) nogil:
    return a if a > b else b

cdef inline unsigned int cuimin(unsigned int a, unsigned int b) nogil:
    return a if a < b else b
cdef inline unsigned int cuimax(unsigned int a, unsigned int b) nogil:
    return a if a > b else b

