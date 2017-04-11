# -*- coding: utf-8 -*-
#
# min/max functions for C code in Cython.

from __future__ import absolute_import

from pydgq.solver.pydgq_types cimport DTYPE_t

# fast inline min/max for C code
#
cdef inline DTYPE_t cfmin(DTYPE_t a, DTYPE_t b) nogil:
    return a if a < b else b
cdef inline DTYPE_t cfmax(DTYPE_t a, DTYPE_t b) nogil:
    return a if a > b else b

cdef inline int cimin(int a, int b) nogil:
    return a if a < b else b
cdef inline int cimax(int a, int b) nogil:
    return a if a > b else b

cdef inline unsigned int cuimin(unsigned int a, unsigned int b) nogil:
    return a if a < b else b
cdef inline unsigned int cuimax(unsigned int a, unsigned int b) nogil:
    return a if a > b else b

