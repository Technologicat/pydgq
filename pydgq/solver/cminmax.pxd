# -*- coding: utf-8 -*-
#
# fast inline min/max functions for C code in Cython.

from __future__ import absolute_import

from pydgq.solver.types cimport RTYPE_t

# real
#
cdef inline RTYPE_t cfmin(RTYPE_t a, RTYPE_t b) nogil:
    return a if a < b else b
cdef inline RTYPE_t cfmax(RTYPE_t a, RTYPE_t b) nogil:
    return a if a > b else b

# int
#
cdef inline int cimin(int a, int b) nogil:
    return a if a < b else b
cdef inline int cimax(int a, int b) nogil:
    return a if a > b else b

# unsigned int
#
cdef inline unsigned int cuimin(unsigned int a, unsigned int b) nogil:
    return a if a < b else b
cdef inline unsigned int cuimax(unsigned int a, unsigned int b) nogil:
    return a if a > b else b

