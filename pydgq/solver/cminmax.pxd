# -*- coding: utf-8 -*-
#
# min/max functions for C code in Cython.

cimport pydgq_types

# fast inline min/max for C code
#
cdef inline pydgq_types.DTYPE_t cfmin(pydgq_types.DTYPE_t a, pydgq_types.DTYPE_t b) nogil:
    return a if a < b else b
cdef inline pydgq_types.DTYPE_t cfmax(pydgq_types.DTYPE_t a, pydgq_types.DTYPE_t b) nogil:
    return a if a > b else b

cdef inline int cimin(int a, int b) nogil:
    return a if a < b else b
cdef inline int cimax(int a, int b) nogil:
    return a if a > b else b

cdef inline unsigned int cuimin(unsigned int a, unsigned int b) nogil:
    return a if a < b else b
cdef inline unsigned int cuimax(unsigned int a, unsigned int b) nogil:
    return a if a > b else b

