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

from pydgq.solver.kernel_interface cimport CythonKernel
from pydgq.solver.types cimport RTYPE_t

cdef class MyKernel(CythonKernel):
    # custom cdef data attributes go here
    cdef RTYPE_t omega

    # tell Cython that we would like to override callback() with our implementation
    cdef void callback(self, RTYPE_t t) nogil

