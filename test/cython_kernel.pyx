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

import numpy as np

from pydgq.solver.types cimport RTYPE_t
from pydgq.solver.kernel_interface cimport CythonKernel

from libc.math cimport sin, cos, log, exp, sqrt

cdef extern from "math.h":
    double M_PI

# A simple cosine kernel with phase-shifted components.
#
# The custom kernel only needs to override callback(); even __init__ is not strictly needed,
# unless adding some custom parameters (like here).
#
class MyKernel(CythonKernel):
    # there's no need to cimport this module, so we can put the declaration here.
    cdef RTYPE_t omega

    def __init__(self, n, omega):  # omega : rad/s
        # super
        CythonKernel.__init__(self, n)

        # custom init
        self.omega = omega

    cdef void callback(self, RTYPE_t t) nogil:
        cdef RTYPE_t phi0_j
        cdef int j
        for j in range(self.n):
            phi0_j = (float(j+1) / self.n) * 2. * M_PI
            self.out[j] = cos(phi0_j + self.omega*t)

    # known analytical solution, for testing the integrators
    #
    def reference_solution(self, RTYPE_t[::1] tt):
        cdef RTYPE_t[::1]   tt = np.atleast_1d(tt)
        cdef DTYPE_t[:,::1] ww = np.empty( (tt.shape[0],self.n), dtype=DTYPE, order="C" )
 
        def sol(RTYPE_t[::1] t, RTYPE_t phi0):
            return 1./self.omega * np.sin(phi0 + self.omega*t)
 
        cdef RTYPE_t phi0_j
        cdef int j
        for j in range(self.n):
            phi0_j  = (float(j+1) / self.n) * 2. * M_PI
            ww[:,j] = sol(tt,phi0_j) - sol(0.,phi0_j)  # shift to account for the initial condition (all solution components start at zero)

        return np.asanyarray(ww)

