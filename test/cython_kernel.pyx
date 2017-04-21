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

from pydgq.solver.types cimport RTYPE_t, DTYPE_t
from pydgq.solver.types  import RTYPE,   DTYPE
from pydgq.solver.kernel_interface cimport CythonKernel

from libc.math cimport sin, cos, log, exp, sqrt

cdef extern from "math.h":
    double M_PI

# A simple cosine kernel with phase-shifted components.
#
# See python_kernel_test.py for a pure-Python version of this.
#
# Note that Cython requires us to put the cdef class declarations into cython_kernel.pxd.
#
# The custom kernel only needs to override callback(); even __init__ is not strictly needed,
# unless adding some custom parameters (like here).
#
cdef class MyKernel(CythonKernel):
    def __init__(self, n, omega):  # omega : rad/s
        # super
        CythonKernel.__init__(self, n)

        # custom init
        self.omega = omega

    # implementation of callback()
    #
    cdef void callback(self, RTYPE_t t) nogil:
        cdef RTYPE_t phi0_j
        cdef int j
        for j in range(self.n):
            phi0_j = (<RTYPE_t>(j+1) / self.n) * 2. * M_PI
            self.out[j] = cos(phi0_j + self.omega*t)

    # known analytical solution, for testing the integrators
    #
    # (only needed by the testing script; usual custom kernels do not have this part)

    # one component
    def __sol(self, RTYPE_t[::1] tt, RTYPE_t phi0):
        cdef int nt = tt.shape[0]
        cdef DTYPE_t[::1] out = np.empty( (nt,), dtype=DTYPE, order="C" )
        cdef int j
        for j in range(nt):
            out[j] = 1./self.omega * sin(phi0 + self.omega*tt[j])
        return np.asanyarray(out)  # np.ndarray needed to support broadcast

    # all components
    def reference_solution(self, RTYPE_t[::1] tt):
        cdef int nt = tt.shape[0]
        cdef DTYPE_t[:,::1] ww = np.empty( (nt,self.n), dtype=DTYPE, order="C" )
        cdef RTYPE_t[::1] zero = np.array( (0.,), dtype=RTYPE, order="C" )  # the tt input to __sol() must be an array
 
        cdef DTYPE_t[::1] tmp
        cdef RTYPE_t phi0_j
        cdef int j, k
        for j in range(self.n):
            phi0_j = (<RTYPE_t>(j+1) / self.n) * 2. * M_PI
            tmp = self.__sol(tt,phi0_j) - self.__sol(zero,phi0_j)  # shift to account for the initial condition (all solution components start at zero)
            for k in range(nt):
                ww[k,j] = tmp[k]

        return np.asanyarray(ww)

