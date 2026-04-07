# -*- coding: utf-8 -*-
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True
#
"""Example Cython kernel: cosine ODE with phase-shifted components.

This demonstrates how to write a custom Cython kernel for the pydgq solver.
The ODE system is  w'_j = cos(phi0_j + omega*t),  j = 0, ..., n-1,
with known analytical solution  w_j = (1/omega) * sin(phi0_j + omega*t).

See also the PythonKernel version in tests/test_integrators.py.
"""

import numpy as np

from pydgq.solver.types cimport RTYPE_t, DTYPE_t
from pydgq.solver.types  import RTYPE,   DTYPE
from pydgq.solver.kernel_interface cimport CythonKernel

from libc.math cimport sin, cos, M_PI


cdef class MyKernel(CythonKernel):
    def __init__(self, n, omega):
        """A cosine kernel with phase-shifted components.

        Parameters:
            n : int
                Number of DOFs.
            omega : float
                Angular frequency (rad/s).
        """
        CythonKernel.__init__(self, n)
        self.omega = omega

    cdef void callback(self, RTYPE_t t) noexcept nogil:
        cdef RTYPE_t phi0_j
        cdef int j
        for j in range(self.n):
            phi0_j = (<RTYPE_t>(j+1) / self.n) * 2. * M_PI
            self.out[j] = cos(phi0_j + self.omega*t)

    def reference_solution(self, RTYPE_t[::1] tt):
        """Analytical solution for testing."""
        cdef int nt = tt.shape[0]
        cdef DTYPE_t[:,::1] ww = np.empty( (nt,self.n), dtype=DTYPE, order="C" )
        cdef RTYPE_t[::1] zero = np.array( (0.,), dtype=RTYPE, order="C" )

        cdef DTYPE_t[::1] tmp
        cdef RTYPE_t phi0_j
        cdef int j, k
        for j in range(self.n):
            phi0_j = (<RTYPE_t>(j+1) / self.n) * 2. * M_PI
            tmp = self.__sol(tt,phi0_j) - self.__sol(zero,phi0_j)
            for k in range(nt):
                ww[k,j] = tmp[k]

        return np.asanyarray(ww)

    def __sol(self, RTYPE_t[::1] tt, RTYPE_t phi0):
        cdef int nt = tt.shape[0]
        cdef DTYPE_t[::1] out = np.empty( (nt,), dtype=DTYPE, order="C" )
        cdef int j
        for j in range(nt):
            out[j] = 1./self.omega * sin(phi0 + self.omega*tt[j])
        return np.asanyarray(out)
