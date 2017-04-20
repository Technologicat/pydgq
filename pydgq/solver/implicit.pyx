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
"""Classical implicit integrators to update w by one timestep."""

from __future__ import division, print_function, absolute_import

import numpy as np

from pydgq.solver.types cimport DTYPE_t, RTYPE_t
from pydgq.solver.types import DTYPE, RTYPE
from pydgq.solver.kernel_interface cimport KernelBase
from pydgq.solver.integrator_interface cimport ImplicitIntegrator


cdef class IMR(ImplicitIntegrator):
    def __init__(self, KernelBase rhs, int maxit):
        """def __init__(self, KernelBase rhs, int maxit):

Implicit midpoint rule.

Parameters:
    as in ancestor (pydgq.solver.integrator_interface.ImplicitIntegrator).
"""
        # super
        ImplicitIntegrator.__init__(self, name="IMR", rhs=rhs, maxit=maxit)

        cdef int n_space_dofs = rhs.n
        self.wrk_arr = np.empty( (4*n_space_dofs,), dtype=DTYPE, order="C" )
        self.wrk     = &(self.wrk_arr[0])

    cdef int call(self, DTYPE_t* w, RTYPE_t t, RTYPE_t dt) nogil:
        cdef unsigned int j, m=-1, m2, nequals
        cdef int n_space_dofs = self.rhs.n
        cdef int success = 0

        cdef DTYPE_t* whalf = self.wrk                   # iterative approximation of w(k+1/2)
        cdef DTYPE_t* wp    = &(self.wrk[n_space_dofs])  # iterative approximation of w'(k+1/2)

        # Iterative approximations of w(k+1)
        cdef DTYPE_t* wcur = &(self.wrk[2*n_space_dofs])
        cdef DTYPE_t* wnew = &(self.wrk[3*n_space_dofs])

        cdef RTYPE_t thalf = t + 0.5*dt

        # Trivial initial guess: w(k+1) = w(k)
        #
        # This leads to the *second* iterate being the Forward Euler prediction.
        #
        for j in range(n_space_dofs):
            wcur[j] = w[j]

        # Implicit iteration (Banach fixed point iteration)
        #
        for m in range(self.maxit):
            self.rhs.begin_iteration(m)  # inform RHS kernel that a new iteration starts

            for j in range(n_space_dofs):
                whalf[j] = 0.5 * (w[j] + wcur[j])  # estimate midpoint value of w
            self.rhs.call(whalf, wp, thalf)
            for j in range(n_space_dofs):
                wnew[j] = w[j] + dt*wp[j]  # new estimate of w(k+1)

            # Check convergence; break early if converged to within machine precision.
            #
            # If the timestep is reasonable i.e. small enough to make the truncation error small
            # and actually produce correct results, typically the solution converges to within
            # machine precision in a large majority of cases.
            #
            # If your problem is too large for this to be feasible, set self.maxit accordingly.
            #
            # See
            #
            #  https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
            #  http://stackoverflow.com/questions/3281237/is-the-use-of-machine-epsilon-appropriate-for-floating-point-equality-tests
            #
            nequals = 0
            for j in range(n_space_dofs):
                if wnew[j] == wcur[j]:
                    nequals += 1
                wcur[j] = wnew[j]  # update w(k+1) (old value no longer needed)
            if nequals == n_space_dofs:
                success = 1
                break

        # if timestep failed, try a few more steps with relaxation
        #
        if not success:
            for m2 in range(5):  # we want to be able to return the original value of m
                self.rhs.begin_iteration(self.maxit + m2)  # inform RHS kernel that a new iteration starts

                for j in range(n_space_dofs):
                    whalf[j] = 0.5 * (w[j] + wnew[j])  # estimate midpoint value of w
                self.rhs.call(whalf, wp, thalf)
                for j in range(n_space_dofs):
                    wnew[j] = w[j] + dt*wp[j]  # new estimate of w(k+1)

                nequals = 0
                for j in range(n_space_dofs):
                    if wnew[j] == wcur[j]:
                        nequals += 1
                    wnew[j] = 0.5*wnew[j] + 0.5*wcur[j]  # update w(k+1), with relaxation (old value no longer needed)
                if nequals == n_space_dofs:
                    break

        # w(k+1) computed, update it into w
        for j in range(n_space_dofs):
            w[j] = wnew[j]

        return (m+1)  # Return the number of iterations taken (note that the numbering starts from m=0). If self.maxit iterations were taken, the convergence failed.


cdef class BE(ImplicitIntegrator):
    def __init__(self, KernelBase rhs, int maxit):
        """def __init__(self, KernelBase rhs, int maxit):

Backward Euler (implicit Euler).

Parameters:
    as in ancestor (pydgq.solver.integrator_interface.ImplicitIntegrator).
"""
        # super
        ImplicitIntegrator.__init__(self, name="BE", rhs=rhs, maxit=maxit)

        cdef int n_space_dofs = rhs.n
        self.wrk_arr = np.empty( (3*n_space_dofs,), dtype=DTYPE, order="C" )
        self.wrk     = &(self.wrk_arr[0])

    cdef int call(self, DTYPE_t* w, RTYPE_t t, RTYPE_t dt) nogil:
        cdef unsigned int j, m=-1, m2, nequals
        cdef int n_space_dofs = self.rhs.n
        cdef int success = 0

        cdef DTYPE_t* wp   = self.wrk     # iterative approximation of w'(k+1)

        # Iterative approximations of w(k+1)
        cdef DTYPE_t* wcur = &(self.wrk[n_space_dofs])
        cdef DTYPE_t* wnew = &(self.wrk[2*n_space_dofs])

        cdef RTYPE_t tend = t + dt

        # Trivial initial guess: w(k+1) = w(k)
        #
        # This leads to the *second* iterate being the Forward Euler prediction.
        #
        for j in range(n_space_dofs):
            wcur[j] = w[j]

        # Implicit iteration (Banach fixed point iteration)
        #
        for m in range(self.maxit):
            self.rhs.begin_iteration(m)  # inform RHS kernel that a new iteration starts

            self.rhs.call(wcur, wp, tend)
            for j in range(n_space_dofs):
                wnew[j] = w[j] + dt*wp[j]  # new estimate of w(k+1)

            # Check convergence; break early if converged to within machine precision.
            #
            nequals = 0
            for j in range(n_space_dofs):
                if wnew[j] == wcur[j]:
                    nequals += 1
                wcur[j] = wnew[j]  # update w(k+1) (old value no longer needed)
            if nequals == n_space_dofs:
                success = 1
                break

        # if timestep failed, try a few more steps with relaxation
        #
        if not success:
            for m2 in range(5):  # we want to be able to return the original value of m
                self.rhs.begin_iteration(self.maxit + m2)  # inform RHS kernel that a new iteration starts

                self.rhs.call(wnew, wp, tend)
                for j in range(n_space_dofs):
                    wnew[j] = w[j] + dt*wp[j]  # new estimate of w(k+1)

                nequals = 0
                for j in range(n_space_dofs):
                    if wnew[j] == wcur[j]:
                        nequals += 1
                    wnew[j] = 0.5*wnew[j] + 0.5*wcur[j]  # update w(k+1), with relaxation (old value no longer needed)
                if nequals == n_space_dofs:
                    break

        # w(k+1) computed, update it into w
        for j in range(n_space_dofs):
            w[j] = wnew[j]

        return (m+1)  # Return the number of iterations taken (note that the numbering starts from m=0). If self.maxit iterations were taken, the convergence failed.

