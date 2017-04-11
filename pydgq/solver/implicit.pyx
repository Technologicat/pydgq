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

cimport pydgq.solver.pydgq_types as pydgq_types
cimport pydgq.solver.kernels as kernels


# Implicit midpoint rule.
#
# wrk must have space for 4*n_space_dofs items.
#
cdef int IMR( kernels.kernelfuncptr f, pydgq_types.DTYPE_t* w, void* data, int n_space_dofs, pydgq_types.DTYPE_t t, pydgq_types.DTYPE_t dt, pydgq_types.DTYPE_t* wrk, int maxit ) nogil:
    cdef unsigned int j, m, m2, nequals
    cdef int success = 0

    cdef pydgq_types.DTYPE_t* whalf = wrk  # iterative approximation of w(k+1/2)
    cdef pydgq_types.DTYPE_t* wp = &wrk[n_space_dofs]     # iterative approximation of w'(k+1/2)

    # Iterative approximations of w(k+1)
    cdef pydgq_types.DTYPE_t* wcur = &wrk[2*n_space_dofs]
    cdef pydgq_types.DTYPE_t* wnew = &wrk[3*n_space_dofs]

    cdef pydgq_types.DTYPE_t thalf = t + 0.5*dt

    # Trivial initial guess: w(k+1) = w(k)
    #
    # This leads to the *second* iterate being the Forward Euler prediction.
    #
    for j in range(n_space_dofs):
        wcur[j] = w[j]

    # Implicit iteration (Banach fixed point iteration)
    #
    for m in range(maxit):
        for j in range(n_space_dofs):
            whalf[j] = 0.5 * (w[j] + wcur[j])  # estimate midpoint value of w
        f(whalf, wp, n_space_dofs, thalf, data)
        for j in range(n_space_dofs):
            wnew[j] = w[j] + dt*wp[j]  # new estimate of w(k+1)

        # Check convergence; break early if converged to within machine precision.
        #
        # If the timestep is reasonable i.e. small enough to make the truncation error small
        # and actually produce correct results, typically the solution converges to within
        # machine precision in a large majority of cases.
        #
        # If your problem is too large for this to be feasible, set maxit accordingly.
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
            for j in range(n_space_dofs):
                whalf[j] = 0.5 * (w[j] + wnew[j])  # estimate midpoint value of w
            f(whalf, wp, n_space_dofs, thalf, data)
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

    return (m+1)  # Return the number of iterations taken (note that the numbering starts from m=0). If maxit iterations were taken, the convergence failed.


# Backward Euler (implicit Euler).
#
# wrk must have space for 3*n_space_dofs items.
#
cdef int BE( kernels.kernelfuncptr f, pydgq_types.DTYPE_t* w, void* data, int n_space_dofs, pydgq_types.DTYPE_t t, pydgq_types.DTYPE_t dt, pydgq_types.DTYPE_t* wrk, int maxit ) nogil:
    cdef unsigned int j, m, m2, nequals
    cdef int success = 0

    cdef pydgq_types.DTYPE_t* wp = wrk     # iterative approximation of w'(k+1)

    # Iterative approximations of w(k+1)
    cdef pydgq_types.DTYPE_t* wcur = &wrk[n_space_dofs]
    cdef pydgq_types.DTYPE_t* wnew = &wrk[2*n_space_dofs]

    cdef pydgq_types.DTYPE_t tend = t + dt

    # Trivial initial guess: w(k+1) = w(k)
    #
    # This leads to the *second* iterate being the Forward Euler prediction.
    #
    for j in range(n_space_dofs):
        wcur[j] = w[j]

    # Implicit iteration (Banach fixed point iteration)
    #
    for m in range(maxit):
        f(wcur, wp, n_space_dofs, tend, data)
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
            f(wnew, wp, n_space_dofs, tend, data)
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

    return (m+1)  # Return the number of iterations taken (note that the numbering starts from m=0). If maxit iterations were taken, the convergence failed.

