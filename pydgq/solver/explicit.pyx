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
"""Classical explicit integrators to update w by one timestep.

These routines use compensated summation to obtain maximal accuracy
(at the cost of 4x math in the final summation step).
"""

from __future__ import division, print_function, absolute_import

from pydgq.solver.pydgq_types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase
from pydgq.solver.compsum cimport accumulate


# Fourth-order Runge-Kutta (RK4).
#
# wrk must have space for 5*n_space_dofs items.
#
cdef int RK4( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil:
    cdef unsigned int j
    cdef int n_space_dofs = rhs.n
    cdef DTYPE_t* wstar = wrk  # updated w, only needed for computing the next k

    # standard RK4 temp variables
    cdef DTYPE_t* k1 = &wrk[n_space_dofs]
    cdef DTYPE_t* k2 = &wrk[2*n_space_dofs]
    cdef DTYPE_t* k3 = &wrk[3*n_space_dofs]
    cdef DTYPE_t* k4 = &wrk[4*n_space_dofs]

    cdef DTYPE_t dtp2  = dt/2.0
    cdef DTYPE_t dtp6  = dt/6.0

    cdef DTYPE_t thalf = t + dtp2
    cdef DTYPE_t tend  = t + dt

    cdef DTYPE_t s
    cdef DTYPE_t c

    # Compute k1, ..., k4
    #
    rhs.begin_iteration(0)  # explicit method, single iteration only
    rhs.call(w, k1, t)

    for j in range(n_space_dofs):
        wstar[j] = w[j] + dtp2 * k1[j]
    rhs.call(wstar, k2, thalf)

    for j in range(n_space_dofs):
        wstar[j] = w[j] + dtp2 * k2[j]
    rhs.call(wstar, k3, thalf)

    for j in range(n_space_dofs):
        wstar[j] = w[j] + dt   * k3[j]
    rhs.call(wstar, k4, tend)

    # Update
    #
    #   w(new) = w(old) + (dt/6) * ( k1 + 2*k2 + 2*k3 + k4 )
    #
    # Use compensated summation for maximal accuracy. This is important for problems sensitive to initial conditions
    # (such as mechanical problems with low damping).
    #
    for j in range(n_space_dofs):
        s = k1[j]  # the first operand is always exactly representable; here both LHS and RHS are DTYPE_t
        c = 0.0
        accumulate( &s, &c, 2.0*k2[j] )
        accumulate( &s, &c, 2.0*k3[j] )
        accumulate( &s, &c,     k4[j] )

        # we want  w = w + (dt/6) * s,  do the multiplication now
        s *= dtp6
        c *= dtp6

        # s = s + w
        accumulate( &s, &c,      w[j] )
        w[j] = s

    return 1  # explicit method, always one iteration


# Kutta's third-order method (commonly known as RK3).
#
# wrk must have space for 4*n_space_dofs items.
#
cdef int RK3( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil:
    cdef unsigned int j
    cdef int n_space_dofs = rhs.n
    cdef DTYPE_t* wstar = wrk  # updated w, only needed for computing the next k

    # RK temp variables
    cdef DTYPE_t* k1 = &wrk[n_space_dofs]
    cdef DTYPE_t* k2 = &wrk[2*n_space_dofs]
    cdef DTYPE_t* k3 = &wrk[3*n_space_dofs]

    cdef DTYPE_t dtp2  = dt/2.0
    cdef DTYPE_t twodt = 2.0*dt
    cdef DTYPE_t dtp6  = dt/6.0

    cdef DTYPE_t thalf = t + dtp2
    cdef DTYPE_t tend  = t + dt

    cdef DTYPE_t s
    cdef DTYPE_t c

    # Compute k1, ..., k3
    #
    rhs.begin_iteration(0)  # explicit method, single iteration only
    rhs.call(w, k1, t)

    for j in range(n_space_dofs):
        wstar[j] = w[j] + dtp2 * k1[j]
    rhs.call(wstar, k2, thalf)

    for j in range(n_space_dofs):
        wstar[j] = w[j]  - dt * k1[j]  + twodt * k2[j]
    rhs.call(wstar, k3, tend)

    # Update
    #
    #   w(new) = w(old) + (dt/6) * ( k1 + 4*k2 + k3 )
    #
    # Use compensated summation for maximal accuracy. This is important for problems sensitive to initial conditions
    # (such as mechanical problems with low damping).
    #
    for j in range(n_space_dofs):
        s = k1[j]  # the first operand is always exactly representable; here both LHS and RHS are DTYPE_t
        c = 0.0
        accumulate( &s, &c, 4.0*k2[j] )
        accumulate( &s, &c,     k3[j] )

        # we want  w = w + (dt/6) * s,  do the multiplication now
        s *= dtp6
        c *= dtp6

        # s = s + w
        accumulate( &s, &c,      w[j] )
        w[j] = s

    return 1  # explicit method, always one iteration


# Parametric second-order Runge-Kutta.
#
# The beta parameter controls where inside the timestep the second evaluation of f() is taken. It must be in the half-open interval (0, 1].
# Very small values will cause problems (beta appears in the denominator in the final summation formula).
#
# Popular choices:
#   beta = 1/2, explicit midpoint method
#   beta = 2/3, Ralston's method
#   beta = 1,   Heun's method, also known as the explicit trapezoid rule
#
# wrk must have space for 3*n_space_dofs items.
#
cdef int RK2( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk, DTYPE_t beta ) nogil:
    cdef unsigned int j
    cdef int n_space_dofs = rhs.n
    cdef DTYPE_t* wstar = wrk  # updated w, only needed for computing the next k

    # RK temp variables
    cdef DTYPE_t* k1 = &wrk[n_space_dofs]
    cdef DTYPE_t* k2 = &wrk[2*n_space_dofs]

    cdef DTYPE_t weight2 = 1.0 / (2.0 * beta)
    cdef DTYPE_t weight1 = 1.0 - weight2
    weight1 *= dt
    weight2 *= dt

    cdef DTYPE_t betadt = beta*dt
    cdef DTYPE_t tmid   = t + betadt

    cdef DTYPE_t s
    cdef DTYPE_t c

    # Compute k1, k2
    #
    rhs.begin_iteration(0)  # explicit method, single iteration only
    rhs.call(w, k1, t)

    for j in range(n_space_dofs):
        wstar[j] = w[j] + betadt * k1[j]
    rhs.call(wstar, k2, tmid)

    # Update
    #
    #   w(new) = w(old) +  weight1 * k1  +  weight2 * k2
    #
    # Use compensated summation for maximal accuracy. This is important for problems sensitive to initial conditions
    # (such as mechanical problems with low damping).
    #
    for j in range(n_space_dofs):
        s = weight1 * k1[j]  # the first operand is always exactly representable; here both LHS and RHS are DTYPE_t
        c = 0.0
        accumulate( &s, &c, weight2*k2[j] )

        # s = s + w
        accumulate( &s, &c,      w[j] )
        w[j] = s

    return 1  # explicit method, always one iteration


# Forward Euler.
#
# The most unstable integrator known. Requires a very small timestep. Accuracy O(dt).
#
# Provided for reference only.
#
# wrk must have space for n_space_dofs items.
#
cdef int FE( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil:
    cdef int n_space_dofs = rhs.n
    cdef DTYPE_t* wp = wrk  # w prime (output from RHS)

    rhs.begin_iteration(0)  # explicit method, single iteration only
    rhs.call(w, wp, t)

    for j in range(n_space_dofs):
        w[j] += dt*wp[j]

    return 1  # explicit method, always one iteration


# Symplectic Euler.
#
# Accuracy is O(dt).
#
# This method is applicable only to second-order ODE systems
# which have been reduced to first order using the companion method.
#
# http://en.wikipedia.org/wiki/Semi-implicit_Euler_method
# http://www.phy.uct.ac.za/courses/opencontent/phylab2/worksheet9_09.pdf
# http://umu.diva-portal.org/smash/get/diva2:140361/FULLTEXT01.pdf   p.29 ff.
#
# Let v = u'. Storage format of w:  u1, v1, u2, v2, [...], um, vm
#
# where  m = n_space_dofs / 2  is the size of the original 2nd-order system.
#
# Note that the method takes in n_space_dofs, i.e. the size of the **1st-order** system.
#
# wrk must have space for n_space_dofs items.
#
cdef int SE( KernelBase rhs, DTYPE_t* w, DTYPE_t t, DTYPE_t dt, DTYPE_t* wrk ) nogil:
    cdef unsigned int j
    cdef int n_space_dofs = rhs.n
    cdef DTYPE_t* wp = wrk  # w prime (output from RHS)

    rhs.begin_iteration(0)  # explicit method, single iteration only
    rhs.call(w, wp, t)  # of this, SE uses only the v' components corresponding to the v degrees of freedom

    for j in range(1,n_space_dofs,2):
        w[j] += dt*wp[j]  # qdot, forward diff using current qdotdot
    for j in range(0,n_space_dofs,2):
        w[j] += dt*w[j+1]  # q, backward diff using updated qdot

#    # e.g. for n_space_dofs == 4:
#    w[1] += dt*wp[1]  # v1, forward diff using current v'1
#    w[3] += dt*wp[3]  # v2, forward diff using current v'2
#    w[0] += dt*w[1]   # u1, backward diff using updated u'1 = v1
#    w[2] += dt*w[3]   # u2, backward diff using updated u'2 = v2

    return 1  # explicit method, always one iteration

