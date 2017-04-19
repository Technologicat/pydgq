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
"""Solve the initial value problem of a first-order ordinary differential equation (ODE) system

  w'     = f(w,t)
  w(t=0) = w0  (initial condition)

where f may be nonlinear.
"""

from __future__ import division, print_function, absolute_import

#from libc.stdlib cimport malloc, free

# use fast math functions from <math.h>, available via Cython
#from libc.math cimport fabs as c_abs
#from libc.math cimport sin, cos, log, exp, sqrt

#import cython

import numpy as np

from pydgq.solver.pydgq_types cimport DTYPE_t
from pydgq.solver.pydgq_types import DTYPE
from pydgq.solver.kernel_interface cimport KernelBase  # f() for w' = f(w)
from pydgq.solver.cminmax cimport cuimax  # "C unsigned int max"
cimport pydgq.solver.fputils as fputils    # nan/inf checks for arrays
cimport pydgq.solver.explicit as explicit  # classical explicit integrators
cimport pydgq.solver.implicit as implicit  # classical implicit integrators
cimport pydgq.solver.galerkin as galerkin  # Galerkin integrators
import  pydgq.solver.galerkin as galerkin

### Tell Cython that GCC's __float128 behaves like a double
### (this only concerns the Cython to C compile process and doesn't generate an actual C typedef)
###
### (There is also np.float128, which we could use.)
###
##cdef extern from "math.h":
##    ctypedef double __float128

#cdef extern from "complex.h":
#    double creal(double complex z) nogil
#    double cimag(double complex z) nogil
#    double complex conj(double complex z) nogil


#########################################################################################
# End-of-timestep boilerplate
#########################################################################################

# Store final value from this timestep to result array, if we have passed the start-of-recording point.
#
# This covers the most common case where interp=1 (save value at end of each timestep only).
#
# Galerkin integrators have their own implementation that accounts for interp.
#
# wrk must have space for n_space_dofs items.
#
cdef inline void store( DTYPE_t* w, int n_space_dofs, int timestep, double t, int save_from, DTYPE_t* ww, KernelBase rhs, DTYPE_t* ff, int* pfail, int failure, DTYPE_t* wrk ) nogil:
    cdef unsigned int n, j
    cdef DTYPE_t* wp = wrk

    # Note indexing:
    #  - save_from=1 means "save from first timestep onward" (1-based index 1, although we use 0-based indices).
    #  - save_from=0 means that also the initial condition is saved into the results.
    #
    if timestep >= save_from:
        # Compute the output slot number.
        n = timestep - save_from

        # Write output.
        for j in range(n_space_dofs):
            ww[n*n_space_dofs + j] = w[j]

        # Optionally output the time derivative of the state vector (obtained via f()).
        #
        # This takes some extra time due to the extra rhs.call(), but can be useful for some visualizations.
        #
        if ff:
            rhs.begin_iteration(-1)  # iteration -1 = evaluating final result from this timestep
            rhs.call(w, wp, t)
            for j in range(n_space_dofs):
                ff[n*n_space_dofs + j] = wp[j]

        # Save failure flag if an array was provided for this.
        #
        if pfail:
            pfail[n] = failure


#########################################################################################
# Helper functions for modules using odesolve
#########################################################################################

def n_saved_timesteps( nt, save_from ):
    """def n_saved_timesteps( nt, save_from ):

    Determine number of timesteps that will be saved.

    Note that this is not the length of the result array; for that, see result_len().
    The number returned by this function matches the length of the output arrays from timestep_boundaries().

    For an explanation of parameters, see result_len().

    If save_from == 0, the initial condition qualifies as one timestep.

    Returns:
        int:
            the number of timesteps that will be saved.

    """
    if nt < 1:
        raise ValueError( "nt must be >= 1, got %d" % (nt) )
    if save_from < 0:
        raise ValueError( "save_from must be >= 0, got %d" % (save_from) )
    if save_from > nt:
        raise ValueError( "save_from must be <= nt, otherwise nothing to do; got save_from = %d, nt = %d" % (save_from, nt) )

    n = nt - (max(1, save_from) - 1)  # this many actual timesteps will be saved
    if save_from == 0:
        return n + 1  # the initial condition takes one output slot
    else:
        return n


def result_len( int nt, int save_from, int interp=1 ):
    """def result_len( int nt, int save_from, int interp=1 ):

    Determine length of storage needed on the time axis for ivp().

    Parameters:

        nt        = int, number of timesteps to take.

        save_from = int, first timestep index to save, 0-based.
                    This allows discarding part of the data at the beginning.

                    The special value 0 means that also the initial condition
                    will be copied into the results. The initial condition
                    always produces exactly one item.

                    A value of 1 means "save results from first actual timestep onward";
                    likewise for higher values (e.g. 2 -> second timestep onward).

        interp    = int. Galerkin integrators, such as dG, have the possibility of
                    evaluating the solution at points inside the timestep,
                    by evaluating the computed Galerkin approximation.

                    This sets the number of result points that will be generated
                    per computed timestep.

                    The maximum allowed value for interp is the "maxnx"
                    reported when galerkin.init() loads its data file.

                    interp=1 means that only the value at the endpoint
                    of each timestep will be saved.

                    For all non-Galerkin integrators, interp=1 is the only valid setting.
                    (Because the other integrators are based on collocation methods
                     (point-based methods), the value of the solution at points
                     other than the timestep boundaries is undefined for them.)

                    For dG:
                        A value of interp >= 2 means that n values equally spaced in time,
                        from the start of the timestep to its end, will be saved.

                        E.g.:
                          - interp=2 means to save the start and end values
                            for each timestep.
                          - interp=3 saves these and also the midpoint value.
                          - interp=11 gives a nice spacing of 10 equal intervals
                            (i.e. 11 values!) across each timestep.

                        Note that due to *discontinuous* Galerkin,
                        the start value of a timestep will be different
                        from the end value of the previous timestep,
                        although these share the same time coordinate!

                        The solution is defined to be left-continuous
                        i.e. the "end value" is the actual value.
                        The "start value" is actually a one-sided limit
                        from the right, toward the start of the timestep.

                    For cG:
                        In cG, the solution is continuous, so the endpoint of timestep n
                        is the start point of timestep n+1.

                        The values are equally spaced, but avoiding the duplicate.
                        Effectively this takes the visualization points for interp
                        one larger than specified, discarding the first one.

                        Thus, e.g.:
                          - interp=2 means to save the midpoint and end values
                            for each timestep.
                          - interp=4 saves values at relative offsets dt/4,
                            dt/2, 3dt/4 and dt from the timestep start.
                            There are effectively five "fenceposts" and
                            four intervals in [n*dt, (n+1)*dt].

    Returns:

        Number of storage slots (i.e. array length) needed along the time axis.

    """
    if nt < 1:
        raise ValueError( "nt must be >= 1, got %d" % (nt) )
    if save_from < 0:
        raise ValueError( "save_from must be >= 0, got %d" % (save_from) )
    if save_from > nt:
        raise ValueError( "save_from must be <= nt, otherwise nothing to do; got save_from = %d, nt = %d" % (save_from, nt) )
    if interp < 1:
        raise ValueError( "interp must be >= 1, got %d" % (interp) )

    if save_from == 0:
        # 1 = the initial condition
        return int( 1 + nt*interp )
    else:
        # save_from = 1 means that the initial condition is not saved,
        # but the first timestep (n=0 in the timestep loop) and onward are saved.
        return int( (nt - (save_from - 1))*interp )


def timestep_boundaries( int nt, int save_from, int interp=1 ):
    """def timestep_boundaries( int nt, int save_from, int interp=1 ):

    Return start and one-past-end indices for each timestep in the result. These can be used to index tt, ww and ff on the time axis.

    This is useful with Galerkin integrators, which support several visualization points per timestep (interp > 1).

    Parameters are the same as for result_len().

    Returns:

        Tuple (startj, endj), where startj (endj) is a rank-1 np.array containing the start and one-past-end indices for each timestep.

        The indices for timestep n are range(startj[n], endj[n]).

        If save_from == 0, the initial condition (always exactly one point) counts as "timestep 0"; otherwise "timestep 0" is the first saved timestep.

    """
    if nt < 1:
        raise ValueError( "nt must be >= 1, got %d" % (nt) )
    if save_from < 0:
        raise ValueError( "save_from must be >= 0, got %d" % (save_from) )
    if save_from > nt:
        raise ValueError( "save_from must be <= nt, otherwise nothing to do; got save_from = %d, nt = %d" % (save_from, nt) )
    if interp < 1:
        raise ValueError( "interp must be >= 1, got %d" % (interp) )

    cdef unsigned int n, offs
    cdef unsigned int n_saved_timesteps = nt - (cuimax(1, save_from) - 1)
    cdef unsigned int n_output          = n_saved_timesteps

    if save_from == 0:
        n_output += 1  # the initial condition takes one output slot

    cdef int[::1] startj = np.empty( [n_output], dtype=np.intc, order="C" )
    cdef int[::1] endj   = np.empty( [n_output], dtype=np.intc, order="C" )  # one-past-end

    if save_from == 0:
        offs = 1  # one output slot was taken; shift the rest

        # The initial condition always produces exactly one point
        startj[0] = 0
        endj[0]   = 1
    else:
        offs = 0

    with nogil:
        for n in range(n_saved_timesteps):
            # Loop over visualization points in the timestep.
            #
            startj[offs+n] = offs           + n*interp
            endj[offs+n]   = startj[offs+n] +   interp  # actually one-past-end

    return (startj, endj)


def make_tt( double dt, int nt, int save_from, int interp=1, out=None ):
    """def make_tt( double dt, int nt, int save_from, int interp=1, out=None ):

    Generate rank-1 np.array of the time values that correspond to the solution values output by ivp().

    Parameters:

        out    = rank-1 np.array or None.
                 If None, tt will be created by this call.
                 If supplied, the user-given array will be filled. (No bounds checking - make sure it is large enough!)

    For the other parameters, see result_len().

    Returns:

        rank-1 np.array of double, the time values.

    """
    if dt == 0.0:
        raise ValueError( "dt cannot be zero" )
    if nt < 1:
        raise ValueError( "nt must be >= 1, got %d" % (nt) )
    if save_from < 0:
        raise ValueError( "save_from must be >= 0, got %d" % (save_from) )
    if save_from > nt:
        raise ValueError( "save_from must be <= nt, otherwise nothing to do; got save_from = %d, nt = %d" % (save_from, nt) )
    if interp < 1:
        raise ValueError( "interp must be >= 1, got %d" % (interp) )

    cdef unsigned int n, k, offs, start, end

    # TODO: Use vis_x from integrator (gives the actually used time offsets on the reference element [-1,1], map this to [0,1] to reliably get what we need here)
    # TODO: Currently this function has no access to it, because the galerkin.Helper instance does not exist at this time.

    # "local" time values, i.e. offsets in [0,1] inside one timestep
    cdef DTYPE_t[::1] tloc = np.linspace(0.0, 1.0, interp)
    if interp == 1:  # special case: for one point, linspace gives the beginning of the range, but we want the end
        tloc[0] = 1.0

    # global time values
    cdef DTYPE_t[::1] tt
    if out is None:
        nvals = result_len( nt, save_from, interp )
        tt = np.empty( [nvals], dtype=DTYPE, order="C" )
    else:
        tt = out

    if save_from == 0:
        tt[0] = 0.0  # initial condition occurs at t=0
        offs = 1  # one output slot was taken; shift the rest when writing
    else:
        offs = 0

    # avoid allocating extra memory using a compiled C loop
    cdef DTYPE_t startt
    with nogil:
        # Loop over the timesteps.
        #
        # save_from = 0  -->  initial condition, nt steps (corrected for by offs)
        # save_from = 1  -->  nt steps (discard IC)
        # save_from = 2  -->  nt-1 steps (discard IC and first step)
        # ...
        #
        startt = (cuimax(1, save_from) - 1)*dt
        for n in range(nt - (cuimax(1, save_from) - 1)):
            # Loop over visualization points in the timestep.
            #
            start = offs  + n*interp
            end   = start +   interp  # actually one-past-end
            for k in range(end - start):
                tt[start + k] = startt + (<DTYPE_t>(n) + tloc[k])*dt

    return tt


#########################################################################################
# Integrator
#########################################################################################

# TODO: add convergence tolerance (needs some changes in implicit.pyx and galerkin.pyx (basically wherever "maxit" is used))
def ivp( str integrator, int allow_denormals, DTYPE_t[::1] w0, double dt, int nt, int save_from, int interp,
         KernelBase rhs, DTYPE_t[:,::1] ww, DTYPE_t[:,::1] ff, int[::1] fail, double RK2_beta=1.0,
         int maxit=100 ):
    """def ivp( str integrator, int allow_denormals, DTYPE_t[::1] w0, double dt, int nt, int save_from, int interp,
         KernelBase rhs, DTYPE_t[:,::1] ww, DTYPE_t[:,::1] ff, int[::1] fail, double RK2_beta=1.0,
         int maxit=100 ):

    Solve initial value problem.

    This routine integrates first-order ordinary differential equation (ODE) systems of the form

        w'     = f(w, t),  0 < t <= t_end
        w(t=0) = w0

    where f is a user-provided kernel for computing the RHS.

    Parameters:

        integrator : str
            Time integration algorithm. One of:

                SE : Symplectic Euler (also known as semi-implicit Euler)
                    1st order accuracy, symplectic, conserves energy approximately,
                    very fast, may require a smaller timestep than the others.

                    !!! Only for second-order problems which have been reduced
                        to first-order form. See the user manual for details. !!!

                BE : Backward Euler (implicit Euler).
                    1st order accuracy, high numerical dissipation.

                    A-stable for linear problems, but due to implementation constraints,
                    in this nonlinear solver arbitrarily large timesteps cannot be used.

                    (The timestep size is limited by the loss of contractivity in the
                     Banach iteration as the timestep becomes larger than some
                     situation-specific critical value.)

                IMR : Implicit Midpoint Rule.
                    2nd order accuracy, symplectic, conserves energy approximately.
                    Slow, but may work with a larger timestep than others.

                RK4 : 4th order Runge-Kutta.
                    4th order accuracy, but not symplectic and does not conserve energy;
                    computed orbits may drift arbitrarily far from the true orbits
                    in a long simulation (esp. in a vibration simulation). Moderately fast.

                RK3 : Kutta's third-order method.

                RK2 : parametric second-order Runge-Kutta.
                    Takes the optional parameter RK2_beta, which controls where inside the timestep
                    the second evaluation of f() is taken.

                    RK2_beta must be in the half-open interval (0, 1]. Very small values
                    will cause problems (beta appears in the denominator in the final summation formula).

                    Popular choices:
                        beta = 1/2          , explicit midpoint method
                        beta = 2/3          , Ralston's method
                        beta = 1   (default), Heun's method, also known as the explicit trapezoid rule

                FE : Forward Euler (explicit Euler).
                    1st order accuracy, very unstable, requires a very small timestep.

                    Provided for reference only.

                dG : discontinuous Galerkin (recommended)
                    An advanced implicit method. Finds a weak solution that is finitely
                    discontinuous (C^{-1}) across timestep boundaries.
                    Typically works with large-ish timesteps.

                    The solution satisfies the Galerkin orthogonality property:
                    the residual of the result is L2-orthogonal to the basis functions.
                    Roughly, this means that the numerical solution is, in the least-squares sense,
                    the best representation (in the given basis) of the unknown true solution.

                    See galerkin.init() for configuration (polynomial degree of basis).

                cG : continuous Galerkin.
                    Like dG, but the solution is C^0 continuous across timestep boundaries.

                    See galerkin.init() for configuration (polynomial degree of basis).

        allow_denormals : bool
                    If True,  allow denormal numbers in computation (10...100x performance penalty
                              on most modern processors).

                    If False, stop computation and fill the rest of the solution with zeroes
                              when denormal numbers are reached.

                    Denormal numbers are usually encountered in cases with high damping (beta) and
                    low external load (mu_m, Pi_F); in such a case the system quickly spirals in onto
                    the equilibrium point at the origin, with the magnitude of the numbers decreasing
                    very rapidly.

                    Setting this to True gives a more beautiful plot in such cases, but that can be
                    traded for a large performance increase by setting this to False.

                    In cases where denormal numbers are not encountered, this option has no effect
                    on the results.

        w0 : rank-1 np.array
            initial state (w1, w2, ..., wn).
            This also automatically sets n_space_dofs (for the current problem) to len(w0).

        dt : double, != 0
            timestep size

            Negative values can be used to integrate backward in time.

            !!! The time t always starts at zero; if your RHS explicitly depends on t,
                take this into account! !!!

        nt : int, >= 1
            number of timesteps to take

        save_from : int, >= 0, <= nt
            first timestep index to save, 0-based (allows discarding part of the data at the beginning)

            0 = save initial condition and all timesteps in results
            1 = save all timesteps (but don't save initial condition)
            2 = save from second timestep onward
            ...

        interp : int
            For Galerkin methods: how many visualization points to produce per computed timestep.
            For all other integrators, interp must be 1.

        rhs : instance of class derived from KernelBase
            Kernel implementing the right-hand side of  u' = f(u, t)

        ww : DTYPE_t[:,::1] of size [result_len(),n_space_dofs]
            Output array for w

        ff : DTYPE_t[:,::1] of size [result_len(),n_space_dofs] or None
            If not None, output array for w' (the time derivative of w).

        fail : int[::1] of size [result_len(),] or None.
            If not None, output array for status flag for each timestep:
                0 = converged to machine precision
                1 = did not converge to machine precision

            This data is only meaningful for implicit methods (IMR, BE, dG, cG); explicit methods will simply flag success for each timestep.

            If save_from == 0, the initial condition counts as the zeroth timestep, and is always considered as converged.

        maxit : int, >= 1
            Maximum number of Banach/Picard iterations to take at each timestep.

            Only meaningful if an implicit integrator is used (BE, IMR, dG, cG).
    """
    # Parameter validation
    #
    known_integrators    = ["IMR", "BE", "RK4", "RK3", "RK2", "FE", "dG", "cG", "SE"]
    galerkin_integrators = ["dG", "cG"]  # integrators, of those already listed in known_integrators, which are based on Galerkin methods.

    if integrator not in known_integrators:
        raise ValueError("Unknown integrator '%s'; valid: %s" % ( integrator, ", ".join(known_integrators) ))

    if integrator not in galerkin_integrators and interp != 1:
        raise ValueError("For non-Galerkin integrators (such as the chosen integrator='%s'), interp must be 1, but interp=%d was given." % (integrator, interp))

    if dt == 0.0:
        raise ValueError( "dt cannot be zero" )
    if nt < 1:
        raise ValueError( "nt must be >= 1, got %d" % (nt) )
    if save_from < 0:
        raise ValueError( "save_from must be >= 0, got %d" % (save_from) )
    if save_from > nt:
        raise ValueError( "save_from must be <= nt, otherwise nothing to do; got save_from = %d, nt = %d" % (save_from, nt) )
    if interp < 1:
        raise ValueError( "interp must be >= 1, got %d" % (interp) )
    if maxit < 1:
        raise ValueError( "maxit must be >= 1, got %d" % (maxit) )

    cdef int n_slots = result_len( nt, save_from, interp )  # only needed for sanity check
    if ww.shape[0] != n_slots:
        raise ValueError( "shape of output array ww not compatible with length of output: shape(ww)[0] = %d, but %d timesteps are to be saved" % (ww.shape[0], n_slots) )
    cdef int n_space_dofs = w0.shape[0]  # this is actually needed below
    if ww.shape[1] != n_space_dofs:
        raise ValueError( "shape of output array ww not compatible with n_space_dofs: shape(ww)[1] = %d, but n_space_dofs = %d" % (ww.shape[1], n_space_dofs) )

    if ff is not None:
        if ff.shape[0] != n_slots:
            raise ValueError( "shape of output array ff not compatible with length of output: shape(ff)[0] = %d, but %d timesteps are to be saved" % (ff.shape[0], n_slots) )
        if ff.shape[1] != n_space_dofs:
            raise ValueError( "shape of output array ff not compatible with n_space_dofs: shape(ff)[1] = %d, but n_space_dofs = %d" % (ff.shape[1], n_space_dofs) )

    # Runtime sanity checking of the result
    #
    cdef int do_denormal_check  = <int>(not allow_denormals)  # if denormals not allowed, check them
    cdef int denormal_triggered = 0
    cdef int naninf_triggered   = 0

    cdef DTYPE_t* pff
    if ff is not None:
        pff = &ff[0,0]
    else:
        pff = <DTYPE_t*>0  # store() knows to omit saving w' if the pointer is NULL

    cdef int* pfail
    if fail is not None:
        pfail = &fail[0]
    else:
        pfail = <int*>0

    # Initial condition: initialize the value of w at the end of the previous timestep to the initial condition of the problem.
    #
    cdef DTYPE_t[::1] w_arr = np.empty( (n_space_dofs,), dtype=DTYPE, order="C" )
    cdef DTYPE_t* w = &w_arr[0]  # we only need a raw pointer
    cdef unsigned int j
    for j in range(n_space_dofs):
        w[j] = w0[j]

    # Fill in stuff from the initial condition to the results, if saving all the way from the start.
    #
    # Temporary storage for w' as output by rhs.call(). This is needed later anyway, but we may need it already here.
    cdef DTYPE_t[::1] wp_arr = np.empty( (n_space_dofs,), dtype=DTYPE, order="C" )
    cdef DTYPE_t* wp = &wp_arr[0]
    if save_from == 0:
        # State vector w
        ww[0,:] = w0

        # w'
        if ff is not None:
            rhs.begin_timestep(0)  # timestep 0 = initial condition
            rhs.begin_iteration(-1)  # iteration -1 = evaluating final result from this timestep
            rhs.call(&w0[0], wp, 0.0)  # t at beginning = 0.0
            for j in range(n_space_dofs):
                ff[0,j] = wp[j]

        # success/fail information (initial condition is always successful)
        if fail is not None:
            fail[0] = 0

    # Timestep number and current time
    #
    cdef unsigned int n = 0
    cdef double t = 0.0

    # Work space for classical integrators (will be allocated below, needed size depends on algorithm)
    #
    cdef DTYPE_t[::1] wrk_arr
    cdef DTYPE_t* wrk = <DTYPE_t*>0

    # Implicit methods support
    #
    cdef unsigned int nits                     # number of implicit iterations (Banach fixed point iterations) taken at this timestep
    cdef unsigned int nfail         = 0        # number of last failed timestep (failed = did not converge to machine precision)
    cdef unsigned int totalfailed   = 0        # how many failed timesteps in total
    cdef unsigned int totalnits     = 0        # total number of iterations taken, across all timesteps
    cdef unsigned int max_taken_its = 0        # maximum number of iterations (seen) that was taken for one timestep
    cdef unsigned int min_taken_its = 2*maxit  # minimum number of iterations (seen) that was taken for one timestep. (Invalid value used to force initialization at first timestep.)

    # Galerkin methods support
    #
    cdef galerkin.Params gp
    cdef unsigned int offs, out_start, out_end, l, noutput
    cdef DTYPE_t* pww

    cdef explicit.integrator_ptr timestep_explicit = <explicit.integrator_ptr>0
    cdef implicit.integrator_ptr timestep_implicit = <implicit.integrator_ptr>0
    cdef galerkin.integrator_ptr timestep_galerkin = <galerkin.integrator_ptr>0

    # Account for the output slot possibly used for the initial condition.
    #
    # We need this for Galerkin methods, but also for filling the empty slots when the solver fails.
    #
    if save_from == 0:
        offs = 1
    else:
        offs = 0


    # Integration loops
    #
    if integrator in ["SE", "RK4", "RK3", "FE"]:
        if integrator == "SE":  # symplectic Euler
            if n_space_dofs % 2 != 0:
                raise ValueError("SE: Symplectic Euler (SE) only makes sense for second-order systems transformed to first-order ones, but got odd number of n_space_dofs = %d" % (n_space_dofs))
            timestep_explicit = explicit.SE
            wrk_arr = np.empty( (1 * n_space_dofs,), dtype=DTYPE, order="C" )
        elif integrator == "RK4":
            timestep_explicit = explicit.RK4  # classical fourth-order Runge-Kutta
            wrk_arr = np.empty( (5 * n_space_dofs,), dtype=DTYPE, order="C" )
        elif integrator == "RK3":
            timestep_explicit = explicit.RK3  # Kutta's third-order method
            wrk_arr = np.empty( (4 * n_space_dofs,), dtype=DTYPE, order="C" )
        else: # integrator == "FE":
            timestep_explicit = explicit.FE   # forward Euler
            wrk_arr = np.empty( (1 * n_space_dofs,), dtype=DTYPE, order="C" )

        wrk = &wrk_arr[0]

        # We release the GIL for the integration loop to let another Python thread execute
        # while this one is running through a lot of timesteps (possibly several million per solver run).
        #
        with nogil:
            for n in range(1,nt+1):
                t = (n-1)*dt  # avoid accumulating error (don't sum; for very large t, this version will tick as soon as the floating-point representation allows it)

                rhs.begin_timestep(n)
                timestep_explicit( rhs, w, t, dt, wrk )

                # end-of-timestep boilerplate
                #
                t = n*dt
                store( w, n_space_dofs, n, t, save_from, &ww[0,0], rhs, pff, pfail, 0, wrk )
                if do_denormal_check:
                    denormal_triggered = fputils.all_denormal( w, n_space_dofs )
                naninf_triggered = fputils.any_naninf( w, n_space_dofs )
                if denormal_triggered or naninf_triggered:
                    break

    elif integrator == "RK2":  # parametric second-order Runge-Kutta
        # different function signature; otherwise the handling is identical to the above.

        wrk_arr = np.empty( (3 * n_space_dofs,), dtype=DTYPE, order="C" )
        wrk = &wrk_arr[0]

        with nogil:
            for n in range(1,nt+1):
                t = (n-1)*dt

                rhs.begin_timestep(n)
                explicit.RK2( rhs, w, t, dt, wrk, RK2_beta )

                # end-of-timestep boilerplate
                #
                t = n*dt
                store( w, n_space_dofs, n, t, save_from, &ww[0,0], rhs, pff, pfail, 0, wrk )
                if do_denormal_check:
                    denormal_triggered = fputils.all_denormal( w, n_space_dofs )
                naninf_triggered = fputils.any_naninf( w, n_space_dofs )
                if denormal_triggered or naninf_triggered:
                    break

    elif integrator in ["BE", "IMR"]:
        if integrator == "BE":  # backward Euler
            timestep_implicit = implicit.BE
            wrk_arr = np.empty( (3 * n_space_dofs,), dtype=DTYPE, order="C" )
        else:
            timestep_implicit = implicit.IMR  # implicit midpoint rule
            wrk_arr = np.empty( (4 * n_space_dofs,), dtype=DTYPE, order="C" )

        wrk = &wrk_arr[0]

        with nogil:
            for n in range(1,nt+1):
                t = (n-1)*dt

                rhs.begin_timestep(n)
                nits = timestep_implicit( rhs, w, t, dt, wrk, maxit )

                # update the iteration statistics
                #
                if nits == maxit:
                    totalfailed += 1
#                if nfail == 0  and  nits == maxit:  # store first failed timestep
                if nits == maxit:  # store last failed timestep (update at each, last one left standing)
                    nfail = n
                if nits > max_taken_its:
                    max_taken_its = nits
                if nits < min_taken_its:
                    min_taken_its = nits
                totalnits += nits

                # end-of-timestep boilerplate
                #
                t = n*dt
                store( w, n_space_dofs, n, t, save_from, &ww[0,0], rhs, pff, pfail, <int>(nits == maxit), wrk )
                if do_denormal_check:
                    denormal_triggered = fputils.all_denormal( w, n_space_dofs )
                naninf_triggered = fputils.any_naninf( w, n_space_dofs )
                if denormal_triggered or naninf_triggered:
                    break

        nt_taken = max(1, n)
        failed_str = "" if totalfailed == 0 else "; last non-converged timestep %d" % (nfail)
        print( "    min/avg/max iterations taken = %d, %g, %d; total number of non-converged timesteps %d (%g%%)%s" % (int(min_taken_its), float(totalnits)/nt_taken, int(max_taken_its), totalfailed, 100.0*float(totalfailed)/nt_taken, failed_str) )

    else:  # integrator in galerkin_integrators:  # Galerkin integrators
        my_unique_id = id(ww)  # HACK: the output array is probably unique across simultaneously running instances. (TODO: odesolve could benefit from more object-orientedness)
        gp = galerkin.setup_instance(my_unique_id, integrator, rhs, w_arr, maxit, interp)

        if integrator == "dG":  # discontinuous Galerkin (recommended!)
            timestep_galerkin = galerkin.dG
        else: # integrator == "cG":  # continuous Galerkin (doesn't work that well in practice)
            timestep_galerkin = galerkin.cG

        with nogil:
            for n in range(1,nt+1):
                gp.t = (n-1)*dt

                rhs.begin_timestep(n)
                nits = timestep_galerkin( gp )

                # update the iteration statistics
                #
                if nits == maxit:
                    totalfailed += 1
#                if nfail == 0  and  nits == maxit:  # store first failed timestep
                if nits == maxit:  # store last failed timestep (update at each, last one left standing)
                    nfail = n
                if nits > max_taken_its:
                    max_taken_its = nits
                if nits < min_taken_its:
                    min_taken_its = nits
                totalnits += nits

                # end-of-timestep boilerplate
                #
                if interp == 1:
                    # In this case, we already have what we need. Just do what the other methods do,
                    # saving the end-of-timestep value of the solution from w.
                    #
                    t = n*dt
                    store( w, n_space_dofs, n, t, save_from, &ww[0,0], rhs, pff, pfail, <int>(nits == maxit), gp.wrk )
                else:
                    # Inline custom store() implementation accounting for interp (and how to assemble a Galerkin series)
                    #
                    if n >= save_from:
                        # Interpolate inside timestep using the Galerkin representation of the solution, obtaining more visualization points.
                        #
                        pww = &ww[0,0]
                        galerkin.assemble( gp.u, gp.psivis, gp.uvis, gp.ucvis, n_space_dofs, gp.n_time_dofs, interp )
                        noutput = n - cuimax(1, save_from)  # 0-based timestep number starting from the first saved one.
                                                                    # Note that n = 1, 2, ... (also store() depends on this numbering!)
                        out_start = offs + noutput*interp
                        for l in range(interp):
                            for j in range(n_space_dofs):
                                # uvis: [nx,n_space_dofs]
                                pww[(out_start+l)*n_space_dofs + j] = gp.uvis[l*n_space_dofs + j]

                        # Optionally output the time derivative of the state vector (obtained via f()).
                        #
                        # This takes some extra time, but can be useful for some visualizations.
                        #
                        if pff:
                            rhs.begin_iteration(-1)  # iteration -1 = evaluating final result from this timestep
                            for l in range(interp):
                                t = ((n-1) + gp.tvis[l]) * dt
                                rhs.call(&gp.uvis[l*n_space_dofs + 0], wp, t)
                                for j in range(n_space_dofs):
                                    pff[(out_start+l)*n_space_dofs + j] = wp[j]

                        if pfail:
                            pfail[offs + noutput] = <int>(nits == maxit)

                if do_denormal_check:
                    denormal_triggered = fputils.all_denormal( w, n_space_dofs )
                naninf_triggered = fputils.any_naninf( w, n_space_dofs )
                if denormal_triggered or naninf_triggered:
                    break

        nt_taken = max(1, n)
        failed_str = "" if totalfailed == 0 else "; last non-converged timestep %d" % (nfail)
        print( "    min/avg/max iterations taken = %d, %g, %d; total number of non-converged timesteps %d (%g%%)%s" % (int(min_taken_its), float(totalnits)/nt_taken, int(max_taken_its), totalfailed, 100.0*float(totalfailed)/nt_taken, failed_str) )

        galerkin.teardown_instance(my_unique_id)


    # DEBUG/INFO: final value of w'
    #
    t = nt_taken*dt
    rhs.begin_iteration(-1)  # iteration -1 = evaluating final result from this timestep
    rhs.call(w, wp, t)

    lw  = [ "%0.18g" % w[j] for j in range(n_space_dofs) ]
    sw  = ", ".join(lw)
    lwp = [ "%0.18g" % wp[j] for j in range(n_space_dofs) ]
    swp = ", ".join(lwp)
    print( "    final w = %s\n    final f(w) = %s" % (sw, swp) )

    # If a failure check triggered, mark the rest of the solution accordingly.
    #
    if denormal_triggered  or  naninf_triggered:
        if n < save_from:
            noutput = 0
        else:
            noutput = n - cuimax(1, save_from)  # 0-based timestep number starting from the first saved one.
                                                # Note that n = 1, 2, ... (also store() depends on this numbering!)
        out_start = offs + noutput*interp

        if denormal_triggered:
            print( "    denormal check triggered at timestep %d, rest of the solution is zero." % n )
            ww[out_start:,:] = 0.0
            ff[out_start:,:] = 0.0  # no more change in w  =>  w' = 0
            fail[(offs + noutput):] = 0  # successful

        if naninf_triggered:
            print( "    nan/inf check triggered at timestep %d, rest of the solution is nonsense." % n )

            ww[out_start:,:] = np.nan

            if ff is not None:
                ff[out_start:, :] = np.nan  # not evaluated

            if fail is not None:
                fail[(offs + noutput):] = 1  # all the rest failed

