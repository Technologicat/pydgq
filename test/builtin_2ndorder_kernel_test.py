# -*- coding: utf-8 -*-
#
# Tests/usage examples for kernels built-in to the solver.

from __future__ import division, print_function

import numpy as np

import matplotlib.pyplot as plt

from pydgq.solver.types import DTYPE
from pydgq.solver.galerkin import init
import pydgq.solver.builtin_kernels
import pydgq.solver.odesolve
from pydgq.utils.discontify import discontify  # for plotting dG results


#####################
# config for testing
#####################

q = 2  # degree of basis for dG and cG

# How many visualization (interpolation) points to use within each timestep for Galerkin methods.
#
# Note that the dG solution has the best accuracy at the endpoint of the timestep;
# to compare apples-to-apples with classical integrators, this should be set to 1.
#
# Larger values (e.g. 11) are useful for visualizing the behavior of the dG solution inside
# the timestep (something the classical integrators do not model at all).
#
nt_vis_galerkin = 1

nt = 100   # number of timesteps
dt = 0.02  # timestep size

save_from = 0  # see pydgq.solver.odesolve.ivp()


# known analytical solution, for testing the integrators
#
# # see http://docs.sympy.org/dev/modules/solvers/ode.html
# from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols
# from sympy.abc import t
# w = Function('w')
# dsolve(Derivative(w(t), t, t) + w(t), w(t))   # w'' = -w
#
# ==>  w(t) == C1*sin(t) + C2*cos(t)
#
# where  C1 = w'(0), C2 = w(0)  account for the initial condition.
#
# Hence we have also
#
#      w'(t) == C1*cos(t) - C2*sin(t)
#
def reference_solution(tt, n, w0):  # n = number of DOFs
    tt = np.atleast_1d(tt)
    ww = np.empty( (tt.shape[0],n), dtype=DTYPE, order="C" )
    m  = n//2
    for j in range(m):
        # u
        ww[:,2*j]   = w0[2*j+1] * np.sin(tt) + w0[2*j] * np.cos(tt)
        # v
        ww[:,2*j+1] = w0[2*j+1] * np.cos(tt) - w0[2*j] * np.sin(tt)
    return ww


#####################
# main program
#####################

# rel_tol : how close the numerical solution must be to the exact analytical one, in l-infinity norm (max abs)
def test(integrator, nt_vis, rel_tol=1e-2, vis=False):
    n_saved_timesteps = pydgq.solver.odesolve.n_saved_timesteps( nt, save_from )
    result_len        = pydgq.solver.odesolve.result_len( nt, save_from, interp=nt_vis )
    startj,endj       = pydgq.solver.odesolve.timestep_boundaries( nt, save_from, interp=nt_vis )

    n   = 4  # number of DOFs in the **1st-order** system

    # set IC
    w0  = np.empty( (n,), dtype=DTYPE, order="C" )
    w0[0] =  0.  # u1
    w0[1] =  1.  # v1
    w0[2] =  2.  # u2
    w0[3] = -1.  # v2

    # set up the M0 and M1 matrices for  u'' = M0 u + M1 u'
    #
    m  = n//2
    M0 = -np.eye(   m,     dtype=DTYPE)
    M1 =  np.zeros( (m,m), dtype=DTYPE)

    # instantiate kernel
    rhs = pydgq.solver.builtin_kernels.Linear2ndOrderKernel(n, M0, M1)  # note n, not m

    # create output arrays
    ww   = None #np.empty( (result_len,n), dtype=DTYPE, order="C" )    # result array for w; if None, will be created by ivp()
    ff   = np.empty( (result_len,n), dtype=DTYPE, order="C" )          # optional,  result array for w', could be None
    fail = np.empty( (n_saved_timesteps,), dtype=np.intc, order="C" )  # optional,  fail flag for each timestep, could be None

    # solve problem
    ww,tt = pydgq.solver.odesolve.ivp( integrator=integrator, allow_denormals=False,
                                       w0=w0, dt=dt, nt=nt,
                                       save_from=save_from, interp=nt_vis,
                                       rhs=rhs,
                                       ww=ww, ff=ff, fail=fail,
                                       maxit=100 )

    # check result
    ww_ref = reference_solution(tt, n, w0)
    relerr_linfty = np.linalg.norm(ww - ww_ref, ord=np.inf) / np.linalg.norm(ww_ref, ord=np.inf)
    if (relerr_linfty < rel_tol).all():
        passed = 1
        if not vis:
            print("PASS, %03s, tol=% 7g, relerr=%g" % (integrator, rel_tol, relerr_linfty))
    else:
        passed = 0
        if not vis:
            print("FAIL, %03s, tol=% 7g, relerr=%g" % (integrator, rel_tol, relerr_linfty))

    # visualize if requested
    if vis:
        plt.figure(1)
        plt.clf()

        # show the discontinuities at timestep boundaries if using dG (and actually have something to draw within each timestep)
        if integrator == "dG" and nt_vis > 1:
            tt = discontify( tt, endj - 1, fill="nan" )
            for j in range(n):
                # we need the copy() to get memory-contiguous data for discontify() to process
                wtmp = discontify( ww_ref[:,j].copy(), endj - 1, fill="nan" )
                plt.plot( tt, wtmp, 'r--' )  # known exact solution

                wtmp = discontify( ww[:,j].copy(), endj - 1, fill="nan" )
                plt.plot( tt, wtmp, 'k-'  )  # numerical solution
        else:
            plt.plot( tt, ww_ref, 'r--' )  # known exact solution
            plt.plot( tt, ww,     'k-'  )  # numerical solution

        plt.grid(b=True, which="both")
        plt.axis("tight")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$w(t)$")

    return passed


if __name__ == '__main__':
    print("** Testing integrators with 1st-order linear kernel **")

    stuff_to_test = ( ("SE",  1e-1,  False),  # 2nd-order problem, also "SE" is applicable
                      ("IMR", 1e-3,  False),
                      ("BE",  1e-1,  False),
                      ("RK4", 1e-8,  False),
                      ("RK3", 1e-6,  False),
                      ("RK2", 1e-3,  False),
                      ("FE",  1e-1,  False),
                      ("dG",  1e-12, True ),
                      ("cG",  1e-4,  True )
                    )
    n_passed = 0

    for integrator,rel_tol,is_galerkin in stuff_to_test:
        if is_galerkin:
            nt_vis = nt_vis_galerkin  # visualization points per timestep in Galerkin methods
            init(q=q, method=integrator, nt_vis=nt_vis, rule=None)
        else:
            nt_vis = 1   # other methods compute only the end value for each timestep

        n_passed += test(integrator, nt_vis, rel_tol)
    print("** %d/%d tests passed **" % (n_passed, len(stuff_to_test)))

    # draw something
    #
    nt_vis = nt_vis_galerkin
    init(q=q, method="dG", nt_vis=nt_vis, rule=None)
    test(integrator="dG", nt_vis=nt_vis, vis=True)
    plt.show()

