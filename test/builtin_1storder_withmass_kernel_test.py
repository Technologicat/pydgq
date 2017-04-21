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


#####################
# main program
#####################

# rel_tol : how close the numerical solution must be to the exact analytical one, in l-infinity norm (max abs)
def test(integrator, nt_vis, rel_tol=1e-2, vis=False):
    n_saved_timesteps = pydgq.solver.odesolve.n_saved_timesteps( nt, save_from )
    result_len        = pydgq.solver.odesolve.result_len( nt, save_from, interp=nt_vis )
    startj,endj       = pydgq.solver.odesolve.timestep_boundaries( nt, save_from, interp=nt_vis )

    n   = 2  # number of DOFs in the 1st-order system
    w0  = np.arange(1, n+1, dtype=DTYPE)  # a trivial IC

    # set up "M" matrix (arbitrary example)
    M   = (np.arange(n*n, dtype=DTYPE) + 1.).reshape( (n,n) )

    # set up "A" matrix
    #
    # let's make it swap the components (so that u1' = u2,  u2' = u1)
    #
    A   = np.zeros( (n,n), dtype=DTYPE, order="C" )
    A[0,1] = 1.
    A[1,0] = 1.

    # for checking the result
    invA_times_M = (np.linalg.inv(A)).dot(M)

    # instantiate kernels
    rhs1 = pydgq.solver.builtin_kernels.Linear1stOrderKernelWithMassMatrix(n, M, A)
    rhs2 = pydgq.solver.builtin_kernels.Linear1stOrderKernel(n, invA_times_M)  # different algorithm, but should give the same result

    # create output arrays
    ww   = None #np.empty( (result_len,n), dtype=DTYPE, order="C" )    # result array for w; if None, will be created by ivp()
    ff   = np.empty( (result_len,n), dtype=DTYPE, order="C" )          # optional,  result array for w', could be None
    fail = np.empty( (n_saved_timesteps,), dtype=np.intc, order="C" )  # optional,  fail flag for each timestep, could be None

    # solve problem
    ww,tt = pydgq.solver.odesolve.ivp( integrator=integrator, allow_denormals=False,
                                       w0=w0, dt=dt, nt=nt,
                                       save_from=save_from, interp=nt_vis,
                                       rhs=rhs1,
                                       ww=ww, ff=ff, fail=fail,
                                       maxit=100 )

    # check result
    ww_ref,dummy = pydgq.solver.odesolve.ivp( integrator=integrator, allow_denormals=False,
                                       w0=w0, dt=dt, nt=nt,
                                       save_from=save_from, interp=nt_vis,
                                       rhs=rhs2,
                                       ww=ww, ff=ff, fail=fail,
                                       maxit=100 )

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
    print("** Testing integrators with 1st-order linear kernel with mass matrix **")

    # "SE" is not applicable, since we are testing a 1st-order problem
    stuff_to_test = ( ("IMR", 1e-3,  False),
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

#    # DEBUG: draw something
#    #
#    nt_vis = nt_vis_galerkin
#    init(q=q, method="dG", nt_vis=nt_vis, rule=None)
#    test(integrator="dG", nt_vis=nt_vis, vis=True)
#    plt.show()

