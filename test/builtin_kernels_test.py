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

integrator_to_test = "dG"
q = 2  # degree of basis for dG and cG

nt = 10
save_from = 0
dt = 1.0  # dG can typically use huge timesteps

if integrator_to_test in ["dG", "cG"]:
    nt_vis = 11  # visualization points per timestep in Galerkin methods
else:
    nt_vis = 1   # other methods compute only the end value for each timestep


def test():
    n_saved_timesteps = pydgq.solver.odesolve.n_saved_timesteps( nt, save_from )
    result_len        = pydgq.solver.odesolve.result_len( nt, save_from, interp=nt_vis )
    startj,endj       = pydgq.solver.odesolve.timestep_boundaries( nt, save_from, interp=nt_vis )

    n   = 3  # number of DOFs in the 1st-order system
    w0  = np.arange(1, n+1, dtype=DTYPE)  # a trivial IC
    M   = np.eye(n, dtype=DTYPE)          # a trivial "M" matrix

    # instantiate kernel
    rhs = pydgq.solver.builtin_kernels.Linear1stOrderKernel(n, M)

    # create output arrays
    ww   = None #np.empty( (result_len,n), dtype=DTYPE, order="C" )    # result array for w; if None, will be created by ivp()
    ff   = np.empty( (result_len,n), dtype=DTYPE, order="C" )          # optional,  result array for w', could be None
    fail = np.empty( (n_saved_timesteps,), dtype=np.intc, order="C" )  # optional,  fail flag for each timestep, could be None

    # solve problem
    ww,tt = pydgq.solver.odesolve.ivp( integrator=integrator_to_test, allow_denormals=False,
                                       w0=w0, dt=dt, nt=nt,
                                       save_from=save_from, interp=nt_vis,
                                       rhs=rhs,
                                       ww=ww, ff=ff, fail=fail,
                                       maxit=100 )

    # visualize
    plt.figure(1)
    plt.clf()

    # show the discontinuities at timestep boundaries if using dG
    if integrator_to_test == "dG":
        tt = discontify( tt, endj - 1, fill="nan" )
        for j in range(n):
            # we need the copy() to get memory-contiguous data for discontify() to process
            wtmp = discontify( ww[:,j].copy(), endj - 1, fill="nan" )
            plt.plot( tt, wtmp )
    else:
        plt.plot( tt, ww )

    plt.grid(b=True, which="both")
    plt.axis("tight")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$w(t)$")

if __name__ == '__main__':
    # initialize Galerkin integrators
    if integrator_to_test in ["dG", "cG"]:
        init(q=q, method=integrator_to_test, nt_vis=nt_vis, rule=None)

    test()
    plt.show()

