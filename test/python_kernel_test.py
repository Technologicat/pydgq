# -*- coding: utf-8 -*-
#
# Tests/usage examples for a custom Python-based kernel.

from __future__ import division, print_function

import numpy as np

import matplotlib.pyplot as plt

from pydgq.solver.types import DTYPE
from pydgq.solver.galerkin import init
from pydgq.solver.kernel_interface import PythonKernel
import pydgq.solver.odesolve
from pydgq.utils.discontify import discontify  # for plotting dG results


# A simple cosine kernel with phase-shifted components.
#
# The custom kernel only needs to override callback(); even __init__ is not strictly needed,
# unless adding some custom parameters (like here).
#
class MyKernel(PythonKernel):
    def __init__(self, n, omega):  # omega : rad/s
        # super
        PythonKernel.__init__(self, n)

        # custom init
        self.omega = omega

    def callback(self, t):
        for j in range(self.n):
            phi0_j = (float(j+1) / self.n) * 2. * np.pi
            self.out[j] = np.cos(phi0_j + self.omega*t)


integrator_to_test = "dG"
q = 2  # degree of basis for dG and cG

nt = 10
save_from = 0
dt = 1.0  # dG can typically use huge timesteps

if integrator_to_test in ["dG", "cG"]:
    nx_vis = 11  # visualization points per timestep in Galerkin methods
else:
    nx_vis = 1   # other methods compute only the end value for each timestep


def test():
    n_saved_timesteps = pydgq.solver.odesolve.n_saved_timesteps( nt, save_from )
    result_len        = pydgq.solver.odesolve.result_len( nt, save_from, interp=nx_vis )
    startj,endj       = pydgq.solver.odesolve.timestep_boundaries( nt, save_from, interp=nx_vis )

    n   = 3  # number of DOFs in the 1st-order system
    w0  = np.zeros( (n,), dtype=DTYPE, order="C" )  # a trivial IC

    # instantiate kernel
    rhs = MyKernel(n, omega=0.1 * (2. * np.pi))

    # create output arrays
    ww   = np.empty( (result_len,n), dtype=DTYPE, order="C" )          # mandatory, result array for w
    ff   = np.empty( (result_len,n), dtype=DTYPE, order="C" )          # optional,  result array for w', could be None
    fail = np.empty( (n_saved_timesteps,), dtype=np.intc, order="C" )  # optional,  fail flag for each timestep, could be None

    # solve problem
    pydgq.solver.odesolve.ivp( integrator=integrator_to_test, allow_denormals=False,
                               w0=w0, dt=dt, nt=nt,
                               save_from=save_from, interp=nx_vis,
                               rhs=rhs,
                               ww=ww, ff=ff, fail=fail,
                               maxit=100 )

    # visualize
    plt.figure(1)
    plt.clf()

    tt = pydgq.solver.odesolve.make_tt( dt, nt, save_from, interp=nx_vis, out=None )  # out=None --> create a new array and return it

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
        init(q=q, method=integrator_to_test, nx=nx_vis, rule=None)

    test()
    plt.show()

