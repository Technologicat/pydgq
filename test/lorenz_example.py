# -*- coding: utf-8 -*-
#
# Usage example: solve the Lorenz system using a custom Python kernel.

from __future__ import division, print_function

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from pydgq.solver.types import DTYPE
from pydgq.solver.galerkin import init
from pydgq.solver.kernel_interface import PythonKernel
import pydgq.solver.odesolve
from pydgq.utils.discontify import discontify  # for plotting dG results


#####################
# config
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
nt_vis_galerkin = 11

nt = 3500  # number of timesteps
dt = 0.1   # timestep size

save_from = 0  # see pydgq.solver.odesolve.ivp()


#####################
# custom kernel
#####################

# A kernel for the Lorenz system.
#
# The custom kernel only needs to override callback(); even __init__ is not strictly needed,
# unless adding some custom parameters (like here).
#
class LorenzKernel(PythonKernel):
    def __init__(self, rho, sigma, beta):
        # super
        PythonKernel.__init__(self, n=3)

        # custom init
        self.rho   = rho
        self.sigma = sigma
        self.beta  = beta

    def callback(self, t):
        # dxdt = sigma (y - x)
        # dydt = x (rho - z) - y
        # dzdt = x y - beta z
        self.out[0] = self.sigma * (self.w[1] - self.w[0])
        self.out[1] = self.w[0]  * (self.rho  - self.w[2]) - self.w[1]
        self.out[2] = self.w[0]  * self.w[1] - self.beta * self.w[2]  # this is nonlinear, so we can't use a built-in linear kernel


#####################
# main program
#####################

def test(integrator, nt_vis):
    n_saved_timesteps = pydgq.solver.odesolve.n_saved_timesteps( nt, save_from )
    result_len        = pydgq.solver.odesolve.result_len( nt, save_from, interp=nt_vis )
    startj,endj       = pydgq.solver.odesolve.timestep_boundaries( nt, save_from, interp=nt_vis )

    n  = 3  # the Lorenz system has 3 DOFs

    # we use the same values as in the example at https://en.wikipedia.org/wiki/Lorenz_system
    #
    rho   = 28.
    sigma = 10.
    beta  = 8./3.

    # set IC
    #
    w0 = np.empty( (n,), dtype=DTYPE, order="C" )
    w0[0] = 0.
    w0[1] = 2.
    w0[2] = 20.

    # instantiate kernel
    rhs = LorenzKernel(rho=rho, sigma=sigma, beta=beta)

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
                                       maxit=10 )

    # visualize
    #
    # http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    #
    print( "** Plotting solution **" )
    fig = plt.figure(1)
    plt.clf()

    # Axes3D has a tendency to underestimate how much space it needs; it draws its labels
    # outside the window area in certain orientations.
    #
    # This causes the labels to be clipped, which looks bad. We prevent this by creating the axes
    # in a slightly smaller rect (leaving a margin). This way the labels will show - outside the Axes3D,
    # but still inside the figure window.
    #
    # The final touch is to set the window background to a matching white, so that the
    # background of the figure appears uniform.
    #
    fig.patch.set_color( (1,1,1) )
    fig.patch.set_alpha( 1.0 )
    x0y0wh = [ 0.02, 0.02, 0.96, 0.96 ]  # left, bottom, width, height      (here as fraction of subplot area)

    ax = Axes3D(fig, rect=x0y0wh)

    # show the discontinuities at timestep boundaries if using dG (and actually have something to draw within each timestep)
    if integrator == "dG" and nt_vis > 1:
        tt = discontify( tt, endj - 1, fill="nan" )

        wtmp = np.empty( (tt.shape[0],n), dtype=DTYPE, order="C" )
        for j in range(n):
            # we need the copy() to get memory-contiguous data for discontify() to process
            wtmp[:,j] = discontify( ww[:,j].copy(), endj - 1, fill="nan" )

        ax.plot( wtmp[:,0], wtmp[:,1], wtmp[:,2], linewidth=0.5  )
    else:
        ax.plot( ww[:,0], ww[:,1], ww[:,2], linewidth=0.5  )

    plt.grid(b=True, which="both")
    plt.axis("tight")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    plt.suptitle(r"Lorenz system: $\rho = %g$, $\sigma = %g$, $\beta = %g$, $x_0 = %g$, $y_0 = %g$, $z_0 = %g$" % (rho, sigma, beta, w0[0], w0[1], w0[2]))

if __name__ == '__main__':
    print("** Solving the Lorenz system **")

    nt_vis = nt_vis_galerkin
    init(q=q, method="dG", nt_vis=nt_vis, rule=None)
    test(integrator="dG", nt_vis=nt_vis)
    plt.show()

