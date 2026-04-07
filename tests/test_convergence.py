# -*- coding: utf-8 -*-
"""Convergence tests: higher-order methods should give smaller errors on smooth problems."""

import numpy as np

from pydgq.solver.types import DTYPE
from pydgq.solver.galerkin import init as galerkin_init
from pydgq.solver.kernel_interface import PythonKernel
from pydgq.solver.odesolve import ivp, n_saved_timesteps, result_len
import pydgq.solver.builtin_kernels as builtin_kernels


# ---------------------------------------------------------------------------
# Common settings
# ---------------------------------------------------------------------------

N = 3
SAVE_FROM = 0
# Use a coarser timestep so the methods produce distinguishable errors
# rather than all bottoming out at machine precision.
DT_CONV = 0.1
NT_CONV = 100


def _compute_error_exp(integrator, is_galerkin=False, q=2):
    """Solve w' = w (exponential growth) and return relative L-inf error.

    This problem has w-dependent RHS, which is essential for distinguishing
    the convergence orders of RK2, RK3, and RK4. (A pure f(t) problem
    degenerates because the w-updates in the RK stages become irrelevant.)
    """
    nt_vis = 1
    if is_galerkin:
        galerkin_init(q=q, method=integrator, nt_vis=nt_vis, rule=None)

    n_saved = n_saved_timesteps(NT_CONV, SAVE_FROM)
    rlen = result_len(NT_CONV, SAVE_FROM, interp=nt_vis)

    w0 = np.arange(1, N + 1, dtype=DTYPE)
    M = np.eye(N, dtype=DTYPE, order="C")
    rhs = builtin_kernels.Linear1stOrderKernel(N, M)

    ww = np.empty((rlen, N), dtype=DTYPE, order="C")
    fail = np.empty((n_saved,), dtype=np.intc, order="C")

    ww, tt = ivp(integrator=integrator, allow_denormals=False,
                 w0=w0, dt=DT_CONV, nt=NT_CONV,
                 save_from=SAVE_FROM, interp=nt_vis,
                 rhs=rhs,
                 ww=ww, ff=None, fail=fail,
                 maxit=100)

    # Exact solution: w(t) = w0 * exp(t)
    tt = np.asarray(tt)
    ww_ref = w0[np.newaxis, :] * np.exp(tt)[:, np.newaxis]
    return np.linalg.norm(ww - ww_ref, ord=np.inf) / np.linalg.norm(ww_ref, ord=np.inf)


# ---------------------------------------------------------------------------
# Runge-Kutta convergence order: RK4 < RK3 < RK2
# ---------------------------------------------------------------------------

def test_rk_convergence_order():
    err_rk2 = _compute_error_exp("RK2")
    err_rk3 = _compute_error_exp("RK3")
    err_rk4 = _compute_error_exp("RK4")

    assert err_rk4 < err_rk3, f"RK4 ({err_rk4:.2e}) should beat RK3 ({err_rk3:.2e})"
    assert err_rk3 < err_rk2, f"RK3 ({err_rk3:.2e}) should beat RK2 ({err_rk2:.2e})"


# ---------------------------------------------------------------------------
# Galerkin convergence order: dG q=2 < dG q=1
# ---------------------------------------------------------------------------

def test_dg_convergence_order():
    err_q1 = _compute_error_exp("dG", is_galerkin=True, q=1)
    err_q2 = _compute_error_exp("dG", is_galerkin=True, q=2)

    assert err_q2 < err_q1, f"dG q=2 ({err_q2:.2e}) should beat dG q=1 ({err_q1:.2e})"
