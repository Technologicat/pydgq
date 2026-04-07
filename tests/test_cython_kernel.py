# -*- coding: utf-8 -*-
"""Test all integrators with the example CythonKernel (cosine ODE)."""

import numpy as np
import pytest

from pydgq.solver.types import DTYPE
from pydgq.solver.galerkin import init as galerkin_init
from pydgq.solver.odesolve import ivp, n_saved_timesteps, result_len
from pydgq.examples.example_kernel import MyKernel


# ---------------------------------------------------------------------------
# Common settings
# ---------------------------------------------------------------------------

N = 3
NT = 100
DT = 0.1
SAVE_FROM = 0
Q = 2
OMEGA = 0.1 * (2.0 * np.pi)


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrator, rel_tol, is_galerkin", [
    ("IMR", 1e-3, False),
    ("BE", 1e-1, False),
    ("RK4", 1e-8, False),
    ("RK3", 1e-6, False),
    ("RK2", 1e-3, False),
    ("FE", 1e-1, False),
    ("dG", 1e-12, True),
    ("cG", 1e-4, True),
])
def test_integrator_cython_kernel(integrator, rel_tol, is_galerkin):
    nt_vis = 1
    if is_galerkin:
        galerkin_init(q=Q, method=integrator, nt_vis=nt_vis, rule=None)

    n_saved = n_saved_timesteps(NT, SAVE_FROM)
    rlen = result_len(NT, SAVE_FROM, interp=nt_vis)

    w0 = np.zeros((N,), dtype=DTYPE, order="C")
    rhs = MyKernel(N, omega=OMEGA)

    ww = np.empty((rlen, N), dtype=DTYPE, order="C")
    ff = np.empty((rlen, N), dtype=DTYPE, order="C")
    fail = np.empty((n_saved,), dtype=np.intc, order="C")

    ww, tt = ivp(integrator=integrator, allow_denormals=False,
                 w0=w0, dt=DT, nt=NT,
                 save_from=SAVE_FROM, interp=nt_vis,
                 rhs=rhs,
                 ww=ww, ff=ff, fail=fail,
                 maxit=100)

    ww_ref = rhs.reference_solution(np.ascontiguousarray(tt))
    relerr = np.linalg.norm(ww - ww_ref, ord=np.inf) / np.linalg.norm(ww_ref, ord=np.inf)
    assert relerr < rel_tol, f"{integrator}: relative L-inf error {relerr:.2e} >= tolerance {rel_tol:.2e}"
