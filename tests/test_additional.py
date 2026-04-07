# -*- coding: utf-8 -*-
"""Additional tests for coverage gaps: SE integrator, interp>1, save_from>0, data loading."""

import numpy as np
import pytest

from pydgq.solver.types import DTYPE
from pydgq.solver.galerkin import init as galerkin_init
from pydgq.solver.odesolve import ivp, n_saved_timesteps, result_len, timestep_boundaries
import pydgq.solver.builtin_kernels as builtin_kernels
from pydgq.solver.kernel_interface import PythonKernel


# ---------------------------------------------------------------------------
# Symplectic Euler — needs a 2nd-order system
# ---------------------------------------------------------------------------

class HarmonicOscillatorKernel(PythonKernel):
    """w' = f(w) for the harmonic oscillator u'' = -u in companion form.

    State: w = (u1, v1, u2, v2, ...) with v = u'.
    RHS:   u' = v, v' = -u.
    """
    def callback(self, t):
        for j in range(0, self.n, 2):
            self.out[j] = self.w[j + 1]      # u' = v
            self.out[j + 1] = -self.w[j]     # v' = -u


def test_symplectic_euler():
    """SE on a harmonic oscillator: u'' = -u, u(0) = 0, u'(0) = 1."""
    m = 2
    n = 2 * m
    w0 = np.zeros((n,), dtype=DTYPE, order="C")
    w0[1::2] = 1.0  # v initial = 1

    rhs = HarmonicOscillatorKernel(n)
    dt = 0.01
    nt = 200

    n_saved = n_saved_timesteps(nt, 0)
    rlen = result_len(nt, 0)

    ww, tt = ivp(integrator="SE", allow_denormals=False,
                 w0=w0, dt=dt, nt=nt,
                 save_from=0, interp=1,
                 rhs=rhs,
                 ww=None, ff=None, fail=None,
                 maxit=1)

    tt = np.asarray(tt)
    # SE is 1st order, so tolerance is loose
    for j in range(m):
        u_ref = np.sin(tt)
        u_num = ww[:, 2 * j]
        relerr = np.max(np.abs(u_num - u_ref)) / np.max(np.abs(u_ref))
        assert relerr < 0.05, f"SE DOF {j}: relerr {relerr:.2e}"


def test_symplectic_euler_builtin_2nd_order():
    """SE with the built-in Linear2ndOrderKernel."""
    m = 2
    n = 2 * m
    M0 = -np.eye(m, dtype=DTYPE, order="C")
    M1 = np.zeros((m, m), dtype=DTYPE, order="C")

    w0 = np.zeros((n,), dtype=DTYPE, order="C")
    w0[1::2] = 1.0

    rhs = builtin_kernels.Linear2ndOrderKernel(n, M0, M1)
    dt = 0.01
    nt = 200

    ww, tt = ivp(integrator="SE", allow_denormals=False,
                 w0=w0, dt=dt, nt=nt,
                 save_from=0, interp=1,
                 rhs=rhs,
                 ww=None, ff=None, fail=None,
                 maxit=1)

    tt = np.asarray(tt)
    for j in range(m):
        u_ref = np.sin(tt)
        u_num = ww[:, 2 * j]
        relerr = np.max(np.abs(u_num - u_ref)) / np.max(np.abs(u_ref))
        assert relerr < 0.05, f"SE builtin DOF {j}: relerr {relerr:.2e}"


# ---------------------------------------------------------------------------
# Galerkin interp > 1 — verify interpolation inside timesteps
# ---------------------------------------------------------------------------

class DecayKernel(PythonKernel):
    """w' = -w, solution w(t) = w0 * exp(-t)."""
    def callback(self, t):
        for j in range(self.n):
            self.out[j] = -self.w[j]


def test_galerkin_interp():
    """dG with interp=11 — verify interpolated values match analytical solution."""
    n = 1
    w0 = np.array([1.0], dtype=DTYPE, order="C")
    rhs = DecayKernel(n)

    dt = 0.1
    nt = 20
    interp = 11
    save_from = 0

    galerkin_init(q=2, method="dG", nt_vis=interp, rule=None)

    rlen = result_len(nt, save_from, interp=interp)
    n_saved = n_saved_timesteps(nt, save_from)

    ww, tt = ivp(integrator="dG", allow_denormals=False,
                 w0=w0, dt=dt, nt=nt,
                 save_from=save_from, interp=interp,
                 rhs=rhs,
                 ww=None, ff=None, fail=None,
                 maxit=100)

    tt = np.asarray(tt)
    ww_ref = np.exp(-tt)
    relerr = np.max(np.abs(ww[:, 0] - ww_ref)) / np.max(np.abs(ww_ref))
    # Interior interpolation is limited by the Galerkin polynomial degree (q=2),
    # so accuracy is much less than endpoint accuracy (~1e-12).
    assert relerr < 1e-4, f"dG interp=11: relerr {relerr:.2e}"
    # Also check that we got the right number of output points
    assert ww.shape[0] == rlen


# ---------------------------------------------------------------------------
# save_from > 0 — verify partial saves work correctly
# ---------------------------------------------------------------------------

def test_save_from_nonzero():
    """Solve with save_from=50 and verify result matches the tail of a full solve."""
    n = 1
    w0 = np.array([1.0], dtype=DTYPE, order="C")
    dt = 0.01
    nt = 100

    # Full solve
    rhs_full = DecayKernel(n)
    ww_full, tt_full = ivp(integrator="RK4", allow_denormals=False,
                           w0=w0, dt=dt, nt=nt,
                           save_from=0, interp=1,
                           rhs=rhs_full,
                           ww=None, ff=None, fail=None)

    # Partial solve
    rhs_part = DecayKernel(n)
    ww_part, tt_part = ivp(integrator="RK4", allow_denormals=False,
                           w0=w0, dt=dt, nt=nt,
                           save_from=50, interp=1,
                           rhs=rhs_part,
                           ww=None, ff=None, fail=None)

    # Partial result should match the tail of the full result
    # save_from=0 gives IC + 100 timesteps = 101 points
    # save_from=50 gives timesteps 50..100 = 51 points
    assert ww_part.shape[0] == 51
    np.testing.assert_allclose(ww_part, ww_full[50:], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(tt_part), np.asarray(tt_full)[50:], rtol=1e-14)


# ---------------------------------------------------------------------------
# Data loading — verify structure of loaded data
# ---------------------------------------------------------------------------

def test_data_file_loads():
    """Verify pydgq.data.load_data() returns correct structure."""
    from pydgq.data import load_data, data_file_path

    data = load_data()
    path = data_file_path()

    assert isinstance(path, str)
    assert path.endswith("pydgq_data.npz")

    assert "maxq" in data
    assert "integ" in data
    assert "vis" in data

    assert data["maxq"] == 10
    assert data["integ"]["maxrule"] == 11
    assert data["vis"]["maxnx"] == 101

    # Check a sample rule
    rule = data["integ"][11]
    assert "x" in rule and "w" in rule and "y" in rule
    assert rule["x"].shape == (11,)
    assert rule["w"].shape == (11,)
    assert rule["y"].shape[0] == 11  # n_basis_functions

    # Check a sample vis
    vis = data["vis"][2]
    assert "x" in vis and "y" in vis
    assert vis["x"].shape == (2,)
