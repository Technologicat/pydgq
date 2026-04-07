# -*- coding: utf-8 -*-
"""Test built-in linear kernels with known analytical solutions."""

import numpy as np
import pytest

from pydgq.solver.types import DTYPE
from pydgq.solver.galerkin import init as galerkin_init
from pydgq.solver.odesolve import ivp, n_saved_timesteps, result_len
import pydgq.solver.builtin_kernels as builtin_kernels


# ---------------------------------------------------------------------------
# Common settings
# ---------------------------------------------------------------------------

DT = 0.02
NT = 100
SAVE_FROM = 0
Q = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solve(integrator, n, w0, rhs, dt=DT, nt=NT, is_galerkin=False):
    """Run ivp() and return (ww, tt)."""
    nt_vis = 1
    if is_galerkin:
        galerkin_init(q=Q, method=integrator, nt_vis=nt_vis, rule=None)

    n_saved = n_saved_timesteps(nt, SAVE_FROM)
    rlen = result_len(nt, SAVE_FROM, interp=nt_vis)

    ww = np.empty((rlen, n), dtype=DTYPE, order="C")
    fail = np.empty((n_saved,), dtype=np.intc, order="C")

    ww, tt = ivp(integrator=integrator, allow_denormals=False,
                 w0=w0, dt=dt, nt=nt,
                 save_from=SAVE_FROM, interp=nt_vis,
                 rhs=rhs,
                 ww=ww, ff=None, fail=fail,
                 maxit=100)
    return ww, tt


# ---------------------------------------------------------------------------
# Linear1stOrderKernel:  w' = M*w  with M = I  =>  w(t) = w0 * exp(t)
# ---------------------------------------------------------------------------

INTEGRATORS_1ST = [
    ("IMR", 1e-3, False),
    ("BE", 1e-1, False),
    ("RK4", 1e-8, False),
    ("RK3", 1e-5, False),
    ("RK2", 1e-3, False),
    ("FE", 1e-1, False),
    ("dG", 1e-10, True),
    ("cG", 1e-3, True),
]


@pytest.mark.parametrize("integrator, rel_tol, is_galerkin", INTEGRATORS_1ST)
def test_linear_1st_order_kernel(integrator, rel_tol, is_galerkin):
    n = 3
    w0 = np.arange(1, n + 1, dtype=DTYPE)
    M = np.eye(n, dtype=DTYPE, order="C")
    rhs = builtin_kernels.Linear1stOrderKernel(n, M)

    ww, tt = _solve(integrator, n, w0, rhs, is_galerkin=is_galerkin)

    # Known solution: w(t) = w0 * exp(t)
    ww_ref = w0[np.newaxis, :] * np.exp(np.asarray(tt))[:, np.newaxis]
    relerr = np.linalg.norm(ww - ww_ref, ord=np.inf) / np.linalg.norm(ww_ref, ord=np.inf)
    assert relerr < rel_tol, f"{integrator}: relative L-inf error {relerr:.2e} >= tolerance {rel_tol:.2e}"


# ---------------------------------------------------------------------------
# Linear1stOrderKernelWithMassMatrix:  A*w' = M*w
# Result should match Linear1stOrderKernel with inv(A)*M as the matrix.
# ---------------------------------------------------------------------------

INTEGRATORS_MASS = [
    ("IMR", 1e-10, False),
    ("RK4", 1e-10, False),
    ("RK3", 1e-10, False),
    ("RK2", 1e-10, False),
    ("BE", 1e-10, False),
    ("FE", 1e-10, False),
    ("dG", 1e-10, True),
    ("cG", 1e-10, True),
]


@pytest.mark.parametrize("integrator, rel_tol, is_galerkin", INTEGRATORS_MASS)
def test_linear_1st_order_kernel_with_mass_matrix(integrator, rel_tol, is_galerkin):
    n = 2
    M = np.array([[2.0, 1.0],
                   [0.5, 3.0]], dtype=DTYPE, order="C")
    A = np.array([[1.5, 0.3],
                   [0.2, 1.0]], dtype=DTYPE, order="C")
    w0 = np.array([1.0, 2.0], dtype=DTYPE)

    # Solve with mass matrix kernel
    rhs_mass = builtin_kernels.Linear1stOrderKernelWithMassMatrix(n, M, A)
    ww_mass, tt = _solve(integrator, n, w0, rhs_mass, is_galerkin=is_galerkin)

    # Solve with equivalent plain kernel using inv(A)*M
    Ainv_M = np.ascontiguousarray(np.linalg.solve(A, M), dtype=DTYPE)
    rhs_plain = builtin_kernels.Linear1stOrderKernel(n, Ainv_M)
    ww_plain, _ = _solve(integrator, n, w0, rhs_plain, is_galerkin=is_galerkin)

    # The two should give the same result (up to floating-point)
    abserr = np.linalg.norm(ww_mass - ww_plain, ord=np.inf)
    scale = max(np.linalg.norm(ww_plain, ord=np.inf), 1.0)
    assert abserr / scale < rel_tol, f"{integrator}: mass vs plain mismatch {abserr / scale:.2e}"


# ---------------------------------------------------------------------------
# Linear2ndOrderKernel:  u'' = M0*u + M1*u'
# Harmonic oscillator:   u'' = -u  (M0 = -I, M1 = 0)
# Solution:  u(t) = C1*sin(t) + C2*cos(t)
# With IC u(0) = 0, u'(0) = 1 => u(t) = sin(t)
# ---------------------------------------------------------------------------

INTEGRATORS_2ND = [
    ("IMR", 1e-3, False),
    ("BE", 2e-1, False),
    ("RK4", 1e-7, False),
    ("RK3", 1e-5, False),
    ("RK2", 1e-3, False),
    ("FE", 2e-1, False),
    ("dG", 1e-8, True),
    ("cG", 1e-3, True),
]


@pytest.mark.parametrize("integrator, rel_tol, is_galerkin", INTEGRATORS_2ND)
def test_linear_2nd_order_kernel(integrator, rel_tol, is_galerkin):
    m = 2  # 2 second-order DOFs
    n = 2 * m  # 4 first-order DOFs
    M0 = -np.eye(m, dtype=DTYPE, order="C")  # u'' = -u
    M1 = np.zeros((m, m), dtype=DTYPE, order="C")  # no damping

    # IC: u(0) = 0, u'(0) = 1 for each DOF
    w0 = np.zeros((n,), dtype=DTYPE, order="C")
    w0[1::2] = 1.0  # u' components

    rhs = builtin_kernels.Linear2ndOrderKernel(n, M0, M1)
    ww, tt = _solve(integrator, n, w0, rhs, is_galerkin=is_galerkin)
    tt = np.asarray(tt)

    # Reference: u(t) = sin(t), u'(t) = cos(t) for each DOF
    for j in range(m):
        u_ref = np.sin(tt)
        v_ref = np.cos(tt)
        u_num = ww[:, 2 * j]
        v_num = ww[:, 2 * j + 1]

        relerr_u = np.max(np.abs(u_num - u_ref)) / np.max(np.abs(u_ref))
        relerr_v = np.max(np.abs(v_num - v_ref)) / np.max(np.abs(v_ref))
        assert relerr_u < rel_tol, f"{integrator} DOF {j} u: relerr {relerr_u:.2e} >= {rel_tol:.2e}"
        assert relerr_v < rel_tol, f"{integrator} DOF {j} v: relerr {relerr_v:.2e} >= {rel_tol:.2e}"


# ---------------------------------------------------------------------------
# Linear2ndOrderKernelWithMassMatrix:  M2*u'' = M0*u + M1*u'
# Verify equivalence with the non-mass-matrix version using inv(M2)*M0, inv(M2)*M1.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrator, rel_tol, is_galerkin", INTEGRATORS_MASS)
def test_linear_2nd_order_kernel_with_mass_matrix(integrator, rel_tol, is_galerkin):
    m = 2
    n = 2 * m
    M0 = np.array([[-2.0, 0.5],
                     [0.3, -1.5]], dtype=DTYPE, order="C")
    M1 = np.array([[-0.1, 0.0],
                     [0.0, -0.2]], dtype=DTYPE, order="C")
    M2 = np.array([[1.5, 0.3],
                     [0.1, 2.0]], dtype=DTYPE, order="C")

    w0 = np.zeros((n,), dtype=DTYPE, order="C")
    w0[0::2] = 0.5  # u initial
    w0[1::2] = 1.0  # u' initial

    # Solve with mass matrix kernel
    rhs_mass = builtin_kernels.Linear2ndOrderKernelWithMassMatrix(n, M0, M1, M2)
    ww_mass, tt = _solve(integrator, n, w0, rhs_mass, is_galerkin=is_galerkin)

    # Solve with equivalent plain kernel: u'' = inv(M2)*M0*u + inv(M2)*M1*u'
    M2inv = np.linalg.inv(M2)
    M0_eff = np.ascontiguousarray(M2inv @ M0, dtype=DTYPE)
    M1_eff = np.ascontiguousarray(M2inv @ M1, dtype=DTYPE)
    rhs_plain = builtin_kernels.Linear2ndOrderKernel(n, M0_eff, M1_eff)
    ww_plain, _ = _solve(integrator, n, w0, rhs_plain, is_galerkin=is_galerkin)

    abserr = np.linalg.norm(ww_mass - ww_plain, ord=np.inf)
    scale = max(np.linalg.norm(ww_plain, ord=np.inf), 1.0)
    # Dense M2 means two different code paths (LU solve vs explicit inv(M2)*M),
    # so we allow O(eps)-accumulated differences rather than exact match.
    assert abserr / scale < 1e-8, f"{integrator}: 2nd-order mass vs plain mismatch {abserr / scale:.2e}"


# ---------------------------------------------------------------------------
# Gyroscopic system: M2*u'' = M0*u + M1*u'  with skew-symmetric M1
#
# This exercises the full off-diagonal structure of the matvec in compute().
# The skew-symmetric damping (gyroscopic coupling) means all matrix entries
# participate; a bug that uses the wrong index would break the result.
# ---------------------------------------------------------------------------

INTEGRATORS_GYRO = [
    ("RK4", 1e-6, False),
    ("dG", 1e-7, True),
    ("IMR", 1e-2, False),
]


@pytest.mark.parametrize("integrator, rel_tol, is_galerkin", INTEGRATORS_GYRO)
def test_gyroscopic_system(integrator, rel_tol, is_galerkin):
    """Gyroscopic 2-DOF oscillator with skew-symmetric coupling."""
    m = 2
    n = 2 * m

    # Stiffness (restoring force)
    M0 = np.array([[-4.0, 1.0],
                     [1.0, -9.0]], dtype=DTYPE, order="C")
    # Gyroscopic (skew-symmetric) damping
    M1 = np.array([[0.0, -0.5],
                     [0.5,  0.0]], dtype=DTYPE, order="C")
    # Mass matrix
    M2 = np.array([[2.0, 0.3],
                     [0.3, 1.5]], dtype=DTYPE, order="C")

    w0 = np.zeros((n,), dtype=DTYPE, order="C")
    w0[0] = 1.0   # u1(0) = 1
    w0[3] = -0.5  # v2(0) = -0.5

    # Solve with mass matrix kernel
    rhs_mass = builtin_kernels.Linear2ndOrderKernelWithMassMatrix(n, M0, M1, M2)
    ww_mass, tt = _solve(integrator, n, w0, rhs_mass, dt=0.01, nt=200,
                         is_galerkin=is_galerkin)

    # Solve with equivalent plain kernel: u'' = inv(M2)*M0*u + inv(M2)*M1*u'
    M2inv = np.linalg.inv(M2)
    M0_eff = np.ascontiguousarray(M2inv @ M0, dtype=DTYPE)
    M1_eff = np.ascontiguousarray(M2inv @ M1, dtype=DTYPE)
    rhs_plain = builtin_kernels.Linear2ndOrderKernel(n, M0_eff, M1_eff)
    ww_plain, _ = _solve(integrator, n, w0, rhs_plain, dt=0.01, nt=200,
                         is_galerkin=is_galerkin)

    abserr = np.linalg.norm(ww_mass - ww_plain, ord=np.inf)
    scale = max(np.linalg.norm(ww_plain, ord=np.inf), 1.0)
    assert abserr / scale < rel_tol, f"{integrator}: gyroscopic mass vs plain mismatch {abserr / scale:.2e}"
