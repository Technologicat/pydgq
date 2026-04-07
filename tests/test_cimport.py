# -*- coding: utf-8 -*-
"""Verify that .pxd files are installed and cimport works."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import pydgq


def test_pxd_installed():
    """Verify .pxd files are present in the installed package."""
    solver_dir = Path(pydgq.__file__).parent / "solver"
    expected = [
        "types.pxd",
        "compsum.pxd",
        "fputils.pxd",
        "cminmax.pxd",
        "kernel_interface.pxd",
        "integrator_interface.pxd",
        "explicit.pxd",
        "implicit.pxd",
        "galerkin.pxd",
        "builtin_kernels.pxd",
    ]
    for name in expected:
        assert (solver_dir / name).exists(), f"{name} not found in {solver_dir}"


def test_cimport_compiles(tmp_path):
    """Verify that cimport pydgq.solver.types succeeds at the Cython level."""
    pytest.importorskip("Cython")

    pyx = tmp_path / "check_cimport.pyx"
    pyx.write_text(textwrap.dedent("""\
        from pydgq.solver.types cimport DTYPE_t, RTYPE_t
        from pydgq.solver.kernel_interface cimport CythonKernel
    """))

    result = subprocess.run(
        [sys.executable, "-m", "cython", str(pyx)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"cython failed:\n{result.stderr}"
    assert (tmp_path / "check_cimport.c").exists()
