# -*- coding: utf-8 -*-
"""Basic import and version tests for pydgq."""


def test_import():
    import pydgq
    from pathlib import Path
    expected = (Path(__file__).resolve().parent.parent / "pydgq" / "VERSION").read_text().strip()
    assert pydgq.__version__ == expected


def test_submodule_imports():
    from pydgq.solver import types, compsum, kernel_interface  # noqa: F401
    from pydgq.solver import integrator_interface, explicit, implicit  # noqa: F401
    from pydgq.solver import galerkin, builtin_kernels, odesolve  # noqa: F401
    from pydgq.utils import discontify  # noqa: F401
    from pydgq.examples import example_kernel  # noqa: F401
    from pydgq import data  # noqa: F401
