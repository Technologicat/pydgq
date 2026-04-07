# -*- coding: utf-8 -*-
#
"""Integrate first-order ODE system  u'(t) = f(u, t).

The main point of interest in this library is dG(q), i.e. the
time-discontinuous Galerkin method using a Lobatto basis
(a.k.a. hierarchical polynomial basis). See ivp().

For preparing the data file used by the integrator (pydgq_data.npz),
run the module pydgq.utils.precalc as the main program.

Note also that since the precalc module is not needed once the data file
has been generated, it is not automatically imported.

When this module is imported, it imports all symbols from pydgq.solver.odesolve
into the local namespace.
"""


from pathlib import Path as _Path
__version__ = (_Path(__file__).parent / "VERSION").read_text().strip()

from .solver.odesolve import *

