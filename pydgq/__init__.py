# -*- coding: utf-8 -*-
#
"""Integrate first-order ODE system  u'(t) = f(u, t).

The main point of interest in this library is dG(q), i.e. the
time-discontinuous Galerkin method using a Lobatto basis
(a.k.a. hierarchical polynomial basis). See ivp().

For preparing the data file used by the integrator (pydgq_data.bin),
run the module pydgq.utils.precalc as the main program.

Note also that since the precalc module is not needed once the data file
has been generated, it is not automatically imported.

When this module is imported, it imports all symbols from pydgq.solver.odesolve
into the local namespace.
"""

from __future__ import absolute_import  # https://www.python.org/dev/peps/pep-0328/

__version__ = '0.1.2'

from .solver.odesolve import *

