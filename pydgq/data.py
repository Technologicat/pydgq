# -*- coding: utf-8 -*-
"""Load precalculated data for the Galerkin solver.

The data file contains Lobatto basis function values, Gauss-Legendre
quadrature points and weights, precomputed using extended-precision
arithmetic (via mpmath) to avoid catastrophic cancellation.

See pydgq.utils.precalc for regeneration.
"""

import numpy as np
from pathlib import Path
from importlib.resources import files, as_file

__all__ = ["load_data", "data_file_path"]

_DATA_FILENAME = "pydgq_data.npz"


def _find_data_file():
    """Find pydgq_data.npz, checking local, user config, and package locations.

    Returns a path-like object. For the package location, the caller must
    use `importlib.resources.as_file` as a context manager.
    """
    # 1. Local override
    local = Path(_DATA_FILENAME)
    if local.exists():
        return local

    # 2. User config override
    user = Path.home() / ".config" / "pydgq" / _DATA_FILENAME
    if user.exists():
        return user

    # 3. Installed package data
    return files("pydgq") / _DATA_FILENAME


def data_file_path():
    """Return a human-readable string describing where the data file was found."""
    ref = _find_data_file()
    if isinstance(ref, Path):
        return str(ref)
    return str(ref)  # Traversable.__str__ gives a useful representation


def load_data():
    """Load precalculated data, returning the nested dict expected by DataManager.

    Returns a dict with the same structure as the old pickle format::

        {"maxq": int,
         "integ": {"maxrule": int,
                   rule_int: {"x": array, "w": array, "y": array}, ...},
         "vis":   {"maxnx": int,
                   nx_int:  {"x": array, "y": array}, ...}}
    """
    ref = _find_data_file()

    # importlib Traversable needs as_file(); plain Path does not.
    if isinstance(ref, Path):
        npz = np.load(ref)
    else:
        with as_file(ref) as path:
            npz = np.load(path)

    # Reconstruct the nested dict from flat npz keys.
    maxq = int(npz["maxq"])
    maxrule = int(npz["integ/maxrule"])
    maxnx = int(npz["vis/maxnx"])

    integ = {"maxrule": maxrule}
    for rule in range(1, maxrule + 1):
        prefix = f"integ/{rule}/"
        if prefix + "x" in npz:
            integ[rule] = {
                "x": npz[prefix + "x"],
                "w": npz[prefix + "w"],
                "y": npz[prefix + "y"],
            }

    vis = {"maxnx": maxnx}
    for nx in range(1, maxnx + 1):
        prefix = f"vis/{nx}/"
        if prefix + "x" in npz:
            vis[nx] = {
                "x": npz[prefix + "x"],
                "y": npz[prefix + "y"],
            }

    return {"maxq": maxq, "integ": integ, "vis": vis}
