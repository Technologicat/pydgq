# -*- coding: utf-8 -*-
#
# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True
#
"""Helper for processing C^{-1} (finitely discontinuous) functions for correct plotting (i.e. without drawing a vertical line at the discontinuities).

Effectively, this is equivalent to the following NumPy snippet, using the idea from

    http://stackoverflow.com/questions/10377593/how-to-drop-connecting-lines-where-the-function-is-discontinuous

def discontify( data, idxs, fill="nan" ):  # data, idxs: rank-1 np.arrays
    # prepare
    nz = idxs.tolist()
    nz.sort()
    nz.reverse()

    if len(nz) == 0:
        return data

    # process
    out = data.copy()
    if fill == "nan":
        for pos in nz:
            out = np.insert(out, pos+1, np.nan)
    elif fill == "prev":
        for pos in nz:
            out = np.insert(out, pos+1, out[pos])
    else:
        raise ValueError("Unknown fill '%s'; available: 'nan', 'prev'" % fill)

    return out

but (even if the preparation was done only once) this Cython version works much faster when idxs contains a lot of elements.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from pydgq.solver.types cimport DTYPE_t
from pydgq.solver.types  import DTYPE, DNAN


def discontify( DTYPE_t[::1] data, int[::1] idxs, str fill="nan" ):
    """Make a rank-1 np.array discontinuous by inserting fill values.

    This is useful for plotting C^{-1} functions correctly (i.e. without drawing a vertical line at the discontinuities).

    The implementation uses Cython and runs at native (C) speed, so if there are a lot of discontinuities (as in e.g. a dG numerical solution),
    it is much faster than using np.insert() for each discontinuity individually.

    Parameters:
        data : rank-1 np.array
            input data
        idxs : rank-1 np.array
            indices in original data *after which* a copy of the fill value is to be inserted. Must be already sorted in increasing order.
        fill : str, one of:
            "nan"  :
                fill by NaN. Useful for y data of a C^{-1} function, as it disables plotting (in Matplotlib) for that data point.
            "prev" :
                fill by copying the element at the immediately preceding position in data. Useful for x data of a C^{-1} function,
                as Matplotlib will check the x values for their increasingness and emits a runtime warning if there are any NaNs.

    Returns:
        rank-1 np.array
            the discontified data. Note that the indexing no longer matches the original data (due to the inserted fill values).

    """
    cdef unsigned int ndata, nidxs
    ndata = data.shape[0]
    nidxs = idxs.shape[0]

    if nidxs == 0:  # if no idxs, this is a no-op
        return np.asanyarray(data)

    cdef unsigned int j, k
    cdef unsigned int offs = 0
    cdef DTYPE_t nan = DNAN
    cdef DTYPE_t[::1] out = np.empty( (ndata + nidxs,), dtype=DTYPE, order="C" )

    DEF MODE_NAN  = 1
    DEF MODE_PREV = 2
    cdef int fill_mode
    if fill == "nan":
        fill_mode = MODE_NAN
    elif fill == "prev":
        fill_mode = MODE_PREV
    else:
        raise ValueError("Unknown fill '%s'; available: 'nan', 'prev'" % (fill))

    with nogil:
        # copy data up to first discontinuity
        for k in range(idxs[0]+1):
            out[k] = data[k]

        # add the fill value
        if fill_mode == MODE_NAN:
            out[idxs[0]+1] = nan
        else: # fill_mode == MODE_PREV:
            out[idxs[0]+1] = out[idxs[0]]

        offs += 1  # one discontinuity handled; increase orig -> output index offset by one

        # handle the rest of discontinuities (if > 1 specified)
        if nidxs > 1:
            for j in range(1, nidxs):
                # copy data between discontinuities j-1 and j
                for k in range(idxs[j-1]+1, idxs[j]+1):
                    out[k+offs] = data[k]

                # add the fill value
                if fill_mode == MODE_NAN:
                    out[idxs[j]+offs+1] = nan
                else: # fill_mode == MODE_PREV:
                    out[idxs[j]+offs+1] = out[idxs[j]+offs]

                offs += 1  # one more discontinuity handled

        # copy data after the last discontinuity
        for k in range(idxs[nidxs-1]+1, ndata):
            out[k+offs] = data[k]

    return np.asanyarray(out)

