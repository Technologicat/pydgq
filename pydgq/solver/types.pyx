# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from numpy import float64, complex128, nan

# real
RTYPE = float64
RNAN  = nan

# complex
ZTYPE = complex128
ZNAN  = nan * (1. + 1j)  # both real and imaginary parts nan (to avoid pitfalls for things like isnan(imag(w)))

# problem data (choose real or complex here)
DTYPE = RTYPE
DNAN  = RNAN  # appropriate NaN value for DTYPE

