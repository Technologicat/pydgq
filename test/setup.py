# -*- coding: utf-8 -*-
#
"""Setup script for compiling the Cython example.

Usage:
    python -m setup build_ext --inplace
"""

from __future__ import division, print_function, absolute_import

#########################################################
# Config
#########################################################

# choose build type here
#
build_type="optimized"
#build_type="debug"


#########################################################
# Init
#########################################################

# check for Python 2.7 or later
# http://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
import sys
if sys.version_info < (2,7):
    sys.exit('Sorry, Python < 2.7 is not supported')

import os

from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build the extension modules for pydgq.")


#########################################################
# Definitions
#########################################################

extra_compile_args_math_optimized    = ['-fopenmp', '-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
extra_compile_args_math_debug        = ['-fopenmp', '-march=native', '-O0', '-g']

extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug     = ['-O0', '-g']

extra_link_args_optimized    = ['-fopenmp']
extra_link_args_debug        = ['-fopenmp']


if build_type == 'optimized':
    my_extra_compile_args_math    = extra_compile_args_math_optimized
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
    my_extra_link_args            = extra_link_args_optimized
    debug = False
    print( "build configuration selected: optimized" )
else: # build_type == 'debug':
    my_extra_compile_args_math    = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args            = extra_link_args_debug
    debug = True
    print( "build configuration selected: debug" )


#########################################################
# Helpers
#########################################################

my_include_dirs = ["."]  # IMPORTANT, see https://github.com/cython/cython/wiki/PackageHierarchy

def ext(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension( extName,
                      [extPath],
                      extra_compile_args=my_extra_compile_args_nonmath
                    )
def ext_math(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension( extName,
                      [extPath],
                      extra_compile_args=my_extra_compile_args_math,
                      extra_link_args=my_extra_link_args,
                      libraries=["m"]  # "m" links libm, the math library on unix-likes; see http://docs.cython.org/src/tutorial/external.html
                    )


#########################################################
# Modules
#########################################################

ext_module_ckernel  = ext_math( "cython_kernel" )

#########################################################

setup(
    ext_modules = cythonize( [ ext_module_ckernel ],
                             include_path = my_include_dirs,
                             gdb_debug = debug )
)

