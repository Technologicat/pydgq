# -*- coding: utf-8 -*-
#
"""Setuptools-based setup script for pydgq."""

from __future__ import absolute_import

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
    print "build configuration selected: optimized"
else: # build_type == 'debug':
    my_extra_compile_args_math    = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args            = extra_link_args_debug
    debug = True
    print "build configuration selected: debug"


#########################################################
# Long description
#########################################################

DESC="""Integrate first-order ODE systems  u'(t) = f(u, t).

The main feature of this library is dG(q), i.e. the
time-discontinuous Galerkin method using a Lobatto basis
(a.k.a. hierarchical polynomial basis).

dG(q) is a very accurate implicit method that often allows
using a rather large timestep. Due to its Galerkin nature,
it also allows inspecting the behavior of the solution
inside the timestep.

Arbitrary q is supported, but often best results are
obtained for q=1 or q=2.

Some classical integrators (RK2, RK3, RK4, IMR, BE)
are also provided for convenience.

The focus is on arbitrary nonlinear problems; all implicit
methods are implemented using fixed-point (Banach/Picard)
iteration.
"""


#########################################################
# Helpers
#########################################################

my_include_dirs = [".", "/usr/local/lib/python2.7/dist-packages"]  # IMPORTANT, see https://github.com/cython/cython/wiki/PackageHierarchy

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

# http://stackoverflow.com/questions/13628979/setuptools-how-to-make-package-contain-extra-data-folder-and-all-folders-inside
datadir = "test"
datafiles = [(root, [os.path.join(root, f) for f in files if f.endswith(".py")])
    for root, dirs, files in os.walk(datadir)]

datafiles.append( ('.', ["README.md", "LICENSE.md"]) )
datafiles.append( ('doc', ["pydgq_user_manual.lyx", "pydgq_user_manual.pdf"]) )


#########################################################
# Modules
#########################################################

ext_module_ptrwrap  = ext(      "pydgq.utils.ptrwrap"      )
ext_module_types    = ext(      "pydgq.solver.pydgq_types" )
ext_module_explicit = ext(      "pydgq.solver.explicit"    )
ext_module_implicit = ext(      "pydgq.solver.implicit"    )
ext_module_galerkin = ext(      "pydgq.solver.galerkin"    )
ext_module_odesolve = ext_math( "pydgq.solver.odesolve"    )
ext_module_kernels  = ext_math( "pydgq.solver.kernels"     )

#########################################################

# Extract __version__ from the package __init__.py
# (since it's not a good idea to actually run __init__.py during the build process).
#
# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
#
import ast
with file('pydgq/__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            version = ast.parse(line).body[0].value.s
            break
    else:
        version = '0.0.unknown'
        print "WARNING: Version information not found, using placeholder '%s'" % (version)


setup(
    name = "pydgq",
    version = version,
    author = "Juha Jeronen",
    author_email = "juha.jeronen@jyu.fi",
    url = "https://github.com/Technologicat/pydgq",

    description = "ODE system solver using dG(q) (time-discontinuous Galerkin w/ Lobatto basis)",
    long_description = DESC,

    license = "BSD",
    platforms = ["Linux"],  # free-form text field; http://stackoverflow.com/questions/34994130/what-platforms-argument-to-setup-in-setup-py-does

    classifiers = [ "Development Status :: 4 - Beta",
                    "Environment :: Console",
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: BSD License",
                    "Operating System :: POSIX :: Linux",
                    "Programming Language :: Cython",
                    "Programming Language :: Python",
                    "Programming Language :: Python :: 2",
                    "Programming Language :: Python :: 2.7",
                    "Topic :: Scientific/Engineering",
                    "Topic :: Scientific/Engineering :: Mathematics",
                    "Topic :: Software Development :: Libraries",
                    "Topic :: Software Development :: Libraries :: Python Modules"
                  ],

    setup_requires = ["cython", "numpy"],
    install_requires = ["numpy", "mpi4py", "pylu"],  # mpi4py for precalc only
    provides = ["pydgq"],

    # same keywords as used as topics on GitHub
    keywords = ["numerical integration ordinary-differential-equations initial-value-problems cython"],  # TODO update this (check GitHub for common keywords for this topic)

    ext_modules = cythonize( [ ext_module_ptrwrap, ext_module_types,
                               ext_module_explicit, ext_module_implicit, ext_module_galerkin,
                               ext_module_odesolve, ext_module_kernels ],
                             include_path = my_include_dirs,
                             gdb_debug = debug ),

    # Declare packages so that  python -m setup build  will copy .py files (especially __init__.py).
    packages = ["pydgq", "pydgq.solver", "pydgq.utils"],

    # Install also Cython headers so that other Cython modules can cimport ours
    # FIXME: force sdist, but sdist only, to keep the .pyx files (this puts them also in the bdist)
    package_data={'pydgq': ['*.pxd', '*.pyx']},  # note: paths relative to each package

    # Usage examples; not in a package
    data_files = datafiles
)

