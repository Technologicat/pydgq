# -*- coding: utf-8 -*-
#
"""Setuptools-based setup script for pydgq."""

from __future__ import division, print_function, absolute_import

import os

# remove file, ignore error if it did not exist
#
# portable version for Python 2.7 and 3.x
#
def myremove(filename):
    try:
        exc = FileNotFoundError  # this does not exist in Python 2.7
    except:
        exc = OSError

    try:
        os.remove( redirect_name )
    except exc:
        pass


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

from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build the extension modules for pydgq.")


#########################################################
# HACK: Attempt to package the correct data file on POSIX
#########################################################

redirect_name = os.path.join("pydgq", "pydgq_data.bin")
file_27_name  = os.path.join("pydgq", "pydgq_data_27.bin")
file_34_name  = os.path.join("pydgq", "pydgq_data_34.bin")

# remove existing symlink or file if any
#
myremove( redirect_name )

# symlink correct file depending on Python version
#
# (FIXME/TODO: for now, we assume that setup.py is running under the same Python version the packaged library will run under)
#
if sys.version_info < (3,0):
    print("Packaging %s as %s" % (file_27_name, redirect_name))
    os.symlink( file_27_name, redirect_name )
else:
    print("Packaging %s as %s" % (file_34_name, redirect_name))
    os.symlink( file_34_name, redirect_name )


#########################################################
# Definitions
#########################################################

extra_compile_args_math_optimized    = ['-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
extra_compile_args_math_debug        = ['-march=native', '-O0', '-g']

extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug     = ['-O0', '-g']

extra_link_args_optimized    = []
extra_link_args_debug        = []


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

# http://stackoverflow.com/questions/13628979/setuptools-how-to-make-package-contain-extra-data-folder-and-all-folders-inside
datadirs  = ("doc", "test")
dataexts  = (".py", ".pyx", ".pxd", ".c", ".sh", ".lyx", ".pdf")
datafiles = []
getext = lambda filename: os.path.splitext(filename)[1]
for datadir in datadirs:
    datafiles.extend( [(root, [os.path.join(root, f) for f in files if getext(f) in dataexts])
                       for root, dirs, files in os.walk(datadir)] )

datafiles.append( ('.', ["README.md", "LICENSE.md", "TODO.md", "CHANGELOG.md"]) )

#########################################################
# Modules
#########################################################

ext_module_types      = ext(      "pydgq.solver.types"                )

ext_module_compsum    = ext_math( "pydgq.solver.compsum"              )

ext_module_discontify = ext_math( "pydgq.utils.discontify"            )

ext_module_kernintf   = ext_math( "pydgq.solver.kernel_interface"     )
ext_module_bkernels   = ext_math( "pydgq.solver.builtin_kernels"      )

ext_module_intgintf   = ext_math( "pydgq.solver.integrator_interface" )
ext_module_explicit   = ext_math( "pydgq.solver.explicit"             )
ext_module_implicit   = ext_math( "pydgq.solver.implicit"             )
ext_module_galerkin   = ext_math( "pydgq.solver.galerkin"             )

ext_module_odesolve   = ext_math( "pydgq.solver.odesolve"             )

#########################################################

# Extract __version__ from the package __init__.py
# (since it's not a good idea to actually run __init__.py during the build process).
#
# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
#
import ast
with open('pydgq/__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            version = ast.parse(line).body[0].value.s
            break
    else:
        version = '0.0.unknown'
        print( "WARNING: Version information not found, using placeholder '%s'" % (version) )


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
                    "Programming Language :: Python :: 3",
                    "Programming Language :: Python :: 3.4",
                    "Topic :: Scientific/Engineering",
                    "Topic :: Scientific/Engineering :: Mathematics",
                    "Topic :: Software Development :: Libraries",
                    "Topic :: Software Development :: Libraries :: Python Modules"
                  ],

    setup_requires = ["cython", "numpy"],
    # pydgq.utils.precalc can optionally use mpi4py, but it is not mandatory.
    # Also, thre no need to run precalc in common use cases, since we include
    # the data file pydgq_data.bin (computed using default options) in the package.
    # Thus, we just leave out mpi4py.
    install_requires = ["numpy", "pylu"],
    provides = ["pydgq"],

    # same keywords as used as topics on GitHub
    keywords = ["numerical integration ordinary-differential-equations ode ivp ode-solver solver galerkin discontinuous-galerkin cython numpy"],

    ext_modules = cythonize( [ ext_module_types,
                               ext_module_compsum,
                               ext_module_discontify,
                               ext_module_kernintf, ext_module_bkernels,
                               ext_module_intgintf, ext_module_explicit, ext_module_implicit, ext_module_galerkin,
                               ext_module_odesolve,  ],
                             include_path = my_include_dirs,
                             gdb_debug = debug ),

    # Declare packages so that  python -m setup build  will copy .py files (especially __init__.py).
    packages = ["pydgq", "pydgq.solver", "pydgq.utils"],

    # Install also Cython headers so that other Cython modules can cimport ours
    # FIXME: force sdist, but sdist only, to keep the .pyx files (this puts them also in the bdist)
    package_data={'pydgq':        ['*.pxd', '*.pyx', '*.bin'],  # note: paths relative to each package
                  'pydgq.solver': ['*.pxd', '*.pyx'],
                  'pydgq.utils':  ['*.pxd', '*.pyx']},

    # Disable zip_safe, because:
    #   - Cython won't find .pxd files inside installed .egg, hard to compile libs depending on this one
    #   - dynamic loader may need to have the library unzipped to a temporary folder anyway (at import time)
    zip_safe = False,

    # Usage examples; not in a package
    data_files = datafiles
)


# remove symlink created earlier
#
myremove( redirect_name )

