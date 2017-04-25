# -*- coding: utf-8 -*-
"""Generate data file (pydgq_data.bin) for pydgq.

The generation is very slow; MPI parallelization is supported to boost performance.

Run this module as the main program (with or without mpiexec) to perform the precalculation.
Command-line options are available; pass the standard --help flag to see them.
"""

from __future__ import division, print_function, absolute_import

import time

try:
    import cPickle as pickle  # Python 2.7
except ImportError:
    import pickle  # Python 3.x

import functools  # reduce (Python 3 compatibility)

import numpy as np

try:
    import mpmath  # Python 3.x
except ImportError:
    import sympy.mpmath as mpmath  # Python 2.7

from pydgq.solver.types import RTYPE  # the precalc data is always real-valued regardless of DTYPE
import pydgq.utils.mpi_shim as mpi_shim
import pydgq.utils.listutils as listutils

# version of precalc.py (this is NOT the version of the pydgq package)
__version__ = "1.0.2"


############################################################################################################
# Worker classes
############################################################################################################

# Memoization
class Cache:
    """Cache for values that have been computed so far. Useful if precomputing for a large number of different divisions of [-1,1] some of which contain identical points.

    The cache is meant to be shared across all instances of Precalc in the current process.

    """
    def __init__(self, q):
        """q = maximum degree (see Precalc)"""

        self.q     = q
        self._data = {}

        # create empty caches for all polynomial degrees
        for j in range(q+1):
            self._data[j] = {}

    # syntactic sugar

    # e.g. "if cache contains 3" --> a sub-cache exists for 3rd degree polynomials
    def __contains__(self, key):
        return key in self._data

    # e.g. "cache[3]" --> the sub-cache for 3rd degree polynomials
    def __getitem__(self, key):
        return self._data[key]

    # the caller is supposed to write only to the individual sub-caches, not directly into the top level one
    def __setitem__(self, key, value):
        raise NotImplementedError("The top-level cache in Cache does not support __setitem__(), maybe you meant to write into cache[j]?")


class Precalc:
    """Precalculate hierarchical (Lobatto) basis functions up to the given degree q (>= 1) at the given points (rank-1 np.array of length >= 1) on the reference element [-1,1]."""

    # Legendre polynomials, high precision from sympy.mpmath to avoid cancellation in hierarchical basis functions
    _P = mpmath.legendre  # lambda j, x: ...

    def __init__(self, q, xx, cache=None):
        """q = maximum degree, xx = vector of points, cache = Cache instance or None

        The cache is used to memoize results for each unique value in xx. This is mainly useful
        for sharing evaluations to future instances, which may get some of the same elements in their xx.
        """

        self.x  = None
        self.y  = None  # "y = f(x)"

        assert(q >= 1)
        assert(np.size(xx) >= 1)

        self.q     = q
        self.xx    = xx
        self.cache = cache

        self._build_hierarchical_basis()

    def _build_hierarchical_basis(self):
        """Build a list of Python functions that evaluate the hierarchical basis functions.

        After running this, self.N[j] is a one-argument lambda that evaluates N_j at x. (Not vectorized!)

        """
        q = self.q

        N = []
        N.append( lambda x: (1./2.) * (1. - x) )  # linear, left endpoint
        N.append( lambda x: (1./2.) * (1. + x) )  # linear, right endpoint

        # bubble functions (see user manual)
        for j in range(2,q+1):
            # HACK: Python 3 compatibility: we must float(j), because some part of the toolchain here wants to convert all arguments to mpf, which does not work for int.
            N.append(  (  lambda j: lambda x : ( self._P(j, x) - self._P(j-2, x) ) / np.sqrt( 2. * (2.*j - 1.) )  )(float(j))  )  # use factory to bind j at define time

        self.N = N

    def run(self):
        """Do the precomputation. MPI-fied."""

        # Create indices and corresponding x values.
        #
        nx    = np.size(self.xx)
        all_i = range(nx)
        all_x = self.xx

        # Divide the work among the tasks.
        #
        split_i = listutils.load_balance_list( all_i, mpi_shim.get_size() )
        my_i    = split_i[ mpi_shim.get_rank() ]

        # Do the local work for task-local items.
        #
        N = self.N
        ly = np.empty( [self.q+1, len(my_i)], dtype=RTYPE )  # local y (basis function values)
        # use caching (if available) to avoid slow re-evaluation of the polynomials for x values already seen.
        if self.cache is not None:
            for li,gi in enumerate(my_i):  # local i, corresponding global i
                x = all_x[gi]
                for j in range(self.q+1):
                    # yes, we use floats as keys, and we do really want floating-point equality down to the last ulp in order to accept the entry as matched.
                    if x in self.cache[j]:
                        ly[j,li] = self.cache[j][x]
                    else:
                        ly[j,li] = N[j]( x )  # Compute. Writing into the ly array forces conversion to RTYPE, as we want.
                        self.cache[j][x] = ly[j,li]

        else:
            for li,gi in enumerate(my_i):  # local i, corresponding global i
                x = all_x[gi]
                for j in range(self.q+1):
                    ly[j,li] = N[j]( x )

        # Gather results.
        #
        if mpi_shim.get_size() > 1:
            # Allocate an array to receive the global data.
            #
            # This will need to be reordered after receiving, because any leftover items (modulo nprocs) are processed by some of the first tasks.
            # We have the correct x indices in split_i; at the first step we just glue together the data arrays in sequence using MPI's Gatherv().
            #
            gshape  = [nx, self.q+1]  # transposed global shape (see below)
            recv    = np.empty( np.prod(gshape), dtype=RTYPE )  # linear buffer for global data (we'll reshape this after receiving)

            counts  = [len(lis)*(self.q+1) for lis in split_i]  # lis contains the task-local item indices; all tasks handle q+1 basis functions
            disps   = [0] + np.cumsum( np.array(counts, dtype=int)[:-1] ).tolist()   #  e.g. [3,3,3,2] -> [0,3,6,9]

            assert( np.prod(np.size(ly)) == counts[mpi_shim.get_rank()] )  # in each task, computed local data size must match actual size of local data

            # The number of basis functions is constant (q+1), while the length of my_i may vary across the tasks.
            #
            # Thus, we transpose the local data array before linearizing it for sending, so that the "slices" are of constant length.
            # This makes it simpler to de-linearize the global array after the gather.
            #
            # For Gatherv(), Allgatherv(), sendbuf, recvbuf, see:
            #    https://wiki.gwdg.de/index.php/Mpi4py
            #
            sendbuf = [ np.reshape(np.transpose(ly),-1), counts[mpi_shim.get_rank()] ]  # local_data, local_count (must be counts[mpi_rank])

            # TODO: other dtypes?
            if RTYPE == np.float64:
                MPI_datatype = mpi_shim.get_mpi().DOUBLE
            elif RTYPE == np.float32:
                MPI_datatype = mpi_shim.get_mpi().SINGLE
            else:
                raise NotImplementedError("Unknown RTYPE %s, cannot transmit data buffer" % (RTYPE))
            recvbuf = [ recv, counts, disps, MPI_datatype ]  # data, counts, displacements, mpi_datatype
            comm = mpi_shim.get_comm_world()
            comm.Allgatherv( sendbuf, recvbuf )

            # De-linearize the received array.
            #
            # This utilizes the fact that in the transposed data, each "item" (one x value for all basis functions) has the same length.
            #
            recv = np.reshape(recv, gshape)

            # Get the permutation of rows that was applied by splitting all_i -> split_i and then gluing together the split_i in sequence.
            #
            # E.g. with nx=11, nprocs=4, we could have
            #
            # MPI rank 0: [0,1,8]
            # MPI rank 1: [2,3,9]
            # MPI rank 2: [4,5,10]
            # MPI rank 3: [6,7]
            #
            # After gather this becomes:
            #
            # [0,1,8,2,3,9,4,5,10,6,7]
            #
            # These are the indices in all_x corresponding to each row in the transposed data array.
            #
            perm = np.array( functools.reduce( lambda x,y: x+y,  split_i ), dtype=int )

            # Get the inverse permutation.
            #
            invperm       = np.empty_like( perm )
            invperm[perm] = np.arange( np.size(perm), dtype=int )  # inverse permutation of range(N): invperm[perm] = range(N)

            # Remap by inverse permutation. This orders the data correctly, so that the ith row of the transposed data corresponds to basis function values at all_x[i].
            #
            # Undo the transpose (applying it again) to obtain the final result. Then self.y[j,i] is N[j]( all_x[i] ).
            #
            self.y = np.transpose( recv[invperm] )
            self.x = all_x

        else:  # only one task, which processes everything
            self.y = ly
            self.x = all_x


############################################################################################################
# Main program
############################################################################################################

def main(q, nx, **kwargs):
    """Create precomputed arrays of basis function values.

    Parameters are the maximum degree q (>= 1) and the maximum number of visualization points nx (>= 1).

    Since the basis is hierarchical, any lower degree is obtained by simply chopping off the extraneous rows, as y[:(desired_q + 1),:].

    Thus, for visualization, different arrays are needed only for different numbers of points, which are taken to be equally spaced
    on the reference element [-1,1]. Arrays are generated for 1, 2, ..., nx points. Caching (memoization) is used to accelerate
    the computation so that only unique points are actually evaluated (e.g. arrays with 3 and 5 points share the point at 0.0)

    For integration, separate arrays are created, of function values at Gauss-Legendre points for integration rules of order 1,2,...,q+1.
    The q+1 rule is sufficient for evaluation of mass matrices of the form N(x)*N(x) (where N has at most degree q).
    The q rule is sufficient for N'(x)*N(x) (needed in dG(q)) and lower orders. (This is assuming an affine coordinate mapping
    from the reference element to the actual timestep, so that the Jacobian of the coordinate mapping is constant on each element.)
    """

    if mpi_shim.get_size() < 2:
        msg = " This script can benefit from MPI parallelization; consider running with 'mpiexec -n <some reasonable number> python -m precalc\n"
        sep = ( "=" * len(msg) ) + "\n"
        mpi_shim.mpi_print("\n\n%s%s%s\n" % (sep, msg, sep))

    data = {}

    # common metadata
    data["maxq"] = q  # note: one less than number of rows in y items

    # memoize for faster computation
    cache = Cache(q)

    #########################################################
    # Create array for visualization (equally spaced points)
    #########################################################

    mpi_shim.mpi_print( "Maximum degree %d (%d basis functions)" % (q, q+1) )

    mpi_shim.mpi_print( "\nGenerating visualization data" )
    mpi_shim.mpi_print( "Equally spaced points: max points = %d" % nx )

    data["vis"] = {}  # data at visualization points
    data["vis"]["maxnx"] = nx

    for k in range(1,nx+1):
        if k == 1:
            # Considering the use of this data in ODE system integration (odesolve.pyx),
            # the end of the timestep is the most reasonable choice in the case of a single point.
            #
            xx = np.array( [1.], dtype=RTYPE )
        else:
            xx = np.linspace( -1., 1., k, dtype=RTYPE )

        mpi_shim.mpi_print( "    Computing for %d points" % k )

        # The cache avoids the need to re-compute the basis functions for any values of x
        # already seen at an earlier iteration of this loop, making this run much faster.
        #
        precalc = Precalc(q, xx, cache)
        precalc.run()

        data["vis"][k] = { "x" : precalc.x, "y" : precalc.y }

#    # NOTE: the data is accessed by indexing by ["vis"][how many points], like this:
#    print data["vis"][3]["x"]  # points (x values)
#    print data["vis"][3]["y"]  # corresponding function values (y values)


    #########################################################
    # Create array for integration (Gauss-Legendre points)
    #########################################################

    mpi_shim.mpi_print( "\nGenerating integrator data (Gauss-Legendre rule)" )

    data["integ"] = {}  # data at Gauss-Legendre integration points
    data["integ"]["maxrule"] = q+1

    for d in range(1,q+2):
        mpi_shim.mpi_print( "    Computing for GL rule of order %d" % d )

        xx,ww = np.polynomial.legendre.leggauss( d )

        # We run this from order 1 up to order q+1, even though the lower-degree rules may not be high enough for exact integration
        # of the highest-degree basis functions, in order to allow for under-integration by the user later.
        #
        # Note that if a lower-degree basis is being used (by chopping off rows of the data array, see above),
        # one can then chop this data, too, using a lower-degree (desired_q + 1) rule to still obtain exact integration.
        #
        # E.g.
        #   o = data["integ"][3]  # Gauss-Legendre data for rule order 3, with basis functions up to degree data["maxq"].
        #                         # This under-integrates most of the higher-degree functions.
        #   b = o["y"][:3,:]  # basis functions 0, 1, 2 only - using the order 3 rule, for these the integration of the mass matrix is exact
        #   x = o["x"]  # the integration points (for information only, not actually needed for computing the integral)
        #   w = o["w"]  # the integration weights
        #
        # Recall that b[j,i] = N_j at x_i.
        #
        # For computing the matrices in Galerkin methods, it is much better to use the analytical results (see legtest3.py) than to use Gauss-Legendre numerically,
        # but the b array is very useful for e.g. evaluating Galerkin series in this basis. Let c be a rank-1 np.array with Galerkin coefficients (in this example, of length 3).
        # Then:
        #
        #   u = np.dot( b, c )
        #
        # gives the values of u at the integration points, where u is a function expressed as a linear combination of the basis functions.
        #
        precalc = Precalc(q, xx, cache)  # the cache likely doesn't help here, but it doesn't hurt to use it, either.
        precalc.run()

        # We save a copy of the Gauss-Legendre weights as "w".
        data["integ"][d] = { "x" : precalc.x, "y" : precalc.y, "w" : ww }

#    # NOTE: the data is accessed by the order of the Gauss-Legendre rule, e.g.:
#    print data["integ"][2]

    #########################################################
    # Save results
    #########################################################

    # In the root process: save results to disk
    #
    if mpi_shim.get_rank() == 0:
        # http://stackoverflow.com/questions/10075661/how-to-save-dictionaries-and-arrays-in-the-same-archive-with-numpy-savez
        with open('pydgq_data.bin', 'wb') as outfile:
            pickle.dump( data, outfile, protocol=pickle.HIGHEST_PROTOCOL )
        print( "Wrote pydgq_data.bin" )

# TODO: how to find the data file in an actual installation? (need to save it relative to the package directory)
# TODO: change to .mat format to make the data file more self-documenting?


############################################################################################################
# Command line parser
############################################################################################################

if __name__ == '__main__':

    # MPI support: only the root process parses command-line arguments
    #
    kwargs = None
    comm = mpi_shim.get_comm_world()
    if mpi_shim.get_rank() == 0:
        # We wrap the parsing in try/finally and always broadcast *something*
        # so that the non-root processes can silently exit if the parsing fails.
        #
        # http://stackoverflow.com/questions/25087360/parsing-arguments-using-argparse-and-mpi4py
        #
        try:
            import argparse
            parser = argparse.ArgumentParser(description="""Precalculate hierarchical (Lobatto) basis functions for Galerkin integrators.

For high degrees, the definition of the bubble functions exhibits numerical cancellation, and must thus be computed at increased precision before casting the result to target precision. This is done using arbitrary-precision floating point math, which relies on a pure software implementation and is thus very slow. This script performs the required precomputation and saves the result to disk.

This script supports MPI for parallelization.""", formatter_class=argparse.RawDescriptionHelpFormatter)

            parser.add_argument( '-v', '--version', action='version', version=('%(prog)s ' + __version__) )

            group_behavior = parser.add_argument_group('behavior', 'Precalculator behavior options.')

            group_behavior.add_argument( '-q', '--degree',
                                         dest='q',
                                         default=10,
                                         type=int,
                                         metavar='n',
                                         help='Sets the highest degree (the "q" in "dG(q)") to precompute. Must be >= 1. (Very high degrees, such as 50, are supported by this software, but usually dG(q) gives the best results for q=1 or q=2.) Default %(default)s.' )

            group_behavior.add_argument( '-nx', '--points',
                                         dest='nx',
                                         default=101,
                                         type=int,
                                         metavar='n',
                                         help='For visualization use: sets how many evenly spaced points (on the reference element [-1,1]) each basis function will be evaluated at. Must be >= 1. Default %(default)s.' )

            # http://parezcoydigo.wordpress.com/2012/08/04/from-argparse-to-dictionary-in-python-2-7/
            kwargs = vars( parser.parse_args() )

        finally:
            # Broadcast to all ranks (even if parsing failed, so that the other ranks won't hang)
            #
            if mpi_shim.get_size() > 1:
                kwargs = comm.bcast(kwargs, root=0)

    else:
        # Other ranks: receive broadcast, and exit if empty
        #
        kwargs = comm.bcast(kwargs, root=0)

        if kwargs is None:
            exit(0)

    main(**kwargs)

