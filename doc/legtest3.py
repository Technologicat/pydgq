# -*- coding: utf-8 -*-
#
# Trying out the NumPy API for Legendre polynomials and Gauss--Legendre quadrature,
# with an eye toward the modern hierarchical (Lobatto) basis functions for Galerkin methods
# (B. Szabó, I. Babuška, Finite element analysis, John Wiley & Sons, 1991).
#
# JJ 2016-02-16

from __future__ import division, print_function, absolute_import

import time

import numpy as np

try:
    import mpmath  # Python 3.x
except ImportError:
    import sympy.mpmath as mpmath  # Python 2.7

import matplotlib.pyplot as plt

import pylu.dgesv as dgesv


class RandomPileOfTestStuff:
    # Create a NumPy wrapper for high-precision Legendre polynomials from mpmath.
    #
    # Note that vectorize() is just a convenience wrapper; the implementation is
    # essentially a Python for loop, so it won't increase performance over manually
    # looping over the items.
    #
    # See
    #   http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.vectorize.html
    #
    #
    # Static method!
    #
    # (Actually, technically speaking this is a function object bound to the class scope;
    #  see the rules for creating static members and the discussion on Python's class system,
    #   http://stackoverflow.com/questions/3506150/static-class-members-python
    #   https://docs.python.org/2/tutorial/classes.html#class-definition-syntax
    # )
    #
    _P = np.vectorize( mpmath.legendre )

    def __init__(self, q=15):
        assert( q >= 1 )

        # max polynomial degree
        self.q = q

        self.P   = None  # Legendre polynomials
        self.N   = None  # hierarchical basis functions (FEM, dG)

        # Galerkin matrices
        self.K   = None  # N' * N'
        self.C   = None  # N' * N
        self.M   = None  # N  * N

        self.get_legendre_polynomials()
        self.build_hierarchical_basis()
        self.build_K()
        self.build_C()
        self.build_M()

    def get_legendre_polynomials(self):
        q = self.q

        # We use a function factory to freeze j to the given value (i.e. bind it at define time instead of at call time).
        #
        # See
        #   http://stackoverflow.com/questions/1107210/python-lambda-problems
        #
        # Note that in Python, it is valid to call a static method by self.f() (as well as the traditional ClassName.f()).
        #
        P = []
        for j in range(q):
            P.append(  ( lambda j: lambda x : self._P(j, x) )(j)  )

        self.P = P

    def build_hierarchical_basis(self):
        q = self.q

        N = []
        N.append( lambda x: (1./2.) * (1. - x) )
        N.append( lambda x: (1./2.) * (1. + x) )
        # explicit solution, using mpmath's high-precision routines (slow, but works fine at least to q=300)
        for j in range(2,q+1):
            # HACK: Python 3 compatibility: we must float(j), because some part of the toolchain here wants to convert all arguments to mpf, which does not work for int.
            N.append(  (  lambda j: lambda x : ( self._P(j, x) - self._P(j-2, x) ) / np.sqrt( 2. * (2.*j - 1.) )  )(float(j))  )

        self.N = N

    # stiffness matrix, integrand N' * N'
    def build_K(self):
        q = self.q
        n = q+1

        K = np.eye( n, dtype=np.float64 )
        K[0,0] =  1./2.
        K[0,1] = -1./2.
        K[1,0] = -1./2.
        K[1,1] =  1./2.

        self.K = K

    # damping or gyroscopic matrix, integrand N' * N
    def build_C(self):
        q = self.q
        n = q+1

        C = np.zeros( (n,n), dtype=np.float64 )
        C[0,0] = -1./2.
        C[0,1] =  1./2.
        C[1,0] = -1./2.
        C[1,1] =  1./2.

        if q >= 2:
            t = 1./np.sqrt(6.)
            C[0,2] = -t
            C[1,2] =  t
            C[2,0] =  t
            C[2,1] = -t

        # General formula for C_ji for j,i >= 2.
        for j in range(2,n):
            i = j + 1  # i-1 = j  <=>  i = j+1
            if i >= 2 and i < n:
                C[j,i] =  2. * np.sqrt( 1. / ( ( 2.*j - 1. ) * ( 2.*j + 1. ) ) )
            i = j - 1  # i+1 = j  <=>  i = j-1
            if i >= 2 and i < n:
                C[j,j-1] = -2. * np.sqrt( 1. / ( ( 2.*j - 1. ) * ( 2.*j - 3. ) ) )

        self.C = C

    # mass matrix, integrand N * N
    def build_M(self):
        q = self.q
        n = q+1

        M = np.zeros( (n,n), dtype=np.float64 )
        M[0,0] = 2./3.
        M[0,1] = 1./3.
        M[1,0] = 1./3.
        M[1,1] = 2./3.
        if q >= 2:
            t = 1./np.sqrt(2.)
            M[0,2] = -t
            M[1,2] = -t
            M[2,0] = -t
            M[2,1] = -t
        if q >= 3:
            t = 1. / (3. * np.sqrt(10) )
            M[0,3] = -t
            M[1,3] =  t
            M[3,0] = -t
            M[3,1] =  t

        # General formula for M_ji for j,i >= 2.
        for j in range(2,n):
            M[j,j]   = 1. / (2.*j - 1.) * ( 1. / (2.*j + 1.)  +  1. / (2.*j - 3.) )
            i = j - 2  # i+2 = j  <=>  i = j-2
            if i >= 2 and i < n:
                M[j,i] = 1. / ( np.sqrt(2.*j - 5.) * (2.*j - 3.) * np.sqrt(2.*j - 1.) )
            i = j + 2  # i-2 = j  <=>  i = j+2
            if i >= 2 and i < n:
                M[j,i] = 1. / ( np.sqrt(2.*j - 1.) * (2.*j + 1.) * np.sqrt(2.*j + 3.) )

        self.M = M


def main():
    stuff = RandomPileOfTestStuff(q=100)

    # From the API docs for numpy.polynomial.legendre.leggauss:
    #    Computes the sample points and weights for Gauss-Legendre quadrature.
    #    These sample points and weights will correctly integrate polynomials of degree 2*deg - 1 or less over the interval [-1, 1] with the weight function f(x) = 1.
    #
    # Hence, in Galerkin methods, to exactly handle a mass matrix where neither of the terms is differentiated, using affine mapping to the reference element [-1,1]
    # (implying piecewise constant Jacobian), we need to have
    #
    #   2*deg - 1 = 2*d
    #
    # i.e.
    #
    #   deg = (2*d + 1) / 2
    #
    # where d is the degree of the highest-degree polynomial present in the Galerkin basis, and deg is the order of the Gauss-Legendre rule.
    # Obviously, since only integer deg are available, we must round up (if rounded down, 2*deg-1 is less than the smallest needed, 2*d).
    # Thus the actual practical result is
    #
    #   deg = ceil( (2*d + 1) / 2 ) = d + ceil( 1/2 ) = d+1
    #
    # (Observe that a rule of this order can do one degree more than the matrix M (integrand N*N) needs. With this, we could exactly integrate x*N*N, if needed.)
    #
    #
    # For the purposes of solving the first-order problem  u' = f(u, t)  by dG, the matrix is not our M, but instead our C (N'*N = degree d-1 plus degree d), so
    #
    #   2*deg - 1 = 2*d - 1
    #
    # i.e.
    #
    #   deg = d
    #
    # Thus, we can solve this problem with a Gauss-Legendre rule of one order lower than in the case where the matrix M is needed.
    #
    #
#    deg = d+1
#    q,w = np.polynomial.legendre.leggauss( deg )
#    print( deg,(2*deg-1),q,w )

    # matrix, name, figure number to plot, bugcheck (max(abs(bugcheck)) should evaluate to 0 for mat[2:,2:])
    data = ( (stuff.K, "K", 2, lambda v: v - np.transpose(v)),
             (stuff.C, "C", 3, lambda v: v + np.transpose(v)),
             (stuff.M, "M", 4, lambda v: v - np.transpose(v)) )

    for mat,name,figno,bugcheck in data:
        print( mat )
        plt.figure(figno)

        plt.subplot(1,2, 1)

        plt.spy(mat)  # spy() doesn't work for a full matrix without any zero entries! (try stuff.M with q=2)

#        plt.imshow(mat, interpolation="nearest", cmap="Oranges")
#        plt.colorbar()

        plt.plot( [0,stuff.q], [0,stuff.q], 'r--' )  # mark diagonal
        plt.title(r"$\mathbf{%s}$" % name)

        if stuff.q >= 2:
            v = mat[2:,2:]
            b = np.max( np.abs( bugcheck(v) ) )
            assert b == 0.0, "bugcheck fail for matrix %s; should be 0, got %g" % (name, b)


        # LU decomposition (sort of)
        #
        plt.subplot(1,2, 2)
        A = mat.copy()
        A[0,0] += 1.0  # K and C are rank-deficient by one; simulate effect of boundary conditions (or dG jump term)
        LU,p = dgesv.lup_packed(A)
        plt.spy(LU)
        plt.plot( [0,stuff.q], [0,stuff.q], 'r--' )
        plt.title(r"$\mathbf{LU}$ (packed format)")


##    L,U,p = dgesv.lup(stuff.M)
##    print( np.transpose(np.nonzero(L)) )
##    print( np.transpose(np.nonzero(U)) )
##    print( p )
##    plt.figure(3)
##    plt.subplot(1,2, 1)
##    plt.spy(L)
##    plt.plot( [0,stuff.q-1], [0,stuff.q-1], 'r--' )
###    plt.imshow(L, interpolation="nearest", cmap="Oranges")
###    plt.colorbar(orientation="horizontal")
##    plt.title(r"$\mathbf{L}$")
##    plt.subplot(1,2, 2)
##    plt.spy(U)
##    plt.plot( [0,stuff.q-1], [0,stuff.q-1], 'r--' )
###    plt.imshow(U, interpolation="nearest", cmap="Oranges")
###    plt.colorbar(orientation="horizontal")
##    plt.title(r"$\mathbf{U}$")


##    LU,p = dgesv.lup_packed(stuff.M)
##    plt.figure(4)
##    plt.spy(LU)
##    plt.plot( [0,stuff.q-1], [0,stuff.q-1], 'r--' )
##    plt.title(r"$\mathbf{LU}$ (packed format)")

##    mincols,maxcols = dgesv.find_bands(LU, 1e-15)
##    print( mincols, maxcols )


##    # old Python-based mincols, maxcols finding code
##
##    # Find the smallest column index with nonzero data on each row in L.
##    #
##    # We can use this to "sparsify" the backsolve even though the data structure is dense.
##    #
##    # This assumes that each row has at least one nonzero entry (which is always the case for an invertible matrix).
##    #
##    Lnz = np.nonzero(L)
##    mincols = []
##    rowprev = -1
##    n = len(Lnz[0])
##    i = 0
##    while i < n:
##        if Lnz[0][i] != rowprev:
##            mincols.append(Lnz[1][i])
##            rowprev = Lnz[0][i]
##        i += 1
##    mincols = np.array( mincols, dtype=np.intc, order="C" )
##    print( L )
##    print( mincols )

##    # Find the largest column index with nonzero data on each row in U.
##    #
##    # We can use this to "sparsify" the backsolve even though the data structure is dense.
##    #
##    # This assumes that each row has at least one nonzero entry (which is always the case for an invertible matrix).
##    #
##    Unz = np.nonzero(U)
##    maxcols = []
##    rowprev = -1
##    n = len(Unz[0])
##    i = n - 1
##    while i >= 0:
##        if Unz[0][i] != rowprev:
##            maxcols.append(Unz[1][i])
##            rowprev = Unz[0][i]
##        i -= 1
##    maxcols.reverse()
##    maxcols = np.array( maxcols, dtype=np.intc, order="C" )
##    print( U )
##    print( maxcols )


    # Visualize
    #
    xx = np.linspace(-1., 1., 101)
    plt.figure(1)
    plt.clf()
    for func in stuff.N:
        plt.plot( xx, func(xx) )

    plt.axis('tight')
    a = plt.axis()
    plt.axis( [ a[0], a[1], a[2]*1.05, a[3]*1.05 ] )

    plt.grid(b=True, which='both')
    plt.title('Hierarchical basis functions')


# naive solve (repeat the LU decomposition process each time)
#
def method1(reps, A, b, x):
    for j in range(reps):
#        dgesv.solve( A, b[j,:], x )
        dgesv.solve( A, b, x )

# decompose once, then solve
#
def method2(reps, A, b, x):
    LU,p = dgesv.lup_packed(A)
    for j in range(reps):
#        dgesv.solve_decomposed( LU, p, b[j,:], x )
        dgesv.solve_decomposed( LU, p, b, x )

# decompose once, then solve, utilize banded structure
#
def method3(reps, A, b, x):
    LU,p = dgesv.lup_packed(A)
    mincols,maxcols = dgesv.find_bands(LU, tol=1e-15)
    for j in range(reps):
#        dgesv.solve_decomposed_banded( LU, p, mincols, maxcols, b[j,:], x )
        dgesv.solve_decomposed_banded( LU, p, mincols, maxcols, b, x )


class MyTimer:
    t0 = None
    l  = None
    n  = None  # number of runs

    def __init__(self, label="", n=None):
        self.label = label
        self.n     = n

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, type, value, traceback):
        dt  = time.time() - self.t0
        l   = ("%s: " % self.label) if len(self.label) else "time taken: "
        avg = (", avg. %gs per run" % (dt/self.n)) if self.n is not None else ""
        print( "%s%gs%s" % (l, dt, avg) )


if __name__ == '__main__':
    main()
    plt.show()

#    # Running the benchmark loop at the Python end makes the banded version look slower (for our matrix "C", the C code is actually ~3x faster than the generic non-banded version),
#    # because a large majority of the execution time is taken up by data conversion from Python to C and back (and Python asserts, if enabled).
#    #
#    # For small matrices (q = 15 or so), to get reliable results on the C code only (which is a realistic use case if used from inside a Cython-accelerated solver, which is the whole point of dgesv.pyx),
#    # the benchmark looping must be done inside dgesv.pyx.
#    #
#    # Naive benchmarking results start becoming reliable around q >= 100.
#    #
#    reps   = 10000
##    qstart = 3
##    qend   = 16  # actually one-past-end
#    qstart = 300
#    qend   = 301

#    for q in range(qstart, qend):
#        stuff = RandomPileOfTestStuff(q)
#        A = stuff.M

#        n = np.shape(A)[0]
##        A[0,0] += 1.0  # simulate effect of dG jump term (this is the only basis function which is nonzero at x = -1)
##        b = np.random.uniform(0.0, 1.0, size=(reps,n))  # this makes slicing part of the performance measurement - not good
#        b = np.random.uniform(0.0, 1.0, size=(n,))
#        x = np.empty( [n], dtype=np.float64, order="C" )

#        print( "Timings for %d runs" % reps )
##        with MyTimer("%dx%d naive" % (n,n), reps) as mt:
##            print( mt.label )
##            method1(reps, A, b, x)
#        with MyTimer("%dx%d decompose-once" % (n,n), reps) as mt:
#            method2(reps, A, b, x)
#        with MyTimer("%dx%d decompose-once-banded" % (n,n), reps) as mt:
#            method3(reps, A, b, x)
#        print( "Residual from last run %g" % np.max(np.abs( np.dot(A,x) - b )) )

