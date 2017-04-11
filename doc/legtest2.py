# -*- coding: utf-8 -*-
#
# Trying out the NumPy API for Legendre polynomials and Gauss--Legendre quadrature,
# with an eye toward the modern hierarchical (Lobatto) basis functions for Galerkin methods
# (B. Szabó, I. Babuška, Finite element analysis, John Wiley & Sons, 1991).
#
# JJ 2016-02-16

from __future__ import division

import time

import numpy as np
import scipy.integrate
import pylab as pl

import dgesv


class RandomPileOfTestStuff:
    def __init__(self, q=15, tol=1e-8):
        assert( q >= 2 )  # we don't have special case handling for q=1 in build_hierarchical_basis()

        self.q   = q     # max polynomial degree for Legendre polynomials; number of basis functions for hierarchical basis (as in "dG(q)")
        self.tol = tol   # tolerance for nonzero check

        self.P   = None  # Legendre polynomials
        self.N   = None  # hierarchical basis functions (FEM, dG)

        self.C   = None  # dG mass matrix for the first-order problem u' = f(u, t)

        self.get_legendre_polynomials()
        self.build_hierarchical_basis()
        self.dgmass()

    def get_legendre_polynomials(self):
        q = self.q

        P = []

        # For each degree d, get the polynomial coefficients of a Legendre series
        # that has only the dth degree term. Construct the corresponding Polynomial object.
        #
        # The coefficients are listed from the lowest order to highest.
        #
        for d in range(q):
            # d zeroes followed by a one
            #
            series_coeffs = [ 0. ] * d
            series_coeffs.append( 1. )

            # coefficients for a standard power series 1, x, x**2, ...
            #
            c = np.polynomial.legendre.leg2poly( series_coeffs )

            P.append( np.polynomial.Polynomial( c ) )

        self.P = P

    def build_hierarchical_basis(self):
        assert( self.P is not None )

        q = self.q
        P = self.P

        N = []
        N.append( np.polynomial.Polynomial( [0.5, -0.5] ) )  # N_1, will become N[0] in the code, using Polynomial instead of explicit lambda gets us support for .deriv()
        N.append( np.polynomial.Polynomial( [0.5,  0.5] ) )  # N_2
        for j in range(2,q):
            #N.append( np.sqrt( (2.*j - 1.)/2.) * P[j-1].integ(lbnd=-1, k=0) )  # surely this approach makes no numerical sense

            # Explicit solution, using NumPy to evaluate the sum of Legendre polynomials.
            #
            # Much better (and still fast), but not nearly as accurate as evaluating using higher precision internally. See legtest3.py.
            #
            series_coeffs = [ 0. ] * (j-2)
            series_coeffs.extend( [-1., 0., 1.] )  # -P_{j-2} + P_{j}
            c = np.polynomial.legendre.leg2poly( series_coeffs )
            Nj = np.polynomial.Polynomial(c)  /  np.sqrt( 2. * (2.*j - 1.) )
            N.append( Nj )

        self.N = N

    # This numerical approach for generating the matrix is prone to roundoff and obsolete (not to mention stupid
    # since we know that most of the matrix entries should be zero); see the analytical solution in legtest3.py.
    #
    def dgmass(self):
        assert( self.N is not None )

        q = self.q
        N = self.N

        C = np.empty( (q,q), dtype=np.float64 )
        for i in range(q):
            for j in range(q):
                C[i,j] = scipy.integrate.quad( N[j].deriv(1)*N[i], -1., 1. )[0]
        C[ np.abs(C) < self.tol ] = 0.0
        C[0,0] += 1.0  # simulate the effect of the jump term (N_1 is the only function that is nonzero at xi=-1)

        self.C = C


def main():
    # Up to q=24, the full script works despite warnings from quad() in dgmass().
    #
    # For evaluating the hierarchical basis functions only (no dgmass()):
    #
    #   q = 30, still sort of works, small deviations (1e-7) can be seen in the endpoint values of the few highest-order Nj
    #   q = 40, almost works, high-order Nj start getting wobbly
    #   q = 50, completely broken, out of precision
    #
    # By comparison, legtest3.py, which uses SymPy's mpmath (arbitrary precision floating point), works at least up to q=300, but is very slow.
    #
    stuff = RandomPileOfTestStuff(q=24, tol=1e-3)

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
#    deg = int(np.ceil( (2*d + 1)/2. ))
#    q,w = np.polynomial.legendre.leggauss( deg )
#    print deg,(2*deg-1),q,w

    print stuff.C
    print np.linalg.matrix_rank(stuff.C)  # should be full rank
    pl.figure(2)
    pl.spy(stuff.C)
    pl.plot( [0,stuff.q-1], [0,stuff.q-1], 'r--' )
#    pl.imshow(M, interpolation="nearest", cmap="Oranges")
#    pl.colorbar()
    pl.title(r"$\mathbf{M}$")


##    L,U,p = dgesv.lup(stuff.C)
##    print np.transpose(np.nonzero(L))
##    print np.transpose(np.nonzero(U))
##    print p
##    pl.figure(3)
##    pl.subplot(1,2, 1)
##    pl.spy(L)
##    pl.plot( [0,stuff.q-1], [0,stuff.q-1], 'r--' )
###    pl.imshow(L, interpolation="nearest", cmap="Oranges")
###    pl.colorbar(orientation="horizontal")
##    pl.title(r"$\mathbf{L}$")
##    pl.subplot(1,2, 2)
##    pl.spy(U)
##    pl.plot( [0,stuff.q-1], [0,stuff.q-1], 'r--' )
###    pl.imshow(U, interpolation="nearest", cmap="Oranges")
###    pl.colorbar(orientation="horizontal")
##    pl.title(r"$\mathbf{U}$")


    LU,p = dgesv.lup_packed(stuff.C)
    pl.figure(4)
    pl.spy(LU)
    pl.plot( [0,stuff.q-1], [0,stuff.q-1], 'r--' )
    pl.title(r"$\mathbf{LU}$ (packed format)")

    mincols,maxcols = dgesv.find_bands(LU, 1e-15)
    print mincols, maxcols


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
##    print L
##    print mincols

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
##    print U
##    print maxcols


    # Visualize
    #
    xx = np.linspace(-1., 1., 100001)  # the good thing about the fast approach... smooth curves!
    pl.figure(1)
    pl.clf()
    for func in stuff.N:
        pl.plot( xx, func(xx) )

    pl.axis('tight')
    a = pl.axis()
    pl.axis( [ a[0], a[1], a[2]*1.05, a[3]*1.05 ] )

    pl.grid(b=True, which='both')
    pl.title('Hierarchical basis functions')


    # Try some operations on the original Legendre polynomials
    #
    # As long as we keep the Polynomial objects, we can multiply them the intuitive way, producing a new Polynomial:
    #
    print stuff.P[2]*stuff.P[3]  # => poly([ 0.    0.75  0.   -3.5   0.    3.75])

    # We can also differentiate them, which is useful for constructing the mass matrix:
    #
    print stuff.P[2].deriv(1)*stuff.P[3]  # => poly([  0.   0.  -9.   0.  15.])

    # Also integration is supported.
    #
    # p.integ() returns the definite integral, as a Polynomial object, from lbnd to an unspecified upper limit x, adding the integration constant k.
    # The value of x is chosen when calling the resulting object.
    #
    # Legendre polynomials are L2-orthogonal on [-1,1]:
    print ( (stuff.P[2]*stuff.P[2]).integ(lbnd=-1, k=0) )(1.0)  # 2/(2 n + 1);  here n = 2, so this = 2/5 = 0.4
    print ( (stuff.P[2]*stuff.P[3]).integ(lbnd=-1, k=0) )(1.0)  # zero

    # The integral of  dPn/dx * Pm  over the interval is zero if:
    #
    #  - n + m is even
    #  - n < m  (and by the previous condition, also  n <= m)
    #
    # These observations are based on the L2-orthogonality and the relation
    #
    #   (2 n + 1) P_n = (d/dx)( P_{n+1} - P_{n-1} )         (*)
    #
    # which can be used to get rid of the derivative. The relation (*) follows from Bonnet’s recursion formula,
    #
    #   (n + 1) P_{n+1} = (2 n + 1) P_n - n P_{n-1}
    #
    # By recursive application, (*) leads to the representation
    #
    #   (d/dx) P_{n+1} = (2 n + 1) P_n + ( 2 (n - 2) + 1 ) P_{n-2} + ( 2 (n - 4) + 1 ) P_{n-4} + ...
    #
    # which is guaranteed to bottom out at P_1 and P_0 (by using  P_0 = 1  and  P_1 = x  in (*)).
    #
    # See
    #  https://en.wikipedia.org/wiki/Legendre_polynomials#Additional_properties_of_Legendre_polynomials
    #
    print ( (stuff.P[3].deriv(1)*stuff.P[3]).integ(lbnd=-1, k=0) )(1.0)  # zero, n + m even
    print ( (stuff.P[3].deriv(1)*stuff.P[1]).integ(lbnd=-1, k=0) )(1.0)  # zero, n + m even
    print ( (stuff.P[2].deriv(1)*stuff.P[3]).integ(lbnd=-1, k=0) )(1.0)  # zero, n < m
    print ( (stuff.P[3].deriv(1)*stuff.P[2]).integ(lbnd=-1, k=0) )(1.0)  # nonzero (derivative of p3 contains p2, p0)


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
    mincols,maxcols = dgesv.find_bands(LU, 1e-15)
    for j in range(reps):
#        dgesv.solve_decomposed_banded( LU, p, mincols, maxcols, b[j,:], x )
        dgesv.solve_decomposed_banded( LU, p, mincols, maxcols, b, x )


class MyTimer:
    t0 = None
    l  = None

    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, type, value, traceback):
        dt = time.time() - self.t0
        l  = ("%s: " % self.label) if len(self.label) else "time taken: "
        print "%s%gs" % (l, dt)


if __name__ == '__main__':
    main()
    pl.show()

#    # Running the benchmark loop at the Python end makes the banded version look slower (for our matrix M, the C code is actually ~3x faster than the generic non-banded version),
#    # because a large majority of the execution time is taken up by data conversion from Python to C and back (and Python asserts, if enabled).
#    #
#    # To get reliable results on the C code only (which is a realistic use case if used from inside a Cython-accelerated solver, which is the whole point of dgesv.pyx),
#    # the looping must be done inside dgesv.pyx.
#    #
#    reps = 100000
#    for q in range(3, 16):
#        stuff = RandomPileOfTestStuff(q)
#        n = np.shape(stuff.C)[0]
##        b = np.random.uniform(0.0, 1.0, size=(reps,n))  # this makes slicing part of the performance measurement - not good
#        b = np.random.uniform(0.0, 1.0, size=(n,))
#        x = np.empty( [n], dtype=np.float64, order="C" )

#        print "Timings for %d runs" % reps
#        with MyTimer("%dx%d naive" % (n,n)) as mt:
#            method1(reps, stuff.C, b, x)
#        with MyTimer("%dx%d decompose-once" % (n,n)) as mt:
#            method2(reps, stuff.C, b, x)
#        with MyTimer("%dx%d decompose-once-banded" % (n,n)) as mt:
#            method3(reps, stuff.C, b, x)

