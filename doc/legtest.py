# -*- coding: utf-8 -*-
#
# Trying out the NumPy API for Legendre polynomials and Gauss--Legendre quadrature.
#
# JJ 2016-02-16

from __future__ import division

import numpy as np
import pylab as pl


def get_legendre_polynomials(max_degree):
    polynomials = []

    # For each degree d, get the polynomial coefficients of a Legendre series
    # that has only the dth degree term. Construct the corresponding Polynomial object,
    #
    # The coefficients are listed from the lowest order to highest.
    #
    for d in range(max_degree):
        # d zeroes followed by a one
        #
        series_coeffs = [ 0. ] * d
        series_coeffs.append( 1. )

        # coefficients for a standard power series 1, x, x**2, ...
        #
        c = np.polynomial.legendre.leg2poly( series_coeffs )

        # create the Polynomial object, remapping the input range to [0,1] for convenience
        #
        polynomials.append( np.polynomial.Polynomial( c, domain=[0., 1.], window=[-1., 1.] ) )

    return polynomials


def main():
    # Set the maximum degree.
    #
    # For our purposes, for accurate results (all polynomials staying within [-1,1] for the whole interval)
    # it seems d = 30 is about the upper limit of what this implementation can do.
    #
    d = 30

    # From the API docs for numpy.polynomial.legendre.leggauss:
    #    Computes the sample points and weights for Gauss-Legendre quadrature.
    #    These sample points and weights will correctly integrate polynomials of degree 2*deg - 1 or less over the interval [-1, 1] with the weight function f(x) = 1.
    #
    # Hence, in Galerkin methods, to exactly handle a mass matrix where neither of the terms is differentiated, using affine mapping to the reference element [0,1]
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

    P = get_legendre_polynomials(d)
    xx = np.linspace(0., 1., 501)

    pl.figure(1)
    pl.clf()
    for p in P:
        pl.plot( xx, p(xx) )

    pl.axis('tight')
    a = pl.axis()
    pl.axis( [ a[0], a[1], a[2]*1.05, a[3]*1.05 ] )

    pl.grid(b=True, which='both')
    pl.title('Legendre polynomials')


    # Try some operations

    # As long as we keep the Polynomial objects, we can multiply them the intuitive way, producing a new Polynomial:
    #
    print P[2]*P[3]  # => poly([ 0.    0.75  0.   -3.5   0.    3.75])

    # We can also differentiate them, which is useful for constructing the mass matrix:
    #
    print P[2].deriv(1)*P[3]  # => poly([  0.   0.  -9.   0.  15.])

    # Also integration is supported.
    #
    # p.integ() returns the definite integral, as a Polynomial object, from lbnd to an unspecified upper limit x, adding the integration constant k.
    # The value of x is chosen when calling the resulting object.
    #
    # Legendre polynomials are L2-orthogonal on [-1,1]; the scaled ones are orthogonal on [0,1]:
    print ( (P[2]*P[2]).integ(lbnd=0, k=0) )(1.0)  # 1/(2 n + 1);  here n = 2, so this = 1/5 = 0.2
    print ( (P[2]*P[3]).integ(lbnd=0, k=0) )(1.0)  # zero

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
    print ( (P[3].deriv(1)*P[3]).integ(lbnd=0, k=0) )(1.0)  # zero, n + m even
    print ( (P[3].deriv(1)*P[1]).integ(lbnd=0, k=0) )(1.0)  # zero, n + m even
    print ( (P[2].deriv(1)*P[3]).integ(lbnd=0, k=0) )(1.0)  # zero, n < m
    print ( (P[3].deriv(1)*P[2]).integ(lbnd=0, k=0) )(1.0)  # nonzero (derivative of p3 contains p2, p0)


if __name__ == '__main__':
    main()
    pl.show()

