# -*- coding: utf-8 -*-
"""Support routines for Galerkin integrators."""

from . cimport types

# Helper routines.
#
cdef void assemble( DTYPE_t* u, DTYPE_t* psi, DTYPE_t* uass, DTYPE_t* ucorr, int n_space_dofs, int n_time_dofs, int n_points ) nogil
cdef void final_value( DTYPE_t* u, DTYPE_t* uass, int n_space_dofs, int n_time_dofs ) nogil
cdef DTYPE_t do_quadrature( DTYPE_t* funcvals, DTYPE_t* qw, int n, DTYPE_t dt ) nogil

# Parameters for integrator.
#
#    # array shapes and types:
#
#    # instance arrays (see galerkin.Helper.allocate_storage())
#    cdef types.DTYPE_t[:,:,::1] g     = np.empty( [n_space_dofs,n_time_dofs,n_quad], dtype=DTYPE, order="C" )  # effective load vector, for each space DOF, for each time DOF, at each integration point
#    cdef types.DTYPE_t[:,::1]   b     = np.empty( [n_space_dofs,n_time_dofs],        dtype=DTYPE, order="C" )  # right-hand sides (integral, over the timestep, of g*psi)
#    cdef types.DTYPE_t[:,::1]   u     = np.empty( [n_space_dofs,n_time_dofs],        dtype=DTYPE, order="C" )  # Galerkin coefficients (unknowns)
#    cdef types.DTYPE_t[:,::1]   uprev = np.empty( [n_space_dofs,n_time_dofs],        dtype=DTYPE, order="C" )  # Galerkin coefficients from previous iteration
#    cdef types.DTYPE_t[:,::1]   uass  = np.empty( [n_quad,n_space_dofs],             dtype=DTYPE, order="C" )  # u, assembled for integration (this ordering needed for speed!)
#    cdef types.DTYPE_t[::1]     ucorr = np.empty( [n_quad],                          dtype=DTYPE, order="C" )  # correction for compensated summation in galerkin.assemble() (for integration)
#    cdef types.DTYPE_t[:,::1]   uvis  = np.empty( [nx,n_space_dofs],                 dtype=DTYPE, order="C" )  # u, assembled for visualization
#    cdef types.DTYPE_t[::1]     ucvis = np.empty( [nx],                              dtype=DTYPE, order="C" )  # correction for compensated summation in galerkin.assemble() (for visualization)
#
#    # global arrays, same for each solver instance (see galerkin.Helper.load_data(), galerkin.Helper.prep_solver())
#    cdef types.DTYPE_t[:,::1] LU      = galerkin.helper_instance.LU       # LU decomposed mass matrix (packed format), for one space DOF, shape (n_time_dofs, n_time_dofs)
#    cdef int[::1]       p       = galerkin.helper_instance.p        # row permutation information, length n_time_dofs
#    cdef int[::1]       mincols = galerkin.helper_instance.mincols  # band information for L, length n_time_dofs
#    cdef int[::1]       maxcols = galerkin.helper_instance.maxcols  # band information for U, length n_time_dofs
#    cdef types.DTYPE_t[::1]   qw      = galerkin.helper_instance.integ_w  # quadrature weights (Gauss-Legendre)
#    cdef types.DTYPE_t[:,::1] psi     = galerkin.helper_instance.integ_y  # basis function values at the quadrature points, psi[j,i] is N[j]( x[i] )
#    cdef types.DTYPE_t[:,::1] psivis  = galerkin.helper_instance.vis_y    # basis function values at the visualization points, psivis[j,i] is N[j]( x[i] )
#
ctypedef struct params:
    cdef kernels.kernelfuncptr f  # RHS compute kernel
    cdef types.DTYPE_t* w         # state vector, value at end of previous timestep (dG() and cG() update this)
    cdef void* data               # user data for f

    cdef types.DTYPE_t t          # time value at start of current timestep
    cdef types.DTYPE_t dt         # size of current timestep

    cdef int n_space_dofs         # size of the 1st order problem
    cdef int n_time_dofs          # how many time DOFs to use for *each* space DOF (this is determined by the choice of q in dG(q))

    cdef int maxit                # maximum number of Banach/Picard iterations in implicit solve

    # instance arrays (specific to this problem instance)
    cdef types.DTYPE_t* g
    cdef types.DTYPE_t* b
    cdef types.DTYPE_t* u
    cdef types.DTYPE_t* uprev
    cdef types.DTYPE_t* uass
    cdef types.DTYPE_t* ucorr
    cdef types.DTYPE_t* uvis

    # global arrays (shared across all problems when using the same settings)

    # LU decomposed mass matrix for dG(q), for one space DOF
    cdef types.DTYPE_t* LU
    cdef int* p
    cdef int* mincols
    cdef int* maxcols

    cdef int n_quad            # number of integration points in the Gauss-Legendre rule
    cdef types.DTYPE_t* qw     # quadrature weights
    cdef types.DTYPE_t* tquad  # integration points, **scaled to [0,1]**

    cdef types.DTYPE_t* psi
    cdef types.DTYPE_t* psivis

    cdef types.DTYPE_t* tvis  # visualization points (accounting for the interp parameter), **scaled to [0,1]**


cdef int dG( params* gp ) nogil
cdef int cG( params* gp ) nogil

