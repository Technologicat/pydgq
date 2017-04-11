# -*- coding: utf-8 -*-
"""Support routines for Galerkin integrators."""

cimport pydgq_types
cimport kernels

# Helper routines.
#
cdef void assemble( pydgq_types.DTYPE_t* u, pydgq_types.DTYPE_t* psi, pydgq_types.DTYPE_t* uass, pydgq_types.DTYPE_t* ucorr, int n_space_dofs, int n_time_dofs, int n_points ) nogil
cdef void final_value( pydgq_types.DTYPE_t* u, pydgq_types.DTYPE_t* uass, int n_space_dofs, int n_time_dofs ) nogil
cdef pydgq_types.DTYPE_t do_quadrature( pydgq_types.DTYPE_t* funcvals, pydgq_types.DTYPE_t* qw, int n, pydgq_types.DTYPE_t dt ) nogil

# Parameters for integrator.
#
#    # array shapes and types:
#
#    # instance arrays (see galerkin.Helper.allocate_storage())
#    cdef pydgq_types.DTYPE_t[:,:,::1] g     = np.empty( [n_space_dofs,n_time_dofs,n_quad], dtype=DTYPE, order="C" )  # effective load vector, for each space DOF, for each time DOF, at each integration point
#    cdef pydgq_types.DTYPE_t[:,::1]   b     = np.empty( [n_space_dofs,n_time_dofs],        dtype=DTYPE, order="C" )  # right-hand sides (integral, over the timestep, of g*psi)
#    cdef pydgq_types.DTYPE_t[:,::1]   u     = np.empty( [n_space_dofs,n_time_dofs],        dtype=DTYPE, order="C" )  # Galerkin coefficients (unknowns)
#    cdef pydgq_types.DTYPE_t[:,::1]   uprev = np.empty( [n_space_dofs,n_time_dofs],        dtype=DTYPE, order="C" )  # Galerkin coefficients from previous iteration
#    cdef pydgq_types.DTYPE_t[:,::1]   uass  = np.empty( [n_quad,n_space_dofs],             dtype=DTYPE, order="C" )  # u, assembled for integration (this ordering needed for speed!)
#    cdef pydgq_types.DTYPE_t[::1]     ucorr = np.empty( [n_quad],                          dtype=DTYPE, order="C" )  # correction for compensated summation in galerkin.assemble() (for integration)
#    cdef pydgq_types.DTYPE_t[:,::1]   uvis  = np.empty( [nx,n_space_dofs],                 dtype=DTYPE, order="C" )  # u, assembled for visualization
#    cdef pydgq_types.DTYPE_t[::1]     ucvis = np.empty( [nx],                              dtype=DTYPE, order="C" )  # correction for compensated summation in galerkin.assemble() (for visualization)
#
#    # global arrays, same for each solver instance (see galerkin.Helper.load_data(), galerkin.Helper.prep_solver())
#    cdef pydgq_types.DTYPE_t[:,::1] LU      = galerkin.helper_instance.LU       # LU decomposed mass matrix (packed format), for one space DOF, shape (n_time_dofs, n_time_dofs)
#    cdef int[::1]       p       = galerkin.helper_instance.p        # row permutation information, length n_time_dofs
#    cdef int[::1]       mincols = galerkin.helper_instance.mincols  # band information for L, length n_time_dofs
#    cdef int[::1]       maxcols = galerkin.helper_instance.maxcols  # band information for U, length n_time_dofs
#    cdef pydgq_types.DTYPE_t[::1]   qw      = galerkin.helper_instance.integ_w  # quadrature weights (Gauss-Legendre)
#    cdef pydgq_types.DTYPE_t[:,::1] psi     = galerkin.helper_instance.integ_y  # basis function values at the quadrature points, psi[j,i] is N[j]( x[i] )
#    cdef pydgq_types.DTYPE_t[:,::1] psivis  = galerkin.helper_instance.vis_y    # basis function values at the visualization points, psivis[j,i] is N[j]( x[i] )
#
cdef struct params:
    kernels.kernelfuncptr f  # RHS compute kernel
    pydgq_types.DTYPE_t* w         # state vector, value at end of previous timestep (dG() and cG() update this)
    void* data               # user data for f

    pydgq_types.DTYPE_t t          # time value at start of current timestep
    pydgq_types.DTYPE_t dt         # size of current timestep

    int n_space_dofs         # size of the 1st order problem
    int n_time_dofs          # how many time DOFs to use for *each* space DOF (this is determined by the choice of q in dG(q))

    int maxit                # maximum number of Banach/Picard iterations in implicit solve

    # instance arrays (specific to this problem instance)
    pydgq_types.DTYPE_t* g
    pydgq_types.DTYPE_t* b
    pydgq_types.DTYPE_t* u
    pydgq_types.DTYPE_t* uprev
    pydgq_types.DTYPE_t* uass
    pydgq_types.DTYPE_t* ucorr
    pydgq_types.DTYPE_t* uvis
    pydgq_types.DTYPE_t* wrk   # work space for n_space_dofs items

    # global arrays (shared across all problems when using the same settings)

    # LU decomposed mass matrix for dG(q), for one space DOF
    pydgq_types.DTYPE_t* LU
    int* p
    int* mincols
    int* maxcols

    int n_quad            # number of integration points in the Gauss-Legendre rule
    pydgq_types.DTYPE_t* qw     # quadrature weights
    pydgq_types.DTYPE_t* tquad  # integration points, **scaled to [0,1]**

    pydgq_types.DTYPE_t* psi
    pydgq_types.DTYPE_t* psivis

    pydgq_types.DTYPE_t* tvis  # visualization points (accounting for the interp parameter), **scaled to [0,1]**

cdef int dG( params* gp ) nogil
cdef int cG( params* gp ) nogil

