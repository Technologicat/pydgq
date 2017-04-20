# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pydgq.solver.types cimport DTYPE_t, RTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase
from pydgq.solver.integrator_interface cimport ImplicitIntegrator

# Base class for Galerkin integrators.
#
# Handles data array allocation and provides helper methods common to all Galerkin integrators.
#
#    # array shapes and types:
#
#    # instance arrays (see galerkin.DataManager.allocate_storage())
#    cdef DTYPE_t[:,:,::1] g     = np.empty( [n_space_dofs,n_time_dofs,n_quad], dtype=DTYPE, order="C" )  # effective load vector, for each space DOF, for each time DOF, at each integration point
#    cdef DTYPE_t[:,::1]   b     = np.empty( [n_space_dofs,n_time_dofs],        dtype=DTYPE, order="C" )  # right-hand sides (integral, over the timestep, of g*psi)
#    cdef DTYPE_t[:,::1]   u     = np.empty( [n_space_dofs,n_time_dofs],        dtype=DTYPE, order="C" )  # Galerkin coefficients (unknowns)
#    cdef DTYPE_t[:,::1]   uprev = np.empty( [n_space_dofs,n_time_dofs],        dtype=DTYPE, order="C" )  # Galerkin coefficients from previous iteration
#    cdef DTYPE_t[:,::1]   uass  = np.empty( [n_quad,n_space_dofs],             dtype=DTYPE, order="C" )  # u, assembled for integration (this ordering needed for speed!)
#    cdef DTYPE_t[::1]     ucorr = np.empty( [n_quad],                          dtype=DTYPE, order="C" )  # correction for compensated summation in assemble() (for integration)
#    cdef DTYPE_t[:,::1]   uvis  = np.empty( [nt_vis,n_space_dofs],             dtype=DTYPE, order="C" )  # u, assembled for visualization
#    cdef DTYPE_t[::1]     ucvis = np.empty( [nt_vis],                          dtype=DTYPE, order="C" )  # correction for compensated summation in assemble() (for visualization)
#
#    # global arrays, same for each solver instance (see galerkin.DataManager.load_data(), galerkin.DataManager.prep_solver())
#    cdef RTYPE_t[:,::1] LU      = galerkin.datamanager.LU       # LU decomposed mass matrix (packed format), for one space DOF, shape (n_time_dofs, n_time_dofs)
#    cdef int[::1]       p       = galerkin.datamanager.p        # row permutation information, length n_time_dofs
#    cdef int[::1]       mincols = galerkin.datamanager.mincols  # band information for L, length n_time_dofs
#    cdef int[::1]       maxcols = galerkin.datamanager.maxcols  # band information for U, length n_time_dofs
#    cdef RTYPE_t[::1]   qw      = galerkin.datamanager.integ_w  # quadrature weights (Gauss-Legendre)
#    cdef RTYPE_t[:,::1] psi     = galerkin.datamanager.integ_y  # basis function values at the quadrature points, psi[j,i] is N[j]( x[i] )
#    cdef RTYPE_t[:,::1] psivis  = galerkin.datamanager.vis_y    # basis function values at the visualization points, psivis[j,i] is N[j]( x[i] )
#
cdef class GalerkinIntegrator(ImplicitIntegrator):
    cdef int n_time_dofs    # how many time DOFs to use for *each* space DOF (this is determined by the choice of q in dG(q))

    # instance arrays (specific to this problem instance)
    cdef DTYPE_t* g
    cdef DTYPE_t* b
    cdef DTYPE_t* u
    cdef DTYPE_t* uprev
    cdef DTYPE_t* uass
    cdef DTYPE_t* ucorr
    cdef DTYPE_t* uvis
    cdef DTYPE_t* ucvis
    # also wrk, but that is already declared in IntegratorBase.

    # global arrays (shared across all problems when using the same settings)

    # dG(q) mass matrix for one space DOF, LU decomposed
    cdef RTYPE_t* LU
    cdef int* p
    cdef int* mincols
    cdef int* maxcols

    cdef int n_quad         # number of integration points in the Gauss-Legendre rule
    cdef RTYPE_t* tquad     # integration points, **scaled to [0,1]**
    cdef RTYPE_t* qw        # quadrature weights
    cdef RTYPE_t* psi       # basis function values at integration points

    cdef int n_vis
    cdef RTYPE_t* tvis      # visualization points (accounting for the interp parameter), **scaled to [0,1]**
    cdef RTYPE_t* psivis    # basis function values at visualization points

    # helper methods:

    # assemble Galerkin series of u at given points
    cdef void assemble( self, RTYPE_t* psi, DTYPE_t* uass, DTYPE_t* ucorr, int n_points ) nogil

    # get the value of u at the end of this timestep
    cdef void final_value( self, DTYPE_t* uass ) nogil

    # integrate a function (provided as values at the quadrature points) over the timestep
    cdef DTYPE_t do_quadrature( self, DTYPE_t* funcvals, RTYPE_t dt ) nogil

# discontinuous Galerkin
#
cdef class DG(GalerkinIntegrator):
    cdef int call(self, DTYPE_t* w, RTYPE_t t, RTYPE_t dt) nogil

# continuous Galerkin
#
cdef class CG(GalerkinIntegrator):
    cdef int call(self, DTYPE_t* w, RTYPE_t t, RTYPE_t dt) nogil

