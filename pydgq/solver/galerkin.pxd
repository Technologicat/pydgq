# -*- coding: utf-8 -*-
"""Support routines for Galerkin integrators."""

from __future__ import absolute_import

from pydgq.solver.types cimport DTYPE_t
from pydgq.solver.kernel_interface cimport KernelBase
from pydgq.solver.integrator_interface cimport ImplicitIntegrator

# base class for Galerkin integrators
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
#    cdef DTYPE_t[:,::1]   uvis  = np.empty( [nx,n_space_dofs],                 dtype=DTYPE, order="C" )  # u, assembled for visualization
#    cdef DTYPE_t[::1]     ucvis = np.empty( [nx],                              dtype=DTYPE, order="C" )  # correction for compensated summation in assemble() (for visualization)
#
#    # global arrays, same for each solver instance (see galerkin.DataManager.load_data(), galerkin.DataManager.prep_solver())
#    cdef DTYPE_t[:,::1] LU      = galerkin.datamanager.LU       # LU decomposed mass matrix (packed format), for one space DOF, shape (n_time_dofs, n_time_dofs)
#    cdef int[::1]       p       = galerkin.datamanager.p        # row permutation information, length n_time_dofs
#    cdef int[::1]       mincols = galerkin.datamanager.mincols  # band information for L, length n_time_dofs
#    cdef int[::1]       maxcols = galerkin.datamanager.maxcols  # band information for U, length n_time_dofs
#    cdef DTYPE_t[::1]   qw      = galerkin.datamanager.integ_w  # quadrature weights (Gauss-Legendre)
#    cdef DTYPE_t[:,::1] psi     = galerkin.datamanager.integ_y  # basis function values at the quadrature points, psi[j,i] is N[j]( x[i] )
#    cdef DTYPE_t[:,::1] psivis  = galerkin.datamanager.vis_y    # basis function values at the visualization points, psivis[j,i] is N[j]( x[i] )
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
    cdef DTYPE_t* LU
    cdef int* p
    cdef int* mincols
    cdef int* maxcols

    cdef int n_quad         # number of integration points in the Gauss-Legendre rule
    cdef DTYPE_t* qw        # quadrature weights
    cdef DTYPE_t* tquad     # integration points, **scaled to [0,1]**

    cdef DTYPE_t* psi
    cdef DTYPE_t* psivis

    cdef DTYPE_t* tvis      # visualization points (accounting for the interp parameter), **scaled to [0,1]**

    # helper methods
    #
    cdef void assemble( self, DTYPE_t* psi, DTYPE_t* uass, DTYPE_t* ucorr, int n_points ) nogil  # assemble Galerkin series
    cdef void final_value( self, DTYPE_t* uass ) nogil  # get final value at this timestep
    cdef DTYPE_t do_quadrature( self, DTYPE_t* funcvals, DTYPE_t dt ) nogil  # integrate over timestep

# discontinuous Galerkin
#
cdef class DG(GalerkinIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

# continuous Galerkin
#
cdef class CG(GalerkinIntegrator):
    cdef int call(self, DTYPE_t* w, DTYPE_t t, DTYPE_t dt) nogil

