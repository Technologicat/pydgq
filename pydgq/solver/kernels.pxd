# -*- coding: utf-8 -*-
#
# Interface and examples for Cython-based computational kernels for evaluating the f() in  w' = f(w).

# Note that our requirement to release the Python GIL in the computational kernels
# is considered in Cython as part of the function pointer type.
#
# (Releasing the GIL allows us to run the time integration loop "with nogil", and
#  thus release another Python thread for execution while the integrator is running.)
#
# <state vector in>, <result out>, <n_dofs in>, <current t value in>, <user data in/out (read/write access!)>
#
ctypedef void (*kernelfuncptr)(double*, double*, int, double, void*) nogil


# examples

cdef struct lin_mass_matrix_data:
    cdef double* LU
    cdef int* p
    cdef double* M
    cdef double* work

cdef void f_lin_1st(double* w, double* out, int n, double t, void* data) nogil
cdef void f_lin_2nd(double* w, double* out, int n, double t, void* data) nogil
cdef void f_lin_1st_with_mass(double* w, double* out, int n, double t, void* data) nogil
cdef void f_lin_2nd_with_mass(double* w, double* out, int n, double t, void* data) nogil

