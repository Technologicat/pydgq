# -*- coding: utf-8 -*-
#
# Interface and examples for Cython-based computational kernels for evaluating the f() in  w' = f(w).

###############
# Base classes
###############

# TODO: have an update hook for updating the matrices?
#
# The problem is that in a nonlinear problem, f() is called in the innermost loop:
#
#    timestep (base value of t changes)
#      nonlinear iteration (w changes)
#        sub-evaluation at some t inside the timestep (w changes; actual t used for f() evaluation changes)
#
# so when should the solver call the matrix update methods? (assuming "at every call" is too expensive)

# cpdef methods cannot be nogil, so we must work around as follows:
#
# KernelBase --> cdef call() nogil... (interface) (NOTE: this class **does not** define callback())
#   CythonKernel --> implements call(), provides hook for Cython callback() (nogil)
#   PythonKernel --> implements call(), provides hook for Python callback() (called in a "with gil" block)

cdef class KernelBase:
    cdef double* w    # old state vector (memory owned by caller)
    cdef double* out  # new state vector (memory owned by caller)
    cdef int n        # n_space_dofs
    cdef int timestep
    cdef int iteration

    cdef void update_metadata(self, int timestep, int iteration) nogil
    cdef void call(self, double* w, double* out, double t) nogil

# Base class for kernels implemented in Cython.
#
# Cython kernels will run in nogil mode.
#
cdef class CythonKernel(KernelBase):
    cdef void call(self, double* w, double* out, double t) nogil
    cdef void callback(self, double t) nogil

# Base class for kernels implemented in pure Python.
#
# Python kernels will acquire the gil for calling callback().
#
cdef class PythonKernel(KernelBase):
    cdef double[::1] w_arr    # Python-accessible view into w
    cdef double[::1] out_arr  # Python-accessible view into out

    cdef void call(self, double* w, double* out, double t) nogil
    # here callback() is a Python (regular def) function


##############################
# Specific kernels (examples)
##############################

cdef class Linear1stOrderKernel(CythonKernel):
    cdef double* M
    cdef double[:,::1] M_arr

    cdef void callback(self, double t) nogil
    cdef void compute(self, double* w_in, double* wp_out) nogil

cdef class Linear1stOrderKernelWithMassMatrix(Linear1stOrderKernel):
    cdef double* LU
    cdef int* p
    cdef double* wrk
    cdef double[:,::1] LU_arr
    cdef int[::1] p_arr
    cdef double[::1] wrk_arr

    cdef void callback(self, double t) nogil

cdef class Linear2ndOrderKernel(CythonKernel):
    cdef int m  # size of original 2nd-order system, m = n/2
    cdef double* M0
    cdef double* M1
    cdef double[:,::1] M0_arr
    cdef double[:,::1] M1_arr

    cdef void callback(self, double t) nogil
    cdef void compute(self, double* w_in, double* wp_out) nogil

# cdef classes are single inheritance only, so we have some duplication here
# (since this is both a "linear 2nd-order kernel" as well as a "kernel with mass matrix").
#
cdef class Linear2ndOrderKernelWithMassMatrix(Linear2ndOrderKernel):
    cdef double* LU
    cdef int* p
    cdef double* wrk1  # n elements
    cdef double* wrk2  # m elements
    cdef double* wrk3  # m elements
    cdef double[:,::1] LU_arr
    cdef int[::1] p_arr
    cdef double[::1] wrk_arr

    cdef void callback(self, double t) nogil

