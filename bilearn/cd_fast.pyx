# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=True
# cython: wraparound=True
#
# Author: Vlad Niculae
# License: BSD

from libc.math cimport fabs

cimport numpy as np

from lightning.impl.dataset_fast cimport ColumnDataset

cpdef _cd_bilinear_epoch(np.ndarray[double, ndim=2] U,
                         ColumnDataset Xl,
                         np.ndarray[double, ndim=2] XrV,
                         np.ndarray[double, ndim=1] y,
                         np.ndarray[double, ndim=1] y_pred,
                         np.ndarray[double, ndim=2] VtGsq,
                         double alpha):

    cdef int n_features = U.shape[0]
    cdef int n_components = U.shape[1]
    cdef int n_samples = XrV.shape[0]

    cdef int s, j, i, ii

    cdef double update
    cdef double viol = 0

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz


    for j in range(n_features):
        for s in range(n_components):

            Xl.get_column_ptr(j, &indices, &data, &n_nz)

            update = 0
            for ii in range(n_nz):
                i = indices[ii]
                update += (y_pred[i] - y[i]) * data[ii] * XrV[i, s]

            update += alpha * U[j, s]

            update /= (VtGsq[s, j] + alpha)

            viol += fabs(update)
            U[j, s] -= update

            # synchronize predictions
            for ii in range(n_nz):
                i = indices[ii]
                y_pred[i] -= update * data[ii] * XrV[i, s]

    return viol