import numpy as np
cimport numpy as np
from cython import boundscheck, nonecheck, wraparound
from autograd.core import primitive


def csr_binary_dot_left(rows, cols, inputs):
    """
    The binary matrix is on the left of the dot product.
    """
    out = np.zeros_like(inputs)
    _csr_binary_dot_left(rows, cols, inputs, out)
    return out


@nonecheck(False)
@wraparound(False)
@boundscheck(False)
cdef inline void _csr_binary_dot_left(int[::1] rows,
                                      int[::1] cols,
                                      double[:,::1] inputs,
                                      double[:,::1] out):
    cdef int idx, i, j, k
    for idx in range(rows.shape[0]):
        i = rows[idx]
        k = cols[idx]
        for j in range(inputs.shape[1]):
            out[i, j] += inputs[k, j]


def csr_binary_dot_right(rows, cols, inputs):
    """
    The binary matrix is on the right of the dot product.
    """
    out = np.zeros_like(inputs)
    _csr_binary_dot_right(rows, cols, inputs, out)
    return out


@nonecheck(False)
@wraparound(False)
@boundscheck(False)
cdef inline void _csr_binary_dot_right(int[:] rows,
                                       int[:] cols,
                                       double[:,:] inputs,
                                       double[:,:] out):

    cdef int i, j, k, idx
    for idx in range(cols.shape[0]):
        j = cols[idx]
        i = rows[idx]
        for k in range(inputs.shape[0]):
            out[k, j] += inputs[k, i]
