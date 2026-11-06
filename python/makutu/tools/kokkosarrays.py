"""Kokkos arrays in the format expected by the FUnTiDES Python bindings."""

import numpy as np

import kokkos


def funtides_array(shape, dtype=kokkos.float32):
    """
    Allocate a zero-filled Kokkos array that can be passed to ``pyfuntides``.

    FUnTiDES views use the memory space and layout of the default Kokkos
    execution space: CudaUVMSpace + LayoutLeft on a CUDA build, HostSpace +
    LayoutRight on a CPU build. The numpy view of a LayoutLeft array has
    column-major strides; element-wise numpy assignment handles it.

    Parameters
    ----------
        shape : list of int
        dtype : kokkos dtype, default kokkos.float32

    Returns
    -------
        kokkos array (Kokkos must be initialized)
    """
    space = kokkos.DefaultMemorySpace
    layout = (kokkos.LayoutRight if space == kokkos.HostSpace
              else kokkos.LayoutLeft)
    kk = kokkos.array(list(shape), dtype=dtype, space=space, layout=layout)
    np.array(kk, copy=False)[...] = 0
    return kk
