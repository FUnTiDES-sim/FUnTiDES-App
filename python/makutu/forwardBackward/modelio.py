"""Model files of the optimisation (HDF5)."""

import h5py

from ..imaging import globals as gb
from .parallel import isDecomposed


def writeModel(filename, x, comm):
    """
    Write the model ``x`` (dataset "velocity") to an HDF5 file, from rank 0 of ``comm``.

    With shot parallelism every rank holds the whole (same) model. With the MPI domain
    decomposition each rank holds its part of the model, which would have to be gathered
    (not implemented).
    """
    if comm.Get_size() > 1 and isDecomposed(getattr(gb, "solver", None)):
        raise NotImplementedError("model files (func, callback) with the MPI domain decomposition")
    if comm.Get_rank() == 0:
        with h5py.File(filename, "w") as h:
            h.create_dataset("velocity", data=x[:], dtype='d', chunks=True, maxshape=(x.shape[0],))
