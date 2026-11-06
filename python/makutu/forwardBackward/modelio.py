"""Model files of the optimisation (HDF5)."""

import h5py


def writeModel(filename, x, comm):
    """
    Write the model ``x`` (dataset "velocity") to an HDF5 file, single process only:
    with the MPI domain decomposition each rank holds its part of the model, which
    would have to be gathered (not implemented).
    """
    if comm.Get_size() > 1:
        raise NotImplementedError("model files (func, callback) with the MPI domain decomposition")
    with h5py.File(filename, "w") as h:
        h.create_dataset("velocity", data=x[:], dtype='d', chunks=True, maxshape=(x.shape[0],))
