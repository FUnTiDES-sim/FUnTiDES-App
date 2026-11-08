"""Model files of the optimisation (HDF5)."""

import h5py

from .parallel import isDecomposed


def gridAttributes(solver):
    """
    HDF5 attributes describing a field flat per element (seismic order, x fastest):
    ``shape`` = (nz, nx) in 2-D or (nz, ny, nx) in 3-D, and ``lengths`` = the domain
    lengths [m] in the same order. Used by :class:`makutu.display.Hdf5Field` to display
    the files without giving the shape. Empty if the solver is unknown or decomposed (local
    sizes only).
    """
    if solver is None or getattr(solver, "ex", None) is None or isDecomposed(solver):
        return {}
    if solver.dim == 2:
        return {"shape": (solver.ez, solver.ex), "lengths": (solver.lz, solver.lx)}
    return {"shape": (solver.ez, solver.ey, solver.ex), "lengths": (solver.lz, solver.ly, solver.lx)}


def writeFields(filename, fields, solver=None):
    """
    Write the arrays of ``fields`` (name -> flat array per element) to an HDF5 file, with the
    grid attributes of ``solver`` (:func:`gridAttributes`) on every dataset. Call it from
    one rank only.
    """
    attrs = gridAttributes(solver)
    with h5py.File(filename, "w") as h:
        for name, data in fields.items():
            if data is None:
                continue
            d = h.create_dataset(name, data=data)
            for k, v in attrs.items():
                d.attrs[k] = v


def writeModel(filename, x, comm, solver=None):
    """
    Write the model ``x`` (dataset "velocity") to an HDF5 file, from rank 0 of ``comm``, with
    the grid attributes of ``solver`` (:func:`gridAttributes`).

    With shot parallelism every rank holds the whole (same) model. With the MPI domain
    decomposition each rank holds its part of the model, which would have to be gathered
    (not implemented).
    """
    if comm.Get_size() > 1 and isDecomposed(solver):
        raise NotImplementedError("model files (func, callback) with the MPI domain decomposition")
    if comm.Get_rank() == 0:
        writeFields(filename, {"velocity": x[:].astype("d")}, solver)
