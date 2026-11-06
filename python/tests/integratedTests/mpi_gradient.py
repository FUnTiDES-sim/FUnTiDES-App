"""
Gradient (solver.computeGradient) with the X-axis MPI domain decomposition.

Run with ``mpirun -np N python mpi_gradient.py WORKDIR``: homogeneous SEP model
written to WORKDIR (rank 0), observed data from a model with a velocity anomaly
between the source and the receivers, gradient of every rank gathered by rank 0
in WORKDIR/gradient_<N>.npy (seismic order, shape (nz, ny, nx)).
Used by test_mpi_decomposition.py.
"""

import os
import sys

import numpy as np
from mpi4py import MPI

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import kokkos  # noqa: E402

kokkos.initialize()

import pyfuntides.solver as FTSolver  # noqa: E402

from makutu.input.createSepModel import _write_sep  # noqa: E402
from makutu.mesh.InternalMesh import StructuredMesh  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402

L, NE, ORDER, H = 1500.0, 30, 2, 50.0


def main(workdir):
    comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    sep = os.path.join(workdir, "v1500.H")
    if rank == 0 and not os.path.exists(sep):
        _write_sep(sep, np.full((NE, NE, NE), 1500.0, np.float32), H, H, H)
    comm.Barrier()
    mesh = StructuredMesh(L, L, L, NE, NE, NE, order=ORDER, comm=comm if size > 1 else None, vp_file=sep)
    solver = AcousticSolver(mesh_type=FTSolver.MeshType.UNSTRUCT, implem_type=FTSolver.ImplemType.MAKUTU,
                            model_location=FTSolver.ModelLocationType.ONELEMENTS, dt=1e-3, minTime=0.0,
                            maxTime=0.6, dtSeismo=2e-3, sourceType="ricker2", sourceFreq=5.0,
                            comm=comm if size > 1 else None, **mesh.solver_kwargs())
    solver.initialize()

    c = (np.arange(NE) + 0.5) * H
    xc = mesh.origin_x + (np.arange(mesh.ex) + 0.5) * H
    Z, Y, X = np.meshgrid(c, c, xc, indexing="ij")             # local part, z down
    v0 = np.full(X.shape, 1500.0)
    vtrue = v0 + 300.0 * np.exp(-((X - 900) ** 2 + (Y - 775) ** 2 + (Z - 250) ** 2) / 150.0 ** 2)
    src = [[525.0, 775.0, 300.0]]
    rcv = [[x, 776.3, 108.7] for x in np.linspace(100.0, 1400.0, 14)]

    def shoot(v, step=None):
        solver.updateVelocityModel(v.ravel())
        solver.updateSourceAndReceiversFromCoords(src, rcv)
        solver.evaluateSource()
        solver.resetWaveField()
        solver.runForward(step)
        return solver.getPressureAtReceivers().astype(np.float64)

    obs = shoot(vtrue)
    sim = shoot(v0, solver.snapshotStepFor())
    grad = solver.computeGradient(obs - sim, "c").reshape(X.shape)
    parts = comm.gather(grad, root=0)
    if rank == 0:
        np.save(os.path.join(workdir, f"gradient_{size}.npy"), np.concatenate(parts, axis=2))
    del solver, mesh


if __name__ == "__main__":
    main(sys.argv[1])
    kokkos.finalize()
