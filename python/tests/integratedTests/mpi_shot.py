"""
Acoustic shot with the X-axis MPI domain decomposition of StructuredMesh.

Run with ``mpirun -np N python mpi_shot.py OUTPUT.npy``: rank 0 writes the
pressure at the receivers (already reduced over the ranks).
Used by test_mpi_decomposition.py.
"""

import os
import sys

import numpy as np
from mpi4py import MPI

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(__file__))

import kokkos  # noqa: E402

kokkos.initialize()

import pyfuntides.solver as FTSolver  # noqa: E402

import shots  # noqa: E402
from makutu.mesh.InternalMesh import StructuredMesh  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402

DT, TMAX = 0.001, 1.0
# Receivers on both sides of every split of 2 or 3 ranks, two of them on the
# 3-rank subdomain interfaces (x = 500 m and 1000 m)
RECEIVERS = [[x, 776.3, 808.7] for x in np.linspace(100.0, 1400.0, 14)]
SOURCE = [525.0, 775.0, 775.0]


def main(output):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    mesh = StructuredMesh(shots.L, shots.L, shots.L, shots.NE, shots.NE, shots.NE,
                          order=shots.ORDER, comm=comm if size > 1 else None)
    kwargs = shots.solver_kwargs(DT, TMAX)
    kwargs.update(lx=mesh.lx, ex=mesh.ex, origin_x=mesh.origin_x,
                  comm=comm if size > 1 else None)
    solver = AcousticSolver(
        model=mesh.model,
        mesh_type=FTSolver.MeshType.UNSTRUCT if mesh.heterogeneous else FTSolver.MeshType.STRUCT,
        **kwargs)
    solver.initialize()
    solver.updateSourceAndReceiversFromCoords([SOURCE], RECEIVERS)
    shots.run(solver)
    traces = solver.getPressureAtReceivers()   # MPI Allreduce inside
    if comm.Get_rank() == 0:
        np.save(output, traces)
    del solver, mesh


if __name__ == "__main__":
    main(sys.argv[1])
    kokkos.finalize()
