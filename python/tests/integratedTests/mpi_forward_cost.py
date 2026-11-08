"""
Forward chain of makutu.forwardBackward with the X-axis MPI domain decomposition:
two shots exported to SEG-Y (rank 0), then the cost function against these traces.

Run with ``mpirun -np N python mpi_forward_cost.py WORKDIR``: rank 0 writes
``WORKDIR/result.npz`` (first gather, J and the data energy).
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

import segyio  # noqa: E402
import pyfuntides.solver as FTSolver  # noqa: E402

import shots  # noqa: E402
from makutu.acquisition import Acquisition  # noqa: E402
from makutu.forwardBackward import computeFullCostFunction, directSimulation  # noqa: E402
from makutu.imaging import FWIProblem  # noqa: E402
from makutu.mesh.InternalMesh import StructuredMesh  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402

RECEIVERS = [[x, 776.3, 808.7] for x in np.linspace(100.0, 1400.0, 14)]
SOURCES = [[525.0, 775.0, 775.0], [1025.0, 775.0, 775.0]]


def main(workdir):
    comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    os.chdir(workdir)
    mesh = StructuredMesh(shots.L, shots.L, shots.L, shots.NE, shots.NE, shots.NE,
                          order=shots.ORDER, comm=comm if size > 1 else None)
    kwargs = shots.solver_kwargs(0.001, 0.6)
    kwargs.update(lx=mesh.lx, ex=mesh.ex, origin_x=mesh.origin_x, dtSeismo=0.002,
                  comm=comm if size > 1 else None)
    solver = AcousticSolver(
        model=mesh.model,
        mesh_type=FTSolver.MeshType.UNSTRUCT if mesh.heterogeneous else FTSolver.MeshType.STRUCT,
        **kwargs)
    solver.initialize()
    acq = Acquisition(sources=SOURCES, receivers=RECEIVERS)
    problem = FWIProblem(solver, acq, minModel=1000.0, maxModel=3000.0, freqList=["all"])

    directSimulation(problem, 0, exportSeismicTrace=True, traceDirectory="obs")
    if rank == 0:
        for i in (1, 2):
            os.replace(f"obs/seismoTrace_shot{i:05d}_{int(shots.F0)}.sgy", f"obs/seismoTrace_shot{i:05d}.sgy")
    comm.Barrier()
    directSimulation(problem, 0, exportSeismicTrace=False, computeCostFunction=True, dataDirectory="obs", pCostDir="pcost")
    comm.Barrier()
    J = computeFullCostFunction(len(SOURCES), "pcost", save=False)
    if rank == 0:
        with segyio.open("obs/seismoTrace_shot00001.sgy", ignore_geometry=True) as f:
            data = f.trace.raw[:].T
        np.savez("result.npz", data=data, J=J, energy=solver.dtSeismoRecorded * np.sum(data.astype(float) ** 2) / 2)
    problem.release()
    del solver, mesh


if __name__ == "__main__":
    main(sys.argv[1])
    kokkos.finalize()
