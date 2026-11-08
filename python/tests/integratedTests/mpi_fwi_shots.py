"""
FWI functions (gradf, func) with the shots dealt out to the MPI ranks (shot parallelism:
the solver is not decomposed, every rank holds the whole model).

Run with ``mpirun -np N python mpi_fwi_shots.py WORKDIR``: 2-D model with an anomaly, 3 shots;
the observed data are computed by the ranks (each exports its shots), then gradf and func on
the background model. Every rank saves its gradient, J of gradf and J of func in
WORKDIR/result_<N>_<rank>.npz. Used by test_mpi_decomposition.py.
"""

import os
import sys

import numpy as np
from mpi4py import MPI

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import kokkos  # noqa: E402

kokkos.initialize()

import pyfuntides.solver as FTSolver  # noqa: E402

from makutu.acquisition import Acquisition  # noqa: E402
from makutu.forwardBackward import directSimulation, func  # noqa: E402
from makutu.imaging import FWIProblem  # noqa: E402
from makutu.imaging import gradf  # noqa: E402
from makutu.mesh import CartesianMesh2D  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402

EX, EZ = 40, 20


def main(workdir):
    comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    os.chdir(workdir)
    X, Z = np.meshgrid((np.arange(EX) + 0.5) * 1000.0 / EX, (np.arange(EZ) + 0.5) * 500.0 / EZ)
    v0 = np.full(X.shape, 2000.0)
    vtrue = v0 + 300.0 * np.exp(-((X - 600) ** 2 + (Z - 300) ** 2) / 80.0 ** 2)
    mesh = CartesianMesh2D(1000.0, 500.0, EX, EZ, order=2, vp=vtrue, free_surface_on_top=False)
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=4e-4, minTime=0.0, maxTime=0.6, dtSeismo=2e-3, sourceType="ricker2",
                            sourceFreq=10.0, **mesh.solver_kwargs())
    solver.initialize()
    acq = Acquisition(sources=[[x, 0.0, 100.0] for x in (250.0, 500.0, 750.0)],
                      receivers=[[x, 0.0, 60.0] for x in np.linspace(100, 900, 17)])
    problem = FWIProblem(solver, acq, minModel=1000.0, maxModel=4000.0, freqList=["all"])

    obs = f"obs_{size}"
    directSimulation(problem, 0, exportSeismicTrace=True, traceDirectory=obs)          # each rank its shots
    comm.Barrier()
    if rank == 0:
        for shot in acq.shots:
            os.replace(f"{obs}/seismoTrace_shot{shot.id}_10.sgy", f"{obs}/seismoTrace_shot{shot.id}.sgy")
    comm.Barrier()

    g, J = gradf(problem, v0.ravel(), obs, exportSeismicTrace=False, computeFullCF=True)
    Jf = func(problem, v0.ravel(), obs, exportSeismicTrace=False)
    np.savez(os.path.join(workdir, f"result_{size}_{rank}.npz"), gradient=g, J=J, Jf=Jf,
             illumination=problem.illumination)
    problem.release()
    del solver, mesh


if __name__ == "__main__":
    main(sys.argv[1])
    kokkos.finalize()
