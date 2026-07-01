import numpy as np

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py.util import pkl5

from ..output.SEGYTraceOutput import SEGYTraceOutput
from ..imaging import globals as gb


def backwardComputation(solver, shot, comm, exportSeismicTrace=True, traceDirectory="."):
    """ Backward computation

    Parameters
    ----------
        solver : AcousticSolver
            AcousticSolver object
        shot : Shot
            Contains all informations on current shot
        comm : MPI_COMM
            MPI communicators
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy. If not set default is True
        traceDirectory: string
            Directory where to export the seismic traces. If not set default is current directory
    """
    backward(solver, shot, comm)

    if exportSeismicTrace:
        rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}"
        pressureAtReceivers = solver.getPressureAtReceivers()

        pressOut = SEGYTraceOutput(pressureAtReceivers,
                                   rootname=rootname,
                                   directory=traceDirectory)
        pressOut.export(receiverCoords=shot.getReceiverCoords(),
                        sourceCoords=shot.getSourceCoords()[0],
                        dt=solver.dtSeismo,
                        comm=comm)


def backward(solver, shot, comm, gradDirectory="partialGradient", outputWaveField=-1):
    """Solves backward (adjoint) problem.

    .. note::
        Reverse-time gradient computation via pyfuntides is not yet
        implemented.  ``computePartialGradient`` raises NotImplementedError.

    Parameters
    ----------
        solver : AcousticSolver
            AcousticSolver object
        shot : Shot
            Contains all informations on current shot
        comm : MPI_COMM
            MPI communicators
        gradDirectory : str
            Output directory for partial gradients
        outputWaveField : int
            Output interval for the Wavefield (not used with pyfuntides)
    """
    rank = comm.Get_rank()

    if rank == 0:
        print("\nBackward ", shot.id)

    # Reverse time loop — step indices count down
    dt = solver.dt
    n_steps = solver.n_time_steps

    for cycle in range(n_steps - 1, -1, -1):
        if rank == 0 and cycle % 100 == 0:
            print("time = %.3fs," % (cycle * dt), "dt = %.4f," % dt, "iter =", cycle)

        solver.execute(cycle)

        if cycle == 0:
            if rank == 0:
                print("Outputing Gradient...")
            minDepth = gb.minDepth
            solver.computePartialGradient(shot.id, minDepth, comm, gradDirectory)

    solver.resetWaveField()


def adjointSimulation(solver, acquisition, computeSeismicTrace, traceDirectory, residual, comm, startModel=None):
    rank = comm.Get_rank()

    ishot = 0
    for shot in acquisition.shots:
        # Reset wavefield for each shot (replaces initialize + applyInitialConditions)
        solver.resetWaveField()

        if ishot == 0:
            if startModel is not None:
                solver.updateVelocityModel(startModel)

        solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())
        residual = np.reshape(residual, (residual.shape[0], 1))
        solver.updateSourceValue(residual)

        backwardComputation(solver, shot, comm, computeSeismicTrace, traceDirectory)

        #Update shot flag
        shot.flag = "Done"
        if rank == 0:
            print("Shot", shot.id, "done\n")

        ishot += 1
        comm.Barrier()
