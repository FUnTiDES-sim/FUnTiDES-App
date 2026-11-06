import os
import numpy as np

import mpi4py
from mpi4py import MPI
from mpi4py.util import pkl5
from ..imaging import globals as gb
from ..output.SEGYTraceOutput import SEGYTraceOutput
from .costFunction import ComputeCostFunction, \
                          ComputeResidual


def directSimulation(iacq, exportSeismicTrace=True, traceDirectory=".", computeCostFunction=False, dataDirectory=None, exportResidual=False, model="c", startModelFile=None, ifreq=0, pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy"):
    """
    Performs a direct propagation of Acoustic Solver. Note that you must specify some global variables before being able to use this routine (see globals.py).

    Parameters:
    -----------
        iacq : int
            Index of Acquisition in gb.listOfAcquisition
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy \
            Default is True
        traceDirectory : string
            Directory where to export the seismic traces \
            Default is current directory
        computeCostFunction : bool
            Whether or not to compute the costFunction \
            If True, you need to specify dataDirectory
        dataDirectory : string
            Directory containing original data for residual/costFunction computation
        exportResidual : bool
            Whether or not to export the residual \
            Default is False \
            Implies that computeCostFunction is set to True. \
        model : str
            Model to use \
            Default is "c"
        startModelFile : string hdf5 file
            File that contains the velocity model to start with \
            Only needed if you wish to use a model different from the acquisition default
        ifreq : int
            Index of the frequency in gb.freqList \
            Default is 0
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`
    """
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    solver = gb.solver
    low = gb.minModel
    high = gb.maxModel
    freqFilter = gb.freqList[ifreq]
    solver.setModelForGradient(model)

    ishot = 0
    print( f"{rank} IACQ {iacq}" )
    for shot in gb.listOfAcquisition[iacq].shots:
        # Reset wavefield for each shot (replaces initialize + applyInitialConditions)
        solver.resetWaveField()

        if ishot == 0:
            if startModelFile is not None:
                solver.updateModel(startModelFile, low, high)

        solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())
        solver.evaluateSource()
        solver.filterSource(freqFilter)

        if computeCostFunction:
            # set cost function
            computeCostFunctionObj = ComputeCostFunction( directory=pCostDir,
                                                          dt=solver.dtSeismoRecorded,
                                                          rootname="partialCost_"+shot.id
            ).computePartialCostFunction

            # set computeResidual fonction
            drootname, dformat, dext = dinfo.split(";")
            filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"

            computeResidual = ComputeResidual(filename, dt=solver.dtSeismoRecorded, fmax=freqFilter).computeResidual

        else:
            exportResidual=False
            computeCostFunctionObj=None
            computeResidual=None

        forwardComputation(solver, shot, comm, exportSeismicTrace, traceDirectory, computeCostFunctionObj, computeResidual, exportResidual=exportResidual)

        #Update shot flag
        shot.flag = "Done"
        if rank == 0:
            print("Shot", shot.id, "done\n")

        ishot += 1
        comm.Barrier()


def forwardComputation(solver, shot, comm, exportSeismicTrace=True, traceDirectory=".", computeCostFunction=None, computeResidual=None, exportResidual=False):
    """Performs forward simulation and handle exports for one shot using Acoustic Solver.
    Parameters:
    -----------
        solver : AcousticSolver
            AcousticSolver object
        shot : Shot
            Shot object
        comm : MPI_COMM
            MPI communicators
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy. If not set default is True
        traceDirectory: string
            Directory where to export the seismic traces. If not set default is current directory
        computeCostFunction : callback
            Callback to compute the cost function.
            Default is None (not computed)
        computeResidual : callback
            Callback to compute the residual.
            Default is None (not computed)
        exportResidual : bool
            Whether or not to export the residual if computeCostFunction is set to True. Default is False
    """
    rank = comm.Get_rank()

    forward(solver, shot, comm)

    # Get seismogram at receivers
    wavefield = solver.getWaveField()
    if exportSeismicTrace:
        rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}"
        pressOut = SEGYTraceOutput( wavefield,
                                   rootname=rootname,
                                   directory=traceDirectory,
                                   )
        pressOut.export(receiverCoords=shot.getReceiverCoords(),
                        sourceCoords=shot.getSourceCoords()[0],
                        dt=solver.dtSeismoRecorded,
                        comm=comm)

    # residual and cost function calculation
    if computeCostFunction is not None:
        simulatedData = solver.getFullWaveFieldAtReceivers(comm)

        residual = computeResidual( simulatedData )

        if rank == 0:
            computeCostFunction( residual )

        if exportResidual:
            rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}"
            residualOut = SEGYTraceOutput(residual,
                                          rootname=rootname,
                                          directory="residual",
                        )
            residualOut.export(receiverCoords=shot.getReceiverCoords(),
                               sourceCoords=shot.getSourceCoords()[0],
                               dt=solver.dtSeismoRecorded,
                               comm=comm)


def forward(solver, shot, comm, outputWaveField=-1):
    """Solves forward problem: runs the time loop of the current shot.

    The wavefield and the seismograms are kept at the end (read them with
    ``solver.getWaveField()``); reset the solver (``solver.resetWaveField()``)
    before the next shot.

    Parameters
    ----------
        solver : AcousticSolver
            AcousticSolver object
        shot : Shot
            Contains all informations on current shot
        comm : MPI_COMM
            MPI communicators
        outputWaveField : int
            Output interval for the Wavefield (not used with pyfuntides)
    """
    rank = comm.Get_rank()

    if rank == 0 :
        print("\nForward ", shot.id)

    # Time loop (step index 0-based)
    dt = solver.dt
    n_steps = solver.n_time_steps
    shot.flag = "In Progress"

    for cycle in range(n_steps):
        if rank == 0 and cycle % 100 == 0:
            print("time = %.3fs," % (cycle * dt), "dt = %.4f," % dt, "iter =", cycle + 1)

        solver.execute(cycle)
