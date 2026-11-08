import os
import numpy as np

import mpi4py
from mpi4py import MPI
from mpi4py.util import pkl5
from ..output.SEGYTraceOutput import SEGYTraceOutput
from .costFunction import ComputeCostFunction, \
                          ComputeResidual
from .dataprocessing import applySourceCorrection, sourceCorrection, sourceCorrectionTerms
from .parallel import localShots, outputComm, shotParallel, sumOverShots


def directSimulation(problem, iacq, exportSeismicTrace=True, traceDirectory=".", computeCostFunction=False, dataDirectory=None, exportResidual=False, model="c", startModelFile=None, ifreq=0, pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy"):
    """
    Forward runs of the shots of an acquisition of the problem.

    Parameters:
    -----------
        problem : makutu.imaging.FWIProblem
            Solver, acquisitions, model bounds and filter frequencies
        iacq : int
            Index of the acquisition in problem.listOfAcquisition
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
        startModelFile : str or np.ndarray, optional
            Model to start with: HDF5 file (dataset "velocity") or array, in the
            parametrisation ``model``; default None keeps the solver model
        ifreq : int
            Index of the frequency in problem.freqList \
            Default is 0
        pCostDir : str, optional
            Partial cost function directory (None: no file) \
            Default is `partialCostFunction`

    The shots are dealt out to the MPI ranks when the solver is not decomposed
    (makutu.forwardBackward.parallel): every rank computes and exports its shots.

    Returns:
    --------
        float : cost function of the shots of this rank (0 without computeCostFunction;
        with the domain decomposition, on rank 0 only)
    """
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    solver = problem.solver
    low = problem.minModel
    high = problem.maxModel
    freqFilter = problem.freqList[ifreq]
    solver.setModelForGradient(model)
    if startModelFile is not None:
        solver.updateModel(startModelFile, low, high)
    outComm = outputComm(solver)               # its rank 0 writes the outputs of a shot
    sameShots = not shotParallel(solver)       # every rank works on every shot

    cost = 0.0
    for shot in localShots(problem.listOfAcquisition[iacq].shots, solver):
        solver.resetWaveField()
        solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())
        prepareSource(problem, solver, ifreq)

        if computeCostFunction:
            computeCostFunctionObj = ComputeCostFunction(directory=pCostDir, dt=solver.dtSeismoRecorded,
                                                         rootname="partialCost_" + shot.id).computePartialCostFunction
            drootname, dformat, dext = dinfo.split(";")
            filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"
            computeResidual = selectedResidual(problem, shot, solver,
                                               ComputeResidual(filename, dt=solver.dtSeismoRecorded,
                                                               fmax=freqFilter).computeResidual)
        else:
            exportResidual = False
            computeCostFunctionObj = None
            computeResidual = None

        partial = forwardComputation(solver, shot, outComm, exportSeismicTrace, traceDirectory,
                                     computeCostFunctionObj, computeResidual, exportResidual=exportResidual)
        cost += partial or 0.0

        shot.flag = "Done"
        if outComm.Get_rank() == 0:
            print("Shot", shot.id, "done\n")
        if sameShots:
            comm.Barrier()
    return cost


def prepareSource(problem, solver, ifreq):
    """Source wavelet of a shot: evaluated, filtered at the frequency ``problem.freqList[ifreq]``
    and corrected by the estimated filter of that frequency (``problem.sourceCorrection``)."""
    solver.evaluateSource()
    solver.filterSource(problem.freqList[ifreq])
    correction = (problem.sourceCorrection or {}).get(ifreq)
    if correction is not None:
        applySourceCorrection(solver, correction)


def selectedResidual(problem, shot, solver, computeResidual):
    """``computeResidual`` followed by the data selection of the problem (W r), if any."""
    selection = problem.dataSelection
    if selection is None:
        return computeResidual
    src, rcv = shot.getSourceCoords()[0], shot.getReceiverCoords()
    return lambda simulated: selection.apply(computeResidual(simulated), solver.dtSeismoRecorded, src, rcv)


def estimateSource(problem, dataDirectory, ifreq=0, dinfo="seismoTrace_shot;5;.sgy", eps=1e-3):
    """
    Least-squares filter of the source wavelet at the frequency ``problem.freqList[ifreq]``
    for the current model (makutu.forwardBackward.dataprocessing), over all the shots and
    the MPI ranks; stored in ``problem.sourceCorrection[ifreq]`` and used by the following
    runs of that frequency.

    Returns
    -------
        (frequencies, c) : the filter
    """
    solver = problem.solver
    freqFilter = problem.freqList[ifreq]
    drootname, dformat, dext = dinfo.split(";")
    dt = solver.dtSeismoRecorded
    terms = None
    for acq in problem.listOfAcquisition:
        for shot in localShots(acq.shots, solver):
            solver.resetWaveField()
            solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())
            solver.evaluateSource()
            solver.filterSource(freqFilter)
            solver.runForward()
            simulated = np.asarray(solver.getPressureAtReceivers(), dtype=np.float64)
            filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"
            observed = ComputeResidual(filename, dt=dt, fmax=freqFilter).computeResidual(simulated) + simulated
            if problem.dataSelection is not None:
                src, rcv = shot.getSourceCoords()[0], shot.getReceiverCoords()
                simulated = problem.dataSelection.apply(simulated, dt, src, rcv)
                observed = problem.dataSelection.apply(observed, dt, src, rcv)
            freqs, cross, power = sourceCorrectionTerms(simulated, observed, dt)
            new = np.concatenate([cross.real, cross.imag, power])
            terms = new if terms is None else terms + new
    terms = sumOverShots(terms, solver)
    m = len(terms) // 3
    freqs = np.fft.rfftfreq(2 * (m - 1), dt)
    correction = sourceCorrection(freqs, terms[:m] + 1j * terms[m:2 * m], terms[2 * m:], eps)
    if problem.sourceCorrection is None:
        problem.sourceCorrection = {}
    problem.sourceCorrection[ifreq] = correction
    return correction


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

    Returns:
    --------
        float or None : cost function of the shot on the rank 0 of comm, None otherwise
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

        partial = computeCostFunction(residual) if rank == 0 else None

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
        return partial
    return None


def forward(solver, shot, comm, snapshotStep=None):
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
        snapshotStep : int, optional
            Keep the second time derivative of the pressure every snapshotStep
            steps, for the gradient (``backward``); default None keeps nothing
    """
    if comm.Get_rank() == 0:
        print("\nForward ", shot.id)
    shot.flag = "In Progress"
    solver.runForward(snapshotStep)
