"""
Gradient of the least-squares cost function (adjoint-state method) over the shots
of the acquisitions, for the optimisers of makutu.imaging.Minimizer.

Per shot: forward run keeping the second time derivative of the pressure
(``solver.runForward``), residual against the observed SEG-Y data, adjoint run with
the residual injected at the receivers and imaging condition
(``solver.computeGradient``). The gradients of the shots are summed in memory.
The model and the gradient are flat arrays per element, in seismic order
(``solver.getVelocityModel``), in the parametrisation ``model`` ("c", "1/c", "1/c2").
The illumination of the shots (``solver.illumination``, source side of the diagonal
pseudo-Hessian) is summed in ``problem.illumination`` by :func:`gradf`, for
makutu.imaging.GradientConditioner.
"""

import os
import numpy as np

from mpi4py import MPI
from mpi4py.util import pkl5


from ..forwardBackward.forward import forward
from ..forwardBackward.backward import backward
from ..forwardBackward.costFunction import frequencyTag, writeCostFunction, \
                                           ComputeCostFunction, \
                                           ComputeResidual
from ..forwardBackward.modelio import writeFields
from ..forwardBackward.parallel import checkShotParallel, localShots, outputComm, shotParallel, sumOverShots

from ..output.SEGYTraceOutput import SEGYTraceOutput


def gradientComputation(problem, iacq, dataDirectory, exportSeismicTrace=True, traceDirectory=".", exportResidual=False,
                        ifreq=0, model="c", startModelFile=None, iterfwi=None, pCostDir="partialCostFunction",
                        dinfo="seismoTrace_shot;5;.sgy", computeCF=False, snapshotStep=None):
    """Gradient of the cost function over the shots of one acquisition

    Parameters:
    -----------
        problem : makutu.imaging.FWIProblem
            State of the inversion
        iacq : int
            Index of the acquisition in problem.listOfAcquisition
        dataDirectory : str
            Directory containing the observed data (one SEG-Y file per shot, see dinfo)
        exportSeismicTrace : bool, optional
            Export the simulated seismograms to SEG-Y \
            Default is True
        traceDirectory : str, optional
            Directory where to export the seismic traces \
            Default is current directory
        exportResidual : bool, optional
            Export the residuals to SEG-Y (directory "residual") \
            Default is False
        ifreq : int, optional
            Index of the frequency in problem.freqList \
            Default is 0
        model : str, optional
            Parametrisation: "c", "1/c" or "1/c2" \
            Default is "c"
        startModelFile : str or np.ndarray, optional
            Model loaded before the first shot (HDF5 file with a dataset "velocity", or
            array), in the parametrisation ``model``; default None keeps the solver model
        iterfwi : int, optional
            Index of the FWI iteration, appended to the trace names
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`
        dinfo : str, optional
            Rootname of the data files, digits of the shot id and extension, separated by ";"
        computeCF : bool, optional
            Write the cost function of every shot to pCostDir \
            Default is False
        snapshotStep : int, optional
            Steps between the snapshots of the forward run \
            Default: ``solver.snapshotStep``, or ``solver.snapshotStepFor(fmax)`` with the
            filter frequency

    With shot parallelism (makutu.forwardBackward.parallel) only the shots of this rank
    are computed: the values returned are the sums over these shots.

    Returns:
    --------
        (gradient, illumination, cost) : sums over the shots of this rank; gradient and
        illumination are None without shot (the cost is summed on rank 0 of the output
        communicator)
    """
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    solver = problem.solver
    freqFilter = problem.freqList[ifreq]
    solver.setModelForGradient(model)
    if snapshotStep is None:
        snapshotStep = getattr(solver, "snapshotStep", None) or \
            solver.snapshotStepFor(None if str(freqFilter) == "all" else float(freqFilter))
    drootname, dformat, dext = dinfo.split(";")
    if startModelFile is not None:
        solver.updateModel(startModelFile, problem.minModel, problem.maxModel)
    outComm = outputComm(solver)               # its rank 0 writes the outputs of a shot
    sameShots = not shotParallel(solver)       # every rank works on every shot

    gradient, illumination = None, None
    cost = 0.0
    for shot in localShots(problem.listOfAcquisition[iacq].shots, solver):
        solver.resetWaveField()
        solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())
        solver.evaluateSource()
        solver.filterSource(freqFilter)

        forward(solver, shot, outComm, snapshotStep)
        simulated = solver.getPressureAtReceivers()

        rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}"
        if iterfwi is not None:
            rootname += f"_{iterfwi}"
        if exportSeismicTrace:
            SEGYTraceOutput(simulated, rootname=rootname, directory=traceDirectory).export(
                receiverCoords=shot.getReceiverCoords(), sourceCoords=shot.getSourceCoords()[0],
                dt=solver.dtSeismoRecorded, comm=outComm)

        filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"
        residual = ComputeResidual(filename, dt=solver.dtSeismoRecorded, fmax=freqFilter).computeResidual(simulated)
        if exportResidual:
            SEGYTraceOutput(residual, rootname=rootname, directory="residual").export(
                receiverCoords=shot.getReceiverCoords(), sourceCoords=shot.getSourceCoords()[0],
                dt=solver.dtSeismoRecorded, comm=outComm)
        if outComm.Get_rank() == 0:
            cost += ComputeCostFunction(directory=pCostDir if computeCF else None, dt=solver.dtSeismoRecorded,
                                        rootname="partialCost_" + shot.id).computePartialCostFunction(residual)

        partial = backward(solver, residual, model)
        gradient = partial if gradient is None else gradient + partial
        illum = getattr(solver, "illumination", None)
        if illum is not None:
            illumination = illum if illumination is None else illumination + illum

        shot.flag = "Done"
        if outComm.Get_rank() == 0:
            print("Shot", shot.id, "done\n")
        if sameShots:
            comm.Barrier()

    return gradient, illumination, cost


def gradf(problem, x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".",
          exportResidual=False, ifreq=0, model="c", pCostDir="partialCostFunction",
          dinfo="seismoTrace_shot;5;.sgy", gDn=0, computeFullCF=False):
    """
    Gradient of the cost function for the model x. The model is applied in memory; with
    several MPI ranks the shots are dealt out to them (shot parallelism, the solver not
    decomposed) and the gradient, the illumination and J are summed over the ranks.

    Parameters
    -----------
        problem : makutu.imaging.FWIProblem
            State of the inversion; ``problem.model`` and ``problem.illumination`` are
            updated
        x : numpy array
            Model, flat per element in seismic order, parametrisation ``model``
        dataDirectory : str
            Directory containing the observed data
        gradDirectory : str, optional
            In debug mode (problem.mode "DBG"), directory where the full gradient of every
            evaluation is written (fullGradient<n>.hdf5) \
            Default is 'partialGradient'
        exportSeismicTrace : bool, optional
            Export the simulated seismograms to .sgy \
            Default is True
        traceDirectory : str, optional
            Directory where to export the seismic traces \
            Default is current directory
        exportResidual : bool, optional
            Export the residuals \
            Default is False
        ifreq : int, optional
            Index of the frequency in problem.freqList \
            Default is 0
        model : str, optional
            Parametrisation: "c", "1/c" or "1/c2" \
            Default is 'c'
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`
        dinfo : str, optional
            Rootname of the data files, digits of the shot id and extension, separated by ";"
        gDn : int, optional
            Normalisation of the gradient: 0 none (default), 1 by its maximum absolute
            value, 2 by its 2-norm
        computeFullCF : bool, optional
            Also compute and return the cost function \
            Default is False

    Returns
    --------
        gradJ : np.ndarray
            Gradient
        J : float
            Cost function, if computeFullCF is True
    """
    solver = problem.solver
    checkShotParallel(solver)
    problem.model = np.array(x, copy=True)
    debug = problem.debug
    ngrad = problem.ngrad
    pdir = f"{pCostDir}_{ngrad}_G"

    gradJ, illumination, J = None, None, 0.0
    for i in range(len(problem.listOfAcquisition)):
        g, ill, cost = gradientComputation(problem, i, dataDirectory, exportSeismicTrace, traceDirectory,
                                           exportResidual, ifreq, model, startModelFile=x if i == 0 else None,
                                           iterfwi=ngrad, pCostDir=pdir, dinfo=dinfo, computeCF=debug)
        if g is not None:
            gradJ = g if gradJ is None else gradJ + g
        if ill is not None:
            illumination = ill if illumination is None else illumination + ill
        J += cost
    # sums over the ranks (shot parallelism): every rank gets the same values
    gradJ = sumOverShots(gradJ, solver)
    if gradJ is None:
        gradJ = np.zeros(np.shape(x))
    problem.illumination = sumOverShots(illumination, solver)
    J = sumOverShots(J, solver)

    rank = MPI.COMM_WORLD.Get_rank()
    if debug and rank == 0:
        os.makedirs(gradDirectory, exist_ok=True)
        writeFields(os.path.join(gradDirectory, f"fullGradient{ngrad}.hdf5"),
                    {"fullGradient": gradJ, "illumination": problem.illumination}, solver)

    if gDn == 1:
        gradJ = gradJ / np.max(np.abs(gradJ))
    elif gDn == 2:
        gradJ = gradJ / np.linalg.norm(gradJ)

    problem.ngrad += 1
    if computeFullCF is True:
        if debug and rank == 0:
            writeCostFunction(J, filename=f"fullCostFunction{frequencyTag(problem.freqList[ifreq])}.txt")
        return gradJ, J
    return gradJ
