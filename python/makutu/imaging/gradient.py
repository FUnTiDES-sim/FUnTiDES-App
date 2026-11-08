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
pseudo-Hessian) is summed in ``gb.illumination`` by :func:`gradf`, for
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
from ..imaging import globals as gb

Nfeval = 1


def gradientComputation(iacq, dataDirectory, exportSeismicTrace=True, traceDirectory=".", exportResidual=False,
                        ifreq=0, model="c", startModelFile=None, iterfwi=None, pCostDir="partialCostFunction",
                        dinfo="seismoTrace_shot;5;.sgy", computeCF=False, snapshotStep=None):
    """Gradient of the cost function over the shots of one acquisition

    Parameters:
    -----------
        iacq : int
            Index of acquisition in gb.listOfAcquisition
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
            Index of the frequency in gb.freqList \
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
        np.ndarray or None : gradient summed over the shots (None without shot); the
        illumination and the cost function summed over the shots are in the attributes
        ``gradientComputation.illumination`` and ``gradientComputation.cost``
    """
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    solver = gb.solver
    freqFilter = gb.freqList[ifreq]
    solver.setModelForGradient(model)
    if snapshotStep is None:
        snapshotStep = getattr(solver, "snapshotStep", None) or \
            solver.snapshotStepFor(None if str(freqFilter) == "all" else float(freqFilter))
    drootname, dformat, dext = dinfo.split(";")
    if startModelFile is not None:
        solver.updateModel(startModelFile, gb.minModel, gb.maxModel)
    outComm = outputComm(solver)               # its rank 0 writes the outputs of a shot
    sameShots = not shotParallel(solver)       # every rank works on every shot

    gradient = None
    cost = 0.0
    gradientComputation.illumination = None
    for shot in localShots(gb.listOfAcquisition[iacq].shots, solver):
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
            ill = gradientComputation.illumination
            gradientComputation.illumination = illum if ill is None else ill + illum

        shot.flag = "Done"
        if outComm.Get_rank() == 0:
            print("Shot", shot.id, "done\n")
        if sameShots:
            comm.Barrier()

    gradientComputation.cost = cost
    return gradient


def gradf(x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".",
          exportResidual=False, ifreq=0, model="c", pCostDir="partialCostFunction",
          dinfo="seismoTrace_shot;5;.sgy", gDn=0, computeFullCF=False):
    """
    Gradient of the cost function for the model x. The model is applied in memory; with
    several MPI ranks the shots are dealt out to them (shot parallelism, the solver not
    decomposed) and the gradient, the illumination and J are summed over the ranks.

    Parameters
    -----------
        x : numpy array
            Model, flat per element in seismic order, parametrisation ``model``
        dataDirectory : str
            Directory containing the observed data
        gradDirectory : str, optional
            In debug mode (gb.mode "DBG"), directory where the full gradient of every
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
            Index of the frequency in gb.freqList \
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
    global Nfeval

    solver = gb.solver
    checkShotParallel(solver)
    if hasattr(gb, "model"):
        gb.model[:] = x[:]
    debug = getattr(gb, "mode", "REL") == "DBG"
    pdir = f"{pCostDir}_{Nfeval}_G"

    gradJ, illumination, J = None, None, 0.0
    for i in range(len(gb.listOfAcquisition)):
        g = gradientComputation(i, dataDirectory, exportSeismicTrace, traceDirectory, exportResidual, ifreq,
                                model, startModelFile=x if i == 0 else None, iterfwi=Nfeval, pCostDir=pdir,
                                dinfo=dinfo, computeCF=debug)
        if g is not None:
            gradJ = g if gradJ is None else gradJ + g
        ill = gradientComputation.illumination
        if ill is not None:
            illumination = ill if illumination is None else illumination + ill
        J += gradientComputation.cost
    # sums over the ranks (shot parallelism): every rank gets the same values
    gradJ = sumOverShots(gradJ, solver)
    if gradJ is None:
        gradJ = np.zeros(np.shape(x))
    gb.illumination = sumOverShots(illumination, solver)
    J = sumOverShots(J, solver)

    rank = MPI.COMM_WORLD.Get_rank()
    if debug and rank == 0:
        os.makedirs(gradDirectory, exist_ok=True)
        writeFields(os.path.join(gradDirectory, f"fullGradient{Nfeval}.hdf5"),
                    {"fullGradient": gradJ, "illumination": gb.illumination}, solver)

    if gDn == 1:
        gradJ = gradJ / np.max(np.abs(gradJ))
    elif gDn == 2:
        gradJ = gradJ / np.linalg.norm(gradJ)

    Nfeval += 1
    if computeFullCF is True:
        if debug and rank == 0:
            writeCostFunction(J, filename=f"fullCostFunction{frequencyTag(gb.freqList[ifreq])}.txt")
        return gradJ, J
    return gradJ
