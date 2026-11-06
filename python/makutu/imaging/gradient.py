"""
Gradient of the least-squares cost function (adjoint-state method) over the shots
of the acquisitions, for the optimisers of makutu.imaging.Minimizer.

Per shot: forward run keeping the second time derivative of the pressure
(``solver.runForward``), residual against the observed SEG-Y data, adjoint run with
the residual injected at the receivers and imaging condition
(``solver.computeGradient``). The gradients of the shots are summed in memory.
The model and the gradient are flat arrays per element, in seismic order
(``solver.getVelocityModel``), in the parametrisation ``model`` ("c", "1/c", "1/c2").
"""

import os
import numpy as np

from mpi4py import MPI
from mpi4py.util import pkl5

import h5py

from ..forwardBackward.forward import forward
from ..forwardBackward.backward import backward
from ..forwardBackward.modelio import writeModel
from ..forwardBackward.costFunction import computeFullCostFunction, frequencyTag, \
                                           ComputeCostFunction, \
                                           ComputeResidual

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
        startModelFile : str, optional
            HDF5 file with the model (dataset "velocity", parametrisation ``model``) \
            loaded before the first shot; default None keeps the solver model
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
            Default: ``solver.snapshotStepFor(fmax)`` with the filter frequency

    Returns:
    --------
        np.ndarray : gradient summed over the shots
    """
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    solver = gb.solver
    freqFilter = gb.freqList[ifreq]
    solver.setModelForGradient(model)
    if snapshotStep is None:
        snapshotStep = solver.snapshotStepFor(None if str(freqFilter) == "all" else float(freqFilter))
    drootname, dformat, dext = dinfo.split(";")

    gradient = None
    for ishot, shot in enumerate(gb.listOfAcquisition[iacq].shots):
        solver.resetWaveField()
        if ishot == 0 and startModelFile is not None:
            solver.updateModel(startModelFile, gb.minModel, gb.maxModel)

        solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())
        solver.evaluateSource()
        solver.filterSource(freqFilter)

        forward(solver, shot, comm, snapshotStep)
        simulated = solver.getPressureAtReceivers()

        rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}"
        if iterfwi is not None:
            rootname += f"_{iterfwi}"
        if exportSeismicTrace:
            SEGYTraceOutput(simulated, rootname=rootname, directory=traceDirectory).export(
                receiverCoords=shot.getReceiverCoords(), sourceCoords=shot.getSourceCoords()[0],
                dt=solver.dtSeismoRecorded, comm=comm)

        filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"
        residual = ComputeResidual(filename, dt=solver.dtSeismoRecorded, fmax=freqFilter).computeResidual(simulated)
        if exportResidual:
            SEGYTraceOutput(residual, rootname=rootname, directory="residual").export(
                receiverCoords=shot.getReceiverCoords(), sourceCoords=shot.getSourceCoords()[0],
                dt=solver.dtSeismoRecorded, comm=comm)
        if computeCF and rank == 0:
            ComputeCostFunction(directory=pCostDir, dt=solver.dtSeismoRecorded,
                                rootname="partialCost_" + shot.id).computePartialCostFunction(residual)

        partial = backward(solver, residual, model)
        gradient = partial if gradient is None else gradient + partial

        shot.flag = "Done"
        if rank == 0:
            print("Shot", shot.id, "done\n")
        comm.Barrier()

    return gradient


def gradf(x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".",
          exportResidual=False, ifreq=0, model="c", pCostDir="partialCostFunction",
          dinfo="seismoTrace_shot;5;.sgy", gDn=0, computeFullCF=False):
    """
    Gradient of the cost function for the model x (single process: the model file
    is written by makutu.forwardBackward.modelio.writeModel)

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

    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    if hasattr(gb, "model"):
        gb.model[:] = x[:]

    modelFile = "velocityModel.hdf5"
    writeModel(modelFile, x, comm)
    pdir = f"{pCostDir}_{Nfeval}_G"

    gradJ = None
    for i in range(len(gb.listOfAcquisition)):
        g = gradientComputation(i, dataDirectory, exportSeismicTrace, traceDirectory, exportResidual, ifreq,
                                model, startModelFile=modelFile, iterfwi=Nfeval, pCostDir=pdir,
                                dinfo=dinfo, computeCF=computeFullCF)
        gradJ = g if gradJ is None else gradJ + g

    if getattr(gb, "mode", "REL") == "DBG" and comm.Get_rank() == 0:
        os.makedirs(gradDirectory, exist_ok=True)
        with h5py.File(os.path.join(gradDirectory, f"fullGradient{Nfeval}.hdf5"), "w") as h:
            h.create_dataset("fullGradient", data=gradJ)

    if gDn == 1:
        gradJ = gradJ / np.max(np.abs(gradJ))
    elif gDn == 2:
        gradJ = gradJ / np.linalg.norm(gradJ)

    Nfeval += 1
    if computeFullCF is True:
        nfiles = len(gb.acquisition.shots)
        save = getattr(gb, "mode", "REL") == "DBG"
        J = computeFullCostFunction(nfiles, pdir, sfile=f"fullCostFunction{frequencyTag(gb.freqList[ifreq])}.txt",
                                    save=save)
        return gradJ, J
    return gradJ
