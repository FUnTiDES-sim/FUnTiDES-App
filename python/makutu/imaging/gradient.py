import os
import numpy as np

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py.util import pkl5

import h5py

from ..forwardBackward.forward import forward
from ..forwardBackward.backward import backward
from ..forwardBackward.costFunction import computeFullCostFunction, \
                                           ComputeCostFunction, \
                                           ComputeResidual

from ..tools.interpolate import linearInterpolation
from ..output.SEGYTraceOutput import SEGYTraceOutput
from ..imaging import globals as gb
from ..tools.filter import frequencyFilter

Nfeval = 1

def gradientComputation(iacq, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".", exportResidual=False, ifreq=0, model="c", startModelFile=None, iterfwi=None, pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy", computeCF=False):
    """ Perform a partial gradient computation on all shots in a given acquisition

    Parameters:
    -----------
        iacq : int
            Index of acquisition in listOfAcquisition
        dataDirectory : str
            Directory containing data for residual/costFunction computation
        gradDirectory : str, optional
            Directory for the output of the partial gradients \
            Default is 'partialGradient'
        exportSeismicTrace : bool, optional
            Whether or not to export seismic trace to .sgy \
            Default is True
        traceDirectory : str, optional
            Directory where to export the seismic traces \
            Default is current directory
        exportResidual : bool, optional
            Export or not the residual \
            Default is False
        ifreq : int, optional
            Index of the frequency in gb.freqList \
            Default is 0
        model : str, optional
            Model for the velocity \
            Default is 'c'
        startModelFile : str, optional
            HDF5 file containing a starting velocity model \
            Default is None (uses the acquisition model)
        iterfwi : int, optional
            Index of FWI iteration \
            Default is None
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
    for shot in gb.listOfAcquisition[iacq].shots:

        solver.resetWaveField()

        if ishot == 0 and startModelFile is not None:
            solver.updateModel(startModelFile, low, high)

        solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())
        solver.evaluateSource()
        solver.filterSource(freqFilter)

        # set compute cost function callback
        if computeCF:
            if iterfwi is not None:
                pCostDir += f"_{iterfwi}_G"
            computeCostFunction = ComputeCostFunction( directory=pCostDir,
                                                       dt=solver.dtSeismo,
                                                       rootname="partialCost_"+shot.id
            ).computePartialCostFunction
        else:
            computeCostFunction = None
            
        # set residual function callback
        drootname, dformat, dext = dinfo.split(";")
        filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"

        computeResidual = ComputeResidual( filename, 
                                           filtering=[frequencyFilter, freqFilter, solver.dtSeismo],
                                           interpolate=[linearInterpolation, solver.maxTime, solver.dt, solver.dtSeismo] 
        ).computeResidual

        # Compute shot gradient
        partialGradientComputation(solver, shot, dataDirectory, comm, exportSeismicTrace, traceDirectory, gradDirectory, computeResidual, exportResidual, iterfwi=iterfwi, computeCF=computeCostFunction)

        #Update shot flag
        shot.flag = "Done"
        if rank == 0:
            print("Shot", shot.id, "done\n")

        ishot += 1
        comm.Barrier()


def partialGradientComputation(solver, shot, dataDirectory, comm, exportSeismicTrace=True, traceDirectory=".", gradDirectory="partialGradient", computeResidual=None, exportResidual=False, iterfwi=None, computeCF=None):
    """Perform a forward simulation and handle exports for one shot using Acoustic Solver

    Parameters
    -----------
        solver : AcousticSolver
            AcousticSolver object
        shot : Shot
            Shot object
        comm : MPI_COMM
            MPI communicators
        exportSeismicTrace : bool, optional
            Export seismic trace to .sgy \
            Default is True
        traceDirectory: str, optional
            Directory where to export the seismic traces. If not set default is current directory
        computeResidual : callback
            Callback to compute the residual.
            Default is None (not computed)
        exportResidual : bool
            Export the residual \
            Default is False
        computeCF : callback
            Callback to compute the cost function. 
            Default is None (not computed)
    """
    global Nfeval

    rank = comm.Get_rank()

    if solver.dtWaveField is None:
        raise ValueError("You must provide a dtWaveField in the solver for the gradient computation.")
    
    outputWaveFieldInterval = int(round(solver.dtWaveField/solver.dt))

    forward(solver, shot, comm, outputWaveFieldInterval)

    wavefield = solver.getWaveField()

    if exportSeismicTrace:
        rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}" 
        if iterfwi is not None:
            rootname += f"_{iterfwi}"

        pressOut = SEGYTraceOutput(wavefield,
                                   rootname=rootname,
                                   directory=traceDirectory,
                    )
        pressOut.export(receiverCoords=shot.getReceiverCoords(),
                        sourceCoords=shot.getSourceCoords()[0],
                        dt=solver.dtSeismo,
                        comm=comm)

    simulatedData = solver.getFullWaveFieldAtReceivers(comm)

    residual = computeResidual( simulatedData )
    
    if exportResidual:
        rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}" 
        if iterfwi is not None:
            rootname += f"_{iterfwi}"
        
        residualOut = SEGYTraceOutput(residual,
                                      rootname=rootname,
                                      directory="residual",
        )
        residualOut.export(receiverCoords=shot.getReceiverCoords(),
                           sourceCoords=shot.getSourceCoords()[0],
                           dt=solver.dtSeismo,
                           comm=comm)
    
    if rank == 0 and computeCF is not None:
        computeCF( residual )

    solver.updateSourceAndReceiversFromCoords(shot.getReceiverCoords(), [])
    solver.updateSourceValue(residual)

    backward(solver, shot, comm, gradDirectory, outputWaveFieldInterval)


def computeFullGradient(pGradDir="partialGradient", iterfwi=None):
    """
    Compute the full gradient by summing all partial gradients. \
    Note that this routine only use one process to run. This is to avoid IO issues and is fast enough to sum all partialGradients since it summarizes as a partialGradient is ready.

    Parameters
    ----------
        pGradDir : str
            Path to the directory containing all partial gradients. Default look for a directory 'partialGradient' in the current location
        iterfwi : int
            If running an FWI simulation, index on step of FWI. Default to None.
    """
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    if rank == 0:
        acquisition = gb.acquisition
        directory = os.fsencode(pGradDir)

        limited_aperture_flag = acquisition.limited_aperture
        nfiles = len(acquisition.shots)
        read = np.zeros((nfiles), dtype=int) 

        if acquisition.mesh.isSet == False:
            acquisition.loadMesh()

        totalCells = acquisition.mesh.numberOfCells

        nf = '' if iterfwi is None else str(iterfwi)
        h5Filename = f"fullGradient{nf}.hdf5"
        h5F = h5py.File(h5Filename, "w")

        if len(list(h5F.keys())) == 0:
            h5F.create_dataset("fullGradient", data=np.zeros(totalCells), dtype='d', chunks=True, maxshape=(totalCells,))

        while not all(read):
            if os.path.exists(pGradDir):

                file_list = os.listdir(directory)
                if len(file_list):
                    for filename in file_list:
                        filename = os.fsdecode(filename)
                        if filename.endswith("hdf5") and filename.startswith("partialGradient_ready"):
                            shotId = int(os.path.splitext(filename.split("_")[-1])[0])
                            if not read[shotId-1]:
                                try:
                                    # Try reading the file
                                    h5p = h5py.File(os.path.join(pGradDir, filename), 'r')
                                    if not limited_aperture_flag:
                                        h5F["fullGradient"][:] += h5p["partialGradient"][:]
                                    else:
                                        shot = acquisition.shots[shotId-1]
                                        shot.mesh.setMeshProperties()
                                        mesh = shot.getMesh()
    
                                        extractToGlobal = mesh.getExtractToGlobal()
                                        h5F["fullGradient"][extractToGlobal[:]] += h5p["partialGradient"][:]
    
                                    shotId = int(os.path.splitext(filename.split("_")[-1])[0])
                                    read[shotId-1] = 1
                                    h5p.close()
    
                                    if gb.mode != "DBG":
                                        os.remove(os.path.join(pGradDir, filename))
                                    else:
                                        os.remove(os.path.join(pGradDir, f"bkp_"+filename))
                                        shutil.move(os.path.join(pGradDir, filename),os.path.join(pGradDir, f"bkp_"+filename))
                                except:
                                    # If not yet accessible
                                    # or being read by another rank
                                    pass 

        h5F.close()
        if gb.mode != "DBG":
            os.rmdir(pGradDir)

    comm.Barrier()


def gradf( x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".", exportResidual=False, ifreq=0, model="c", pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy", gDn=0, computeFullCF=False ):
    """
    Compute the gradient

    Parameters
    -----------
        x : numpy array
            Model
        dataDirectory : str
            Directory containing data for residual/costFunction computation
        gradDirectory : str, optional
            Directory for the output of the partial gradients
            Default is 'partialGradient'
        exportSeismicTrace : bool, optional
            Whether or not to export seismic trace to .sgy
            Default is True
        traceDirectory : str, optional
            Directory where to export the seismic traces
            Default is current directory
        exportResidual : bool, optional
            Export or not the residual
            Default is False
        ifreq : int, optional
            Index of the frequency in gb.freqList
            Default is 0
        model : str, optional
            Model for the velocity
            Default is 'c'
        pCostDir : str, optional
            Partial cost function directory
            Default is `partialCostFunction`
        gDn : int, optional
            Normalize the gradient
            Default is 0, no normalization
            If 1, the gradient is normalized by its maximum absolute value
            If 2, by its 2-norm
        computeFullCF : bool, optional
            Compute and return the full cost function
            Default is False

    Returns
    --------
        gradJ : float
            Full gradient
        J : float
            Full cost function if computeFullCF is set to True
    """
    global Nfeval

    gb.model[:] = x[:]
    gb.acquisition.loadMesh()
    totalCells = gb.acquisition.getMesh().numberOfCells

    modelFile = "velocityModel.hdf5"
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    h = h5py.File(modelFile, "w", driver="mpio", comm=comm)
    
    h.create_dataset("velocity", data=x[:], dtype='d', chunks=True, maxshape=(totalCells,))
    h.close()
   
    if Nfeval == 1 and gb.mode == "DBG":
        with h5py.File("velocityModel0.hdf5", "w", driver="mpio", comm=comm) as hf:
            hf.create_dataset("velocity", data=x[:], dtype="d", chunks=True, maxshape=(totalCells,))

 
    if gb.mode == "DBG":
        gradDirectory += f"_{Nfeval}"
        fullGradFilename = f"fullGradient{Nfeval}.hdf5"
        computeFullGradient(gradDirectory, Nfeval)
    else:
        fullGradFilename = f"fullGradient.hdf5"
        computeFullGradient(gradDirectory)

    
    for i in range(len(gb.listOfAcquisition)):
        gradientComputation(i,
                            dataDirectory,
                            gradDirectory,
                            exportSeismicTrace,
                            traceDirectory,
                            exportResidual,
                            ifreq,
                            model,
                            modelFile,
                            iterfwi=Nfeval,
                            pCostDir=pCostDir,
                            dinfo=dinfo,
                            computeCF=computeFullCF)

    h5F = h5py.File(fullGradFilename, "r", driver="mpio", comm=comm)
    gradJ = h5F["fullGradient"][:]
    h5F.close()
    
    pCostDir += f"_{Nfeval}_G"
    Nfeval += 1

    # Choice to normalize gradient before output
    #gDn=1
    print("Choice of normalization. ")
    print("gDn is: {}\n".format(gDn))

    if gDn == 1:
        # max abs value
        gradJ = gradJ/max( abs(gradJ) )
    elif gDn == 2:
        # l2 norm
        gradJ = gradJ/np.linalg.norm(gradJ)
        
    if computeFullCF is True:
        nfiles = len( gb.acquisition.shots )

        save = True if gb.mode == "DBG" else False
        J = computeFullCostFunction( nfiles, pCostDir, sfile=f"fullCostFunction{int(gb.freqList[ifreq])}.txt", save=save )
        
        return gradJ, J
    
    else:

        return gradJ

