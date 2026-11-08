import os
import numpy as np

import mpi4py
from mpi4py import MPI
from mpi4py.util import pkl5
import h5py

from ..imaging import globals as gb
from .forward import directSimulation
from .modelio import writeModel
from .costFunction import frequencyTag, writeCostFunction
from .parallel import checkShotParallel, sumOverShots

iter = 1
feval = 1


def func(x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".", exportResidual=False, ifreq=0, model="c", pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy",*args):
    """
    Callable function returning the full cost function

    Parameters
    -----------
        x : array
            data
        dataDirectory : str
            Directory containing data for residual/costFunction computation
        gradDirectory : str
            Directory for the output of the partial gradients \
            Default is "partialGradient"
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy \
            Default is True
        traceDirectory : str
            Directory where to export the seismic traces. If not set, default is current directory
        exportResidual : bool
            Wheter or not to export the residual. Default is False
        ifreq : int
            Index of the frequency in gb.freqList. Default to 0
        model : str
            Model to use
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`
    
    Returns
    --------
        J : float
            Full cost function
    """
    global feval

    solver = gb.solver
    checkShotParallel(solver)
    freqFilter = gb.freqList[ifreq]
    debug = getattr(gb, "mode", "REL") == "DBG"
    pdir = f"{pCostDir}_{feval}_F" if debug else None          # partial cost files in debug mode

    J = 0.0
    for i in range(len(gb.listOfAcquisition)):
        J += directSimulation(i, exportSeismicTrace, traceDirectory, True, dataDirectory, exportResidual, model,
                              x, ifreq, pdir, dinfo)
    J = sumOverShots(J, solver)

    if MPI.COMM_WORLD.Get_rank() == 0:
        if debug:
            writeCostFunction(J, filename=f"fullCostFunction{frequencyTag(freqFilter)}.txt")
        print(f"\nFrequency {freqFilter} - F = {J}\n")
    feval += 1
    return J


def callback(x):
    """
    Create the velocity model HDF5 file corresponding to the iteration done

    Parameters
    -----------
        x : array
            Velocity model
    """
    global iter

    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    modelFile = "velocityModel" + str(iter) + ".hdf5"
    writeModel(modelFile, x, comm)
    
    if rank == 0:
        print( "*"*80, f"Velocity model {iter} saved as `{modelFile}`", "*"*80, sep="\n"*2 )

    iter += 1
    
    comm.Barrier()
