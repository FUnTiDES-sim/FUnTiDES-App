import os
import numpy as np

import mpi4py
from mpi4py import MPI
from mpi4py.util import pkl5
import h5py

from ..imaging import globals as gb
from .forward import directSimulation
from .modelio import writeModel
from .costFunction import computeFullCostFunction, frequencyTag

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

    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    freqFilter = gb.freqList[ifreq]

    modelFile = "velocityModel.hdf5"
    writeModel(modelFile, x, comm)

    pCostDir += f"_{feval}_F"
    for i in range(len(gb.listOfAcquisition)):
        directSimulation(i, exportSeismicTrace, traceDirectory, True, dataDirectory, exportResidual, model, modelFile, ifreq, pCostDir, dinfo)

    nfiles = len(gb.acquisition.shots)

    save = True if gb.mode == "DBG" else False
    J = computeFullCostFunction(nfiles, pCostDir, sfile=f"fullCostFunction{frequencyTag(freqFilter)}.txt", save=save)

    if comm.Get_rank() == 0:
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
