import os
import numpy as np

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py.util import pkl5

import segyio

from ..imaging import globals as gb

class ComputeCostFunction:
    """
    Class containing methods specific to the computation of the cost function
    """
    def __init__( self, directory, dt, rootname ):
        """
        Parameters
        ------------
            directory : str
                Saving directory
            dt : float
                Time step in s
            rootname : str
                Rootname for the saved cost function file
        """
        self.directory = directory
        self.dt = dt
        self.rootname = rootname


    def computePartialCostFunction( self, residual ):
        """
        Compute the cost function for one shot given a residual

        Parameters
        ------------
            residual : array of float
                Array containing the residual

        """
        os.makedirs( self.directory, exist_ok=True )

        partialCost = np.sum(np.square(np.linalg.norm(residual,axis=0)))

        partialCost = self.dt * partialCost / 2

        with open(os.path.join(self.directory, self.rootname+".txt"), 'w') as f:
            f.write(str(partialCost))   


class ComputeResidual:
    """
    Contains methods to compute the residual 
    """
    def __init__( self, dataFilename, filtering=None, interpolate=None ):
        """
        Parameters
        -----------
            dataFilename : str
                Filenames of observation data
            filtering : list
                List containing [callback, args] to filter the observation data
                Default is None (no filtering)
            interpolate : list
                List containing [callback, args] to interpolate the residual
                Default is None (no interpolation)
        """
        self.observationData = dataFilename
        self.filter = filtering
        self.interpolate = interpolate


    def __readObservationData( self ):
        """
        Extract the observation data from SEGY file

        Parameters
        -----------
            filename : str
                Filename
        """
        with segyio.open( self.observationData, 'r', ignore_geometry=True) as f:
            data = f.trace.raw[:].T

        self.data = data


    def computeResidual( self, simulatedData):
        """Compute the difference between the result of the forward problem and the real data

        Parameters
        ----------
            simu : array
                Path to the observed data file
            data : array
                Simulated data

        Returns
        --------
            residual : array
                Contains the residual between the simulated data and the observed data
        """
        if not hasattr( self, "data"):
            self.__readObservationData()

        data = self.data

        # Apply filter if requested
        if self.filter is not None:
            filtering, *fargs = self.filter
            data = filtering( data, *fargs )

        residual = data - simulatedData

        # Interpolate if requested
        if self.interpolate is not None:
            interpolate, *args = self.interpolate
            data = interpolate( data, *args )

        return residual


def computeFullCostFunction(nfiles, pCostDir="partialCostFunction", save=True, sfile="fullCostFunction.txt"):
    """Compute the full cost function by summing all partial cost function

    Parameters
    ----------
        nfiles : int
            Number of files/shots in total
        pCostDir : str, optional
            Directory containing the partial cost function files \
            Default is `partialCostFunction`
        save : bool, optional
            Save the full cost function value in a file \
            Default is True
        sfile : str, optional
            If save if True, filename of full cost function \
            Default is `fullCostFunction.txt`

    Returns
    --------
        fullCost : float
            Sum of all partial cost functions values
    """
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    if rank == 0:
        directory_in_str = pCostDir
        directory = os.fsencode(directory_in_str)

        fullCost = 0
        read = np.zeros((nfiles), dtype=int)
        while not all(read):
            if os.path.exists(directory_in_str):
                file_list = os.listdir(directory)

                if len(file_list) != 0:
                    for filename in file_list:
                        filename = os.fsdecode(filename)
                        shotId = int(os.path.splitext(filename.split("_")[-1])[0])
                        if os.stat(os.path.join(directory_in_str, filename)).st_size > 0 and not read[shotId-1]:
                            with open(os.path.join(directory_in_str, filename), "r") as costFile:
                                partialCost = float(costFile.readline())

                                fullCost += partialCost
                                read[shotId-1] = 1

                                if gb.mode != "DBG":
                                    os.remove(os.path.join(directory_in_str, filename))
                else:
                    continue
            else:
                raise ValueError("directory %s with cost functions not found !"%(directory_in_str))

        if gb.mode != "DBG":
            os.rmdir(directory_in_str)
    else:
        fullCost = None
    fullCost = comm.bcast(fullCost, root=0)

    if save is True:
        writeCostFunction(fullCost, filename=sfile)

    return fullCost


def writeCostFunction(J, filename="fullCostFunction.txt"):
    """Write the cost function on disk
    
    Parameters
    -----------
        J : float
            CostFunction value
        filename : str, optional
            Name of the file to write on
    """
    with open(filename, 'a') as f:
        f.write(f"{str(J)}\n")
