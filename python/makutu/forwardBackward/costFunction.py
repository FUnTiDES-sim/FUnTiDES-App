import os
import numpy as np

from mpi4py import MPI
from mpi4py.util import pkl5

import segyio

from ..imaging import globals as gb
from ..tools.filter import frequencyFilter


class ComputeCostFunction:
    """
    Least-squares cost function of one shot, J = dt / 2 * sum(residual^2),
    written to ``<directory>/<rootname>.txt``.
    """
    def __init__(self, directory, dt, rootname):
        """
        Parameters
        ------------
            directory : str
                Saving directory
            dt : float
                Sampling of the residual [s] (``solver.dtSeismoRecorded``)
            rootname : str
                Rootname of the cost function file; it must end with ``_<shot id>``
        """
        self.directory = directory
        self.dt = dt
        self.rootname = rootname

    def computePartialCostFunction(self, residual):
        """
        Compute and write the cost function of one shot.

        Parameters
        ------------
            residual : array of float, shape (n_samples, n_receivers)

        Returns
        --------
            float : the partial cost
        """
        os.makedirs(self.directory, exist_ok=True)
        partialCost = self.dt * float(np.sum(np.square(residual))) / 2.0
        with open(os.path.join(self.directory, self.rootname + ".txt"), 'w') as f:
            f.write(repr(partialCost))
        return partialCost


class ComputeResidual:
    """
    Residual between observed data (SEG-Y file) and simulated seismograms:
    observed - simulated, on the time axis of the simulation.
    """
    def __init__(self, dataFilename, dt, fmax="all"):
        """
        Parameters
        -----------
            dataFilename : str
                SEG-Y file of the observed data (one trace per receiver)
            dt : float
                Sampling of the simulated seismograms [s] (``solver.dtSeismoRecorded``);
                the observed data are resampled on it
            fmax : float or "all"
                Low-pass filter applied to the observed data (as to the source by
                ``solver.filterSource``); default "all" (no filter)
        """
        self.observationData = dataFilename
        self.dt = dt
        self.fmax = fmax

    def __readObservationData(self):
        """Observed traces (n_samples, n_receivers), filtered, and their sampling."""
        with segyio.open(self.observationData, 'r', ignore_geometry=True) as f:
            data = np.asarray(f.trace.raw[:], dtype=np.float64).T
            dtObs = segyio.tools.dt(f) * 1e-6
        if str(self.fmax) != "all":
            data = frequencyFilter(data, float(self.fmax), dtObs)
        self.data, self.dtObs = data, dtObs

    def computeResidual(self, simulatedData):
        """
        Parameters
        ----------
            simulatedData : array, shape (n_samples, n_receivers)
                Simulated seismograms sampled every ``dt``

        Returns
        --------
            residual : array, shape of ``simulatedData``
                observed - simulated; the observed traces are linearly interpolated
                on the simulation times (zero after their last sample)
        """
        if not hasattr(self, "data"):
            self.__readObservationData()
        simulatedData = np.asarray(simulatedData, dtype=np.float64)
        nt, nrcv = simulatedData.shape
        if self.data.shape[1] != nrcv:
            raise ValueError(f"{self.observationData}: {self.data.shape[1]} traces, "
                             f"{nrcv} simulated receivers")
        tSim = np.arange(nt) * self.dt
        tObs = np.arange(self.data.shape[0]) * self.dtObs
        # times beyond the data (beyond rounding: 5 * 0.0004 != 0.002) get zero
        beyond = tSim > tObs[-1] + 1e-6 * min(self.dt, self.dtObs)
        tSim = np.minimum(tSim, tObs[-1])
        observed = np.empty_like(simulatedData)
        for i in range(nrcv):
            observed[:, i] = np.interp(tSim, tObs, self.data[:, i])
        observed[beyond] = 0.0
        return observed - simulatedData


def computeFullCostFunction(nfiles, pCostDir="partialCostFunction", save=True, sfile="fullCostFunction.txt"):
    """Compute the full cost function by summing the partial cost functions of the shots

    Parameters
    ----------
        nfiles : int
            Number of shots; the files are ``<pCostDir>/<name>_<shot id>.txt``
            with shot ids 1 to nfiles
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

    Raises
    ------
        FileNotFoundError
            If the directory or the cost of a shot is missing
    """
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    error = None
    fullCost = None
    if rank == 0:
        costs = {}
        if os.path.isdir(pCostDir):
            for filename in os.listdir(pCostDir):
                path = os.path.join(pCostDir, filename)
                shotId = int(os.path.splitext(filename.split("_")[-1])[0])
                if os.stat(path).st_size > 0:
                    with open(path) as costFile:
                        costs[shotId] = float(costFile.readline())
        missing = sorted(set(range(1, nfiles + 1)) - set(costs))
        if not os.path.isdir(pCostDir):
            error = f"directory {pCostDir} with the cost functions not found"
        elif missing:
            error = f"{pCostDir}: no cost function for the shots {missing}"
        else:
            fullCost = sum(costs.values())
            if getattr(gb, "mode", "REL") != "DBG":
                for filename in os.listdir(pCostDir):
                    os.remove(os.path.join(pCostDir, filename))
                os.rmdir(pCostDir)
    error, fullCost = comm.bcast((error, fullCost), root=0)
    if error is not None:
        raise FileNotFoundError(error)

    if save is True and rank == 0:
        writeCostFunction(fullCost, filename=sfile)

    return fullCost


def frequencyTag(fmax):
    """Text of a filter frequency for file names: "all", or the frequency (10.0 -> "10")."""
    return "all" if str(fmax) == "all" else f"{float(fmax):g}"


def writeCostFunction(J, filename="fullCostFunction.txt"):
    """Append the cost function value to a file

    Parameters
    -----------
        J : float
            CostFunction value
        filename : str, optional
            Name of the file to write on
    """
    with open(filename, 'a') as f:
        f.write(f"{str(J)}\n")
