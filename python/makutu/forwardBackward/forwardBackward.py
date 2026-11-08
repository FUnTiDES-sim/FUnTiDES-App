"""Cost function and callback of the optimisers (:class:`makutu.imaging.FWIProblem`)."""

from mpi4py import MPI
from mpi4py.util import pkl5

from .forward import directSimulation
from .modelio import writeModel
from .costFunction import frequencyTag, writeCostFunction
from .parallel import checkShotParallel, sumOverShots


def func(problem, x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".", exportResidual=False, ifreq=0, model="c", pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy",*args):
    """
    Cost function over all the shots of the problem (summed over the MPI ranks)

    Parameters
    -----------
        problem : makutu.imaging.FWIProblem
            State of the inversion
        x : array
            Model, flat per element in seismic order, parametrisation ``model``
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
            Index of the frequency in problem.freqList. Default to 0
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
    solver = problem.solver
    checkShotParallel(solver)
    freqFilter = problem.freqList[ifreq]
    debug = problem.debug
    pdir = f"{pCostDir}_{problem.nfunc}_F" if debug else None  # partial cost files in debug mode

    J = 0.0
    for i in range(len(problem.listOfAcquisition)):
        J += directSimulation(problem, i, exportSeismicTrace, traceDirectory, True, dataDirectory, exportResidual, model,
                              x, ifreq, pdir, dinfo)
    J = sumOverShots(J, solver)

    if MPI.COMM_WORLD.Get_rank() == 0:
        if debug:
            writeCostFunction(J, filename=f"fullCostFunction{frequencyTag(freqFilter)}.txt")
        print(f"\nFrequency {freqFilter} - F = {J}\n")
    problem.nfunc += 1
    return J


def callback(problem, x):
    """
    Write the model after an iteration to ``velocityModel<problem.iteration>.hdf5`` and
    increment ``problem.iteration``.

    Parameters
    -----------
        problem : makutu.imaging.FWIProblem
            State of the inversion
        x : array
            Model
    """
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()

    modelFile = f"velocityModel{problem.iteration}.hdf5"
    writeModel(modelFile, x, comm, problem.solver)

    if rank == 0:
        print("*" * 80, f"Velocity model {problem.iteration} saved as `{modelFile}`", "*" * 80, sep="\n" * 2)

    problem.iteration += 1

    comm.Barrier()
