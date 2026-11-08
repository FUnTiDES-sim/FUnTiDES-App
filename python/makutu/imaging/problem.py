"""
State of a full-waveform inversion: solver, acquisitions, model bounds, filter frequencies,
run mode, illumination and counters.

It replaces the module of global variables of the former makutu: the functions of the
inversion (:func:`makutu.forwardBackward.directSimulation`, :func:`makutu.forwardBackward.func`,
:func:`makutu.imaging.gradf`, :func:`makutu.forwardBackward.callback`) take the problem as
first argument, and the problem exposes them as methods with the signatures expected by the
optimisers (``func(x, *args)``, ``jac(x, *args, computeFullCF=...)``, ``callback(x)``)::

    problem = FWIProblem(solver, acquisition, minModel=1400, maxModel=5000, freqList=[4, 6])
    problem.directSimulation(0, exportSeismicTrace=True, traceDirectory="obs")
    minimizer = Minimizer(problem.func, method="CGPR", jac=problem.gradf, callback=problem.callback)

Several problems can coexist (each with its own solver). The counters number the output
files: ``iteration`` is the number of the next ``velocityModel<n>.hdf5`` (set it to resume a
run), ``nfunc`` and ``ngrad`` count the cost and gradient evaluations (debug files).
"""

import math


class FWIProblem:
    """
    Attributes
    -----------
        solver : AcousticSolver
            Solver (the whole model on every rank: no domain decomposition for the FWI)
        acquisition : Acquisition
            Full acquisition
        listOfAcquisition : list of Acquisition
            Acquisitions whose shots are computed (default ``[acquisition]``)
        minModel, maxModel : float
            Bounds of the model, in the parametrisation of the inversion
        freqList : list
            Filter frequencies ("all": no filter), indexed by ``ifreq``
        mode : str
            "REL", or "DBG" (keeps the full gradients and the cost files)
        model : np.ndarray or None
            Current model (updated by :meth:`gradf`)
        illumination : np.ndarray or None
            Illumination summed over the shots by the last :meth:`gradf`
        dataSelection : makutu.forwardBackward.dataprocessing.DataSelection or None
            Weights of the residual (offsets, mute, time window)
        sourceCorrection : dict or None
            Estimated source filter per frequency index (:meth:`estimateSource`)
        iteration : int
            Number of the next model file ``velocityModel<iteration>.hdf5``
        nfunc, ngrad : int
            Numbers of the next cost and gradient evaluations
    """

    def __init__(self, solver, acquisition, listOfAcquisition=None, minModel=-math.inf, maxModel=math.inf,
                 freqList=("all",), mode="REL", model=None, iteration=1, dataSelection=None):
        if mode not in ("REL", "DBG"):
            raise ValueError(f"mode {mode!r}: 'REL' or 'DBG'")
        self.solver = solver
        self.acquisition = acquisition
        self.listOfAcquisition = list(listOfAcquisition) if listOfAcquisition is not None else [acquisition]
        self.minModel, self.maxModel = minModel, maxModel
        self.freqList = list(freqList)
        self.mode = mode
        self.model = model
        self.illumination = None
        self.dataSelection = dataSelection
        self.sourceCorrection = None
        self.iteration = int(iteration)
        self.nfunc = 1
        self.ngrad = 1

    @property
    def debug(self):
        return self.mode == "DBG"

    @property
    def totalShots(self):
        return len(self.acquisition.shots)

    def release(self):
        """Drop the solver: every Kokkos view must be released before ``kokkos.finalize``."""
        self.solver = None

    # Functions of the inversion with this problem
    def directSimulation(self, iacq, *args, **kwargs):
        """:func:`makutu.forwardBackward.directSimulation` with this problem."""
        from ..forwardBackward.forward import directSimulation
        return directSimulation(self, iacq, *args, **kwargs)

    def estimateSource(self, dataDirectory, ifreq=0, **kwargs):
        """:func:`makutu.forwardBackward.forward.estimateSource` with this problem."""
        from ..forwardBackward.forward import estimateSource
        return estimateSource(self, dataDirectory, ifreq, **kwargs)

    def func(self, x, *args, **kwargs):
        """Cost function, :func:`makutu.forwardBackward.func` with this problem."""
        from ..forwardBackward.forwardBackward import func
        return func(self, x, *args, **kwargs)

    def gradf(self, x, *args, **kwargs):
        """Gradient (and cost), :func:`makutu.imaging.gradf` with this problem."""
        from .gradient import gradf
        return gradf(self, x, *args, **kwargs)

    def callback(self, x):
        """Model file after an iteration, :func:`makutu.forwardBackward.callback`."""
        from ..forwardBackward.forwardBackward import callback
        return callback(self, x)
