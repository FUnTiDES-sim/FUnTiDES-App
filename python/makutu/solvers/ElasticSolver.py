import numpy as np

import kokkos
import pyfuntides.solver as FTSolver

from .WaveSolver import WaveSolver
from ..tools.kokkosarrays import funtides_array


class ElasticSolver(WaveSolver):
    """
    ElasticSolver: FUnTiDES SEM elastic wave simulation.

    Manages the displacement wavefields (Ux, Uy, Uz; previous and current
    time step, one [n_dof] array each) and accumulates per-component
    seismograms at receiver nodes.

    Attributes
    ----------
        kk_wavefield_x : tuple of kokkos arrays
            Displacement in X, (previous, current), [n_dof] each
        kk_wavefield_y : tuple of kokkos arrays
            Displacement in Y, (previous, current)
        kk_wavefield_z : tuple of kokkos arrays
            Displacement in Z, (previous, current)
        kk_RHSTermx : kokkos array
            Source signal for X component [n_rhs, n_time_steps]
        kk_RHSTermy : kokkos array
            Source signal for Y component [n_rhs, n_time_steps]
        kk_RHSTermz : kokkos array
            Source signal for Z component [n_rhs, n_time_steps]
        sourceComponents : tuple of str
            Components to apply source on. Default is ('z',) (vertical).
        kk_model_vs : kokkos array, optional
            Kokkos array backing the Vs model
        _seismogram_x : np.ndarray
            Displacement at receivers, X component [n_seismo_steps, n_receivers]
        _seismogram_y : np.ndarray
            Displacement at receivers, Y component
        _seismogram_z : np.ndarray
            Displacement at receivers, Z component
        _seismo_step : int
            Number of time steps between seismogram samples
        useDAS : bool
            Flag for DAS acquisition (not yet implemented for FUnTiDES)
    """

    def __init__(self, sourceComponents=("z",), kk_model_vs=None, **kwargs):
        """
        Parameters
        ----------
            sourceComponents : tuple of str, optional
                Which displacement components to drive with the source.
                Any combination of 'x', 'y', 'z'. Default is ('z',).
            kk_model_vs : kokkos array, optional
                Kokkos array of the Vs model for in-place updates
            kwargs : passed to WaveSolver.__init__
                physic_type is set to ELASTIC automatically.
        """
        kwargs.setdefault("physic_type", FTSolver.PhysicType.ELASTIC)
        super().__init__(**kwargs)

        self.sourceComponents = tuple(c.lower() for c in sourceComponents)
        self.kk_model_vs = kk_model_vs

        self.kk_wavefield_x = None
        self.kk_wavefield_y = None
        self.kk_wavefield_z = None

        self.kk_RHSTermx = None
        self.kk_RHSTermy = None
        self.kk_RHSTermz = None

        self._seismogram_x = None
        self._seismogram_y = None
        self._seismogram_z = None
        self._seismo_step = 1
        self._rcv_elem_nodes = None   # receiver element nodes (SEM interpolation)

        self.useDAS = False

    def __repr__(self):
        lines = [
            f"Solver type : {type(self).__name__}",
            f"dt : {self.dt}",
            f"maxTime : {self.maxTime}",
            f"dtSeismo : {self.dtSeismo}",
        ]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _allocateWavefield(self):
        for attr in ("kk_wavefield_x", "kk_wavefield_y", "kk_wavefield_z"):
            setattr(self, attr, (funtides_array([self.n_dof]),
                                 funtides_array([self.n_dof])))

        if self.receiverNodeIndices is not None:
            self._allocateSeismogram()

    def _allocateSeismogram(self):
        n_seismo = int(round((self.maxTimeSim - self.minTimeSim) / self.dtSeismo)) + 1
        n_receivers = len(self.receiverNodeIndices)
        for attr in ("_seismogram_x", "_seismogram_y", "_seismogram_z"):
            setattr(self, attr, np.zeros((n_seismo, n_receivers), dtype=np.float64))
        self._seismo_step = max(1, int(round(self.dtSeismo / self.dt)))
        self._rcv_elem_nodes = None   # receivers changed: recompute on next record

    def updateSourceAndReceivers(self,
                                  sourceElementIndices,
                                  receiverNodeIndices,
                                  sourceWeights=None):
        """
        Set source element indices and receiver node indices.

        Same as :meth:`WaveSolver.updateSourceAndReceivers`; also (re)allocates
        the seismograms when the solver is already initialized.
        """
        super().updateSourceAndReceivers(sourceElementIndices, receiverNodeIndices, sourceWeights)
        if self.alreadyInitialized and self.kk_wavefield_x is not None:
            self._allocateSeismogram()

    def _copySourceToComponents(self):
        """Copy kk_RHSTerm into the RHS of the source components (others: 0)."""
        RHSTerm = np.array(self.kk_RHSTerm, copy=False)
        for comp in ("x", "y", "z"):
            dest = np.array(getattr(self, f"kk_RHSTerm{comp}"), copy=False)
            dest[:, :] = RHSTerm[:, :] if comp in self.sourceComponents else 0.0

    def updateSourceValue(self, value):
        """
        Set the source waveform (see :meth:`WaveSolver.updateSourceValue`)
        and copy it into the RHS of the source components.
        """
        super().updateSourceValue(value)
        if self.kk_RHSTerm is not None and self.kk_RHSTermx is not None:
            self._copySourceToComponents()

    def _createSolverData(self):
        n_rhs = len(self.sourceElementIndices)
        n_steps = self.n_time_steps

        for attr in ("kk_RHSTermx", "kk_RHSTermy", "kk_RHSTermz"):
            setattr(self, attr, funtides_array([n_rhs, n_steps]))

        # Copy the evaluated source into the requested components
        if self.sourceValue is not None:
            self._copySourceToComponents()

        wavefield = FTSolver.WavefieldElastic(
            *self.kk_wavefield_x, *self.kk_wavefield_y, *self.kk_wavefield_z)
        rhs = FTSolver.RhsElastic(
            self.kk_RHSTermx, self.kk_RHSTermy, self.kk_RHSTermz,
            self.kk_RHSElement, self.kk_RHSWeights)
        self.data = FTSolver.SEMsolverDataElastic(wavefield, rhs)

    # ------------------------------------------------------------------
    # Source and receiver management
    # ------------------------------------------------------------------

    def updateSourceAndReceivers(self,
                                  sourceElementIndices,
                                  receiverNodeIndices,
                                  sourceWeights=None):
        """
        Set source element indices and receiver node indices.

        Parameters
        ----------
            sourceElementIndices : array-like of int
                Element index for each source
            receiverNodeIndices : array-like of int
                Global node index for each receiver
            sourceWeights : array-like, optional
                Weights [n_rhs, n_pts_per_elem]. Defaults to uniform.
        """
        super().updateSourceAndReceivers(sourceElementIndices, receiverNodeIndices, sourceWeights)
        if self.alreadyInitialized and self.kk_wavefield_x is not None:
            self._allocateSeismogram()

    # ------------------------------------------------------------------
    # Seismogram
    # ------------------------------------------------------------------

    def _recordSeismogram(self, time_sample, data):
        if self._seismogram_x is None or self.receiverNodeIndices is None:
            return
        if time_sample % self._seismo_step == 0:
            idx = time_sample // self._seismo_step
            if idx < self._seismogram_x.shape[0]:
                # Current field of the data (the buffers alternate at each swap)
                fields = [np.array(data.get_current_field(c), copy=False) for c in range(3)]
                seismos = (self._seismogram_x, self._seismogram_y, self._seismogram_z)
                if self.receiverElementIndices is not None:
                    # SEM interpolation over the receiver element nodes
                    if self._rcv_elem_nodes is None:
                        elems = np.asarray(self.receiverElementIndices)
                        self._rcv_valid_pos = np.where(elems >= 0)[0]
                        self._rcv_elem_nodes = self._receiverElementNodes(
                            elems[self._rcv_valid_pos])
                        self._rcv_weights_arr = np.array(
                            [self.receiverWeights[j] for j in self._rcv_valid_pos],
                            dtype=np.float32)
                    for u, seismo in zip(fields, seismos):
                        seismo[idx, self._rcv_valid_pos] = np.sum(
                            u[self._rcv_elem_nodes] * self._rcv_weights_arr, axis=1)
                else:
                    # Nearest node; -1 flags a receiver on another rank
                    rcv = np.asarray(self.receiverNodeIndices, dtype=np.int64)
                    valid = np.where(rcv >= 0)[0]
                    for u, seismo in zip(fields, seismos):
                        seismo[idx, valid] = u[rcv[valid]]

    def getDisplacementAtReceivers(self, component="z"):
        """
        Return the accumulated seismogram for a single displacement component.

        Parameters
        ----------
            component : str
                'x', 'y' or 'z'

        Returns
        -------
            np.ndarray, shape (n_seismo_steps, n_receivers)
        """
        assert component.lower() in ("x", "y", "z"), "component must be 'x', 'y' or 'z'"
        return getattr(self, f"_seismogram_{component.lower()}")

    def getAllDisplacementAtReceivers(self):
        """
        Return displacement seismograms for all three components.

        Returns
        -------
            tuple of np.ndarray : (displacementX, displacementY, displacementZ)
        """
        return self._seismogram_x, self._seismogram_y, self._seismogram_z

    def getWaveField(self):
        return self.getAllDisplacementAtReceivers()

    def getWaveFieldAtReceivers(self):
        return self.getAllDisplacementAtReceivers()

    def resetWaveField(self):
        """Reset displacement wavefields and seismograms to zero."""
        for attr in ("kk_wavefield_x", "kk_wavefield_y", "kk_wavefield_z"):
            pair = getattr(self, attr)
            if pair is not None:
                for kk in pair:
                    np.array(kk, copy=False)[:] = 0.0
        for attr in ("_seismogram_x", "_seismogram_y", "_seismogram_z"):
            arr = getattr(self, attr)
            if arr is not None:
                arr[:] = 0.0

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------

    def updateVelocityModel(self, vel, component="vp", **kwargs):
        """
        Update the elastic velocity model (Vp or Vs).

        Parameters
        ----------
            vel : np.ndarray
                New velocity values
            component : str
                'vp' or 'vs'
        """
        assert component.lower() in ("vp", "vs"), "component must be 'vp' or 'vs'"
        if component.lower() == "vs":
            assert self.kk_model_vs is not None, \
                "kk_model_vs must be provided at construction time to update Vs"
            vs = np.array(self.kk_model_vs, copy=False)
            mask = np.asarray(vel) > 0
            vs[mask] = np.asarray(vel, dtype=np.float32)[mask]
            self.reinitSolver()
        else:
            super().updateVelocityModel(vel)

    def getVelocityModel(self, component="vp", **kwargs):
        """
        Return a copy of the velocity model.

        Parameters
        ----------
            component : str
                'vp' or 'vs'
        """
        assert component.lower() in ("vp", "vs"), "component must be 'vp' or 'vs'"
        if component.lower() == "vs":
            assert self.kk_model_vs is not None
            return np.array(self.kk_model_vs, copy=False).copy()
        return super().getVelocityModel()

    def updateDensityModel(self, density, **kwargs):
        """
        Update the elastic density model.

        Parameters
        ----------
            density : np.ndarray
                New density values
        """
        super().updateDensityModel(density)

    def getDensityModel(self, **kwargs):
        return super().getDensityModel()
