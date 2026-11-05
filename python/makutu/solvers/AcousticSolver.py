import os
import shutil
import numpy as np
import h5py

import kokkos
import pyfuntides.solver as FTSolver

from .WaveSolver import WaveSolver
from ..tools.kokkosarrays import funtides_array
from .ModelManagement import geos_params_from_model, grad_to_model


class AcousticSolver(WaveSolver):
    """
    AcousticSolver: FUnTiDES SEM acoustic wave simulation.

    Manages a pressure wavefield [n_dof, 2] (two time levels) and
    accumulates a seismogram at receiver nodes during the time loop.

    Attributes
    ----------
        kk_wavefield : kokkos array
            Pressure field [n_dof, 2]
        _seismogram : np.ndarray
            Recorded pressure at receivers [n_seismo_steps, n_receivers]
        _seismo_step : int
            Number of time steps between seismogram samples
        gradient_param : str
            Parametrisation used for FWI gradient ('c', '1/c', '1/c2')
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
            kwargs : passed to WaveSolver.__init__
                physic_type is set to ACOUSTIC automatically.
        """
        kwargs.setdefault("physic_type", FTSolver.PhysicType.ACOUSTIC)
        super().__init__(**kwargs)

        self.kk_pn_prev = None   # pressure at t-1  [n_dof], see funtides_array()
        self.kk_pn_curr = None   # pressure at t     [n_dof], see funtides_array()
        self._seismogram = None
        self._seismo_step = 1
        self.gradient_param = None

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
        n = self.n_dof
        self.kk_pn_prev = funtides_array([n])
        self.kk_pn_curr = funtides_array([n])

        if self.receiverNodeIndices is not None:
            self._allocateSeismogram()

    def _allocateSeismogram(self):
        n_receivers = len(self.receiverNodeIndices)
        n_seismo = int(round((self.maxTimeSim - self.minTimeSim) / self.dtSeismo)) + 1
        self._seismogram = np.zeros((n_seismo, n_receivers), dtype=np.float32)
        self._seismo_step = max(1, int(round(self.dtSeismo / self.dt)))
        self._seismo_idx = 0    # number of time samples actually recorded

        if self.receiverElementIndices is not None:
            # Preferred path: batch SEM interpolation via get_wavefield_at_receivers()
            elems = np.asarray(self.receiverElementIndices, dtype=np.int32)
            valid = elems >= 0
            self._rcv_valid_pos  = np.where(valid)[0]
            # int32 array (n_valid,) — element index per on-rank receiver
            self._rcv_elem_arr   = np.ascontiguousarray(elems[valid], dtype=np.int32)
            # float32 array (n_valid, npe) C-contiguous — weights per receiver
            self._rcv_weights_arr = np.ascontiguousarray(
                np.array([self.receiverWeights[j] for j in self._rcv_valid_pos],
                         dtype=np.float32))
            self._rcv_valid_nodes    = None   # not used in this mode
            self._rcv_batch_api      = hasattr(self.solver, "get_wavefield_at_receivers")
            self._rcv_sample_api     = hasattr(self.solver, "sample_field_at")
            # Without a sampling API in the solver (FUnTiDES): interpolate in
            # Python from the element nodes, in the weight layout ix*n²+iy*n+iz
            self._rcv_elem_nodes = None
            if not (self._rcv_batch_api or self._rcv_sample_api):
                self._rcv_elem_nodes = self._receiverElementNodes(self._rcv_elem_arr)
        else:
            # Fallback: nearest-node read from kk_pn_curr
            rcv = np.asarray(self.receiverNodeIndices, dtype=np.int64)
            valid = rcv >= 0
            self._rcv_valid_pos   = np.where(valid)[0]
            self._rcv_valid_nodes = rcv[valid]
            self._rcv_elem_arr    = None
            self._rcv_weights_arr = None
            self._rcv_batch_api   = False

    def _createSolverData(self):
        wavefield = FTSolver.WavefieldAcoustic(self.kk_pn_prev, self.kk_pn_curr)
        rhs = FTSolver.RhsAcoustic(self.kk_RHSTerm, self.kk_RHSElement, self.kk_RHSWeights)
        self.data = FTSolver.SEMsolverDataAcoustic(wavefield, rhs)

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
        if self.alreadyInitialized and self.kk_pn_curr is not None:
            self._allocateSeismogram()

    # ------------------------------------------------------------------
    # Seismogram
    # ------------------------------------------------------------------

    def _recordSeismogram(self, time_sample, data):
        if self._seismogram is None or self.receiverNodeIndices is None:
            return
        if time_sample % self._seismo_step != 0:
            return
        idx = time_sample // self._seismo_step
        if idx >= self._seismogram.shape[0]:
            return
        if self._rcv_elem_arr is not None:
            # SEM interpolation via FUnTiDES API
            if self._rcv_batch_api:
                # Batch path: single GPU kernel for all receivers
                vals = self.solver.get_wavefield_at_receivers(
                    data, 0, self._rcv_elem_arr, self._rcv_weights_arr)
                self._seismogram[idx, self._rcv_valid_pos] = vals
            elif self._rcv_elem_nodes is not None:
                # Python path: weighted sum over the receiver element nodes.
                # Current field of the data (time t, as sample_field_at): the
                # buffers alternate at each swap, kk_pn_curr is not always it.
                pn = np.array(data.get_current_field(0), copy=False)
                self._seismogram[idx, self._rcv_valid_pos] = np.sum(
                    pn[self._rcv_elem_nodes] * self._rcv_weights_arr, axis=1)
            else:
                # Per-receiver fallback using sample_field_at
                for col, elem, w in zip(self._rcv_valid_pos,
                                        self._rcv_elem_arr,
                                        self._rcv_weights_arr):
                    self._seismogram[idx, col] = self.solver.sample_field_at(
                        data, 0, int(elem), w)
        else:
            # Nearest-node fallback: direct read of the current field (the
            # buffers alternate at each swap, kk_pn_curr is not always it)
            pn = np.array(data.get_current_field(0), copy=False)
            self._seismogram[idx, self._rcv_valid_pos] = pn[self._rcv_valid_nodes]
        self._seismo_idx = idx + 1

    def getPressureAtReceivers(self):
        """
        Return the seismogram recorded at receiver locations.

        Only the time samples actually written during the simulation are
        returned (trailing pre-allocated zeros are stripped).  The returned
        array is a copy — safe to hold across ``resetWaveField`` calls and
        between shots.

        In MPI mode each rank records only the receivers inside its local
        subdomain (off-rank columns remain zero).  ``MPI.Allreduce SUM``
        assembles the full seismogram on every rank before returning.

        Returns
        -------
            np.ndarray, shape (n_recorded_steps, n_receivers), float32
        """
        if self._seismogram is None:
            return None
        recorded = self._seismogram[:self._seismo_idx]
        if self._mpi_active():
            from mpi4py import MPI
            result = np.zeros_like(recorded)
            self.comm.Allreduce(recorded, result, op=MPI.SUM)
            return result
        return recorded.copy()

    def getWaveField(self):
        return self.getPressureAtReceivers()

    def getWaveFieldAtReceivers(self):
        return self.getPressureAtReceivers()

    def resetWaveField(self):
        """Reset pressure wavefield and seismogram to zero for a new shot."""
        if self.kk_pn_prev is not None:
            np.array(self.kk_pn_prev, copy=False)[:] = 0.0
        if self.kk_pn_curr is not None:
            np.array(self.kk_pn_curr, copy=False)[:] = 0.0
        if self._seismogram is not None:
            self._seismogram[:] = 0.0
            self._seismo_idx = 0

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------

    def updateVelocityModel(self, vel, **kwargs):
        """
        Update acoustic velocity model (Vp).

        Parameters
        ----------
            vel : np.ndarray
                New velocity values
        """
        super().updateVelocityModel(vel)

    def updateDensityModel(self, density, **kwargs):
        """
        Update acoustic density model.

        Parameters
        ----------
            density : np.ndarray
                New density values
        """
        super().updateDensityModel(density)

    def getVelocityModel(self, **kwargs):
        return super().getVelocityModel()

    def getDensityModel(self, **kwargs):
        return super().getDensityModel()

    # ------------------------------------------------------------------
    # FWI
    # ------------------------------------------------------------------

    def setModelForGradient(self, gradient_param):
        """
        Set the velocity parametrisation used for gradient computation.

        Parameters
        ----------
            gradient_param : str
                One of 'c', '1/c', '1/c2'
        """
        self.gradient_param = gradient_param

    def updateModel(self, filename, low, high, **kwargs):
        """
        Load a velocity model from HDF5 and apply it.

        Parameters
        ----------
            filename : str
                Path to .hdf5 file containing a 'velocity' dataset
            low : float
                Minimum velocity threshold
            high : float
                Maximum velocity threshold
        """
        with h5py.File(filename, "r") as f:
            x = f["velocity"][:]

        x = np.clip(x, low, high)

        if self.gradient_param == "1/c2":
            x = np.sqrt(1.0 / x)
        elif self.gradient_param == "1/c":
            x = 1.0 / x
        elif self.gradient_param != "c":
            raise ValueError(f"Unknown gradient_param: {self.gradient_param}")

        self.updateVelocityModel(x)

    def computePartialGradient(self, shotId, minDepth, comm=None,
                                gradDirectory="partialGradient", **kwargs):
        """
        Compute and save the partial FWI gradient for this shot.

        .. note::
            FUnTiDES gradient API is not yet defined.
            This method raises NotImplementedError until it is available.
        """
        raise NotImplementedError(
            "Gradient computation via FUnTiDES is not yet implemented."
        )
