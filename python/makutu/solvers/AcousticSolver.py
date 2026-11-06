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
        self._seismo_step = self.seismoStep
        # one sample every _seismo_step time steps, from step 0
        n_seismo = (self.n_time_steps - 1) // self._seismo_step + 1
        self._seismogram = np.zeros((n_seismo, n_receivers), dtype=np.float32)
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

    def getPressureField(self):
        """
        Current pressure (time t) on the mesh nodes as a grid, seismic order:
        shape (nz, nx) in 2-D or (nz, ny, nx) in 3-D, index 0 at the surface
        (z positive downwards). In MPI mode, the local subdomain only.
        """
        return self._grid_field(0)

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

    def runForward(self, snapshotStep=None):
        """
        Time loop of the current shot (sources and receivers set, source evaluated,
        wavefield reset). With ``snapshotStep``, the second time derivative of the
        pressure is kept every ``snapshotStep`` steps for :meth:`computeGradient`.

        Parameters
        ----------
            snapshotStep : int, optional
                Steps between the kept snapshots; the default, None, keeps nothing.
                The integrand of the gradient has frequencies up to about twice the
                highest frequency of the data: snapshotStep * dt < 1 / (5 f0) is safe
                for a Ricker wavelet of peak frequency f0 (see :meth:`snapshotStepFor`).
        """
        n_steps = self.n_time_steps
        self._pdd, self._pddStep = {}, snapshotStep
        stored = set(range(snapshotStep, n_steps, snapshotStep)) if snapshotStep else set()
        keep = {k + d for k in stored for d in (-1, 0, 1)}
        fields = {0: np.zeros(self.n_dof, dtype=np.float32)} if 0 in keep else {}   # p^0 = 0
        for t in range(n_steps):
            self.execute(t)
            k = t + 1                                    # the current field is p^(t+1)
            if k in keep:
                fields[k] = np.array(self.data.get_current_field(0), copy=True)
            n = k - 1
            if n in stored:                              # p^(n-1), p^n, p^(n+1) available
                self._pdd[n] = (fields[n + 1] - 2.0 * fields[n] + fields[n - 1]) / (self.dt * self.dt)
                for old in [i for i in fields if i < n]:     # later snapshots need indices >= n
                    del fields[old]

    def snapshotStepFor(self, fmax=None):
        """Snapshot step for :meth:`runForward`: (snapshotStep * dt) < 1 / (2.5 fmax),
        fmax = 2.5 f0 by default (highest frequency of a Ricker wavelet)."""
        fmax = 2.5 * self.sourceFreq if fmax is None else fmax
        return max(1, int(1.0 / (2.5 * fmax * self.dt)))

    def _elementNodeTable(self):
        """
        Global node indices of every (local) element, shape (n_elem, n_pts), and the
        matching quadrature weights times the Jacobian, shape (n_pts,), for the
        Cartesian meshes (elements of size lx/ex x [ly/ey x] lz/ez).
        """
        if getattr(self, "_nodeTable", None) is not None:
            return self._nodeTable
        o, n = self.order, self.order + 1
        from ..mesh.CartesianModel2D import gll_points
        x = gll_points(o)
        w = np.array([2.0 / (o * n) / np.polynomial.legendre.legval(xi, [0] * o + [1]) ** 2 for xi in x])
        hx, hz = self.lx / self.ex, self.lz / self.ez
        nx = self.ex * o + 1
        a = np.arange(n)
        if self.dim == 2:
            e = np.arange(self.ex * self.ez)
            ix, iz = e % self.ex, e // self.ex
            c, b = np.meshgrid(a, a, indexing="ij")                 # local index b + c n
            gi = ix[:, None] * o + b.ravel()[None, :]
            gk = iz[:, None] * o + c.ravel()[None, :]
            table = gi + gk * nx
            weights = (np.outer(w, w).ravel() * hx * hz / 4.0)        # w_c w_b
            check = [(int(ee), int(bb), 0, int(cc)) for ee in (0, len(e) - 1) for bb, cc in ((0, 0), (o, o), (1 % n, 0))]
            local = lambda bb, yy, cc: bb + cc * n
        else:
            hy = self.ly / self.ey
            ny = self.ey * o + 1
            e = np.arange(self.ex * self.ey * self.ez)
            ix, iy, iz = e % self.ex, (e // self.ex) % self.ey, e // (self.ex * self.ey)
            c, bj, b = np.meshgrid(a, a, a, indexing="ij")          # local index b + bj n + c n^2
            gi = ix[:, None] * o + b.ravel()[None, :]
            gj = iy[:, None] * o + bj.ravel()[None, :]
            gk = iz[:, None] * o + c.ravel()[None, :]
            table = gi + gj * nx + gk * nx * ny
            weights = (np.einsum("i,j,k->ijk", w, w, w).ravel() * hx * hy * hz / 8.0)
            check = [(int(ee), int(bb), int(yy), int(cc)) for ee in (0, len(e) - 1)
                     for bb, yy, cc in ((0, 0, 0), (o, 1 % n, o), (1 % n, 0, o))]
            local = lambda bb, yy, cc: bb + yy * n + cc * n * n
        for ee, bb, yy, cc in check:                                  # the model numbering
            if self.model.global_node_index(ee, bb, yy, cc) != table[ee, local(bb, yy, cc)]:
                raise NotImplementedError("gradient: node numbering of the mesh not supported")
        self._nodeTable = (table.astype(np.int64), weights)
        return self._nodeTable

    def computeGradient(self, residual, parametrization="c"):
        """
        Gradient of the cost function J = dt_r / 2 * sum(residual^2) of the current
        shot (adjoint-state method), after :meth:`runForward` with a snapshot step.

        The residual (observed - simulated) is injected, reversed in time, at the
        receivers with their interpolation weights (adjoint run, same solver); the
        gradient with respect to m = 1 / (vp^2 rho), which weighs the mass matrix,
        is G_e = sum_n dt sum_(i in e) |J| w_i lambda_i^n d2p_i^n / dt2, then converted.
        The absorbing terms are not differentiated (exact away from the boundaries).

        The receivers are replaced by the adjoint sources: read the seismograms
        before; set the sources and receivers again for the next shot.

        Parameters
        ----------
            residual : array, shape (n_samples, n_receivers)
                observed - simulated, sampled every ``dtSeismoRecorded``
            parametrization : str
                "c" (velocity), "1/c" or "1/c2"

        Returns
        -------
            np.ndarray : gradient per element, seismic order (as getVelocityModel);
            the local part of the mesh with the MPI domain decomposition
        """
        if not getattr(self, "_pddStep", None):
            raise RuntimeError("computeGradient needs runForward(snapshotStep=...) first")
        vp = np.array(self.kk_model_vp, copy=False).astype(np.float64)
        table, weights = self._elementNodeTable()
        if vp.size != table.shape[0]:
            raise NotImplementedError("gradient: the model must be defined on the elements")
        rho = (np.array(self.kk_model_rho, copy=False).astype(np.float64)
               if self.kk_model_rho is not None else np.ones_like(vp))

        # Adjoint sources: the receivers of this rank, with their weights
        residual = np.asarray(residual, dtype=np.float64)
        elems = np.asarray(self.receiverElementIndices)
        on_rank = np.where(elems >= 0)[0]
        n = self.order + 1
        def src_layout(w):           # receiver layout ix n^2 + iy n + iz -> source x + y n + z n^2
            w = np.asarray(w, dtype=np.float32)
            return w if self.dim == 2 else w.reshape(n, n, n).transpose(2, 1, 0).ravel()
        n_pts = n ** self.dim
        if len(on_rank):
            src_elems = elems[on_rank]
            src_weights = np.array([src_layout(self.receiverWeights[j]) for j in on_rank])
        else:                         # no receiver here: one silent source
            src_elems, src_weights = np.array([0]), np.zeros((1, n_pts), np.float32)
        self.receiverElementIndices, self.receiverWeights = None, None
        self.sourceValue = None
        self.updateSourceAndReceivers(src_elems, [], src_weights)

        # Residual at every step, reversed: h^j = r(t_(N-j))
        n_steps, dtr = self.n_time_steps, self.dtSeismoRecorded
        t = np.arange(n_steps + 1) * self.dt
        trec = np.arange(residual.shape[0]) * dtr
        rhs = np.array(self.kk_RHSTerm, copy=False)
        rhs[:] = 0.0
        for k, j in enumerate(on_rank):
            r = np.interp(t, trec, residual[:, j], right=0.0)
            rhs[k, :] = r[n_steps:0:-1].astype(np.float32)

        # Adjoint run; lambda^n = current field after execute(N - 1 - n)
        self.resetWaveField()
        grad_m = np.zeros(table.shape[0])
        step = self._pddStep * self.dt
        for j in range(n_steps):
            self.execute(j)
            m = n_steps - 1 - j
            if m in self._pdd:
                lam = np.array(self.data.get_current_field(0), copy=False)
                v = lam * self._pdd.pop(m)
                grad_m += step * (v[table] @ weights)
        self._pdd, self._pddStep = {}, None

        if parametrization == "c":
            grad = grad_m * (-2.0 / (vp ** 3 * rho))
        elif parametrization == "1/c":
            grad = grad_m * (2.0 / (vp * rho))
        elif parametrization == "1/c2":
            grad = grad_m / rho
        else:
            raise ValueError(f"Unknown parametrization: {parametrization}")
        return self._flip_z(grad)
