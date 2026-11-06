import numpy as np
from scipy.fftpack import fftfreq, ifft, fft

import kokkos
import pyfuntides.solver as FTSolver

from .Solver import Solver
from .BoundaryComm import BoundaryComm
from ..tools.kokkosarrays import funtides_array

# ---------------------------------------------------------------------------
# GLL Lagrange interpolation helpers
# (mirror of FUnTiDES gllpoints.h, orders 1-5)
# ---------------------------------------------------------------------------
_sqrt5 = 2.2360679774997897
_GLL_POINTS = {
    1: np.array([-1.0,  1.0]),
    2: np.array([-1.0,  0.0,  1.0]),
    3: np.array([-1.0, -1.0/_sqrt5,  1.0/_sqrt5,  1.0]),
    4: np.array([-1.0, -0.654653670707977,  0.0,  0.654653670707977,  1.0]),
    5: np.array([-1.0, -0.765055323929465, -0.285231516480645,
                        0.285231516480645,  0.765055323929465,  1.0]),
}


def _lagrange_weights_1d(xi, pts):
    """Evaluate all 1-D Lagrange basis functions at xi given GLL nodes pts."""
    n = len(pts)
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                w[i] *= (xi - pts[j]) / (pts[i] - pts[j])
    return w


def _lagrange_weights_2d(xi_x, xi_z, order):
    """
    Compute (order+1)^2 Lagrange basis weights at local coords (xi_x, xi_z) of a
    2-D quadrilateral, in the FUnTiDES local node layout ``node = i_x + i_z * n``.
    """
    pts = _GLL_POINTS.get(order)
    if pts is None:
        raise ValueError(f"GLL points not tabulated for order {order} (supported: 1-5)")
    wx = _lagrange_weights_1d(xi_x, pts)
    wz = _lagrange_weights_1d(xi_z, pts)
    return np.outer(wz, wx).ravel().astype(np.float32)   # [iz, ix] -> ix + iz*n


def _lagrange_weights_3d(xi_x, xi_y, xi_z, order):
    """
    Compute (order+1)^3 Lagrange basis weights at local coords (xi_x, xi_y, xi_z).

    Flat index layout mirrors sampleFieldAt in sem_solver.h:
        node = i_x * n^2 + i_y * n + i_z    (x slow, z fast)
    """
    pts = _GLL_POINTS.get(order)
    if pts is None:
        raise ValueError(f"GLL points not tabulated for order {order} (supported: 1-5)")
    wx = _lagrange_weights_1d(xi_x, pts)
    wy = _lagrange_weights_1d(xi_y, pts)
    wz = _lagrange_weights_1d(xi_z, pts)
    n = order + 1
    weights = np.empty(n ** 3, dtype=np.float32)
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                weights[ix * n * n + iy * n + iz] = wx[ix] * wy[iy] * wz[iz]
    return weights


class WaveSolver(Solver):
    """
    WaveSolver: base class for FUnTiDES SEM wave simulation.

    Manages time parameters, Ricker source evaluation, source/receiver
    indexing, velocity/density model updates, and the per-step time loop.
    Subclasses (AcousticSolver, ElasticSolver) implement wavefield
    allocation, solver data construction, and seismogram recording.

    Attributes
    ----------
        dt : float
            Time step for simulation
        minTime : float
            Simulation start time
        maxTime : float
            Simulation end time
        minTimeSim : float
            Effective start time (may shift after filterSource)
        maxTimeSim : float
            Effective end time (may shift after filterSource)
        dtSeismo : float
            Time step for seismogram sampling
        dtWaveField : float
            Time step to save wavefield snapshots
        sourceType : str
            Ricker order string, e.g. 'ricker2'
        sourceFreq : float
            Peak frequency of the source in Hz
        sourceElementIndices : np.ndarray of int32
            Element index for each source [n_rhs]
        sourceWeights : np.ndarray of float32
            Interpolation weights per source [n_rhs, n_pts_per_elem]
        receiverNodeIndices : np.ndarray of int64
            Global node index for each receiver [n_receivers]
        sourceValue : np.ndarray
            Source waveform [n_time_steps, 1]
        kk_model_vp : kokkos array, optional
            Kokkos array backing the Vp model (for updateVelocityModel)
        kk_model_rho : kokkos array, optional
            Kokkos array backing the density model (for updateDensityModel)
        kk_RHSTerm : kokkos array
            Source signal [n_rhs, n_time_steps]
        kk_RHSElement : kokkos array
            Source element indices [n_rhs]
        kk_RHSWeights : kokkos array
            Source interpolation weights [n_rhs, n_pts_per_elem]
        data : FTSolver.SEMsolverData*
            Solver data object (created by subclass)
    """

    def __init__(self,
                 dt=None,
                 minTime=0,
                 maxTime=None,
                 dtSeismo=None,
                 dtWaveField=None,
                 sourceType=None,
                 sourceFreq=None,
                 kk_model_vp=None,
                 kk_model_rho=None,
                 lx=None, ly=None, lz=None,
                 ex=None, ey=None, ez=None,
                 origin_x=0.0,
                 **kwargs):
        """
        Parameters
        ----------
            dt : float
                Time step for simulation
            minTime : float
                Starting time of simulation. Default is 0
            maxTime : float
                End time of simulation
            dtSeismo : float
                Time step for seismogram sampling.
                Default is dt (sample every step)
            dtWaveField : float
                Time step to save wavefield snapshots
            sourceType : str
                Ricker order string ('ricker0' to 'ricker4')
            sourceFreq : float
                Peak frequency of the source in Hz
            kk_model_vp : kokkos array, optional
                Kokkos array of the Vp model for in-place updates
            kk_model_rho : kokkos array, optional
                Kokkos array of the density model for in-place updates
            lx, ly, lz : float, optional
                **Local** domain lengths in X, Y, Z [m].
                Required for coordinate-based source/receiver placement.
            ex, ey, ez : int, optional
                **Local** number of elements in X, Y, Z.
                Required for coordinate-based source/receiver placement.
            origin_x : float, optional
                X-coordinate of this rank's subdomain origin [m].
                Used for coordinate → local-index conversion with MPI domain
                decomposition.  Default 0.0 (no decomposition).
            kwargs : passed to Solver.__init__
        """
        super().__init__(**kwargs)

        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.ex = ex
        self.ey = ey
        self.ez = ez
        self.origin_x = float(origin_x)

        self.boundary_comm = None  # set in initialize() when comm is active

        self.dt = dt
        self.minTime = minTime
        self.maxTime = maxTime
        self.minTimeSim = minTime
        self.maxTimeSim = maxTime

        self.dtSeismo = dtSeismo
        self.dtWaveField = dtWaveField

        self.sourceType = sourceType
        self.sourceFreq = sourceFreq

        self.kk_model_vp = kk_model_vp
        self.kk_model_rho = kk_model_rho

        # Source / receiver state
        self.sourceElementIndices = None
        self.sourceWeights = None
        self.receiverNodeIndices = None      # global node index per receiver (-1 = off-rank)
        self.receiverElementIndices = None   # element index per receiver (-1 = off-rank)
        self.receiverWeights = None          # list of float32 arrays, one per receiver
        self.sourceValue = None

        # FUnTiDES kokkos arrays (allocated at initialize time)
        self.kk_RHSTerm = None
        self.kk_RHSElement = None
        self.kk_RHSWeights = None
        self.data = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_time_steps(self):
        """Number of time steps derived from minTimeSim, maxTimeSim, dt."""
        return int(round((self.maxTimeSim - self.minTimeSim) / self.dt)) + 1

    @property
    def n_dof(self):
        """Number of degrees of freedom (nodes) in the model."""
        return self.model.get_number_of_nodes()

    @property
    def n_rhs(self):
        """Number of sources (right-hand sides)."""
        return len(self.sourceElementIndices) if self.sourceElementIndices is not None else 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """
        Initialize Kokkos, create and prepare the FUnTiDES solver,
        then allocate wavefield and RHS arrays.

        When an MPI communicator is present and ``size > 1``, a
        :class:`BoundaryComm` is created for 1-D X boundary synchronisation,
        and the mass matrix is synchronised before the time loop starts.
        """
        assert self.dt is not None, "dt must be set before initialize()"
        assert self.maxTime is not None, "maxTime must be set before initialize()"

        if self.dtSeismo is None:
            self.dtSeismo = self.dt

        if self.dim == 2 and self._mpi_active():
            raise NotImplementedError("MPI domain decomposition is not available in 2-D yet")

        # Build BoundaryComm BEFORE super().initialize() so that
        # Solver._sync_mass_matrix() can find it.
        if self._mpi_active() and self.ex is not None:
            nx = self.ex * self.order + 1
            ny = self.ey * self.order + 1
            nz = self.ez * self.order + 1
            self.boundary_comm = BoundaryComm(self.comm, nx, ny, nz)

        super().initialize()

        self._allocateWavefield()

        if self.sourceElementIndices is not None:
            self._allocateRHS()
            self._createSolverData()

    def _allocateWavefield(self):
        """Allocate wavefield kokkos arrays. Implemented by subclasses."""
        pass

    def _allocateRHS(self):
        """Allocate kk_RHSTerm, kk_RHSElement, kk_RHSWeights from current state."""
        n_rhs = len(self.sourceElementIndices)
        n_pts = self.model.get_number_of_points_per_element()
        n_steps = self.n_time_steps

        self.kk_RHSTerm = funtides_array([n_rhs, n_steps])
        self.kk_RHSElement = funtides_array([n_rhs], dtype=kokkos.int32)
        self.kk_RHSWeights = funtides_array([n_rhs, n_pts])

        np.array(self.kk_RHSElement, copy=False)[:] = self.sourceElementIndices

        RHSWeights = np.array(self.kk_RHSWeights, copy=False)
        if self.sourceWeights is not None:
            RHSWeights[:] = np.asarray(self.sourceWeights, dtype=np.float32)
        else:
            RHSWeights[:] = 1.0 / n_pts

        # Populate source signal if already evaluated
        if self.sourceValue is not None:
            RHSTerm = np.array(self.kk_RHSTerm, copy=False)
            RHSTerm[:] = 0.0
            n_copy = min(n_steps, self.sourceValue.shape[0])
            for i in range(n_rhs):
                RHSTerm[i, :n_copy] = self.sourceValue[:n_copy, 0].astype(np.float32)

    def _createSolverData(self):
        """Create FUnTiDES solver data object. Implemented by subclasses."""
        pass

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
                Element index for each source (length = n_rhs)
            receiverNodeIndices : array-like of int
                Global node index for each receiver
            sourceWeights : array-like, optional
                Interpolation weights [n_rhs, n_pts_per_elem].
                Defaults to uniform weights.
        """
        self.sourceElementIndices = np.asarray(sourceElementIndices, dtype=np.int32)
        self.receiverNodeIndices = np.asarray(receiverNodeIndices, dtype=np.int64)
        self.sourceWeights = sourceWeights

        if self.alreadyInitialized:
            self._allocateRHS()
            if self.sourceValue is not None:
                self.evaluateSource()
            self._createSolverData()

    def evaluateSource(self):
        """
        Evaluate the Ricker source waveform and populate kk_RHSTerm.
        Ricker orders 0 through 4 are supported.
        """
        sourceTypes = ("ricker0", "ricker1", "ricker2", "ricker3", "ricker4")
        assert self.sourceType in sourceTypes, f"Only {sourceTypes} are allowed"

        f0 = self.sourceFreq
        delay = 1.0 / f0
        alpha = -(f0 * np.pi) ** 2

        nsamples = self.n_time_steps
        sourceValue = np.zeros((nsamples, 1))

        order = int(self.sourceType[-1])
        sgn = (-1) ** (order + 1)

        time = self.minTime
        for nt in range(nsamples):
            if self.minTime <= -1.0 / f0:
                tmin = -2.9 / f0
                tmax = 2.9 / f0
                time_d = time
            else:
                time_d = time - delay
                tmin = 0.0
                tmax = 2.9 / f0

            if (time > tmin and time < tmax) or (self.minTime < -1 / f0 and time == tmin):
                gaussian = np.exp(alpha * time_d ** 2)

                if order == 0:
                    sourceValue[nt, 0] = sgn * gaussian
                elif order == 1:
                    sourceValue[nt, 0] = sgn * (2 * alpha * time_d) * gaussian
                elif order == 2:
                    sourceValue[nt, 0] = sgn * (2 * alpha + 4 * alpha ** 2 * time_d ** 2) * gaussian
                elif order == 3:
                    sourceValue[nt, 0] = sgn * (12 * alpha ** 2 * time_d + 8 * alpha ** 3 * time_d ** 3) * gaussian
                elif order == 4:
                    sourceValue[nt, 0] = sgn * (12 * alpha ** 2 + 48 * alpha ** 3 * time_d ** 2 + 16 * alpha ** 4 * time_d ** 4) * gaussian

            time += self.dt

        self.updateSourceValue(sourceValue)

    def updateSourceValue(self, value):
        """
        Set the source waveform and update kk_RHSTerm.

        Parameters
        ----------
            value : np.ndarray, shape (n_samples, 1)
                Source waveform values at each time step
        """
        self.sourceValue = value
        self.maxTimeSim = (value.shape[0] - 1) * self.dt + self.minTime

        if self.kk_RHSTerm is not None:
            RHSTerm = np.array(self.kk_RHSTerm, copy=False)
            n_copy = min(value.shape[0], RHSTerm.shape[1])
            for i in range(RHSTerm.shape[0]):
                RHSTerm[i, :n_copy] = value[:n_copy, 0].astype(np.float32)

    def updateSourceFrequency(self, freq):
        """
        Update source peak frequency and re-evaluate the source waveform.

        Parameters
        ----------
            freq : float
                New peak frequency in Hz
        """
        self.sourceFreq = freq
        if self.sourceValue is not None:
            self.evaluateSource()

    def filterSource(self, fmax):
        """
        Low-pass filter the source waveform and update kk_RHSTerm.
        May extend/trim minTimeSim and maxTimeSim to avoid discontinuities.

        Parameters
        ----------
            fmax : float or 'all'
                Maximum frequency to keep. Pass 'all' to skip filtering.
        """
        if str(fmax) == "all":
            return

        minTime = self.minTime
        maxTime = self.maxTime
        dt = self.dt
        sourceValue = self.sourceValue

        pad = int(round(sourceValue.shape[0] / 2))
        n = sourceValue.shape[0] + 2 * pad

        tf = fftfreq(n, dt)
        y_fft = np.zeros((n, sourceValue.shape[1]), dtype="complex_")
        y = np.zeros(y_fft.shape, dtype="complex_")

        for i in range(y_fft.shape[1]):
            y_fft[pad:n - pad, i] = sourceValue[:, i]
            y_fft[:, i] = fft(y_fft[:, i])

        isup = np.where(tf >= fmax)[0]
        imax = np.where(tf[isup] >= fmax + 1)[0][0]
        i1 = isup[0]
        i2 = isup[imax]

        iinf = np.where(tf <= -fmax)[0]
        imin = np.where(tf[iinf] <= -fmax - 1)[0][-1]
        i3 = iinf[imin]
        i4 = iinf[-1]

        for i in range(y_fft.shape[1]):
            y_fft[i1:i2, i] = np.cos((isup[0:imax] - i1) / (i2 - i1) * np.pi / 2) ** 2 * y_fft[i1:i2, i]
            y_fft[i3:i4, i] = np.cos((iinf[imin:-1] - i4) / (i3 - i4) * np.pi / 2) ** 2 * y_fft[i3:i4, i]
            y_fft[i2:i3, i] = 0

        for i in range(y.shape[1]):
            y[:, i] = ifft(y_fft[:, i])

        it0 = int(round(abs(minTime / dt))) + pad
        d = int(round(1 / fmax / dt))

        i1 = max(it0 - 4 * d, 0)
        i2 = int(round(i1 + d / 4))
        i4 = min(n, n - pad + 4 * d)
        i3 = int(round(i4 - d / 4))

        for i in range(y.shape[1]):
            y[i1:i2, i] = np.cos((np.arange(i1, i2) - i2) / (i2 - i1) * np.pi / 2) ** 2 * y[i1:i2, i]
            y[i3:i4, i] = np.cos((np.arange(i3, i4) - i3) / (i4 - i3) * np.pi / 2) ** 2 * y[i3:i4, i]
            y[max(i1 - d, 0):i1, i] = 0.0
            y[i4:min(i4 + d, n), i] = 0.0

        t = np.arange(minTime - pad * dt, maxTime + pad * dt + dt / 2, dt)

        filtered = np.real(y[max(i1 - d, 0):min(i4 + d, n), :])
        self.updateSourceValue(filtered)
        self.minTimeSim = t[max(i1 - d, 0)]
        self.maxTimeSim = t[min(i4 + d, n - 1)]

    # ------------------------------------------------------------------
    # Model update
    # ------------------------------------------------------------------

    def updateVelocityModel(self, vel):
        """
        Update the Vp model in-place via the kk_model_vp kokkos array,
        then reinitialize the solver matrices.

        Parameters
        ----------
            vel : np.ndarray
                New velocity values. Only positions where vel > 0 are updated.
        """
        assert self.kk_model_vp is not None, \
            "kk_model_vp must be provided at construction time to use updateVelocityModel"
        vp = np.array(self.kk_model_vp, copy=False)
        mask = np.asarray(vel) > 0
        vp[mask] = np.asarray(vel, dtype=np.float32)[mask]
        self.reinitSolver()

    def updateDensityModel(self, density):
        """
        Update the density model in-place via the kk_model_rho kokkos array,
        then reinitialize the solver matrices.

        Parameters
        ----------
            density : np.ndarray
                New density values. Only positions where density > 0 are updated.
        """
        assert self.kk_model_rho is not None, \
            "kk_model_rho must be provided at construction time to use updateDensityModel"
        rho = np.array(self.kk_model_rho, copy=False)
        mask = np.asarray(density) > 0
        rho[mask] = np.asarray(density, dtype=np.float32)[mask]
        self.reinitSolver()

    def getVelocityModel(self):
        """
        Return a copy of the current Vp model.

        Returns
        -------
            np.ndarray
        """
        assert self.kk_model_vp is not None
        return np.array(self.kk_model_vp, copy=False).copy()

    def getDensityModel(self):
        """
        Return a copy of the current density model.

        Returns
        -------
            np.ndarray
        """
        assert self.kk_model_rho is not None
        return np.array(self.kk_model_rho, copy=False).copy()

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Coordinate → mesh index helpers  (Cartesian structured mesh)
    # ------------------------------------------------------------------

    def _check_mesh_params(self):
        if self.dim == 2:
            # 2-D (x, z) mesh: no y direction
            if any(v is None for v in (self.lx, self.lz, self.ex, self.ez)):
                raise ValueError(
                    "lx, lz, ex, ez must be set on the solver to use "
                    "coordinate-based source/receiver placement in 2-D."
                )
            return
        if any(v is None for v in (self.lx, self.ly, self.lz, self.ex, self.ey, self.ez)):
            raise ValueError(
                "lx, ly, lz, ex, ey, ez must be set on the solver to use "
                "coordinate-based source/receiver placement."
            )

    def coordToElementIndex(self, x, y, z):
        """
        Return the **local** flat element index containing (x, y, z).

        Accounts for ``origin_x`` so that coordinates are measured in the
        global frame while indices are local to this rank's subdomain.

        Parameters
        ----------
            x, y, z : float  Physical coordinates [m] in the global frame.

        Returns
        -------
            int  0-based local element index.

        Raises
        ------
            ValueError
                If (x, y, z) does not belong to this rank's subdomain.
        """
        self._check_mesh_params()
        hx = self.lx / self.ex
        x_local = x - self.origin_x
        ix = int(x_local / hx)
        if x_local < 0.0 or ix >= self.ex:
            raise ValueError(
                f"Source x={x} not in this rank's subdomain "
                f"[{self.origin_x}, {self.origin_x + self.lx})"
            )
        ix = min(ix, self.ex - 1)
        if self.dim == 2:
            # 2-D (x, z) mesh: element ix + iz * ex, y ignored
            iz = max(0, min(int(z / (self.lz / self.ez)), self.ez - 1))
            return ix + iz * self.ex
        iy = max(0, min(int(y / (self.ly / self.ey)), self.ey - 1))
        iz = max(0, min(int(z / (self.lz / self.ez)), self.ez - 1))
        return ix + iy * self.ex + iz * self.ex * self.ey

    def coordToNodeIndex(self, x, y, z):
        """
        Return the **local** node index nearest to (x, y, z).

        Returns ``-1`` when the point falls outside this rank's X subdomain,
        which allows callers to silently skip off-rank receivers.

        Parameters
        ----------
            x, y, z : float  Physical coordinates [m] in the global frame.

        Returns
        -------
            int  0-based local node index, or ``-1`` if not on this rank.
        """
        self._check_mesh_params()
        nx = self.ex * self.order + 1
        nz = self.ez * self.order + 1
        # Node spacing is global (identical across ranks)
        dx = self.lx / (nx - 1)
        dz = self.lz / (nz - 1)
        x_local = x - self.origin_x
        ix = int(round(x_local / dx))
        if ix < 0 or ix >= nx:
            return -1   # receiver not on this rank
        if self.dim == 2:
            # 2-D (x, z) mesh: node ix + iz * nx, y ignored (nearest grid line,
            # exact on the element edges; GLL interior nodes are not uniform)
            iz = max(0, min(int(round(z / dz)), nz - 1))
            return ix + iz * nx
        ny = self.ey * self.order + 1
        dy = self.ly / (ny - 1)
        iy = max(0, min(int(round(y / dy)), ny - 1))
        iz = max(0, min(int(round(z / dz)), nz - 1))
        return ix + iy * nx + iz * nx * ny

    def coordToReceiverElementAndWeights(self, x, y, z):
        """
        Return ``(elem_idx, weights)`` for a receiver at physical coordinates.

        Uses GLL Lagrange interpolation to compute weights matching the
        ``sample_field_at`` kernel in FUnTiDES (node flat index = i_x*n²+i_y*n+i_z).

        Parameters
        ----------
            x, y, z : float  Physical coordinates [m] in the global frame.

        Returns
        -------
            elem_idx : int
                Local element index, or ``-1`` if the receiver is off-rank.
            weights : np.ndarray float32 of shape ((order+1)^3,), or ``None`` if off-rank.
        """
        self._check_mesh_params()
        hx = self.lx / self.ex
        hz = self.lz / self.ez
        x_local = x - self.origin_x
        ix = int(x_local / hx)
        if x_local < 0.0 or ix >= self.ex:
            return -1, None    # receiver is on another MPI rank
        ix = min(ix, self.ex - 1)
        if self.dim == 2:
            # 2-D (x, z) quadrilateral: element ix + iz * ex, weights ix + iz * n
            iz = max(0, min(int(z / hz), self.ez - 1))
            xi_x = float(np.clip(2.0 * (x_local - ix * hx) / hx - 1.0, -1.0, 1.0))
            xi_z = float(np.clip(2.0 * (z - iz * hz) / hz - 1.0, -1.0, 1.0))
            return ix + iz * self.ex, _lagrange_weights_2d(xi_x, xi_z, self.order)
        hy = self.ly / self.ey
        iy = max(0, min(int(y / hy), self.ey - 1))
        iz = max(0, min(int(z / hz), self.ez - 1))
        elem_idx = ix + iy * self.ex + iz * self.ex * self.ey
        # Map physical coords to element-local coords in [-1, 1]
        xi_x = float(np.clip(2.0 * (x_local - ix * hx) / hx - 1.0, -1.0, 1.0))
        xi_y = float(np.clip(2.0 * (y       - iy * hy) / hy - 1.0, -1.0, 1.0))
        xi_z = float(np.clip(2.0 * (z       - iz * hz) / hz - 1.0, -1.0, 1.0))
        weights = _lagrange_weights_3d(xi_x, xi_y, xi_z, self.order)
        return elem_idx, weights

    def updateSourceAndReceiversFromCoords(self, source_coords, receiver_coords,
                                            sourceWeights=None):
        """
        Set source and receiver positions from acquisition coordinates.

        Acquisition coordinates are ``[x, y, depth]``: the depth is measured
        downwards from the top face of the mesh (the free surface), as in the
        SEG-Y headers and the SEP models. The mesh has z up, so the point is
        placed at ``z = lz - depth``. In 2-D, y is ignored. The ``coordTo*``
        helpers take mesh coordinates (z up).

        Sources not on this rank are given zero weights so they contribute
        nothing to the RHS.  Receivers not on this rank are stored as -1 and
        silently skipped during seismogram recording; a subsequent MPI reduce
        in :meth:`getWaveFieldAtReceivers` assembles all partial seismograms.

        Parameters
        ----------
            source_coords : list of [x, y, depth]
                Coordinates of each source (global frame, depth positive down).
            receiver_coords : list of [x, y, depth]
                Coordinates of each receiver (global frame, depth positive down).
            sourceWeights : array-like, optional
                Interpolation weights per source.
        """
        self._check_mesh_params()
        # depth (positive down from the top face) -> mesh z (up)
        source_coords = [(x, y, self.lz - d) for x, y, d in source_coords]
        receiver_coords = [(x, y, self.lz - d) for x, y, d in receiver_coords]
        src_elements = []
        src_active = []
        if self.dim == 2:
            # 2-D: point sources with their Lagrange weights (y is ignored)
            src_weights = []
            for x, y, z in source_coords:
                idx, w = self.coordToReceiverElementAndWeights(x, y, z)
                src_active.append(idx >= 0)
                src_elements.append(max(idx, 0))
                src_weights.append(w if idx >= 0 else np.zeros((self.order + 1) ** 2, np.float32))
            if sourceWeights is None:
                sourceWeights = np.array(src_weights, dtype=np.float32)
        else:
            for x, y, z in source_coords:
                try:
                    idx = self.coordToElementIndex(x, y, z)
                    src_elements.append(idx)
                    src_active.append(True)
                except ValueError:
                    src_elements.append(0)   # placeholder; weights zeroed below
                    src_active.append(False)

        # Compute receiver element indices + GLL interpolation weights
        rcv_elems, rcv_weights, rcv_nodes = [], [], []
        for x, y, z in receiver_coords:
            elem_idx, w = self.coordToReceiverElementAndWeights(x, y, z)
            rcv_elems.append(elem_idx)
            rcv_weights.append(w)
            rcv_nodes.append(-1 if elem_idx < 0 else 0)   # -1 flags off-rank

        self.receiverElementIndices = rcv_elems
        self.receiverWeights = rcv_weights
        self.updateSourceAndReceivers(src_elements, rcv_nodes, sourceWeights)

        # Zero weights for sources that are not on this rank
        if any(not a for a in src_active) and self.kk_RHSWeights is not None:
            weights = np.array(self.kk_RHSWeights, copy=False)
            for i, active in enumerate(src_active):
                if not active:
                    weights[i, :] = 0.0

    def applyInitialConditions(self):
        """Reset wavefield to zero (equivalent to applying zero initial conditions)."""
        self.resetWaveField()

    def getFullWaveFieldAtReceivers(self, comm=None):
        """
        Return recorded pressure at receivers.
        The ``comm`` argument is accepted for API compatibility but not used.
        """
        return self.getWaveFieldAtReceivers()

    def execute(self, time_sample):
        """
        Perform one solver time step, record seismogram, then swap wavefields.

        In single-process mode this calls ``compute_one_step``.  In MPI mode
        it uses the split-phase pattern required by domain decomposition:

        1. ``compute_forces``  — compute local element contributions
        2. MPI boundary sync   — accumulate shared boundary nodes across ranks
        3. ``update_solution_forward`` — apply mass⁻¹ × assembled forces

        Parameters
        ----------
            time_sample : int
                Current time step index (0-based)
        """
        if self.boundary_comm is not None:
            # Split time step for domain decomposition
            self.solver.compute_forces(self.dt, time_sample, self.data)
            for c in range(self._numComponents()):
                force_vec = self.solver.get_force_vector(c)
                self.boundary_comm.sync_accumulate(force_vec)
            self.solver.update_solution_forward(self.dt, self.data)
        else:
            self.solver.compute_one_step(self.dt, time_sample, self.data)

        self._recordSeismogram(time_sample, self.data)
        self.data.swap_wavefields()

    def _receiverElementNodes(self, elements):
        """
        Global node indices of the receiver elements, shape (n_rcv, (order+1)^3),
        in the layout of the receiver weights: ix*n² + iy*n + iz; in 2-D shape
        (n_rcv, (order+1)^2), layout ix + iz*n.
        """
        n = self.order + 1
        if self.dim == 2:
            # 2-D (x, z) quadrilateral: layout ix + iz * n (j is ignored by the model)
            nodes = np.empty((len(elements), n ** 2), dtype=np.int64)
            for r, e in enumerate(elements):
                for iz in range(n):
                    for ix in range(n):
                        nodes[r, ix + iz * n] = self.model.global_node_index(int(e), ix, 0, iz)
            return nodes
        nodes = np.empty((len(elements), n ** 3), dtype=np.int64)
        for r, e in enumerate(elements):
            for ix in range(n):
                for iy in range(n):
                    for iz in range(n):
                        nodes[r, ix * n * n + iy * n + iz] = \
                            self.model.global_node_index(int(e), ix, iy, iz)
        return nodes

    # Wavefield components per physics (FUnTiDES PhysicsInfo<...>::kNumFields)
    _NUM_COMPONENTS = {
        FTSolver.PhysicType.ACOUSTIC: 1,
        FTSolver.PhysicType.ELASTIC: 3,
        FTSolver.PhysicType.ACOUSTOELASTIC: 4,
    }

    def _numComponents(self):
        """Number of wavefield components (force vectors) of the solver."""
        return self._NUM_COMPONENTS[self.physic_type]

    def _recordSeismogram(self, time_sample, data):
        """Sample wavefield at receiver locations. Implemented by subclasses."""
        pass

    # ------------------------------------------------------------------
    # Output (implemented by subclasses)
    # ------------------------------------------------------------------

    def getWaveField(self):
        pass

    def getWaveFieldAtReceivers(self):
        pass

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        lines = [
            f"Solver type : {type(self).__name__}",
            f"dt : {self.dt}",
            f"maxTime : {self.maxTime}",
            f"dtSeismo : {self.dtSeismo}",
        ]
        return "\n".join(lines) + "\n"
