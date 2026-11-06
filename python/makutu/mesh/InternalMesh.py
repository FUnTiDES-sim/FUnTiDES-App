"""
FUnTiDES mesh representations.

Two mesh types are supported:

  - StructuredMesh   : axis-aligned Cartesian hexahedral mesh built with
                       ``FTModel.CartesianStructBuilder_f32_i32_O<order>``.
  - UnstructuredMesh : Cartesian tetrahedral mesh built with
                       ``FTModel.CartesianUnstructBuilder_f32_i32`` +
                       ``FTModel.CartesianParams_f32_i32``.

Both support optional MPI 1-D X-axis domain decomposition: pass
``comm=MPI.COMM_WORLD`` and each rank automatically receives its local
subdomain.  The ``origin_x`` and ``x_offset_elem`` attributes record the
rank's position within the global domain.
"""

import numpy as np

try:
    import pyfuntides.model as FTModel
    _HAS_PYFUNTIDES = True
except ImportError:
    _HAS_PYFUNTIDES = False


def _plot_field(ax, data2d, extent, xlabel, ylabel, title, cmap, vmin, vmax):
    """Imshow helper: data2d has shape (nrows, ncols), extent=[xmin,xmax,ymin,ymax]."""
    im = ax.imshow(
        data2d, origin="lower", aspect="auto",
        extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return im

# Structured builder map: order → class.
# FUnTiDES provides CartesianStructBuilder_f32_i32_O1..O9 (the solvers may be
# compiled for fewer orders, see MAX_SOLVER_ORDER); guard each lookup so that
# importing this module works with any pyfuntides build.
_STRUCT_BUILDERS = {}
if _HAS_PYFUNTIDES:
    for _o in range(1, 10):
        _cls = getattr(FTModel, f"CartesianStructBuilder_f32_i32_O{_o}", None)
        if _cls is not None:
            _STRUCT_BUILDERS[_o] = _cls


def _partition_1d(global_n, rank, size):
    """Divide *global_n* elements across *size* ranks, return (local_n, offset)."""
    base = global_n // size
    rem  = global_n % size
    local_n = base + (1 if rank < rem else 0)
    offset  = rank * base + min(rank, rem)
    return local_n, offset


class StructuredMesh:
    """
    Axis-aligned Cartesian structured mesh for FUnTiDES SEM.

    Wraps ``FTModel.CartesianStructBuilder_f32_i32_O<order>`` and exposes the
    attributes expected by the rest of the codebase.

    Parameters
    ----------
    lx, ly, lz : float
        Global domain extents [m].
    ex, ey, ez : int
        Global number of elements in each direction.
    order : int, optional
        Polynomial order (1, 2 or 3).  Default 2.
    on_nodes : bool, optional
        Whether the physical model (Vp, rho …) lives on nodes (True) or on
        elements (False).  Default False.
    is_elastic : bool, optional
        Whether to allocate elastic (True) or acoustic (False) model arrays.
        Default False.
    free_surface_on_top : bool, optional
        Passed to ``get_model()``.  Default True.
    comm : mpi4py communicator, optional
        If provided, the global X-domain is partitioned across ranks.  Each
        rank builds its own local model.  Y and Z are **not** decomposed.

    Attributes
    ----------
    model : pyfuntides model object
        Underlying FUnTiDES model for this rank's subdomain.
    lx, ly, lz : float
        **Local** domain extents [m] (equals global extents when ``comm`` is
        None or ``size == 1``).
    ex, ey, ez : int
        **Local** element counts.
    hx, hy, hz : float
        Element sizes [m] (identical to global values — partitioning preserves
        element size).
    origin_x : float
        X-coordinate of this rank's subdomain origin [m].
    x_offset_elem : int
        Global X-element index of the first local element.
    global_ex : int
        Total number of elements in X across all ranks.
    numberOfCells : int
        Number of local elements.
    numberOfPoints : int
        Number of local DOF nodes.
    bounds : list of [float, float]
        ``[[origin_x, origin_x+lx], [0, ly], [0, lz]]``.
    cellBounds : list of np.ndarray
        Cell-boundary coordinates along X, Y, Z.
    isSet : bool
    """

    def __init__(self, lx, ly, lz, ex, ey, ez,
                 order=2, on_nodes=False, is_elastic=False,
                 free_surface_on_top=True, comm=None,
                 vp_file=None, rho_file=None):
        """
        Parameters
        ----------
        lx, ly, lz : float
            Global domain extents [m].
        ex, ey, ez : int
            Global element counts per direction.
        order : int
            Polynomial order (1, 2 or 3).  Default 2.
        on_nodes : bool
            Model on nodes (True) or elements (False).  Default False.
        is_elastic : bool
            Elastic flag.  Default False.
        free_surface_on_top : bool
            Free-surface BC on top face.  Default True.
        comm : mpi4py communicator, optional
            MPI communicator for 1-D X decomposition.
        vp_file : str, optional
            Path to a SEP ``.H`` header file containing the P-wave velocity
            model [m/s].  When provided the mesh switches to a
            ``ModelUnstruct`` backend so that vp/rho are backed by mutable
            Kokkos arrays (required for FWI and heterogeneous simulations).
            The ``heterogeneous`` attribute is set to ``True`` and
            ``kk_model_vp`` / ``kk_model_rho`` are exposed on the mesh.
        rho_file : str, optional
            Path to a SEP ``.H`` header file for density.  Optional; defaults
            to 1.0 everywhere when omitted.
        """
        if not _HAS_PYFUNTIDES:
            raise ImportError("pyfuntides is required to build a StructuredMesh")
        if order not in _STRUCT_BUILDERS:
            raise ValueError(f"Unsupported order {order}; available: {sorted(_STRUCT_BUILDERS)}")

        self.global_ex = int(ex)
        self.ey = int(ey)
        self.ez = int(ez)
        lx, ly, lz = float(lx), float(ly), float(lz)
        # Global element sizes (preserved by partitioning)
        hx = lx / ex
        self.hy = ly / ey
        self.hz = lz / ez
        self.hx = hx
        self.order = order

        # MPI domain decomposition in X
        if comm is not None and comm.Get_size() > 1:
            rank, size = comm.Get_rank(), comm.Get_size()
            local_ex, x_offset_elem = _partition_1d(self.global_ex, rank, size)
        else:
            local_ex, x_offset_elem = self.global_ex, 0

        self.ex = local_ex
        self.x_offset_elem = x_offset_elem
        self.origin_x = x_offset_elem * hx
        self.lx = local_ex * hx
        self.ly = ly
        self.lz = lz

        # Kokkos model arrays — set when a SEP file is provided
        self.kk_model_vp  = None
        self.kk_model_rho = None
        self.heterogeneous = False   # True when ModelUnstruct backend is used

        # With MPI, the structured builder cannot be given the global domain
        # (pyfuntides passes the local size only): every rank would treat its
        # x faces as domain boundaries. The ModelUnstruct backend is used then.
        partitioned = comm is not None and comm.Get_size() > 1

        if vp_file is not None or partitioned:
            # --- Heterogeneous model: build ModelUnstruct from SEP data ----
            from makutu.input.SEPReader import read_sep, sample_on_elements
            from makutu.mesh.CartesianUnstructModel import (
                build_cartesian_unstruct_model,
            )

            if vp_file is not None:
                sep_vp = read_sep(vp_file)
                vp_elem = sample_on_elements(
                    sep_vp, local_ex, self.ey, self.ez,
                    hx, self.hy, self.hz, self.origin_x,
                )
            else:
                # Homogeneous: default Vp of the FUnTiDES structured builder
                vp_elem = np.full((self.ez, self.ey, local_ex), 1500.0, dtype=np.float32)

            rho_elem = None
            if rho_file is not None:
                sep_rho = read_sep(rho_file)
                rho_elem = sample_on_elements(
                    sep_rho, local_ex, self.ey, self.ez,
                    hx, self.hy, self.hz, self.origin_x,
                )

            self.model, self.kk_model_vp, self.kk_model_rho = \
                build_cartesian_unstruct_model(
                    self.lx, self.ly, self.lz,
                    local_ex, self.ey, self.ez,
                    order,
                    vp_elem, rho_elem,
                    on_nodes=on_nodes,
                    is_elastic=is_elastic,
                    origin_x=self.origin_x,
                    global_x_max=self.global_ex * hx,
                    free_surface_on_top=free_surface_on_top,
                )
            self.heterogeneous = True

        else:
            # --- Homogeneous model: fast structured builder -----------------
            builder_cls = _STRUCT_BUILDERS[order]
            builder = builder_cls(
                local_ex, self.lx,
                self.ey, self.ly,
                self.ez, self.lz,
                on_nodes, is_elastic,
                self.origin_x, 0.0, 0.0,
            )
            self.model = builder.get_model(free_surface_on_top)

        self.numberOfCells  = local_ex * self.ey * self.ez
        self.numberOfPoints = self.model.get_number_of_nodes()

        self.bounds = [
            [self.origin_x, self.origin_x + self.lx],
            [0.0, self.ly],
            [0.0, self.lz],
        ]
        self.cellBounds = [
            np.linspace(self.origin_x, self.origin_x + self.lx, local_ex + 1),
            np.linspace(0.0, self.ly, self.ey + 1),
            np.linspace(0.0, self.lz, self.ez + 1),
        ]
        self.isSet = True

    def plot_model(self, ix=None, iy=None, iz=None, figsize=None, show=True):
        """Display Vp (and rho) cross-sections after heterogeneous model build.

        Three orthogonal slices are shown for each available field:
        XZ (depth section at given *iy*), YZ (at given *ix*) and XY (plan view
        at given *iz*). The mesh z is up (iz = 0 at the bottom); the sections are
        drawn with the depth ``lz - z`` downwards.  Indices default to the midpoint of each axis.

        Parameters
        ----------
        ix, iy, iz : int, optional
            Element indices for the YZ, XZ and XY slices respectively.
            Default: midpoint of each axis.
        figsize : tuple, optional
            Figure size passed to ``plt.figure``.  Default auto-sized.
        show : bool, optional
            Call ``plt.show()`` after plotting.  Default True.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if self.kk_model_vp is None:
            raise RuntimeError(
                "No heterogeneous model loaded — plot_model requires a "
                "vp_file to have been supplied at construction time."
            )

        ix = int(ix) if ix is not None else self.ex // 2
        iy = int(iy) if iy is not None else self.ey // 2
        iz = int(iz) if iz is not None else self.ez // 2

        vp_3d  = np.array(self.kk_model_vp,  copy=False).reshape(self.ez, self.ey, self.ex)
        fields = [("Vp  [m/s]", vp_3d, "hot_r")]
        if self.kk_model_rho is not None:
            rho_3d = np.array(self.kk_model_rho, copy=False).reshape(self.ez, self.ey, self.ex)
            fields.append(("Rho [kg/m³]", rho_3d, "Blues"))

        n_fields = len(fields)
        if figsize is None:
            figsize = (5 * 3, 4 * n_fields)
        fig, axes = plt.subplots(n_fields, 3, figsize=figsize, squeeze=False)

        # Axis extents (metres)
        x0, x1 = self.origin_x, self.origin_x + self.lx
        y0, y1 = 0.0, self.ly
        z0, z1 = 0.0, self.lz

        for row, (label, data3d, cmap) in enumerate(fields):
            vmin, vmax = float(data3d.min()), float(data3d.max())

            # XZ slice — depth section at y=iy  (shape ez×ex)
            ax = axes[row, 0]
            im = _plot_field(ax, data3d[:, iy, :],
                             [x0, x1, z1, z0], "X [m]", "Depth [m]",
                             f"{label}  —  XZ  (iy={iy})",
                             cmap, vmin, vmax)
            fig.colorbar(im, ax=ax, shrink=0.8)

            # YZ slice — depth section at x=ix  (shape ez×ey)
            ax = axes[row, 1]
            im = _plot_field(ax, data3d[:, :, ix],
                             [y0, y1, z1, z0], "Y [m]", "Depth [m]",
                             f"{label}  —  YZ  (ix={ix})",
                             cmap, vmin, vmax)
            fig.colorbar(im, ax=ax, shrink=0.8)

            # XY slice — plan view at z=iz  (shape ey×ex)
            ax = axes[row, 2]
            im = _plot_field(ax, data3d[iz, :, :],
                             [x0, x1, y0, y1], "X [m]", "Y [m]",
                             f"{label}  —  XY  (iz={iz})",
                             cmap, vmin, vmax)
            fig.colorbar(im, ax=ax, shrink=0.8)

        fig.tight_layout()
        if show:
            plt.show()
        return fig


class UnstructuredMesh:
    """
    Cartesian unstructured (tetrahedral) mesh for FUnTiDES SEM.

    Uses ``FTModel.CartesianUnstructBuilder_f32_i32`` with
    ``FTModel.CartesianParams_f32_i32``.

    Parameters
    ----------
    lx, ly, lz : float
        Global domain extents [m].
    ex, ey, ez : int
        Number of elements in each direction.
    order : int, optional
        Polynomial order (1, 2 or 3).  Default 2.
    on_nodes : bool, optional
        Whether the physical model lives on nodes.  Default False.
    is_elastic : bool, optional
        Whether to allocate elastic model arrays.  Default False.
    free_surface_on_top : bool, optional
        Passed to ``get_model()``.  Default True.
    comm : mpi4py communicator, optional
        If provided, the global X-domain is partitioned across ranks.

    Attributes
    ----------
    model : pyfuntides model object
    lx, ly, lz : float
        Local domain extents.
    ex, ey, ez : int
        Local element counts.
    origin_x : float
    x_offset_elem : int
    global_ex : int
    numberOfCells : int
    numberOfPoints : int
    bounds : list of [float, float]
    cellBounds : None
    isSet : bool
    """

    def __init__(self, lx, ly, lz, ex, ey, ez,
                 order=2, on_nodes=False, is_elastic=False,
                 free_surface_on_top=True, comm=None):
        if not _HAS_PYFUNTIDES:
            raise ImportError("pyfuntides is required to build an UnstructuredMesh")
        if order not in (1, 2, 3):
            raise ValueError(f"Unsupported order {order}; available: 1, 2, 3")

        self.global_ex = int(ex)
        self.ey = int(ey)
        self.ez = int(ez)
        lx, ly, lz = float(lx), float(ly), float(lz)
        hx = lx / ex
        self.hy = ly / ey
        self.hz = lz / ez
        self.hx = hx
        self.order = order

        if comm is not None and comm.Get_size() > 1:
            rank, size = comm.Get_rank(), comm.Get_size()
            local_ex, x_offset_elem = _partition_1d(self.global_ex, rank, size)
        else:
            local_ex, x_offset_elem = self.global_ex, 0

        self.ex = local_ex
        self.x_offset_elem = x_offset_elem
        self.origin_x = x_offset_elem * hx
        self.lx = local_ex * hx
        self.ly = ly
        self.lz = lz

        params = FTModel.CartesianParams_f32_i32(
            order, local_ex, ey, ez,
            self.lx, ly, lz,
            on_nodes, is_elastic,
        )
        params.origin_x = self.origin_x
        params.origin_y = 0.0
        params.origin_z = 0.0

        builder = FTModel.CartesianUnstructBuilder_f32_i32(params)
        self.model = builder.get_model(free_surface_on_top)

        self.numberOfCells  = local_ex * ey * ez
        self.numberOfPoints = self.model.get_number_of_nodes()

        self.bounds = [
            [self.origin_x, self.origin_x + self.lx],
            [0.0, self.ly],
            [0.0, self.lz],
        ]
        self.cellBounds = None
        self.isSet = True


# ---------------------------------------------------------------------------
# Back-compat alias
# ---------------------------------------------------------------------------
InternalMesh = StructuredMesh
