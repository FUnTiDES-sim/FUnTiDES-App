"""
Reader for FUnTiDES solver parameter files.

File format
-----------
Plain text, one ``key = value`` per line.  Lines starting with ``#``
(after stripping leading whitespace) are comments and are ignored.
Inline comments (``key = value  # remark``) are also supported.

Example parameter file::

    # --- Mesh ---
    lx = 1500.0
    ly = 1500.0
    lz = 1500.0
    ex = 30
    ey = 30
    ez = 30
    order = 2
    mesh_type = STRUCT          # STRUCT or UNSTRUCT
    on_nodes = False
    is_elastic = False
    free_surface_on_top = True

    # --- Physical model defaults ---
    vp = 1500.0
    rho = 1.0

    # --- Time ---
    dt = 0.001
    minTime = 0.0
    maxTime = 1.0
    dtSeismo = 0.004
    # dtWaveField is optional

    # --- Source ---
    sourceType = ricker2
    sourceFreq = 10.0

    # --- Absorbing boundary ---
    sponge_x = 0.0
    sponge_y = 0.0
    sponge_z = 0.0
    surface_sponge = False
    taper_delta = 0.015

    # --- Physics ---
    physic_type = ACOUSTIC      # ACOUSTIC or ELASTIC
"""

import argparse


def _str2bool(v):
    """Convert common boolean string representations to Python bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "yes", "1"):
        return True
    if v.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def _build_parser():
    p = argparse.ArgumentParser("FUnTiDES parameter file parser", add_help=False)

    # Mesh
    p.add_argument("--lx", type=float, required=True)
    p.add_argument("--ly", type=float, required=True)
    p.add_argument("--lz", type=float, required=True)
    p.add_argument("--ex", type=int, required=True)
    p.add_argument("--ey", type=int, required=True)
    p.add_argument("--ez", type=int, required=True)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--mesh_type", type=str, default="STRUCT",
                   choices=["STRUCT", "UNSTRUCT"])
    p.add_argument("--on_nodes", type=_str2bool, default=False)
    p.add_argument("--is_elastic", type=_str2bool, default=False)
    p.add_argument("--free_surface_on_top", type=_str2bool, default=True)

    # Physical model defaults / SEP files
    p.add_argument("--vp", type=float, default=1500.0)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--vp_file", type=str, default=None,
                   help="SEP .H header file for heterogeneous Vp model")
    p.add_argument("--rho_file", type=str, default=None,
                   help="SEP .H header file for heterogeneous density model")

    # Time
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--minTime", type=float, default=0.0)
    p.add_argument("--maxTime", type=float, default=None)
    p.add_argument("--dtSeismo", type=float, default=None)
    p.add_argument("--dtWaveField", type=float, default=None)

    # Source
    p.add_argument("--sourceType", type=str, default=None,
                   choices=["ricker0", "ricker1", "ricker2", "ricker3", "ricker4"])
    p.add_argument("--sourceFreq", type=float, default=None)

    # Absorbing boundary
    p.add_argument("--sponge_x", type=float, default=0.0)
    p.add_argument("--sponge_y", type=float, default=0.0)
    p.add_argument("--sponge_z", type=float, default=0.0)
    p.add_argument("--surface_sponge", type=_str2bool, default=False)
    p.add_argument("--taper_delta", type=float, default=0.015)

    # Physics
    p.add_argument("--physic_type", type=str, default="ACOUSTIC",
                   choices=["ACOUSTIC", "ELASTIC"])

    return p


def _file_to_argv(filename):
    """Parse a ``key = value`` file into an ``argparse``-compatible list."""
    argv = []
    with open(filename, "r") as f:
        for raw_line in f:
            line = raw_line.split("#")[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(
                    f"Malformed line in parameter file '{filename}': {raw_line.rstrip()!r}"
                )
            key, _, value = line.partition("=")
            argv.append("--" + key.strip())
            argv.append(value.strip())
    return argv


class FUnTiDESParams:
    """
    Parsed FUnTiDES parameter set.

    Attributes
    ----------
    lx, ly, lz : float
        Global domain extents [m].
    ex, ey, ez : int
        Global element counts per direction.
    order : int
        Polynomial order (1, 2 or 3).
    mesh_type : str
        ``'STRUCT'`` or ``'UNSTRUCT'``.
    on_nodes : bool
        Physical model on nodes (True) or elements (False).
    is_elastic : bool
        Elastic (True) or acoustic (False) model arrays.
    free_surface_on_top : bool
        Free-surface boundary condition on the top face.
    vp : float
        Default P-wave velocity [m/s].
    rho : float
        Default density (normalized).
    dt : float or None
        Time step [s].
    minTime : float
        Simulation start time [s].
    maxTime : float or None
        Simulation end time [s].
    dtSeismo : float or None
        Seismogram sampling interval [s]. Defaults to ``dt`` if not set.
    dtWaveField : float or None
        Wavefield snapshot interval [s].
    sourceType : str or None
        Ricker source type, e.g. ``'ricker2'``.
    sourceFreq : float or None
        Source peak frequency [Hz].
    sponge_size : list of float
        Absorbing boundary thickness ``[sponge_x, sponge_y, sponge_z]`` [m].
    surface_sponge : bool
        Apply sponge on free surface.
    taper_delta : float
        Sponge taper width.
    physic_type : str
        ``'ACOUSTIC'`` or ``'ELASTIC'``.
    """

    def __init__(self, args):
        # Mesh
        self.lx = args.lx
        self.ly = args.ly
        self.lz = args.lz
        self.ex = args.ex
        self.ey = args.ey
        self.ez = args.ez
        self.order = args.order
        self.mesh_type = args.mesh_type
        self.on_nodes = args.on_nodes
        self.is_elastic = args.is_elastic
        self.free_surface_on_top = args.free_surface_on_top

        # Physical model defaults
        self.vp = args.vp
        self.rho = args.rho
        self.vp_file = args.vp_file
        self.rho_file = args.rho_file

        # Time
        self.dt = args.dt
        self.minTime = args.minTime
        self.maxTime = args.maxTime
        self.dtSeismo = args.dtSeismo
        self.dtWaveField = args.dtWaveField

        # Source
        self.sourceType = args.sourceType
        self.sourceFreq = args.sourceFreq

        # Absorbing boundary
        self.sponge_size = [args.sponge_x, args.sponge_y, args.sponge_z]
        self.surface_sponge = args.surface_sponge
        self.taper_delta = args.taper_delta

        # Physics
        self.physic_type = args.physic_type

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def build_mesh(self, comm=None):
        """
        Instantiate and return the mesh described by these parameters.

        Parameters
        ----------
        comm : mpi4py communicator, optional
            When provided (and ``size > 1``), the global X-domain is
            partitioned across MPI ranks.

        Returns
        -------
        StructuredMesh or UnstructuredMesh
        """
        from makutu.mesh.InternalMesh import StructuredMesh, UnstructuredMesh

        kwargs = dict(
            lx=self.lx, ly=self.ly, lz=self.lz,
            ex=self.ex, ey=self.ey, ez=self.ez,
            order=self.order,
            on_nodes=self.on_nodes,
            is_elastic=self.is_elastic,
            free_surface_on_top=self.free_surface_on_top,
            comm=comm,
        )
        if self.mesh_type == "STRUCT":
            return StructuredMesh(
                **kwargs,
                vp_file=self.vp_file,
                rho_file=self.rho_file,
            )
        return UnstructuredMesh(**kwargs)

    def build_solver(self, mesh, kk_model_vp=None, kk_model_rho=None, comm=None):
        """
        Instantiate and return the wave solver described by these parameters.

        The caller is responsible for calling ``solver.initialize()`` and
        ``solver.updateSourceAndReceiversFromCoords(...)`` before running the
        time loop.

        Parameters
        ----------
        mesh : StructuredMesh or UnstructuredMesh
            Mesh built by :meth:`build_mesh`.  Used to extract the pyfuntides
            ``model`` object and the local geometry (``lx/ex/origin_x``).
        kk_model_vp : kokkos array, optional
            Kokkos float32 array of shape ``[n_elem]`` or ``[n_nodes]``
            containing the Vp model.  Required for in-place model updates
            (e.g. FWI).
        kk_model_rho : kokkos array, optional
            Kokkos float32 array of shape ``[n_elem]`` or ``[n_nodes]``
            containing the density model.
        comm : mpi4py communicator, optional
            MPI communicator for domain-decomposed runs.

        Returns
        -------
        AcousticSolver or ElasticSolver
        """
        import pyfuntides.solver as FTSolver
        from makutu.solvers.AcousticSolver import AcousticSolver

        _physic_map = {
            "ACOUSTIC": FTSolver.PhysicType.ACOUSTIC,
            "ELASTIC":  FTSolver.PhysicType.ELASTIC,
        }
        _mesh_map = {
            "STRUCT":   FTSolver.MeshType.STRUCT,
            "UNSTRUCT": FTSolver.MeshType.UNSTRUCT,
        }
        _location = (FTSolver.ModelLocationType.ONNODES
                     if self.on_nodes
                     else FTSolver.ModelLocationType.ONELEMENTS)

        # When the mesh was built with a SEP file it uses ModelUnstruct backend
        actual_mesh_type = (FTSolver.MeshType.UNSTRUCT
                            if getattr(mesh, "heterogeneous", False)
                            else _mesh_map[self.mesh_type])

        # Automatically forward the mutable Kokkos arrays from the mesh when
        # the caller did not supply explicit overrides
        if kk_model_vp is None:
            kk_model_vp = getattr(mesh, "kk_model_vp", None)
        if kk_model_rho is None:
            kk_model_rho = getattr(mesh, "kk_model_rho", None)

        solver_kwargs = dict(
            model=mesh.model,
            order=self.order,
            implem_type=FTSolver.ImplemType.MAKUTU,
            mesh_type=actual_mesh_type,
            physic_type=_physic_map[self.physic_type],
            model_location=_location,
            sponge_size=self.sponge_size,
            surface_sponge=self.surface_sponge,
            taper_delta=self.taper_delta,
            # Time
            dt=self.dt,
            minTime=self.minTime,
            maxTime=self.maxTime,
            dtSeismo=self.dtSeismo,
            dtWaveField=self.dtWaveField,
            # Source
            sourceType=self.sourceType,
            sourceFreq=self.sourceFreq,
            # Geometry (for coordinate-based source/receiver placement)
            lx=mesh.lx, ly=mesh.ly, lz=mesh.lz,
            ex=mesh.ex, ey=mesh.ey, ez=mesh.ez,
            origin_x=mesh.origin_x,
            # Model arrays
            kk_model_vp=kk_model_vp,
            kk_model_rho=kk_model_rho,
            # MPI
            comm=comm,
        )

        if self.physic_type == "ACOUSTIC":
            return AcousticSolver(**solver_kwargs)

        # Elastic
        from makutu.solvers.ElasticSolver import ElasticSolver
        return ElasticSolver(**solver_kwargs)

    def __repr__(self):
        lines = [
            "FUnTiDESParams(",
            f"  mesh      : {self.mesh_type}  {self.ex}x{self.ey}x{self.ez} elements"
            f"  [{self.lx}x{self.ly}x{self.lz} m]  order={self.order}",
            f"  physics   : {self.physic_type}  vp={self.vp_file or self.vp} m/s"
            f"  rho={self.rho_file or self.rho}",
            f"  time      : dt={self.dt}  [{self.minTime}, {self.maxTime}] s"
            f"  dtSeismo={self.dtSeismo}",
            f"  source    : {self.sourceType}  f0={self.sourceFreq} Hz",
            f"  sponge    : {self.sponge_size}  surface={self.surface_sponge}"
            f"  taper={self.taper_delta}",
            ")",
        ]
        return "\n".join(lines)


def read_funtides_params(filename):
    """
    Read a FUnTiDES parameter file and return a :class:`FUnTiDESParams` object.

    Parameters
    ----------
    filename : str
        Path to the parameter file.

    Returns
    -------
    FUnTiDESParams

    Raises
    ------
    ValueError
        If the file contains a malformed line.
    SystemExit
        If a required parameter is missing (raised by argparse).
    """
    argv = _file_to_argv(filename)
    parser = _build_parser()
    args, _ = parser.parse_known_args(argv)
    return FUnTiDESParams(args)
