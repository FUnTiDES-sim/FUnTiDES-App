import numpy as np

import kokkos
import pyfuntides.solver as FTSolver

_kokkos_initialized = False


class Solver:
    """
    Base solver using FUnTiDES SEM solver via pyfuntides/kokkos.

    Attributes
    ----------
        model : pyfuntides model object
            The mesh and physical parameters model
        order : int
            Polynomial order of the elements
        implem_type : FTSolver.ImplemType
            Implementation type (MAKUTU, CLASSIC, OPTIM, SHIVA)
        mesh_type : FTSolver.MeshType
            Mesh type (STRUCT or UNSTRUCT)
        physic_type : FTSolver.PhysicType
            Physics type (ACOUSTIC or ELASTIC)
        model_location : FTSolver.ModelLocationType
            Whether model is defined on nodes or elements
        sponge_size : list of float
            Absorbing boundary sizes [x, y, z] in meters
        surface_sponge : bool
            Enable sponge at the free surface
        taper_delta : float
            Taper delta for sponge boundaries
        comm : mpi4py communicator or None
            MPI communicator for domain-decomposed runs.  ``None`` means
            single-process (no MPI synchronisation).
        alreadyInitialized : bool
            Flag indicating if the solver has been initialized
        solver : pyfuntides solver object
    """

    def __init__(self,
                 model,
                 order,
                 implem_type,
                 mesh_type,
                 physic_type,
                 model_location=None,
                 sponge_size=None,
                 surface_sponge=False,
                 taper_delta=0.015,
                 comm=None,
                 **kwargs):
        """
        Parameters
        ----------
            model : pyfuntides model object
                The mesh and physical parameters model
            order : int
                Polynomial order of the elements
            implem_type : FTSolver.ImplemType
                Implementation type
            mesh_type : FTSolver.MeshType
                Mesh type (STRUCT or UNSTRUCT)
            physic_type : FTSolver.PhysicType
                Physics type (ACOUSTIC or ELASTIC)
            model_location : FTSolver.ModelLocationType, optional
                Default is ONELEMENTS
            sponge_size : list of float, optional
                Absorbing boundary sizes [x, y, z]. Default is [0.0, 0.0, 0.0]
            surface_sponge : bool, optional
                Enable sponge at free surface. Default is False
            taper_delta : float, optional
                Taper delta for sponge. Default is 0.015
            comm : mpi4py communicator, optional
                MPI communicator for 1-D X domain decomposition.  When
                provided and ``size > 1``, the mass matrix boundaries are
                synchronised after ``compute_fe_init``.
        """
        self.model = model
        self.order = order
        self.implem_type = implem_type
        self.mesh_type = mesh_type
        self.physic_type = physic_type
        self.model_location = (model_location if model_location is not None
                               else FTSolver.ModelLocationType.ONELEMENTS)
        self.sponge_size = sponge_size if sponge_size is not None else [0.0, 0.0, 0.0]
        self.surface_sponge = surface_sponge
        self.taper_delta = taper_delta
        self.comm = comm

        self.alreadyInitialized = False
        self.solver = None

    def initialize(self):
        """Initialize Kokkos and create the FUnTiDES SEM solver."""
        global _kokkos_initialized
        if not kokkos.is_initialized():
            kokkos.initialize()
        _kokkos_initialized = True

        self.solver = FTSolver.create_solver(
            FTSolver.MethodType.SEM,
            self.implem_type,
            self.mesh_type,
            self.model_location,
            self.physic_type,
            self.order,
        )
        self.solver.compute_fe_init(
            self.model, self.sponge_size, self.surface_sponge, self.taper_delta
        )

        # Synchronise mass-matrix boundary nodes once (domain decomposition)
        if self._mpi_active():
            self._sync_mass_matrix()

        self.alreadyInitialized = True

    def reinitSolver(self):
        """Recompute mass/stiffness matrices from the current model state."""
        self.solver.compute_fe_init(
            self.model, self.sponge_size, self.surface_sponge, self.taper_delta
        )
        if self._mpi_active():
            self._sync_mass_matrix()

    def finalize(self):
        """Finalize Kokkos."""
        global _kokkos_initialized
        if kokkos.is_initialized() and not kokkos.is_finalized():
            kokkos.finalize()
        _kokkos_initialized = False

    # ------------------------------------------------------------------
    # MPI helpers
    # ------------------------------------------------------------------

    def _mpi_active(self):
        """Return True when running with more than one MPI rank."""
        return self.comm is not None and self.comm.Get_size() > 1

    def _sync_mass_matrix(self):
        """Synchronise mass-matrix boundary values across X neighbours."""
        # boundary_comm is created by WaveSolver after it builds local nx/ny/nz
        if hasattr(self, 'boundary_comm') and self.boundary_comm is not None:
            mass = self.solver.get_mass_matrix()
            self.boundary_comm.sync_accumulate(mass)
            # The absorbing-boundary damping is assembled like the mass: the
            # nodes shared by two ranks hold one part from each side
            for c in range(self._numComponents()):
                self.boundary_comm.sync_accumulate(self.solver.get_damping_matrix(c))
