Mesh and model
##############

The meshes are Cartesian grids of spectral elements (quadrilaterals in 2-D, hexahedra in
3-D) of order 1 to the order compiled in FUnTiDES, with Gauss-Lobatto-Legendre (GLL)
points. The model (velocity, density) is constant per element (:code:`ONELEMENTS`) or
given on the nodes.

From a SEP model
================

:func:`makutu.mesh.mesh_from_sep` reads the SEP file, chooses the element size and the time
step for the source frequency, and builds the mesh:

.. code-block:: python

    from makutu.mesh import mesh_from_sep

    m = mesh_from_sep("Vp_xz.H", f0=8.0, order=4, rho_file="Rho_xz.H", ppw=6, cfl=0.3)
    m.mesh, m.dt, m.counts, m.lambda_min     # mesh, time step, element counts, shortest wavelength

- shortest wavelength of the Ricker wavelet: lambda_min = vp_min / (2.5 f0);
- element size h = order * lambda_min / ppw (GLL points per wavelength), or
  :code:`elem_size`;
- dt = cfl * smallest GLL spacing / vp_max (:func:`makutu.mesh.suggest_mesh`).

The model of an element is the SEP sample that contains its centre
(:func:`makutu.mesh.sample_sep_2d`); elements smaller than the SEP sampling are fine.

Classes
=======

- :class:`makutu.mesh.CartesianMesh2D` (2-D, FUnTiDES built with :code:`COMPILE_SEM_2D`):
  :code:`CartesianMesh2D(lx, lz, ex, ez, order, vp=..., rho=..., vp_file=..., free_surface_on_top=True)`;
- :class:`makutu.mesh.StructuredMesh` (3-D), with an optional MPI communicator for the
  domain decomposition along x.

:code:`mesh.solver_kwargs()` gives the arguments of the solvers (model, order, domain,
element counts, Kokkos views of the model). The top of the mesh is a free surface by
default; the other sides have the absorbing (paraxial) condition of FUnTiDES.

Gmsh meshes
===========

:class:`makutu.mesh.GmshMesh` reads a Gmsh mesh through the FUnTiDES readers: linear
quadrilaterals Q4 of the (x, z) plane in 2-D, hexahedra Hex8 in 3-D, any shape (not only a
grid). The GLL nodes are built at the solver order. A file in another format than MSH 2.2
ASCII is converted on reading (:code:`gmsh` Python module).

.. code-block:: python

    from makutu.mesh import GmshMesh
    mesh = GmshMesh("model.msh", order=2,
                    materials={10: (1500.0, 1000.0), 20: (3000.0, 2200.0)},   # tag: (vp, rho)
                    boundaries={1: "surface", 2: "damping"})                  # optional
    solver = AcousticSolver(implem_type=..., mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=..., maxTime=..., **mesh.solver_kwargs())

- Each cell physical group is one material; :code:`materials` can also be a CSV file
  :code:`physical_tag,vp,rho`.
- Each boundary group (lines in 2-D, surfaces in 3-D) is a free surface or absorbing; without
  :code:`boundaries`, a group whose name contains "free" or "surface" is a free surface, any
  other one absorbing. Every outer face must be in a boundary group.
- Coordinates: makutu depth z (down) is placed at the mesh height :code:`mesh.top - z`
  (:code:`top`: highest node). Sources and receivers are located in their element by
  inverting its bilinear / trilinear map (point sources and receivers with Lagrange weights).
- Distorted elements need a smaller time step than a grid of the same size.
- Modelling only for now: the gradient, the conditioning and the model updates of the FWI
  assume a Cartesian grid.

Tools (:code:`apps/mesh/gmsh_mesh.py`): :code:`convert` (to MSH 2.2), :code:`box` (Hex8
box, structured or not), :code:`rectangle` (2-D Q4 rectangle), :code:`info`.
