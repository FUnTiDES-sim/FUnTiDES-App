"""Gmsh meshes in makutu (makutu.mesh.gmshmesh): reading through the FUnTiDES readers,
location of points in general quadrilaterals / hexahedra, and shots on 2-D and 3-D
Gmsh meshes against the exact 2-D solution and the Cartesian 3-D mesh."""

import numpy as np
import pytest

import pyfuntides.model as FTModel
import pyfuntides.solver as FTSolver

pytest.importorskip("gmsh")
pytestmark = pytest.mark.skipif(not hasattr(FTModel, "read_gmsh_3d"),
                                reason="FUnTiDES built without the Gmsh readers")

from makutu.mesh import GmshMesh, generateHexBox, generateQuadRectangle, mshVersion  # noqa: E402
from makutu.mesh.CartesianUnstructModel import build_cartesian_unstruct_model  # noqa: E402
from makutu.mesh.gmshmesh import _shape, physicalNames  # noqa: E402
from makutu.mesh.seismesh import plots  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402

from test_acoustic_2d import exact_2d  # noqa: E402

pytestmark = [pytestmark, pytest.mark.skipif(
    not hasattr(FTModel, "CartesianUnstruct2DBuilder_f32_i32"),
    reason="FUnTiDES built without COMPILE_SEM_2D")]


def solver_kwargs(dt, tmax, f0=5.0):
    return dict(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                model_location=FTSolver.ModelLocationType.ONELEMENTS, dt=dt, minTime=0.0,
                maxTime=tmax, dtSeismo=dt, sourceType="ricker2", sourceFreq=f0)


def shot(solver, source, receivers, sourceWeights=None):
    solver.initialize()
    solver.updateSourceAndReceiversFromCoords([source], receivers, sourceWeights=sourceWeights)
    solver.evaluateSource()
    for t in range(solver.n_time_steps):
        solver.execute(t)
    return solver.getPressureAtReceivers().astype(np.float64)


def check_location(mesh, n=40, seed=0):
    """Points mapped from known (element, xi) are located back in that element."""
    rng = np.random.default_rng(seed)
    for e in rng.choice(mesh.n_elements, n, replace=False):
        xi = rng.uniform(-0.9, 0.9, mesh.dim)
        N, _ = _shape(xi[None])
        p = N[0] @ mesh._corners[e]                               # mesh coordinates
        x, y, h = (p[0], 0.0, p[1]) if mesh.dim == 2 else p
        elem, found = mesh.locate(x, y, mesh.top - h)
        assert elem == e
        np.testing.assert_allclose(found, xi, atol=1e-9)


# ---------------------------------------------------------------------------
# 2-D
# ---------------------------------------------------------------------------

SOURCE_2D, RECEIVER_2D = [1010.0, 0.0, 990.0], [1310.0, 0.0, 690.0]


@pytest.mark.parametrize("order, size, structured, dt", [
    (2, 15.0, True, 1e-3), (2, 15.0, False, 1e-3), (3, 16.0, False, 5e-4)])
def test_2d_shot_matches_exact_solution(tmp_path, order, size, structured, dt, figure_path):
    """Point source and receiver inside general quadrilaterals (absorbing edges far
    enough for no reflection before the end of the trace)."""
    generateQuadRectangle(tmp_path / "rect.msh", 2000.0, 2000.0, size, structured=structured,
                          free_surface=False)
    mesh = GmshMesh(tmp_path / "rect.msh", order, {10: (1500.0, 1.0)})
    assert mesh.dim == 2 and mesh.top == pytest.approx(2000.0)
    if not structured:
        check_location(mesh)
    solver = AcousticSolver(**solver_kwargs(dt, 0.7), **mesh.solver_kwargs())
    trace = shot(solver, SOURCE_2D, [RECEIVER_2D])[:, 0]
    f = solver.sourceValue[:, 0]
    r = np.hypot(RECEIVER_2D[0] - SOURCE_2D[0], RECEIVER_2D[2] - SOURCE_2D[2])
    ref = np.array([exact_2d(n * dt, r, f, dt) for n in range(len(trace))])
    err = np.linalg.norm(trace - ref) / np.linalg.norm(ref)
    print(f"[ gmsh 2-D ] order {order}, {'structured' if structured else 'unstructured'}: error {err:.2e}")
    if figure_path():
        plots.plot_gmsh_mesh(figure_path("mesh"), mesh, f"gmsh 2-D {'structuré' if structured else 'non structuré'}")
        plots.plot_traces(figure_path("trace"), ref[:, None], trace[:, None], dt, ["récepteur"],
                          f"ordre {order} contre la solution exacte 2-D", ("exacte", "gmsh"))
    assert err < 2e-3


def test_2d_free_surface_and_names(tmp_path, figure_path):
    """Boundary types from the group names; the top edge (free surface) stays at zero."""
    generateQuadRectangle(tmp_path / "rect.msh", 1000.0, 500.0, 50.0, structured=False)
    assert {name for (_, name) in physicalNames(tmp_path / "rect.msh").items()} \
        == {"medium", "free_surface", "absorbing"}
    mesh = GmshMesh(tmp_path / "rect.msh", 2, {10: 1500.0})
    assert mesh.boundaries == {1: FTModel.BoundaryFlag.Surface, 2: FTModel.BoundaryFlag.Damping}
    solver = AcousticSolver(**solver_kwargs(1e-3, 0.3, f0=10.0), **mesh.solver_kwargs())
    d = shot(solver, [500.0, 0.0, 100.0], [[300.0, 0.0, 50.0], [700.0, 0.0, 50.0]])
    assert np.isfinite(d).all() and np.abs(d).max() > 0
    if figure_path():
        plots.plot_gmsh_mesh(figure_path("mesh"), mesh, "gmsh 2-D, surface libre en haut")
        plots.plot_traces(figure_path("traces"), d[:, :1], d[:, 1:], 1e-3, ["x = 300 / x = 700"],
                          "récepteurs symétriques", ("x = 300 m", "x = 700 m"))
    p = np.array(solver.data.get_current_field(0), copy=False)
    top = [i for i in range(mesh.model.get_number_of_nodes()) if mesh.model.is_free_surface(i)]
    assert len(top) > 0 and np.all(p[top] == 0.0)
    with pytest.raises(ValueError, match="outside the mesh"):
        solver.updateSourceAndReceiversFromCoords([[500.0, 0.0, 100.0]], [[1200.0, 0.0, 50.0]])


# ---------------------------------------------------------------------------
# 3-D
# ---------------------------------------------------------------------------

L3, NE3 = 1500.0, 15
SOURCE_3D = [737.0, 702.0, 655.0]
RECEIVERS_3D = [[300.0, 410.0, 33.0], [1210.0, 777.0, 501.0], [737.0, 702.0, 1300.0]]


def test_3d_box_matches_cartesian_mesh(tmp_path, figure_path):
    """The same grid read from Gmsh and built by makutu gives the same traces (same point
    source weights), receivers off the nodes, one near the free surface."""
    generateHexBox(tmp_path / "box.msh", (L3, L3, L3), NE3)
    mesh = GmshMesh(tmp_path / "box.msh", 2, tmp_path / "box.csv")   # CSV materials, names
    assert mesh.materials == {10: (1500.0, 1000.0)}
    assert mesh.dim == 3 and mesh.n_elements == NE3 ** 3
    d_gmsh = shot(AcousticSolver(**solver_kwargs(1e-3, 0.8), **mesh.solver_kwargs()),
                  SOURCE_3D, RECEIVERS_3D)

    vp = np.full((NE3,) * 3, 1500.0, np.float32)
    model, *_ = build_cartesian_unstruct_model(L3, L3, L3, NE3, NE3, NE3, 2, vp,
                                               rho_elem=np.full_like(vp, 1000.0))
    h = L3 / NE3
    p = np.array([SOURCE_3D[0], SOURCE_3D[1], L3 - SOURCE_3D[2]])
    xi = 2.0 * (p - np.floor(p / h) * h) / h - 1.0
    cartesian = AcousticSolver(model=model, order=2, lx=L3, ly=L3, lz=L3, ex=NE3, ey=NE3, ez=NE3,
                               **solver_kwargs(1e-3, 0.8))
    d_cart = shot(cartesian, SOURCE_3D, RECEIVERS_3D, sourceWeights=mesh.sourceWeights(xi)[None])
    diff = np.linalg.norm(d_gmsh - d_cart, axis=0) / np.linalg.norm(d_cart, axis=0)
    print(f"[ gmsh 3-D ] relative difference with the Cartesian mesh: {diff}")
    if figure_path():
        plots.plot_gmsh_mesh(figure_path("mesh"), mesh, "gmsh 3-D, cube structuré")
        plots.plot_traces(figure_path("traces"), d_cart, d_gmsh, 1e-3, [str(r) for r in RECEIVERS_3D],
                          "cube gmsh contre maillage cartésien", ("cartésien", "gmsh"))
    assert np.all(diff < 1e-4)


def test_3d_unstructured_box(tmp_path, figure_path):
    """Hexahedra with general horizontal faces (MSH 4.1 file, converted on reading):
    location, and a shot close to the one of the structured mesh of the same size
    (h = 50 m: the difference is the discretization error, 42 % on a receiver at
    h = 100 m, 5 % at 50 m)."""
    ne, dt = 2 * NE3, 2.5e-4
    generateHexBox(tmp_path / "box41.msh", (L3, L3, L3), ne, msh_version=4.1, structured=False)
    assert mshVersion(tmp_path / "box41.msh") == (4.1, False)
    mesh = GmshMesh(tmp_path / "box41.msh", 2, {10: (1500.0, 1000.0)})
    check_location(mesh)
    d = shot(AcousticSolver(**solver_kwargs(dt, 0.8), **mesh.solver_kwargs()), SOURCE_3D, RECEIVERS_3D)
    generateHexBox(tmp_path / "box.msh", (L3, L3, L3), ne)
    ref_mesh = GmshMesh(tmp_path / "box.msh", 2, {10: (1500.0, 1000.0)})
    ref = shot(AcousticSolver(**solver_kwargs(dt, 0.8), **ref_mesh.solver_kwargs()), SOURCE_3D, RECEIVERS_3D)
    diff = np.linalg.norm(d - ref, axis=0) / np.linalg.norm(ref, axis=0)
    print(f"[ gmsh 3-D ] unstructured vs structured: {diff}")
    if figure_path():
        plots.plot_gmsh_mesh(figure_path("mesh"), mesh, "gmsh 3-D, hexaèdres non structurés")
        plots.plot_traces(figure_path("traces"), ref, d, dt, [str(r) for r in RECEIVERS_3D],
                          "non structuré contre structuré", ("structuré", "non structuré"))
    assert np.all(diff < 0.08)
