"""Shots on meshes built by makutu.mesh.seismesh from a SEP model and the source frequency
(2-D quadrilaterals and 3-D hexahedra with 2:1 doubling layers), against the Cartesian makutu
mesh of the same model (makutu convention for both: SEP sample i stands for the cell [o + i d, o + (i+1) d);
the reference elements hold whole cells)."""

import numpy as np
import pytest

import pyfuntides.model as FTModel
import pyfuntides.solver as FTSolver

pytestmark = pytest.mark.skipif(
    not (hasattr(FTModel, "read_gmsh_3d") and hasattr(FTModel, "CartesianUnstruct2DBuilder_f32_i32")),
    reason="FUnTiDES built without the Gmsh readers or COMPILE_SEM_2D")

from makutu.input.createSepModel import _write_sep  # noqa: E402
from makutu.mesh import CartesianMesh2D, GmshMesh, StructuredMesh  # noqa: E402
from makutu.mesh.seismesh import build_mesh_from_sep, grid_from_sep, plots  # noqa: E402
from makutu.mesh.seismesh.sizing import SizingParams, h_required  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402

F0 = 5.0


def _figures(figure_path, tmp_path, r, order, ref, data, dt, labels, suffix=""):
    """Control images: mesh (size, size / required size) and traces against the Cartesian mesh."""
    if not figure_path():
        return
    g = grid_from_sep(tmp_path / "vp.H")
    h = h_required(g, SizingParams(2.5 * F0, order, r["selected"]["ppw_effective"]["target"]))
    plots.plot_mesh_file(figure_path(f"mesh{suffix}"), r["msh"], g, h, f"seismesh {r['selected']['strategy']}")
    plots.plot_traces(figure_path(f"traces{suffix}"), ref, data, dt, labels, "tir seismesh contre cartésien",
                      ("cartésien", "seismesh"))


def shot(mesh_kwargs, dt, tmax, source, receivers, sourceWeights=None):
    s = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                       model_location=FTSolver.ModelLocationType.ONELEMENTS, dt=dt, minTime=0.0, maxTime=tmax,
                       dtSeismo=dt, sourceType="ricker2", sourceFreq=F0, **mesh_kwargs)
    s.initialize()
    s.updateSourceAndReceiversFromCoords([source], receivers, sourceWeights=sourceWeights)
    s.evaluateSource()
    for t in range(s.n_time_steps):
        s.execute(t)
    return s.getPressureAtReceivers().astype(np.float64)


def test_2d_doubling_mesh_matches_cartesian(tmp_path, figure_path):
    """vp = 1500 + 3 z on 2000 x 1000 m: doubling mesh at 10 points per wavelength."""
    d = 20.0                                       # one SEP cell per reference element
    z = (np.arange(50)[:, None] + 0.5) * d * np.ones((1, 100))        # makutu: sample i = cell [i d, (i+1) d)
    _write_sep(str(tmp_path / "vp.H"), (1500.0 + 3.0 * z).astype(np.float32)[None], d, d, 1.0)
    r = build_mesh_from_sep(tmp_path / "vp.H", F0, tmp_path / "mesh", order=4, ppw=10, verbose=False)
    assert r["dim"] == 2 and r["selected"]["strategy"] == "doubling"
    mesh = GmshMesh(r["msh"], 4, r["materials"])
    source, receivers = [1005.0, 0.0, 60.0], [[x, 0.0, 40.0] for x in (205.0, 605.0, 1405.0, 1805.0)] + [[1005.0, 0.0, 800.0]]
    data = shot(mesh.solver_kwargs(), 4e-4, 1.2, source, receivers)
    ref_mesh = CartesianMesh2D(2000.0, 1000.0, 100, 50, order=4, vp_file=str(tmp_path / "vp.H"))
    ref = shot(ref_mesh.solver_kwargs(), 4e-4, 1.2, source, receivers)
    diff = np.linalg.norm(data - ref, axis=0) / np.linalg.norm(ref, axis=0)
    print(f"[ seismesh 2-D ] {mesh.n_elements} elements, relative differences {np.round(diff, 4)}")
    _figures(figure_path, tmp_path, r, 4, ref, data, 4e-4, [f"x = {p[0]:.0f}, z = {p[2]:.0f}" for p in receivers])
    assert np.all(diff < 0.05)


def test_3d_doubling_mesh_matches_cartesian(tmp_path, figure_path):
    """vp = 1500 + 3 z on 1500 x 1500 x 1000 m: doubling mesh at 10 points per wavelength."""
    d = 50.0                                       # one SEP cell per reference element
    z = (np.arange(20)[:, None, None] + 0.5) * d * np.ones((1, 30, 30))
    _write_sep(str(tmp_path / "vp.H"), (1500.0 + 3.0 * z).astype(np.float32), d, d, d)
    r = build_mesh_from_sep(tmp_path / "vp.H", F0, tmp_path / "mesh", order=3, ppw=10,
                            strategies=("tensor", "doubling"), verbose=False)
    assert r["dim"] == 3 and r["selected"]["strategy"] == "doubling"
    mesh = GmshMesh(r["msh"], 3, r["materials"])
    source = [760.0, 740.0, 110.0]
    receivers = [[x, 750.0, 40.0] for x in (160.0, 460.0, 1060.0, 1360.0)] + [[750.0, 750.0, 800.0]]
    data = shot(mesh.solver_kwargs(), 5e-4, 1.0, source, receivers)
    ne, lx, lz = 30, 1500.0, 1000.0
    ref_mesh = StructuredMesh(lx, lx, lz, ne, ne, 20, order=3, vp_file=str(tmp_path / "vp.H"))
    h = lx / ne
    p = np.array([source[0], source[1], lz - source[2]])
    xi = 2.0 * (p - np.floor(p / h) * h) / h - 1.0
    ref = shot(ref_mesh.solver_kwargs(), 5e-4, 1.0, source, receivers, mesh.sourceWeights(xi)[None])
    diff = np.linalg.norm(data - ref, axis=0) / np.linalg.norm(ref, axis=0)
    print(f"[ seismesh 3-D ] {mesh.n_elements} elements, relative differences {np.round(diff, 4)}")
    _figures(figure_path, tmp_path, r, 3, ref, data, 5e-4, [f"x = {p[0]:.0f}, z = {p[2]:.0f}" for p in receivers])
    assert np.all(diff < 0.05)



def test_2d_quadtree_salt_matches_cartesian(tmp_path, figure_path):
    """Large fast body (salt, 4500 m/s) under water and sediments: 2:1 quadtree mesh from the fastest
    velocity (large elements in the salt, small in the water), against the Cartesian mesh (one SEP
    cell per element); the difference decreases when the mesh is refined."""
    d = 20.0
    z = (np.arange(100)[:, None] + 0.5) * d * np.ones((1, 200))
    x = (np.arange(200)[None, :] + 0.5) * d * np.ones((100, 1))
    v = np.where(z < 200, 1500.0, 2000.0 + 0.3 * (z - 200))
    v[((x - 2000) / 1300) ** 2 + ((z - 1000) / 400) ** 2 < 1] = 4500.0
    _write_sep(str(tmp_path / "vp.H"), v.astype(np.float32)[None], d, d, 1.0)
    source = [2010.0, 0.0, 40.0]
    receivers = [[xr, 0.0, 20.0] for xr in np.arange(210.0, 3900.0, 300.0)]
    ref_mesh = CartesianMesh2D(4000.0, 2000.0, 200, 100, order=4, vp_file=str(tmp_path / "vp.H"))
    ref = shot(ref_mesh.solver_kwargs(), 2e-4, 2.0, source, receivers)
    errors = []
    for ppw in (8, 16):
        r = build_mesh_from_sep(tmp_path / "vp.H", F0, tmp_path / f"mesh{ppw}", order=4, ppw=ppw,
                                strategies=("quadtree",), verbose=False)
        assert r["selected"]["strategy"] == "quadtree" and r["selected"]["ok"]
        mesh = GmshMesh(r["msh"], 4, r["materials"])
        data = shot(mesh.solver_kwargs(), 2e-4, 2.0, source, receivers)
        errors.append(np.linalg.norm(data - ref) / np.linalg.norm(ref))
        _figures(figure_path, tmp_path, r, 4, ref[:, ::2], data[:, ::2], 2e-4,
                 [f"x = {p[0]:.0f}" for p in receivers[::2]], suffix=f"ppw{ppw}")
        print(f"[ seismesh quadtree salt ] {ppw} points per wavelength: {mesh.n_elements} elements, "
              f"relative difference {errors[-1]:.3f}")
    assert errors[1] < 0.6 * errors[0] and errors[1] < 0.1
