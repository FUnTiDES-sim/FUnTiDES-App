"""makutu.mesh.seismesh without a solver: SEP grids, 2-D meshes (tensor, 2:1 doubling),
FUnTiDES export (MSH 2.2, z up, one material class per element), SEP -> mesh chain."""

import csv

import numpy as np
import pytest

from makutu.input.createSepModel import _write_sep
from makutu.mesh.seismesh import build_mesh_from_sep, grid_from_sep
from makutu.mesh.seismesh.funtides import material_classes, write_funtides_msh
from makutu.mesh.seismesh.plane import evaluate_2d, mesh_doubling_2d, mesh_tensor_2d, segments_2d
from makutu.mesh.seismesh.sizing import SizingParams, h_required
from makutu.mesh.seismesh import plots


def gradient_sep(path, nx=200, nz=100, d=20.0, v0=1500.0, gradient=2.0):
    z = np.arange(nz)[:, None] * d * np.ones((1, nx))
    _write_sep(str(path), (v0 + gradient * z).astype(np.float32)[None], d, d, 1.0)
    return path


def read_msh(path):
    """Nodes, elements {type: (tags, connectivity)} and physical names of a MSH 2.2 file."""
    lines = open(path).read().split("\n")
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    nodes = np.array([[float(v) for v in lines[i + 2 + k].split()[1:]] for k in range(n)])
    j = lines.index("$Elements")
    elements = {}
    for k in range(int(lines[j + 1])):
        v = [int(t) for t in lines[j + 2 + k].split()]
        tags, conn = elements.setdefault(v[1], ([], []))
        tags.append(v[3])
        conn.append(np.array(v[3 + v[2]:]) - 1)
    return nodes, {t: (np.array(a), np.array(b)) for t, (a, b) in elements.items()}


def edge_counts(quads):
    e = np.sort(np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return cnt


def test_sep_grid_orientation(tmp_path):
    """2-D: axis 1 = x, axis 2 = depth; 3-D: (x, y, z) with z the slowest SEP axis."""
    gradient_sep(tmp_path / "v2.H", nx=5, nz=3, d=10.0, gradient=1.0)
    g = grid_from_sep(tmp_path / "v2.H")
    assert g.shape == (5, 1, 3) and g.spacing == (10.0, 1.0, 10.0)
    assert g.extent[0] == (0.0, 50.0) and g.extent[2] == (0.0, 30.0)   # makutu: sample i = [i d, (i + 1) d)
    np.testing.assert_allclose(g.vel[0, 0], [1500, 1510, 1520])
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(4, 3, 2)          # (z, y, x)
    _write_sep(str(tmp_path / "v3.H"), data, 1.0, 2.0, 3.0)
    g3 = grid_from_sep(str(tmp_path / "v3.H"))
    assert g3.shape == (2, 3, 4)
    assert g3.vel[1, 2, 3] == data[3, 2, 1]


@pytest.mark.parametrize("refine", [False, True])
def test_doubling_2d_is_conforming(tmp_path, refine, figure_path):
    """Doubling (and re-refinement under a slow layer): convex quads, no hanging node, the
    rectangle is covered, the horizontal step doubles at depth."""
    nz = 100
    z = np.arange(nz)[:, None] * 20.0 * np.ones((1, 200))
    v = 1500.0 + 2.0 * z
    if refine:
        v[(z > 1400) & (z < 1600)] = 1500.0                                 # slow layer at depth
    _write_sep(str(tmp_path / "v.H"), v.astype(np.float32)[None], 20.0, 20.0, 1.0)
    grid = grid_from_sep(tmp_path / "v.H")
    p = SizingParams(fmax=12.5, order=4, ppw=5)
    h = h_required(grid, p)
    m = mesh_doubling_2d(grid, h, segments_2d(grid, h, (2, 6)), max_levels=2, allow_refine=refine)
    rep = evaluate_2d(m, grid, p)
    if figure_path():
        plots.plot_quad_mesh(figure_path(), grid, h, m.nodes, m.quads, f"doubling 2-D, re-raffinement {refine}")
    assert rep["scaled_jacobian"]["min"] > 0.3                             # 0.49 with re-refinement
    assert rep["frac_undersampled"] == 0
    kinds = [k for k, _, _ in m.params["plan"]]
    assert "tx" in kinds and (("rx" in kinds) == refine)
    cnt = edge_counts(m.quads)
    assert set(cnt) <= {1, 2}                                               # conforming
    X = m.nodes[m.quads]
    area = 0.5 * np.abs(np.sum(X[:, :, 0] * np.roll(X[:, :, 1], -1, 1) - np.roll(X[:, :, 0], -1, 1) * X[:, :, 1], 1))
    assert area.sum() == pytest.approx(4000.0 * 2000.0)
    e = np.concatenate([m.quads[:, [i, (i + 1) % 4]] for i in range(4)])
    es, idx, c1 = np.unique(np.sort(e, axis=1), axis=0, return_index=True, return_counts=True)
    B = m.nodes[es[c1 == 1]]                                                # (nb, 2, 2): boundary edges
    on_side = (np.all(np.isclose(B[:, :, 0], 0), 1) | np.all(np.isclose(B[:, :, 0], 4000), 1)
               | np.all(np.isclose(B[:, :, 1], 0), 1) | np.all(np.isclose(B[:, :, 1], 2000), 1))
    assert on_side.all()                                                    # no gap inside
    t = mesh_tensor_2d(grid, h, segments_2d(grid, h, (2, 6)))
    if not refine:                     # with a slow layer at depth the tensor grid may be cheaper:
        assert len(m.quads) < len(t.quads)   # the autopilot compares the candidates


def test_export_z_up_and_material_classes(tmp_path):
    """Heights = max depth - depth, free surface on top, absorbing elsewhere, one class per
    (vp, rho); hexahedra keep a positive orientation after the z flip."""
    tags, table = material_classes([1500.2, 1500.4, 2000.0], [1.0, 1.0, 2.0])
    assert len(table) == 2 and tags[0] == tags[1] != tags[2]
    # 2-D: 2 x 1 quads, depth 0 at the top
    nodes = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    quads = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    msh, materials, n = write_funtides_msh(tmp_path / "q.msh", nodes, quads, [1500.0, 3000.0])
    xyz, el = read_msh(msh)
    np.testing.assert_allclose(xyz[:3, 2], 1.0)                               # depth 0 -> height 1 (top)
    tags_line, lines = el[1]
    top = np.all(xyz[lines, 2] == 1.0, axis=1)
    assert set(tags_line[top]) == {1} and set(tags_line[~top]) == {2} and top.sum() == 2
    rows = list(csv.DictReader(open(materials)))
    assert [float(r["vp"]) for r in rows] == [1500.0, 3000.0]
    # 3-D: one unit hexahedron
    c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    msh, _, _ = write_funtides_msh(tmp_path / "h.msh", c, np.arange(8)[None], [2000.0])
    xyz, el = read_msh(msh)
    h = el[5][1][0]
    X = xyz[h]
    assert np.dot(np.cross(X[1] - X[0], X[3] - X[0]), X[4] - X[0]) > 0
    tags_face, faces = el[3]
    assert (tags_face == 1).sum() == 1 and (tags_face == 2).sum() == 5
    assert np.all(xyz[faces[tags_face == 1][0], 2] == 1.0)                  # depth 0 face is on top


def test_build_from_sep_selects_doubling(tmp_path, figure_path):
    """Velocity tripling with depth: the 2:1 doubling mesh is cheaper than the tensor grid."""
    gradient_sep(tmp_path / "v.H")
    r = build_mesh_from_sep(tmp_path / "v.H", f0=5.0, out_dir=tmp_path / "out", order=4, ppw=5, verbose=False)
    sel = r["selected"]
    if figure_path():
        g = grid_from_sep(tmp_path / "v.H")
        plots.plot_mesh_file(figure_path(), r["msh"], g, h_required(g, SizingParams(12.5, 4, 5)), "build_mesh_from_sep, gradient")
    assert r["dim"] == 2 and sel["strategy"] == "doubling" and sel["ok"]
    assert sel["ppw_effective"]["min"] >= 5 * 0.98
    assert (tmp_path / "out" / "mesh.msh").exists() and (tmp_path / "out" / "report.json").exists()


def layered_sep(path, d=10.0):
    """3 layers (1800 / 3000 / 4000 m/s); makutu convention: sample i stands for [i d, (i + 1) d), so the
    interfaces are at 500 and 1000 m."""
    z = np.arange(150)[:, None] * d * np.ones((1, 200))
    v = np.where(z < 500, 1800.0, np.where(z < 1000, 3000.0, 4000.0)).astype(np.float32)
    _write_sep(str(path), v[None], d, d, 1.0)
    return path


def test_element_faces_follow_horizontal_interfaces(tmp_path, figure_path):
    """Faces at the sharp horizontal interfaces (cell boundaries): one velocity per layer, no mixed element."""
    from makutu.mesh.seismesh.sizing import horizontal_interfaces
    layered_sep(tmp_path / "v.H")
    assert horizontal_interfaces(grid_from_sep(tmp_path / "v.H")) == [500.0, 1000.0]
    r = build_mesh_from_sep(tmp_path / "v.H", f0=5.0, out_dir=tmp_path / "out", order=4, ppw=8, verbose=False)
    if figure_path():
        g = grid_from_sep(tmp_path / "v.H")
        plots.plot_mesh_file(figure_path(), r["msh"], g, h_required(g, SizingParams(12.5, 4, 8)), "3 couches, faces sur les interfaces")
    nodes, cells, _ = plots.read_funtides_msh(r["msh"])     # any strategy: no element across an interface
    depth = nodes[:, 2].max() - nodes[:, 2]
    zmin, zmax = depth[cells].min(1), depth[cells].max(1)
    for zi in (500.0, 1000.0):
        assert not np.any((zmin < zi - 1e-6) & (zmax > zi + 1e-6))
        assert np.any(np.isclose(depth, zi))
    rows = list(csv.DictReader(open(r["materials"])))
    assert sorted(float(row["vp"]) for row in rows) == [1800.0, 3000.0, 4000.0]


def test_slowness_average(tmp_path):
    """Element across an interface: harmonic mean (mean slowness) by default, arithmetic on request."""
    layered_sep(tmp_path / "v.H")
    from makutu.mesh.seismesh.build import _element_boxes
    from makutu.mesh.seismesh.velocity import VelocityGrid
    g = grid_from_sep(tmp_path / "v.H")
    nodes = np.array([[0, 450], [10, 450], [10, 550], [0, 550]], float)    # cells 450..550: 5 x 1800, 5 x 3000
    lo, hi = _element_boxes(nodes, np.arange(4)[None], 2)
    vmean = g.box_minmax(lo, hi)[2][0]
    slow = 1.0 / VelocityGrid(1.0 / g.vel, g.origin, g.spacing).box_minmax(lo, hi)[2][0]
    assert vmean == pytest.approx((5 * 1800 + 5 * 3000) / 10)
    assert slow == pytest.approx(10 / (5 / 1800 + 5 / 3000))
    assert slow < vmean


def test_hybrid_2d_local_refinement_is_conforming(tmp_path, figure_path):
    """3 layers + slow inclusion: doubling at depth and local 1:3 refinement around the inclusion
    (vertex and edge transition templates), conforming, covering the domain, sampled everywhere."""
    from makutu.mesh.seismesh.plane import mesh_hybrid_2d
    from makutu.mesh.seismesh.refine3_2d import LOOKUP
    assert sorted(len(v[1]) for k, v in LOOKUP.items() if v) == [3] * 4 + [7] * 4
    d = 10.0
    z = (np.arange(150)[:, None] + 0.5) * d * np.ones((1, 200))
    x = (np.arange(200)[None, :] + 0.5) * d * np.ones((150, 1))
    v = np.where(z < 500, 1800.0, np.where(z < 1000, 3000.0, 4000.0))
    v[(x - 1000) ** 2 + (z - 750) ** 2 < 200 ** 2] = 1500.0
    _write_sep(str(tmp_path / "v.H"), v.astype(np.float32)[None], d, d, 1.0)
    grid = grid_from_sep(tmp_path / "v.H")
    p = SizingParams(fmax=12.5, order=4, ppw=8)
    m = mesh_hybrid_2d(grid, h_required(grid, p), max_levels=1, q_bg=0.3)
    if figure_path():
        plots.plot_quad_mesh(figure_path(), grid, h_required(grid, p), m.nodes, m.quads, "hybrid 2-D, inclusion lente")
    assert m.params["refined_zones"]
    rep = evaluate_2d(m, grid, p)
    assert rep["ok"] and rep["scaled_jacobian"]["min"] > 0.3
    assert set(edge_counts(m.quads)) <= {1, 2}
    X = m.nodes[m.quads]
    area = 0.5 * np.abs(np.sum(X[:, :, 0] * np.roll(X[:, :, 1], -1, 1) - np.roll(X[:, :, 0], -1, 1) * X[:, :, 1], 1))
    assert area.sum() == pytest.approx(2000.0 * 1500.0)
    centre = X.mean(axis=1)
    inside = (centre[:, 0] - 1000) ** 2 + (centre[:, 1] - 750) ** 2 < 150 ** 2
    assert np.ptp(X[inside], axis=1).max() < 60.0                  # refined: required size 4*1500/(12.5*8) = 60 m


def salt_sep(path, d=20.0):
    """Water (1500 m/s, 200 m), sediments 2000 -> 2600 m/s, large salt body (4500 m/s): 4000 x 2000 m."""
    z = (np.arange(100)[:, None] + 0.5) * d * np.ones((1, 200))
    x = (np.arange(200)[None, :] + 0.5) * d * np.ones((100, 1))
    v = np.where(z < 200, 1500.0, 2000.0 + 0.3 * (z - 200))
    v[((x - 2000) / 1300) ** 2 + ((z - 1000) / 400) ** 2 < 1] = 4500.0
    _write_sep(str(path), v.astype(np.float32)[None], d, d, 1.0)
    return path


def test_quadtree_coarsens_fast_body_and_refines_slow_zones(tmp_path, figure_path):
    """From the fastest velocity: 2:1 quadtree, conforming, covering the domain, sampled everywhere;
    the salt keeps the largest elements, the water layer the smallest."""
    from makutu.mesh.seismesh.plane import mesh_quadtree_2d
    from makutu.mesh.seismesh.quadtree_2d import LOOKUP
    assert len(LOOKUP) == 15                       # every set of finer sides except the four of them
    grid = grid_from_sep(salt_sep(tmp_path / "v.H"))
    p = SizingParams(fmax=12.5, order=4, ppw=8)
    m = mesh_quadtree_2d(grid, h_required(grid, p), max_block_elements=0.75 * h_required(grid, p).max())
    if figure_path():
        plots.plot_quad_mesh(figure_path(), grid, h_required(grid, p), m.nodes, m.quads, "quadtree 2-D, sel")
    rep = evaluate_2d(m, grid, p)
    assert rep["ok"] and rep["scaled_jacobian"]["min"] > 0.5
    assert set(edge_counts(m.quads)) <= {1, 2}
    X = m.nodes[m.quads]
    area = 0.5 * np.abs(np.sum(X[:, :, 0] * np.roll(X[:, :, 1], -1, 1) - np.roll(X[:, :, 0], -1, 1) * X[:, :, 1], 1))
    assert area.sum() == pytest.approx(4000.0 * 2000.0)
    c, size = X.mean(axis=1), np.linalg.norm(np.roll(X, -1, 1) - X, axis=2).max(axis=1)
    salt = ((c[:, 0] - 2000) / 1100) ** 2 + ((c[:, 1] - 1000) / 250) ** 2 < 1
    water = c[:, 1] < 150
    assert np.median(size[salt]) >= 1.9 * np.median(size[~salt & ~water])   # coarser in the salt
    assert np.median(size[water]) < np.median(size[~salt & ~water])          # finer in the water


def test_tetsplit_3d_all_hex_adapted(tmp_path, figure_path):
    """3-D salt-like model: gmsh tetrahedra sized by the velocity split into hexahedra: all hex,
    positive jacobian, conforming (every interior face shared by two hexahedra), larger elements
    in the fast body, free surface on top."""
    from makutu.mesh.seismesh.mesher import boundary_quads, mesh_tetsplit
    from makutu.mesh.seismesh.quality import evaluate
    d, n = 50.0, (20, 20, 12)                                   # 1000 x 1000 x 600 m
    z = (np.arange(n[2])[:, None, None] + 0.5) * d * np.ones((1, n[1], n[0]))
    y = (np.arange(n[1])[None, :, None] + 0.5) * d * np.ones((n[2], 1, n[0]))
    x = (np.arange(n[0])[None, None, :] + 0.5) * d * np.ones((n[2], n[1], 1))
    v = np.where(z < 100, 1500.0, 2200.0)
    v[((x - 500) / 300) ** 2 + ((y - 500) / 300) ** 2 + ((z - 350) / 150) ** 2 < 1] = 4500.0
    _write_sep(str(tmp_path / "v.H"), v.astype(np.float32), d, d, d)
    grid = grid_from_sep(tmp_path / "v.H")
    p = SizingParams(fmax=5.0, order=3, ppw=6)
    m = mesh_tetsplit(grid, h_required(grid, p), size_factor=0.5)
    if figure_path():
        plots.plot_hex_mesh(figure_path(), grid, h_required(grid, p), m.nodes, m.hexes, "tetsplit 3-D, sel")
    rep = evaluate(m, grid, p)
    assert m.n_non_hex == 0 and rep["scaled_jacobian"]["min"] > 0
    faces = np.sort(m.hexes[:, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]]
                    .reshape(-1, 4), axis=1)
    _, cnt = np.unique(faces, axis=0, return_counts=True)
    assert set(cnt) <= {1, 2}
    bq = boundary_quads(m.hexes)
    B = m.nodes[bq]
    on_box = np.zeros(len(bq), bool)
    for k, (a, b) in enumerate(grid.extent):
        on_box |= np.all(np.isclose(B[:, :, k], a), 1) | np.all(np.isclose(B[:, :, k], b), 1)
    assert on_box.all()                                         # no internal boundary (interfaces conform)
    X = m.nodes[m.hexes]
    c, size = X.mean(1), np.linalg.norm(X[:, 1] - X[:, 0], axis=1)
    salt = ((c[:, 0] - 500) / 250) ** 2 + ((c[:, 1] - 500) / 250) ** 2 + ((c[:, 2] - 350) / 120) ** 2 < 1
    assert np.median(size[salt]) > 1.5 * np.median(size[~salt])


def _salt_grid(tmp_path, d=50.0, n=(20, 20, 12), rz=150.0):
    z = (np.arange(n[2])[:, None, None] + 0.5) * d * np.ones((1, n[1], n[0]))
    y = (np.arange(n[1])[None, :, None] + 0.5) * d * np.ones((n[2], 1, n[0]))
    x = (np.arange(n[0])[None, None, :] + 0.5) * d * np.ones((n[2], n[1], 1))
    v = np.where(z < 100, 1500.0, 2200.0)
    v[((x - 500) / 300) ** 2 + ((y - 500) / 300) ** 2 + ((z - 350) / rz) ** 2 < 1] = 4500.0
    _write_sep(str(tmp_path / "v.H"), v.astype(np.float32), d, d, d)
    return grid_from_sep(tmp_path / "v.H")


def _conforming_on_box(m, grid):
    faces = np.sort(m.hexes[:, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]]
                    .reshape(-1, 4), axis=1)
    _, cnt = np.unique(faces, axis=0, return_counts=True)
    from makutu.mesh.seismesh.mesher import boundary_quads
    B = m.nodes[boundary_quads(m.hexes)]
    on_box = np.zeros(len(B), bool)
    for k, (a, b) in enumerate(grid.extent):
        on_box |= np.all(np.isclose(B[:, :, k], a), 1) | np.all(np.isclose(B[:, :, k], b), 1)
    return set(cnt) <= {1, 2} and on_box.all()


def test_refine3tz_keeps_coarse_salt(tmp_path, figure_path):
    """3-D salt-like model, strategy « du grossier au fin » with the Tong & Zhang (2026) templates
    (H. Tong, Y. J. Zhang, Element-saving hexahedral 3-refinement templates, CAGD 102550): base grid
    from the fastest velocity, 1:3 refinement in the slow zones; conforming, positive jacobian,
    sampling reached everywhere, coarse regular cells kept inside the salt."""
    from makutu.mesh.seismesh.mesher import mesh_refine3
    from makutu.mesh.seismesh.quality import evaluate
    grid = _salt_grid(tmp_path, rz=220.0)
    p = SizingParams(fmax=25.0, order=3, ppw=6)        # salt ~ 6 base cells wide: a coarse core needs >= 3
    h = h_required(grid, p)
    m = mesh_refine3(grid, h, base_max_segments=(1, 1, 10), base_size=float(h.max()), templates="tz")
    if figure_path():
        plots.plot_hex_mesh(figure_path("y"), grid, h, m.nodes, m.hexes, "refine3tz 3-D, sel")
        plots.plot_hex_mesh(figure_path("z"), grid, h, m.nodes, m.hexes, "refine3tz 3-D, sel", y=350.0, axis=2)
    rep = evaluate(m, grid, p)
    assert rep["scaled_jacobian"]["min"] > 0.15 and rep["frac_undersampled"] == 0.0
    assert _conforming_on_box(m, grid)
    assert m.params["leaves_per_level"].get(0, 0) > 0                 # coarse cells survive
    X = m.nodes[m.hexes]
    c, size = X.mean(1), np.linalg.norm(X[:, 1] - X[:, 0], axis=1)
    coarse = size > 0.95 * size.max()                                 # untouched base cells
    salt = ((c[:, 0] - 500) / 300) ** 2 + ((c[:, 1] - 500) / 300) ** 2 + ((c[:, 2] - 350) / 220) ** 2 < 1
    assert coarse.sum() >= 8 and salt[coarse].all()
    assert size.max() > 2.5 * np.median(size[~salt])


def test_refine3tz_two_levels_conforming(figure_path):
    """Two refinement levels (1/3, 1/9) driven by a point: touch balance (leaves that touch differ by at
    most one level) and Tong & Zhang templates give a conforming mesh with positive jacobian."""
    from makutu.mesh.seismesh.refine3 import Refine3Mesh
    from makutu.mesh.seismesh.mesher import HexMesh
    R = Refine3Mesh(*[np.linspace(0, 270.0, 4)] * 3, lmax=2)
    target = np.array([100.0, 100.0, 100.0])
    R.refine_where(lambda lo, hi, level: bool(np.all(lo - 1e-9 <= target) and np.all(target <= hi + 1e-9)))
    bal = R.balance_touch()
    nodes, hexes, info = R.to_hexes_tz()
    m = HexMesh(nodes, hexes, "refine3tz", info)
    assert set(info["leaves_per_level"]) == {0, 1, 2}
    from makutu.mesh.seismesh.quality import element_metrics
    sj = element_metrics(m)[0]
    assert sj.min() > 0.15

    class _Box:
        extent = [(0.0, 270.0)] * 3
    assert _conforming_on_box(m, _Box)
    vol = 0.0
    X = nodes[hexes]
    for f in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
        fc = X[:, f].mean(1)
        for a, b in zip(f, np.roll(f, -1)):
            vol += np.einsum("ij,ij->i", np.cross(X[:, a] - fc, X[:, b] - fc), X.mean(1) - fc).sum() / 6.0
    assert abs(abs(vol) - 270.0 ** 3) < 1e-6 * 270.0 ** 3
    if figure_path():
        from matplotlib.collections import PolyCollection
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        polys, ids = plots.hex_section(nodes, hexes, 101.0)
        pc = PolyCollection(polys, array=sj[ids], cmap="viridis", edgecolors="k", linewidths=0.3)
        pc.set_clim(0, 1)
        ax.add_collection(pc)
        ax.set_xlim(0, 270)
        ax.set_ylim(270, 0)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, shrink=0.8, label="jacobien normalisé")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(f"Deux niveaux (90 m → 30 m → 10 m) autour du point (100, 100, 100)\n"
                     f"coupe y = 101 m — {len(hexes)} hexaèdres, jacobien min {sj.min():.2f}, "
                     f"{bal['forced_splits']} raffinements d'équilibrage", fontsize=9)
        plots._save(fig, figure_path())



def test_element_values_on_real_geometry(tmp_path):
    """Codex, point 3: a trapezoid (0,0), (2,0), (1,2), (0,2) on 2 x 2 unit cells, 1500 m/s except the cell
    [1,2] x [1,2] at 4500 m/s: slowness average on the real element 1588.24 m/s (0.25 of its area 3 in the
    fast cell), 1800 m/s on its bounding box; rectangles stay exact."""
    from makutu.mesh.seismesh.elements2d import element_stats_2d, element_box_stats_2d
    v = np.full((2, 2), 1500.0)                                   # (depth, x)
    v[1, 1] = 4500.0
    _write_sep(str(tmp_path / "v.H"), v.astype(np.float32)[None], 1.0, 1.0, 1.0)
    grid = grid_from_sep(tmp_path / "v.H")
    nodes = np.array([[0, 0], [2, 0], [1, 2], [0, 2], [2, 2], [1, 0]], float)
    st = element_stats_2d(grid, nodes, np.array([[0, 1, 2, 3]]))
    assert st["harmonic"][0] == pytest.approx(3.0 / (2.75 / 1500 + 0.25 / 4500), rel=0.01)
    assert st["min"][0] == 1500.0 and st["max"][0] == 4500.0
    assert element_box_stats_2d(grid, nodes, np.array([[0, 1, 2, 3]]))[1][0] == 4500.0
    rect = element_stats_2d(grid, nodes, np.array([[5, 1, 4, 2]]))    # (1,0) (2,0) (2,2) (1,2): exact
    assert rect["harmonic"][0] == pytest.approx(2.0 / (1 / 1500 + 1 / 4500))


def test_strict_sampling_validity(tmp_path):
    """Codex, point 2: one undersampled element out of many invalidates the mesh."""
    from makutu.mesh.seismesh.plane import QuadMesh
    grid = grid_from_sep(gradient_sep(tmp_path / "v.H", nx=200, nz=100, gradient=0.0))   # 1500 m/s, 4 x 2 km
    p = SizingParams(fmax=12.5, order=4, ppw=8)                   # required size 60 m
    xs, zs = np.linspace(0, 4000, 81), np.linspace(0, 2000, 41)     # 50 m squares: valid
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    nodes = np.stack([X.ravel(), Z.ravel()], -1)
    n = lambda i, k: i * len(zs) + k
    quads = np.array([[n(i, k), n(i + 1, k), n(i + 1, k + 1), n(i, k + 1)] for i in range(80) for k in range(40)])
    assert evaluate_2d(QuadMesh(nodes, quads, "tensor", {}), grid, p)["ok"]
    nodes2 = nodes.copy()
    nodes2[n(40, 20)] += [30.0, 30.0]                             # one node moved: its elements become too large
    rep = evaluate_2d(QuadMesh(nodes2, quads, "tensor", {}), grid, p)
    assert not rep["ok"] and 1 <= rep["n_undersampled"] <= 4 and rep["frac_undersampled"] < 1e-3


def test_bands_regular_layers_with_deformed_transitions(tmp_path, figure_path):
    """``bands``: regular depth bands, each with its own number of columns, joined in the coarser band by
    a row of 3 -> 1 templates and morphing rows; conforming, covering, sampled in every element, mostly
    rectangles, fewer elements than ``tensor`` on a layered model; selected by the 2-D search."""
    from makutu.mesh.seismesh.bands2d import mesh_bands_2d
    from makutu.mesh.seismesh.elements2d import axis_aligned
    from makutu.mesh.seismesh.plane import autopilot_2d
    grid = grid_from_sep(salt_sep(tmp_path / "v.H"))
    p = SizingParams(fmax=12.5, order=4, ppw=8)
    h = h_required(grid, p)
    m = mesh_bands_2d(grid, h, max_segments=8, coarsen=1.1)
    rep = evaluate_2d(m, grid, p)
    if figure_path():
        plots.plot_quad_mesh_detail(figure_path(), grid, h, m.nodes, m.quads, "bands 2-D, sel", (1000, 3000, 0, 1500))
    assert rep["ok"] and rep["n_undersampled"] == 0 and rep["scaled_jacobian"]["min"] > 0.3
    assert set(edge_counts(m.quads)) <= {1, 2}
    X = m.nodes[m.quads]
    area = 0.5 * np.abs(np.sum(X[:, :, 0] * np.roll(X[:, :, 1], -1, 1) - np.roll(X[:, :, 0], -1, 1) * X[:, :, 1], 1))
    assert area.sum() == pytest.approx(4000.0 * 2000.0)
    assert axis_aligned(X).mean() > 0.7
    assert len(m.params["bands"]) >= 2 and all(t["columns"][0] > t["columns"][1] for t in m.params["transitions"])
    tensor = mesh_tensor_2d(grid, h, segments_2d(grid, h, (6, 8)))
    assert evaluate_2d(tensor, grid, p)["ok"] and rep["n_elements"] < len(tensor.quads)
    (best, brep), _ = autopilot_2d(grid, p)
    assert brep["strategy"] == "bands"


def test_cfl_reaches_the_evaluation(tmp_path):
    """Codex, point 1: ``cfl`` of build_mesh_from_sep reaches the evaluation and the report; for a fixed
    mesh, half the cfl gives half the time step."""
    sep = gradient_sep(tmp_path / "v.H")
    grid = grid_from_sep(sep)
    p = SizingParams(fmax=12.5, order=4, ppw=5)
    h = h_required(grid, p)
    m = mesh_tensor_2d(grid, h, segments_2d(grid, h, (3, 5)))
    r1, r2 = evaluate_2d(m, grid, p, cfl=0.5), evaluate_2d(m, grid, p, cfl=0.25)
    assert r2["dt_cfl"] == pytest.approx(0.5 * r1["dt_cfl"]) and r2["cost_index"] == pytest.approx(2 * r1["cost_index"])
    a = build_mesh_from_sep(sep, f0=5.0, out_dir=tmp_path / "a", order=4, ppw=5, cfl=0.5, verbose=False)
    b = build_mesh_from_sep(sep, f0=5.0, out_dir=tmp_path / "b", order=4, ppw=5, cfl=0.25, verbose=False)
    import json
    ra, rb = json.loads(open(a["report"]).read()), json.loads(open(b["report"]).read())
    assert ra["cfl"] == 0.5 and rb["cfl"] == 0.25 and rb["selected"]["cfl"] == 0.25


def test_meshes_crossing_a_sharp_interface_are_rejected(tmp_path):
    """Codex, point 5: an interface at 50 m in a 1000 m deep model is too close to the surface for the
    quadtree blocks: it is reported as ignored, and the candidates crossing it are rejected."""
    from makutu.mesh.seismesh.plane import mesh_quadtree_2d, autopilot_2d
    from makutu.mesh.seismesh.sizing import horizontal_interfaces
    z = (np.arange(100)[:, None] + 0.5) * 10.0 * np.ones((1, 200))
    _write_sep(str(tmp_path / "v.H"), np.where(z < 50, 1500.0, 3000.0).astype(np.float32)[None], 10.0, 10.0, 1.0)
    grid = grid_from_sep(tmp_path / "v.H")
    assert horizontal_interfaces(grid) == [50.0]
    p = SizingParams(fmax=12.5, order=4, ppw=8)
    h = h_required(grid, p)
    m = mesh_quadtree_2d(grid, h)
    rep = evaluate_2d(m, grid, p)
    assert m.params["ignored_interfaces"] == [50.0]
    assert not rep["ok"] and rep["interfaces_crossed"].get(50.0, 0) > 0
    (best, brep), _ = autopilot_2d(grid, p)                      # the search keeps a mesh with a face at 50 m
    assert brep["ok"] and not brep["interfaces_crossed"]
