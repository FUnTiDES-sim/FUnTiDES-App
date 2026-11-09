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
def test_doubling_2d_is_conforming(tmp_path, refine):
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


def test_build_from_sep_selects_doubling(tmp_path):
    """Velocity tripling with depth: the 2:1 doubling mesh is cheaper than the tensor grid."""
    gradient_sep(tmp_path / "v.H")
    r = build_mesh_from_sep(tmp_path / "v.H", f0=5.0, out_dir=tmp_path / "out", order=4, ppw=5, verbose=False)
    sel = r["selected"]
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


def test_element_faces_follow_horizontal_interfaces(tmp_path):
    """Faces at the sharp horizontal interfaces (cell boundaries): one velocity per layer, no mixed element."""
    from makutu.mesh.seismesh.sizing import horizontal_interfaces
    layered_sep(tmp_path / "v.H")
    assert horizontal_interfaces(grid_from_sep(tmp_path / "v.H")) == [500.0, 1000.0]
    r = build_mesh_from_sep(tmp_path / "v.H", f0=5.0, out_dir=tmp_path / "out", order=4, ppw=8, verbose=False)
    zs = []
    for t0, t1, n in r["selected"]["params"]["segments"][1]:
        zs += list(np.linspace(t0, t1, n + 1))
    assert np.any(np.isclose(zs, 500.0)) and np.any(np.isclose(zs, 1000.0))
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


def test_hybrid_2d_local_refinement_is_conforming(tmp_path):
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


def test_quadtree_coarsens_fast_body_and_refines_slow_zones(tmp_path):
    """From the fastest velocity: 2:1 quadtree, conforming, covering the domain, sampled everywhere;
    the salt keeps the largest elements, the water layer the smallest."""
    from makutu.mesh.seismesh.plane import mesh_quadtree_2d
    from makutu.mesh.seismesh.quadtree_2d import LOOKUP
    assert len(LOOKUP) == 15                       # every set of finer sides except the four of them
    grid = grid_from_sep(salt_sep(tmp_path / "v.H"))
    p = SizingParams(fmax=12.5, order=4, ppw=8)
    m = mesh_quadtree_2d(grid, h_required(grid, p), max_block_elements=0.75 * h_required(grid, p).max())
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
