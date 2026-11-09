"""Gmsh file tools of makutu.mesh.gmshmesh (no solver): generation, conversion,
physical names, materials, reference shape functions."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
from makutu.mesh.gmshmesh import (_shape, convertToMsh22, generateHexBox,  # noqa: E402
                                  generateQuadRectangle, mshVersion, physicalNames, readMaterials)

CLI = Path(__file__).resolve().parents[2] / "apps" / "mesh" / "gmsh_mesh.py"


def test_box_conversion_keeps_groups(tmp_path):
    generateHexBox(tmp_path / "b41.msh", (100.0, 200.0, 300.0), (2, 3, 4), msh_version=4.1, vp=2000.0, rho=2.0)
    assert mshVersion(tmp_path / "b41.msh") == (4.1, False)
    assert readMaterials(tmp_path / "b41.csv") == {10: (2000.0, 2.0)}
    convertToMsh22(tmp_path / "b41.msh", tmp_path / "b22.msh")
    assert mshVersion(tmp_path / "b22.msh") == (2.2, False)
    assert physicalNames(tmp_path / "b22.msh") == {(3, 10): "medium", (2, 1): "free_surface", (2, 2): "absorbing"}
    text = (tmp_path / "b22.msh").read_text()
    n_elements = int(text.split("$Elements\n")[1].split("\n")[0])
    assert n_elements > 2 * 3 * 4                             # hexahedra + boundary quadrilaterals
    with pytest.raises(ValueError, match="distinct"):
        convertToMsh22(tmp_path / "b22.msh", tmp_path / "b22.msh")
    with pytest.raises(ValueError, match="2-D"):
        convertToMsh22(tmp_path / "b41.msh", tmp_path / "x.msh", plane="xy")


def test_xy_rectangle_rotated_into_xz(tmp_path):
    """A mesh of the (x, y) plane becomes a mesh of the (x, z) plane (y -> z)."""
    gmsh.initialize(["gmsh", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        s = gmsh.model.occ.addRectangle(0, 0, 0, 3.0, 2.0)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setRecombine(2, s)
        gmsh.model.addPhysicalGroup(2, [s], 10)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(tmp_path / "xy.msh"))
    finally:
        gmsh.finalize()
    convertToMsh22(tmp_path / "xy.msh", tmp_path / "xz.msh", plane="xy")
    lines = (tmp_path / "xz.msh").read_text().split("$Nodes\n")[1].split("$EndNodes")[0].split("\n")[1:-1]
    xyz = np.array([[float(v) for v in line.split()[1:]] for line in lines])
    np.testing.assert_allclose(xyz[:, 1], 0.0, atol=1e-12)
    assert xyz[:, 2].max() == pytest.approx(2.0) and xyz[:, 0].max() == pytest.approx(3.0)


def test_rectangle_only_quadrilaterals(tmp_path):
    generateQuadRectangle(tmp_path / "r.msh", 1000.0, 500.0, 60.0)
    names = physicalNames(tmp_path / "r.msh")
    assert names == {(2, 10): "medium", (1, 1): "free_surface", (1, 2): "absorbing"}


@pytest.mark.parametrize("dim", [2, 3])
def test_shape_functions(dim):
    xi = np.random.default_rng(1).uniform(-1, 1, (7, dim))
    N, dN = _shape(xi)
    np.testing.assert_allclose(N.sum(axis=1), 1.0)
    np.testing.assert_allclose(dN.sum(axis=1), 0.0, atol=1e-14)
    # vertex v at (+-1, ...) with bit d of v giving the sign along d
    vertices = np.array([[1.0 if (v >> d) & 1 else -1.0 for d in range(dim)] for v in range(2 ** dim)])
    Nv, _ = _shape(vertices)
    np.testing.assert_allclose(Nv, np.eye(2 ** dim), atol=1e-14)


def test_cli_box_and_convert(tmp_path):
    run = [sys.executable, str(CLI)]
    subprocess.run(run + ["box", str(tmp_path / "b.msh"), "--cells", "2", "--msh-version", "4.1"], check=True)
    subprocess.run(run + ["convert", str(tmp_path / "b.msh"), str(tmp_path / "c.msh")], check=True)
    assert mshVersion(tmp_path / "c.msh") == (2.2, False)
