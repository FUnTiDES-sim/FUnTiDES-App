"""
Gmsh meshes for the FUnTiDES SEM solvers: quadrilaterals Q4 in the (x, z) plane (2-D)
and hexahedra Hex8 (3-D), read by the FUnTiDES C++ readers ``read_gmsh_2d`` and
``read_gmsh_3d`` (MSH 2.2 ASCII), which build the GLL nodes at the solver order.

- :func:`mshVersion`, :func:`convertToMsh22`: format of a ``.msh`` file and conversion of
  a linear Q4/Hex8 mesh to MSH 2.2 (needs the ``gmsh`` Python module);
- :func:`generateHexBox`, :func:`generateQuadRectangle`: test meshes (``gmsh`` module);
- :class:`GmshMesh`: FUnTiDES model of a mesh with one acoustic material per physical
  group, and location of sources and receivers in its elements.

Coordinates: the mesh keeps the Gmsh coordinates, z up (2-D: the (x, z) plane, y = 0).
makutu uses the seismic convention, z positive downwards from the free surface: a point
at depth z is placed at the mesh height ``top - z``, with ``top`` the highest node of the
mesh (:attr:`GmshMesh.top`).

Physical groups: each cell group is a material (``vp``, ``rho``); each boundary group
(lines in 2-D, surfaces in 3-D) is a free surface or an absorbing boundary. Every outer
face must belong to a boundary group.
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

import pyfuntides.model as FTModel

_GMSH_Q4, _GMSH_HEX8 = 3, 5

# Boundary names accepted by GmshMesh -> FUnTiDES flags
_BOUNDARY_NAMES = {
    "surface": "Surface", "free": "Surface", "free_surface": "Surface",
    "damping": "Damping", "absorbing": "Damping",
}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def mshVersion(filename):
    """Version of a ``.msh`` file (2.2, 4.1, ...) and whether it is binary."""
    with open(filename, "rb") as f:
        if f.readline().strip() != b"$MeshFormat":
            raise ValueError(f"{filename}: not a Gmsh .msh file")
        version, binary = f.readline().split()[:2]
    return float(version), bool(int(binary))


def physicalNames(filename):
    """``{(dim, tag): name}`` of the ``$PhysicalNames`` section of a ``.msh`` file."""
    names = {}
    with open(filename, "rb") as f:
        for line in f:
            if line.strip() == b"$PhysicalNames":
                for _ in range(int(f.readline())):
                    dim, tag, name = f.readline().decode().split(maxsplit=2)
                    names[(int(dim), int(tag))] = name.strip().strip('"')
                break
    return names


def _gmsh():
    try:
        import gmsh
    except ImportError as e:
        raise ImportError("the gmsh Python module is needed (pip install gmsh)") from e
    return gmsh


def _write(gmsh, output, version):
    gmsh.option.setNumber("Mesh.MshFileVersion", version)
    gmsh.option.setNumber("Mesh.Binary", 0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output))


def convertToMsh22(input, output, plane="xz"):
    """
    Write a linear Q4 (2-D) or Hex8 (3-D) Gmsh mesh in the MSH 2.2 ASCII format of the
    FUnTiDES readers, keeping the physical groups. Triangles, tetrahedra, mixed or
    curved (high-order) meshes are rejected, not converted.

    Parameters
    ----------
        input, output : str or Path
            Mesh files (any format Gmsh reads; distinct paths)
        plane : {"xz", "xy"}
            2-D only: plane of the input mesh; an (x, y) mesh is rotated into the
            (x, z) plane (y becomes z). Default is "xz"
    """
    if plane not in ("xz", "xy"):
        raise ValueError(f"plane must be 'xz' or 'xy', not {plane!r}")
    if Path(input).resolve() == Path(output).resolve():
        raise ValueError("convertToMsh22: use a distinct output path")
    gmsh = _gmsh()
    gmsh.initialize(["gmsh", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(input))
        dim = gmsh.model.getDimension()
        expected = _GMSH_HEX8 if dim == 3 else _GMSH_Q4
        if dim not in (2, 3) or list(gmsh.model.mesh.getElementTypes(dim)) != [expected]:
            raise ValueError("expected only linear Q4 quadrilaterals (2-D) or Hex8 hexahedra (3-D)")
        if dim == 3 and plane == "xy":
            raise ValueError("plane 'xy' applies only to 2-D meshes")
        if plane == "xy":
            gmsh.model.mesh.affineTransform([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        _write(gmsh, output, 2.2)
    finally:
        gmsh.finalize()


def generateHexBox(output, lengths=(1.0, 1.0, 1.0), cells=2, msh_version=2.2,
                   vp=1500.0, rho=1000.0, structured=True):
    """
    Hex8 box [0, lx] x [0, ly] x [0, lz] (z up) with physical groups:
    volume 10 (material), surface 1 = top (free surface), surface 2 = the five other
    sides (absorbing). Also writes ``<output>.csv`` (``physical_tag,vp,rho``).

    Parameters
    ----------
        output : str or Path
        lengths : (lx, ly, lz)
        cells : int or (ex, ey, ez)
            Number of cells along each axis (unstructured: target numbers)
        msh_version : {2.2, 4.1}
        structured : bool
            Regular grid (True), or unstructured quadrilaterals in the (x, y) plane
            (frontal-Delaunay + blossom recombination) extruded along z in ez layers
            (False): hexahedra with general horizontal faces
    """
    ex, ey, ez = (cells,) * 3 if np.isscalar(cells) else cells
    if min(ex, ey, ez) < 1:
        raise ValueError("generateHexBox: cells must be positive")
    lx, ly, lz = lengths
    gmsh = _gmsh()
    gmsh.initialize(["gmsh", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("hex_box")
        geo = gmsh.model.geo
        size = min(lx / ex, ly / ey)
        points = [geo.addPoint(x, y, 0, size) for x, y in [(0, 0), (lx, 0), (lx, ly), (0, ly)]]
        edges = [geo.addLine(points[i], points[(i + 1) % 4]) for i in range(4)]
        surface = geo.addPlaneSurface([geo.addCurveLoop(edges)])
        if structured:
            for i, edge in enumerate(edges):
                geo.mesh.setTransfiniteCurve(edge, (ex if i % 2 == 0 else ey) + 1)
            geo.mesh.setTransfiniteSurface(surface)
        else:
            gmsh.option.setNumber("Mesh.Algorithm", 6)               # frontal-Delaunay
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # blossom
        geo.mesh.setRecombine(2, surface)
        extrusion = geo.extrude([(2, surface)], 0, 0, lz, [ez], recombine=True)
        geo.synchronize()
        top = extrusion[0][1]
        volumes = [tag for dim, tag in extrusion if dim == 3]
        sides = [tag for dim, tag in extrusion[2:] if dim == 2]
        gmsh.model.addPhysicalGroup(3, volumes, 10, name="medium")
        gmsh.model.addPhysicalGroup(2, [top], 1, name="free_surface")
        gmsh.model.addPhysicalGroup(2, [surface] + sides, 2, name="absorbing")
        gmsh.model.mesh.generate(3)
        if list(gmsh.model.mesh.getElementTypes(3)) != [_GMSH_HEX8]:
            raise RuntimeError("generateHexBox: expected only Hex8 cells")
        _write(gmsh, output, msh_version)
    finally:
        gmsh.finalize()
    Path(output).with_suffix(".csv").write_text(f"physical_tag,vp,rho\n10,{vp:g},{rho:g}\n")


def generateQuadRectangle(output, lx, lz, size, structured=False, free_surface=True,
                          msh_version=2.2):
    """
    Q4 mesh of the rectangle [0, lx] x [0, lz] of the (x, z) plane (z up), physical
    groups: surface 10 (material), line 1 = top edge (free surface, or absorbing with
    ``free_surface=False``), line 2 = the other edges (absorbing).

    Parameters
    ----------
        size : float
            Target element size [m]
        structured : bool
            Regular grid (True) or unstructured quadrilaterals (False, Gmsh
            frontal-Delaunay triangles + blossom recombination)
    """
    gmsh = _gmsh()
    gmsh.initialize(["gmsh", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("quad_rectangle")
        geo = gmsh.model.geo
        points = [geo.addPoint(x, 0, z, size) for x, z in [(0, 0), (lx, 0), (lx, lz), (0, lz)]]
        edges = [geo.addLine(points[i], points[(i + 1) % 4]) for i in range(4)]   # bottom, right, top, left
        surface = geo.addPlaneSurface([geo.addCurveLoop(edges)])
        if structured:
            for i, edge in enumerate(edges):
                geo.mesh.setTransfiniteCurve(edge, int(round((lx if i % 2 == 0 else lz) / size)) + 1)
            geo.mesh.setTransfiniteSurface(surface)
        geo.mesh.setRecombine(2, surface)
        geo.synchronize()
        if not structured:
            gmsh.option.setNumber("Mesh.Algorithm", 6)               # frontal-Delaunay
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # blossom
        gmsh.model.addPhysicalGroup(2, [surface], 10, name="medium")
        top, others = [edges[2]], [edges[0], edges[1], edges[3]]
        if free_surface:
            gmsh.model.addPhysicalGroup(1, top, 1, name="free_surface")
            gmsh.model.addPhysicalGroup(1, others, 2, name="absorbing")
        else:
            gmsh.model.addPhysicalGroup(1, top + others, 2, name="absorbing")
        gmsh.model.mesh.generate(2)
        if list(gmsh.model.mesh.getElementTypes(2)) != [_GMSH_Q4]:
            raise RuntimeError("generateQuadRectangle: the recombination left non-quadrilateral cells")
        _write(gmsh, output, msh_version)
    finally:
        gmsh.finalize()


def readMaterials(filename):
    """``{physical_tag: (vp, rho)}`` from a CSV file with columns ``physical_tag,vp,rho``."""
    with open(filename, newline="") as f:
        return {int(row["physical_tag"]): (float(row["vp"]), float(row.get("rho") or 1.0))
                for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Reference element
# ---------------------------------------------------------------------------

def _gll(order):
    from .CartesianModel2D import gll_points
    return np.asarray(gll_points(order), dtype=float)


def _lagrange(xi, pts):
    """Lagrange basis on the points pts at the abscissae xi: shape (len(xi), len(pts))."""
    xi = np.atleast_1d(np.asarray(xi, dtype=float))[:, None]
    w = np.ones((xi.shape[0], len(pts)))
    for i in range(len(pts)):
        for j in range(len(pts)):
            if j != i:
                w[:, i] *= (xi[:, 0] - pts[j]) / (pts[i] - pts[j])
    return w


def _shape(xi):
    """
    Bilinear (2-D) or trilinear (3-D) shape functions and their derivatives at the
    reference points xi (n, dim). Vertices: (a, c) or (a, b, c) in {0, 1}, a fastest.
    Returns N (n, nv) and dN (n, nv, dim).
    """
    n, dim = xi.shape
    lo, hi = 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)
    factors = np.stack([lo, hi], axis=1)                  # (n, 2, dim)
    nv = 2 ** dim
    N = np.ones((n, nv))
    dN = np.ones((n, nv, dim))
    for v in range(nv):
        bits = [(v >> d) & 1 for d in range(dim)]
        for d in range(dim):
            f = factors[:, bits[d], d]
            df = 0.5 if bits[d] else -0.5
            N[:, v] *= f
            for k in range(dim):
                dN[:, v, k] *= df if k == d else f
    return N, dN


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

class GmshMesh:
    """
    FUnTiDES model of a Gmsh mesh (one acoustic material per cell group) and location
    of points in its elements.

    Parameters
    ----------
        filename : str or Path
            ``.msh`` file; another format or version than MSH 2.2 ASCII is converted
            first (``gmsh`` Python module)
        order : int
            Polynomial order of the SEM
        materials : dict or str
            ``{physical_tag: (vp, rho)}`` (or ``{tag: vp}``, rho = 1), or the path of a
            CSV file ``physical_tag,vp,rho``
        boundaries : dict, optional
            ``{physical_tag: "surface" | "damping"}`` (also ``"free"``, ``"absorbing"``)
            for the boundary groups. Default: from the names of the groups, a name
            containing "free" or "surface" is a free surface, any other an absorbing
            boundary
        dim : {2, 3}, optional
            Dimension of the mesh. Default: from the file (3 if it holds hexahedra)

    Attributes
    ----------
        model : pyfuntides ModelUnstruct_f32_i32
        dim, order : int
        top : float
            Mesh height of the depth z = 0 (highest node)
        lower, upper : np.ndarray
            Bounding box of the mesh, (x, y, z), mesh coordinates
    """

    def __init__(self, filename, order, materials, boundaries=None, dim=None):
        self.filename = Path(filename)
        self.order = int(order)
        if isinstance(materials, (str, os.PathLike)):
            materials = readMaterials(materials)
        self.materials = {int(t): (tuple(map(float, m)) if np.ndim(m) else (float(m), 1.0))
                          for t, m in materials.items()}

        with tempfile.TemporaryDirectory() as tmp:
            path = self.filename
            if mshVersion(path) != (2.2, False):
                path = Path(tmp) / "mesh22.msh"
                convertToMsh22(self.filename, path)
            self.dim = dim or self._dimension(path)
            if boundaries is None:
                boundaries = self._boundariesFromNames(physicalNames(path))
            self.boundaries = {int(t): self._flag(f) for t, f in boundaries.items()}
            reader = FTModel.read_gmsh_3d if self.dim == 3 else FTModel.read_gmsh_2d
            self.data = reader(str(path), self.order,
                               {t: FTModel.GmshAcousticMaterial(vp, rho)
                                for t, (vp, rho) in self.materials.items()},
                               self.boundaries)
        self.model = FTModel.ModelUnstruct_f32_i32(self.data)
        if self.dim == 3:
            self.model.build_face_connectivity()   # needed by the 3-D SEM solvers
        self._corners = self._cornerCoordinates()
        pts = self._corners.reshape(-1, self.dim)
        self._lo_e, self._hi_e = self._corners.min(axis=1), self._corners.max(axis=1)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.lower, self.upper = self._xyz(lo), self._xyz(hi)
        self.top = float(hi[-1])
        self._tol = 1e-6 * float(np.max(hi - lo))

    # -- construction -------------------------------------------------------

    @staticmethod
    def _dimension(path):
        with open(path, "rb") as f:
            for line in f:
                if line.strip() == b"$Elements":
                    for _ in range(int(f.readline())):
                        if int(f.readline().split()[1]) == _GMSH_HEX8:
                            return 3
                    return 2
        raise ValueError(f"{path}: no $Elements section")

    def _boundariesFromNames(self, names):
        groups = {tag: name for (d, tag), name in names.items() if d == self.dim - 1}
        if not groups:
            raise ValueError(f"{self.filename}: no named boundary group; give the boundaries")
        return {tag: "surface" if ("free" in name.lower() or "surface" in name.lower()) else "damping"
                for tag, name in groups.items()}

    @staticmethod
    def _flag(flag):
        if isinstance(flag, str):
            key = flag.lower()
            if key not in _BOUNDARY_NAMES:
                raise ValueError(f"unknown boundary type {flag!r}: {sorted(_BOUNDARY_NAMES)}")
            return getattr(FTModel.BoundaryFlag, _BOUNDARY_NAMES[key])
        return flag

    def _cornerCoordinates(self):
        """Vertices of the elements, shape (n_elem, 2**dim, dim), mesh coordinates (x[, y], z)."""
        m, p = self.model, self.order
        ne = m.get_number_of_elements()
        axes = (0, 2) if self.dim == 2 else (0, 1, 2)
        nv = 2 ** self.dim
        corners = np.empty((ne, nv, self.dim))
        for v in range(nv):
            bits = [((v >> d) & 1) * p for d in range(self.dim)]
            ijk = (bits[0], 0, bits[1]) if self.dim == 2 else tuple(bits)
            for e in range(ne):
                n = m.global_node_index(e, *ijk)
                corners[e, v] = [m.node_coord(n, a) for a in axes]
        return corners

    def _xyz(self, point):
        return np.array([point[0], 0.0, point[1]]) if self.dim == 2 else np.asarray(point, float)

    # -- solver -------------------------------------------------------------

    def solver_kwargs(self):
        """Mesh keyword arguments of the makutu solvers (model, order, point location)."""
        return dict(model=self.model, order=self.order, locator=self)

    @property
    def n_elements(self):
        return self.model.get_number_of_elements()

    def materialOfElements(self):
        """Element vp and rho, from the model (shape (n_elem,) each)."""
        m = self.model
        ne = m.get_number_of_elements()
        return (np.array([m.get_model_vp_on_element(e) for e in range(ne)]),
                np.array([m.get_model_rho_on_element(e) for e in range(ne)]))

    # -- point location -----------------------------------------------------

    def meshPoint(self, x, y, z):
        """Mesh coordinates (x[, y], height) of a point given in makutu coordinates (z down)."""
        h = self.top - z
        return np.array([x, h]) if self.dim == 2 else np.array([x, y, h])

    def locate(self, x, y, z, tol=1e-6):
        """
        Element containing the point (x, y, z) (makutu coordinates, z down from the
        surface; y ignored in 2-D) and its reference coordinates xi in [-1, 1]^dim
        (Newton inversion of the bilinear / trilinear map of the candidate elements).

        Returns (element, xi), or (-1, None) when the point is outside the mesh.
        """
        P = self.meshPoint(x, y, z)
        cand = np.where(np.all((self._lo_e - self._tol <= P) & (P <= self._hi_e + self._tol), axis=1))[0]
        if cand.size == 0:
            return -1, None
        X = self._corners[cand]                                   # (nc, nv, dim)
        xi = np.zeros((cand.size, self.dim))
        for _ in range(30):
            N, dN = _shape(xi)
            r = np.einsum("nv,nvd->nd", N, X) - P                # residual x(xi) - P
            J = np.einsum("nvk,nvd->ndk", dN, X)                 # dx_d / dxi_k
            step = np.linalg.solve(J, r[..., None])[..., 0]
            xi -= step
            xi = np.clip(xi, -2.0, 2.0)                           # keep far candidates bounded
            if np.max(np.abs(step)) < 1e-12:
                break
        excess = np.max(np.abs(xi), axis=1) - 1.0
        best = int(np.argmin(excess))
        if excess[best] > tol:
            return -1, None
        return int(cand[best]), np.clip(xi[best], -1.0, 1.0)

    def _weights(self, xi):
        """Lagrange weights at xi per axis: list of (order+1,) arrays (x[, y], z)."""
        pts = _gll(self.order)
        return [_lagrange(x, pts)[0] for x in xi]

    def sourceWeights(self, xi):
        """Weights of a point source at xi, layout of the FUnTiDES right-hand side:
        ``ix + iz n`` (2-D), ``ix + iy n + iz n^2`` (3-D)."""
        w = self._weights(xi)
        if self.dim == 2:
            return np.outer(w[1], w[0]).ravel().astype(np.float32)
        return np.einsum("k,j,i->kji", w[2], w[1], w[0]).ravel().astype(np.float32)

    def receiverWeights(self, xi):
        """Weights of a receiver at xi, layout of the makutu receivers:
        ``ix + iz n`` (2-D), ``ix n^2 + iy n + iz`` (3-D)."""
        w = self._weights(xi)
        if self.dim == 2:
            return np.outer(w[1], w[0]).ravel().astype(np.float32)
        return np.einsum("i,j,k->ijk", w[0], w[1], w[2]).ravel().astype(np.float32)

    def __repr__(self):
        return (f"GmshMesh({self.filename.name}, {self.dim}-D, order {self.order}, "
                f"{self.n_elements} elements, {self.model.get_number_of_nodes()} nodes, "
                f"box {self.lower.tolist()} - {self.upper.tolist()})")
