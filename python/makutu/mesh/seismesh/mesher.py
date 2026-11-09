"""Pilotage de gmsh : génération de maillages tout-hexaèdres conformes.

Stratégies
----------
tensor  : grille tensorielle à pas constant par morceaux (segments optimaux par axe),
          construite par extrusions structurées successives point→ligne→surface→volume.
          100 % hexaèdres, jacobien ≈ 1, très robuste ; pas latéral imposé par le min sur
          tout le plan orthogonal.
extrude : maillage quad non structuré de la surface (x,y) piloté par un champ de taille
          (min sur z), puis extrusion en couches en z (segments optimaux). Gagne quand la
          variabilité latérale est localisée (chenaux lents, proche surface).

Convention : z = profondeur, croissante vers le bas ; z = zmin est la surface libre.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import gmsh

from .velocity import VelocityGrid
from .sizing import Segment, plane_size_map

HEX8 = 5


@dataclass
class HexMesh:
    nodes: np.ndarray                 # (nn,3)
    hexes: np.ndarray                 # (ne,8) indices 0-based dans nodes
    strategy: str
    params: dict = field(default_factory=dict)
    n_non_hex: int = 0
    physical: dict = field(default_factory=dict)   # nom -> tags d'entités
    msh_path: str | None = None


def _layers(segs: list[Segment]):
    L = segs[-1].t1 - segs[0].t0
    num = [s.n for s in segs]
    heights = list(np.cumsum([(s.t1 - s.t0) / L for s in segs]))
    heights[-1] = 1.0
    return num, heights, L


def _init(verbose: bool):
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.option.setNumber("General.Verbosity", 2 if verbose else 0)
    gmsh.model.add("seismesh")


def _physical_groups(ext):
    """Surface libre (z=zmin) / bords absorbants / volume."""
    (x0, x1), (y0, y1), (z0, z1) = ext
    tol = 1e-6 * max(x1 - x0, y1 - y0, z1 - z0)
    free, absorb = [], []
    for _, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(2, tag)
        if abs(bb[2] - z0) < tol and abs(bb[5] - z0) < tol:
            free.append(tag)
        else:
            absorb.append(tag)
    vols = [t for _, t in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, vols, name="medium")
    gmsh.model.addPhysicalGroup(2, free, name="free_surface")
    gmsh.model.addPhysicalGroup(2, absorb, name="absorbing")
    return {"medium": vols, "free_surface": free, "absorbing": absorb}


def _collect(strategy, params, phys) -> HexMesh:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    remap = np.zeros(int(tags.max()) + 1, dtype=np.int64)
    remap[tags.astype(np.int64)] = np.arange(len(tags))
    types, _, conn = gmsh.model.mesh.getElements(3)
    hexes, non_hex = None, 0
    for t, c in zip(types, conn):
        if t == HEX8:
            hexes = remap[c.astype(np.int64)].reshape(-1, 8)
        else:
            non_hex += len(c) // gmsh.model.mesh.getElementProperties(t)[3]
    if hexes is None:
        hexes = np.zeros((0, 8), dtype=np.int64)
    return HexMesh(coords, hexes, strategy, params, non_hex, phys)


# ---------------------------------------------------------------- tensor
def mesh_tensor(grid: VelocityGrid, segs_xyz, verbose=False, keep_gmsh=False,
                out_msh: str | None = None) -> HexMesh:
    _init(verbose)
    ext = grid.extent
    (x0, _), (y0, _), (z0, _) = ext
    nx, hx, Lx = _layers(segs_xyz[0])
    ny, hy, Ly = _layers(segs_xyz[1])
    nz, hz, Lz = _layers(segs_xyz[2])
    g = gmsh.model.geo
    p = g.addPoint(x0, y0, z0)
    e1 = g.extrude([(0, p)], Lx, 0, 0, numElements=nx, heights=hx, recombine=True)
    line = [dt for dt in e1 if dt[0] == 1][0]
    e2 = g.extrude([line], 0, Ly, 0, numElements=ny, heights=hy, recombine=True)
    surf = [dt for dt in e2 if dt[0] == 2][0]
    g.extrude([surf], 0, 0, Lz, numElements=nz, heights=hz, recombine=True)
    g.synchronize()
    phys = _physical_groups(ext)
    gmsh.model.mesh.generate(3)
    params = {"segments": [[(s.t0, s.t1, s.n) for s in segs] for segs in segs_xyz]}
    hm = _collect("tensor", params, phys)
    if out_msh:
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(out_msh)
        hm.msh_path = out_msh
    if not keep_gmsh:
        gmsh.finalize()
    return hm


# --------------------------------------------------------------- extrude
def mesh_extrude(grid: VelocityGrid, h: np.ndarray, segs_z, quad_algo="blossom",
                 size_factor=0.8, verbose=False, keep_gmsh=False,
                out_msh: str | None = None) -> HexMesh:
    """Quad 2D non structuré piloté par la carte de taille min_z(h), extrudé en z."""
    _init(verbose)
    ext = grid.extent
    (x0, x1), (y0, y1), (z0, _) = ext
    hmap = plane_size_map(grid, h) * size_factor
    xs, ys = grid.axis(0), grid.axis(1)
    dx, dy = grid.spacing[0], grid.spacing[1]
    # NB (gmsh 4.15) : les algos "fullquad" ne subdivisent pas → pas de facteur 2.
    # Frontal-quad dépasse la taille demandée d'environ 20 % au p95 → size_factor≈0.8.
    mult = 1.0

    def size_cb(dim, tag, x, y, z, lc):
        i = int(np.clip(round((x - xs[0]) / dx), 0, len(xs) - 1))
        j = int(np.clip(round((y - ys[0]) / dy), 0, len(ys) - 1))
        # min sur un petit voisinage pour éviter de rater un minimum local
        blk = hmap[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
        return float(mult * blk.min())

    g = gmsh.model.geo
    pts = [g.addPoint(x0, y0, z0), g.addPoint(x1, y0, z0), g.addPoint(x1, y1, z0), g.addPoint(x0, y1, z0)]
    lines = [g.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    s = g.addPlaneSurface([g.addCurveLoop(lines)])
    nz, hz, Lz = _layers(segs_z)
    g.extrude([(2, s)], 0, 0, Lz, numElements=nz, heights=hz, recombine=True)
    g.synchronize()
    gmsh.model.mesh.setRecombine(2, s)
    for opt, val in [("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                     ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.Algorithm", 8),
                     ("Mesh.RecombinationAlgorithm",
                      {"simple": 0, "blossom": 1, "simple_fullquad": 2, "blossom_fullquad": 3}[quad_algo]),
                     ("Mesh.Smoothing", 5)]:
        gmsh.option.setNumber(opt, val)
    gmsh.model.mesh.setSizeCallback(size_cb)
    phys = _physical_groups(ext)
    gmsh.model.mesh.generate(3)
    params = {"quad_algo": quad_algo, "size_factor": size_factor,
              "segments_z": [(sg.t0, sg.t1, sg.n) for sg in segs_z]}
    hm = _collect("extrude", params, phys)
    if out_msh:
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(out_msh)
        hm.msh_path = out_msh
    if not keep_gmsh:
        gmsh.finalize()
    return hm


# -------------------------------------------------------------- doubling
_FACES = np.array([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])


def boundary_quads(hexes: np.ndarray) -> np.ndarray:
    f = hexes[:, _FACES].reshape(-1, 4)
    _, idx, cnt = np.unique(np.sort(f, axis=1), axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]]


def inject_into_gmsh(nodes, hexes, z0, verbose=False):
    """Crée un modèle gmsh discret (volume + surface libre + bords absorbants)."""
    _init(verbose)
    tol = 1e-6 * np.ptp(nodes, axis=0).max()
    bq = boundary_quads(hexes)
    is_free = np.all(np.abs(nodes[bq, 2] - z0) < tol, axis=1)
    s_free = gmsh.model.addDiscreteEntity(2)
    s_abs = gmsh.model.addDiscreteEntity(2)
    vol = gmsh.model.addDiscreteEntity(3, -1, [s_free, s_abs])
    tags = np.arange(1, len(nodes) + 1)
    gmsh.model.mesh.addNodes(3, vol, tags, nodes.ravel())
    gmsh.model.mesh.addElementsByType(vol, HEX8, [], (hexes + 1).ravel())
    gmsh.model.mesh.addElementsByType(s_free, 3, [], (bq[is_free] + 1).ravel())
    gmsh.model.mesh.addElementsByType(s_abs, 3, [], (bq[~is_free] + 1).ravel())
    gmsh.model.addPhysicalGroup(3, [vol], name="medium")
    gmsh.model.addPhysicalGroup(2, [s_free], name="free_surface")
    gmsh.model.addPhysicalGroup(2, [s_abs], name="absorbing")
    return {"medium": [vol], "free_surface": [s_free], "absorbing": [s_abs]}


def mesh_doubling(grid: VelocityGrid, h: np.ndarray, segs_xyz, max_levels=3, trans_layers=1,
                  allow_refine=True, verbose=False, out_msh: str | None = None) -> HexMesh:
    from .doubling import build_doubling
    nodes, hexes, info = build_doubling(grid, h, segs_xyz, max_levels, trans_layers, allow_refine)
    phys = inject_into_gmsh(nodes, hexes, grid.extent[2][0], verbose)
    if out_msh:
        gmsh.write(out_msh)
    gmsh.finalize()
    params = {"max_levels": max_levels, "trans_layers": trans_layers, "allow_refine": allow_refine, **info}
    return HexMesh(nodes, hexes, "doubling", params, 0, phys, out_msh)


# --------------------------------------------------------------- refine3
def mesh_refine3(grid: VelocityGrid, h: np.ndarray, base_max_segments=(4, 4, 8), lmax=1,
                 max_grading=2.0, verbose=False, out_msh: str | None = None) -> HexMesh:
    """Grille de base tensorielle dimensionnée pour h_floor = max(h, 3^lmax·h_min), puis
    3-raffinement local (gabarits conformes) là où h_req < taille de cellule."""
    from .refine3 import Refine3Mesh
    from .sizing import segment_all_axes
    from .doubling import _nodes
    hmin = float(h.min())
    h_floor = np.maximum(h, hmin * 3 ** lmax)
    segs = segment_all_axes(grid, h_floor, tuple(base_max_segments), max_grading=max_grading)
    R = Refine3Mesh(*[_nodes(s) for s in segs], lmax=lmax)
    o, d, n = np.asarray(grid.origin), np.asarray(grid.spacing), np.asarray(grid.shape)

    def need(lo, hi, level):
        a = np.clip(np.ceil((lo - o) / d - 1e-6).astype(int), 0, n - 1)
        b = np.maximum(a, np.clip(np.floor((hi - o) / d + 1e-6).astype(int), 0, n - 1))
        hreq = h[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1].min()
        return bool((hi - lo).max() > hreq * (1 + 1e-9))

    n_ref = R.refine_where(need)
    bal = R.balance()
    nodes, hexes, info = R.to_hexes()
    phys = inject_into_gmsh(nodes, hexes, grid.extent[2][0], verbose)
    if out_msh:
        gmsh.write(out_msh)
    gmsh.finalize()
    params = {"lmax": lmax, "base_cells": [len(s) for s in segs], "base_max_segments": list(base_max_segments),
              "refined_cells": n_ref, "balance": bal, **info}
    return HexMesh(nodes, hexes, "refine3", params, 0, phys, out_msh)


def mesh_hybrid(grid: VelocityGrid, h: np.ndarray, segs_xyz, max_levels=3, q_bg=0.1, lmax3=1,
                verbose=False, out_msh: str | None = None) -> HexMesh:
    from .hybrid import build_hybrid
    nodes, hexes, info = build_hybrid(grid, h, segs_xyz, max_levels, q_bg, lmax3)
    phys = inject_into_gmsh(nodes, hexes, grid.extent[2][0], verbose)
    if out_msh:
        gmsh.write(out_msh)
    gmsh.finalize()
    params = {"max_levels": max_levels, "q_bg": q_bg, "lmax3": lmax3, **info}
    return HexMesh(nodes, hexes, "hybrid", params, 0, phys, out_msh)
