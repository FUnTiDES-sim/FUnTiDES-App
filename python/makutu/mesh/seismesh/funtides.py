"""Export d'un maillage seismesh pour les lecteurs gmsh de FUnTiDES (MSH 2.2 ASCII).

- z vers le haut (convention FUnTiDES et :class:`makutu.mesh.GmshMesh`) : la hauteur d'un nœud
  est ``z_max - profondeur``, la surface libre (profondeur minimale) est en haut ;
- une classe de matériau par élément : le groupe physique de chaque élément est le numéro de sa
  classe (vp, rho) quantifiée ; ``materials.csv`` donne ``physical_tag,vp,rho`` ;
- frontières : groupe 1 ``free_surface`` (en haut), groupe 2 ``absorbing`` (autres faces) ;
- 2-D : quadrangles du plan (x, z), y = 0 ; 3-D : hexaèdres (ordre des nœuds gmsh).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .doubling import _orient
from .mesher import boundary_quads

FREE, ABSORBING, FIRST_MATERIAL = 1, 2, 100


def material_classes(vp, rho, vp_step=1.0, rho_step=1e-3):
    """Classes (vp, rho) quantifiées : (tag de chaque élément, {tag: (vp, rho)})."""
    q = np.stack([np.round(np.asarray(vp) / vp_step), np.round(np.asarray(rho) / rho_step)], -1)
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    table = {FIRST_MATERIAL + i: (float(a * vp_step), float(b * rho_step)) for i, (a, b) in enumerate(uniq)}
    return FIRST_MATERIAL + inv.ravel(), table


def _boundary_edges(quads):
    e = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)])
    _, idx, cnt = np.unique(np.sort(e, axis=1), axis=0, return_index=True, return_counts=True)
    return e[idx[cnt == 1]]


def write_funtides_msh(path, nodes, cells, vp, rho=None, vp_step=1.0):
    """
    Écrit ``path`` (.msh) et ``materials.csv`` à côté.

    Parameters
    ----------
        nodes : (n, 2) (x, profondeur) en 2-D ou (n, 3) (x, y, profondeur) en 3-D
        cells : (ne, 4) quadrangles ou (ne, 8) hexaèdres (ordre gmsh)
        vp, rho : valeur par élément (rho par défaut 1)

    Returns
    -------
        (chemin du .msh, chemin de materials.csv, nombre de classes de matériau)
    """
    path = Path(path)
    nodes = np.asarray(nodes, dtype=float)
    cells = np.asarray(cells)
    dim = 2 if cells.shape[1] == 4 else 3
    depth = nodes[:, -1]
    height = depth.max() - depth
    if dim == 2:
        xyz = np.stack([nodes[:, 0], np.zeros(len(nodes)), height], -1)
        faces = _boundary_edges(cells)
        face_type, cell_type = 1, 3
    else:
        xyz = np.stack([nodes[:, 0], nodes[:, 1], height], -1)
        cells = _orient(xyz, cells)                  # z retourné : jacobien positif
        faces = boundary_quads(cells)
        face_type, cell_type = 3, 5
    tol = 1e-6 * max(np.ptp(xyz, axis=0).max(), 1.0)
    top = height.max()
    free = np.all(np.abs(xyz[faces, 2] - top) < tol, axis=1)
    rho = np.ones(len(cells)) if rho is None else np.asarray(rho, dtype=float)
    tags, table = material_classes(vp, rho, vp_step)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$PhysicalNames\n2\n{dim - 1} {FREE} \"free_surface\"\n{dim - 1} {ABSORBING} \"absorbing\"\n"
                "$EndPhysicalNames\n")
        f.write(f"$Nodes\n{len(xyz)}\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(xyz) + 1), xyz]), fmt=["%d", "%.9g", "%.9g", "%.9g"])
        f.write("$EndNodes\n")
        n_el = len(faces) + len(cells)
        f.write(f"$Elements\n{n_el}\n")
        ftag = np.where(free, FREE, ABSORBING)
        rows = np.column_stack([np.arange(1, len(faces) + 1), np.full(len(faces), face_type), np.full(len(faces), 2),
                                ftag, ftag, faces + 1])
        np.savetxt(f, rows, fmt="%d")
        rows = np.column_stack([np.arange(len(faces) + 1, n_el + 1), np.full(len(cells), cell_type),
                                np.full(len(cells), 2), tags, tags, cells + 1])
        np.savetxt(f, rows, fmt="%d")
        f.write("$EndElements\n")
    materials = path.with_name("materials.csv")
    with open(materials, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["physical_tag", "vp", "rho"])
        for tag, (v, r) in table.items():
            w.writerow([tag, f"{v:g}", f"{r:g}"])
    return path, materials, len(table)
