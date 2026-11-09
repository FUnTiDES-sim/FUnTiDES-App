"""Chaîne complète : modèle SEP + fréquence de la source -> maillage adapté pour FUnTiDES.

La fréquence maximale est celle de makutu pour une ondelette de Ricker de fréquence de pic f0 :
``f_max = 2,5 f0`` (longueur d'onde minimale ``v_min / (2,5 f0)``, :func:`makutu.mesh.suggest_mesh`).
La taille requise est ``h = N v / (f_max ppw)`` (N ordre SEM, ppw points GLL par longueur d'onde).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .funtides import write_funtides_msh
from .plane import autopilot_2d
from .quality import public
from .sep import grid_from_sep, is_2d
from .sizing import SizingParams

RICKER_FMAX_FACTOR = 2.5


def _row(r):
    """Summary of a candidate for the report."""
    return {"strategy": r["strategy"], "ok": r["ok"], "n_elements": r["n_elements"],
            "ppw_min": round(r["ppw_effective"]["min"], 2), "sj_min": round(r["scaled_jacobian"]["min"], 3),
            "dt_cfl": r["dt_cfl"], "cost_index": round(r["cost_index"], 4),
            "cost_index_lts": round(r["cost_index_lts"], 4)}


def _element_boxes(nodes, cells, dim, y0=0.0):
    X = nodes[cells]
    lo, hi = X.min(axis=1), X.max(axis=1)
    if dim == 2:                                      # (x, z) -> (x, y0, z)
        lo = np.stack([lo[:, 0], np.full(len(lo), y0), lo[:, 1]], -1)
        hi = np.stack([hi[:, 0], np.full(len(hi), y0), hi[:, 1]], -1)
    return lo, hi


def build_mesh_from_sep(vp_file, f0, out_dir, order=4, ppw=5.0, rho_file=None, rho=1.0, t_sim=4.0,
                        lts=False, strategies=None, cfl=0.5, average="slowness", verbose=True):
    """
    Maillage d'un modèle SEP adapté à une source de Ricker de fréquence de pic ``f0``.

    Les candidats (2-D : ``tensor``, ``doubling`` ; 3-D : ``tensor``, ``doubling``, ``hybrid``,
    ``extrude``) sont générés et évalués ; le meilleur valide (points par longueur d'onde atteints,
    jacobien > 0,2) de plus faible coût (``cost_index`` ou ``cost_index_lts`` avec ``lts``) est
    exporté dans ``out_dir`` : ``mesh.msh``, ``materials.csv``, ``report.json``. La vitesse d'un
    élément est la moyenne des échantillons de la grille qu'il contient : moyenne des lenteurs
    (``average="slowness"``, conserve les temps de trajet) ou des vitesses (``"velocity"``). Des
    faces d'éléments passent par les interfaces horizontales nettes du modèle. ``makutu.mesh.GmshMesh(out_dir/"mesh.msh", order,
    out_dir/"materials.csv")`` le lit pour la modélisation.

    Returns
    -------
        dict : chemins des fichiers, dimension, rapport du maillage retenu
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = grid_from_sep(vp_file)
    rho_grid = grid_from_sep(rho_file, name="rho") if rho_file else None
    p = SizingParams(fmax=RICKER_FMAX_FACTOR * f0, order=order, ppw=ppw)
    if is_2d(grid):
        best, reports = autopilot_2d(grid, p, t_sim=t_sim, lts=lts,
                                     strategies=strategies or ("tensor", "doubling", "hybrid", "quadtree"))
        if best is None:
            raise RuntimeError("aucun maillage 2-D valide (points par longueur d'onde ou jacobien)")
        mesh, rep = best
        nodes, cells, dim = mesh.nodes, mesh.quads, 2
        n_candidates = len(reports)
        table = [_row(r) for r in reports]
    else:
        from .session import Session
        s = Session(out / "candidates")
        s.grid = grid
        s.set_sizing(p.fmax, order, ppw)
        res = s.autopilot(tuple(strategies or ("tensor", "doubling", "hybrid", "refine3", "extrude")), t_sim=t_sim, lts=lts)
        if not res["best_ok"]:
            raise RuntimeError("aucun maillage 3-D valide (points par longueur d'onde ou jacobien)")
        c = s.candidates[res["best_ok"]]
        mesh, rep = c["mesh"], c["report"]
        nodes, cells, dim = mesh.nodes, mesh.hexes, 3
        n_candidates = len(res["candidates"])
        table = [_row(c["report"]) for c in s.candidates.values()]
    boxes = _element_boxes(nodes, cells, dim, grid.origin[1])
    if average == "slowness":
        vp = 1.0 / grid.box_mean(*boxes, values=1.0 / grid.vel)
    elif average == "velocity":
        vp = grid.box_mean(*boxes)
    else:
        raise ValueError("average must be 'slowness' or 'velocity'")
    if rho_grid is not None:
        rho_e = rho_grid.box_mean(*boxes)
    else:
        rho_e = np.full(len(cells), float(rho))
    msh, materials, n_classes = write_funtides_msh(out / "mesh.msh", nodes, cells, vp, rho_e)
    report = {"vp_file": str(vp_file), "rho_file": str(rho_file) if rho_file else None, "f0": f0,
              "fmax": p.fmax, "order": order, "ppw": ppw, "lts": lts, "dim": dim, "average": average,
              "candidates_evaluated": n_candidates, "material_classes": n_classes,
              "surface_depth": float(grid.extent[2][0]), "selected": public(rep), "candidates": table}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    if verbose:
        print(f"{dim}-D, {n_candidates} candidats ; retenu : {rep['strategy']}, {rep['n_elements']} éléments, "
              f"ppw min {rep['ppw_effective']['min']:.2f}, jacobien min {rep['scaled_jacobian']['min']:.2f}, "
              f"dt CFL {rep['dt_cfl']:.2e} s, coût {rep['cost_index']:.3g} (LTS {rep['cost_index_lts']:.3g})")
    return {"msh": msh, "materials": materials, "report": out / "report.json", "dim": dim, "selected": public(rep)}
