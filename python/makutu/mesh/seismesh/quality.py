"""Qualité géométrique et adéquation physique (échantillonnage, CFL, coût) d'un maillage hexa."""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre as L

from .velocity import VelocityGrid
from .sizing import SizingParams, h_required, ideal_element_count
from .mesher import HexMesh

# ordre des nœuds hexa gmsh : 0-3 face inférieure, 4-7 face supérieure
_CORNER = np.array([(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
                    (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)])
_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                   (0, 4), (1, 5), (2, 6), (3, 7)])


def gll_min_fraction(order: int) -> float:
    """Plus petit écart GLL / taille d'élément, pour l'ordre N."""
    if order <= 1:
        return 1.0
    x = np.sort(np.concatenate(([-1.0, 1.0], L.legroots(L.legder([0] * order + [1])))))
    return float(np.min(np.diff(np.real(x))) / 2.0)


def element_metrics(m: HexMesh):
    X = m.nodes[m.hexes]                                   # (ne,8,3)
    c = X[:, _CORNER]                                      # (ne,8,4,3)
    e1, e2, e3 = (c[:, :, k] - c[:, :, 0] for k in (1, 2, 3))
    det = np.einsum("eij,eij->ei", np.cross(e1, e2), e3)
    nrm = np.linalg.norm(e1, axis=2) * np.linalg.norm(e2, axis=2) * np.linalg.norm(e3, axis=2)
    sj = (det / nrm).min(axis=1)
    edges = np.linalg.norm(X[:, _EDGES[:, 1]] - X[:, _EDGES[:, 0]], axis=2)
    return sj, edges.min(axis=1), edges.max(axis=1), X.min(axis=1), X.max(axis=1)


def evaluate(m: HexMesh, grid: VelocityGrid, p: SizingParams, cfl: float = 0.5,
             t_sim: float = 4.0, lts_max_level: int = 10) -> dict:
    """Rapport complet. `cost_index` ∝ (nb points GLL) × (nb de pas de temps)."""
    ne = len(m.hexes)
    if ne == 0:
        return {"ok": False, "error": "aucun hexaèdre généré", "n_non_hex": m.n_non_hex}
    sj, hmin_e, hmax_e, lo, hi = element_metrics(m)
    vmin, vmax, vmean = grid.box_minmax(lo, hi)
    ppw_eff = p.order * vmin / (p.fmax * hmax_e)
    from .sizing import PPW_TOLERANCE
    under = ppw_eff < p.ppw_accept * (1.0 - PPW_TOLERANCE)    # every element (Codex, point 2)
    over = ppw_eff > 2.0 * p.ppw                          # sur-échantillonnage (gaspillage)
    dt_e = cfl * hmin_e * gll_min_fraction(p.order) / vmax
    dt = float(dt_e.min())
    n_gll = ne * p.order ** 3                              # ≈ points GLL uniques
    ideal = ideal_element_count(grid, h_required(grid, p))
    worst = np.argsort(dt_e)[:5]
    rep = {
        # hexahedra split from tetrahedra are general: a positive jacobian is enough (threshold 0.1)
        "ok": bool(m.n_non_hex == 0 and sj.min() > (0.1 if m.strategy == "tetsplit" else 0.2)
                   and not under.any()),
        "strategy": m.strategy, "cfl": cfl,
        "n_elements": int(ne), "n_nodes": int(len(m.nodes)), "n_non_hex": int(m.n_non_hex),
        "n_gll_points_approx": int(n_gll),
        "scaled_jacobian": {"min": float(sj.min()), "p01": float(np.percentile(sj, 1)),
                            "mean": float(sj.mean()), "n_below_0.3": int((sj < 0.3).sum())},
        "aspect_ratio_max": float((hmax_e / hmin_e).max()),
        "ppw_effective": {"min": float(ppw_eff.min()), "p05": float(np.percentile(ppw_eff, 5)),
                          "median": float(np.median(ppw_eff)), "target": p.ppw},
        "frac_undersampled": float(under.mean()),
        "frac_oversampled_x2": float(over.mean()),
        "dt_cfl": dt,
        "n_steps": int(np.ceil(t_sim / dt)),
        "cost_index": float(n_gll * t_sim / dt / 1e9),
        # pas de temps local (LTS par puissances de 2) : chaque élément avance à dt·2^niveau
        "cost_index_lts": float(p.order ** 3 * t_sim * np.sum(
            1.0 / (dt * 2.0 ** np.minimum(np.floor(np.log2(dt_e / dt)), lts_max_level))) / 1e9),
        "lts_levels": {int(k): int(v) for k, v in zip(*np.unique(
            np.minimum(np.floor(np.log2(dt_e / dt)), lts_max_level).astype(int), return_counts=True))},
        "ideal_elements_lower_bound": int(ideal),
        "efficiency_vs_ideal": float(ideal / ne),
        "dt_limiting_elements": [{"center": ((lo[i] + hi[i]) / 2).round(1).tolist(),
                                  "hmin": float(hmin_e[i]), "vmax": float(vmax[i])} for i in worst[:3]],
    }
    rep["_per_element"] = {"vmin": vmin, "vmax": vmax, "vmean": vmean, "ppw": ppw_eff,
                           "sj": sj, "dt": dt_e}
    return rep


def public(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}
