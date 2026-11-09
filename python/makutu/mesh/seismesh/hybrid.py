"""Stratégie hybride conforme : doublements 2:1 (gradient de profondeur) + 3-raffinement local
(anomalies lentes localisées) à l'intérieur des zones à niveau constant.

1. Le plan de doublement est calculé sur un critère de *fond* h_bg : quantile latéral q de
   h_req à chaque profondeur (les anomalies couvrant moins de q du plan sont ignorées).
2. Chaque zone (couches régulières consécutives de même niveau) est une grille tensorielle ;
   si des cellules y violent le vrai h_req, la zone est maillée par Refine3Mesh.
3. Contrainte de raccord : les couches de bord d'une zone raffinée (contiguës à une couche de
   transition) doivent rester non raffinées, et aucune transition ne doit traverser une anomalie.
   Sinon ces couches sont « bloquées » et le plan est recalculé (les transitions s'écartent).
4. Fusion des nœuds par coordonnées (les faces de raccord sont des quads simples identiques).
"""
from __future__ import annotations

import numpy as np

from .velocity import VelocityGrid
from .doubling import plan_levels, build_doubling
from .refine3 import Refine3Mesh


def _need_fn(grid: VelocityGrid, h: np.ndarray):
    o, d, n = np.asarray(grid.origin), np.asarray(grid.spacing), np.asarray(grid.shape)

    def need(lo, hi, level=0):
        a = np.clip(np.ceil((lo - o) / d - 1e-6).astype(int), 0, n - 1)
        b = np.maximum(a, np.clip(np.floor((hi - o) / d + 1e-6).astype(int), 0, n - 1))
        hreq = h[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1].min()
        return bool((hi - lo).max() > hreq * (1 + 1e-9))
    return need


def _cell_min_1d(arr, t0, dt, nodes, axis):
    """min de arr le long d'un axe sur chaque cellule [nodes[i], nodes[i+1]] (bornes incluses)."""
    n = arr.shape[axis]
    a = np.clip(np.ceil((nodes[:-1] - t0) / dt - 1e-6).astype(int), 0, n - 1)
    b = np.clip(np.floor((nodes[1:] - t0) / dt + 1e-6).astype(int), 0, n - 1)
    b = np.maximum(a, b)
    out = np.minimum.reduceat(arr, a, axis=axis)
    return np.minimum(out, np.take(arr, b, axis=axis))


def _layer_violations(xs, ys, zs, plan, need, grid=None, h=None):
    """Couches (index k) contenant au moins une cellule trop grossière vis-à-vis du vrai h."""
    (ox, oy, oz), (dx, dy, dz) = grid.origin, grid.spacing
    zg = grid.axis(2)
    bad = set()
    for kind, kt, kb, c in plan:
        s = 2 ** (c + (kind != "reg"))           # transitions : pas grossier (majorant)
        xl, yl = xs[::s], ys[::s]
        for k in range(kt, kb):
            iz = np.where((zg >= zs[k] - 1e-6) & (zg <= zs[k + 1] + 1e-6))[0]
            if iz.size == 0:
                iz = np.array([int(np.clip(round((0.5 * (zs[k] + zs[k + 1]) - oz) / dz), 0, len(zg) - 1))])
            hz = h[:, :, iz].min(axis=2)
            cm = _cell_min_1d(_cell_min_1d(hz, ox, dx, xl, 0), oy, dy, yl, 1)
            size = np.maximum(np.maximum(np.diff(xl)[:, None], np.diff(yl)[None, :]), zs[k + 1] - zs[k])
            if np.any(size > cm * (1 + 1e-9)):
                bad.add(k)
    return bad


def _zones(plan):
    zones, cur = [], None
    for kind, kt, kb, c in plan:
        if kind == "reg":
            if cur and cur["level"] == c and cur["k1"] == kt:
                cur["k1"] = kb
            else:
                cur = {"level": c, "k0": kt, "k1": kb}
                zones.append(cur)
        else:
            cur = None
    return zones


def build_hybrid(grid: VelocityGrid, h: np.ndarray, segs_xyz, max_levels=3, q_bg=0.1,
                 lmax3=1, max_replan=30):
    h_bg = np.broadcast_to(np.quantile(h, q_bg, axis=(0, 1))[None, None, :], h.shape)
    need = _need_fn(grid, h)
    nz_total = sum(s.n for s in segs_xyz[2])
    blocked: set[int] = set()
    for it in range(max_replan):
        planned = plan_levels(grid, np.ascontiguousarray(h_bg), segs_xyz, max_levels, 1, True,
                              blocked=frozenset(blocked))
        xs, ys, zs, plan, _ = planned
        bad = _layer_violations(xs, ys, zs, plan, need, grid, h)
        trans_layers = {k for kind, kt, kb, _ in plan if kind != "reg" for k in range(kt, kb)}
        new_block = {k for k in bad if k in trans_layers}           # anomalie dans une transition
        zones = _zones(plan)
        zone_meshes = []
        for zn in zones:
            ks = [k for k in range(zn["k0"], zn["k1"]) if k in bad]
            if not ks:
                continue
            top_open = zn["k0"] > 0                    # bord haut contigu à une transition
            bot_open = zn["k1"] < nz_total
            s = 2 ** zn["level"]
            R = Refine3Mesh(xs[::s], ys[::s], zs[zn["k0"]:zn["k1"] + 1], lmax=lmax3)
            R.refine_where(need)
            R.balance()
            nzz = zn["k1"] - zn["k0"]
            # une face de raccord reste un quad simple ssi aucune feuille fine ne la touche
            touch_top = any(c[0] > 0 and c[3] == 0 for c in R.leaves)
            touch_bot = any(c[0] > 0 and c[3] + 1 == nzz * 3 ** c[0] for c in R.leaves)
            if top_open and touch_top:
                new_block |= {zn["k0"], zn["k0"] - 1}
            if bot_open and touch_bot:
                new_block |= {zn["k1"] - 1, zn["k1"]}
            zone_meshes.append((zn, R))
        if not (new_block - blocked):
            break
        blocked |= new_block
    else:
        raise RuntimeError("hybride : impossible d'isoler les anomalies des transitions")

    skip = {k for zn, _ in zone_meshes for k in range(zn["k0"], zn["k1"])}
    nodes_d, hexes_d, info = build_doubling(grid, h_bg, segs_xyz, max_levels, planned=planned,
                                            skip_layers=frozenset(skip))
    blocks = [nodes_d[hexes_d]]
    stats = {"refined_zones": [], "blocked_layers": sorted(blocked)}
    for zn, R in zone_meshes:
        n3, h3, inf3 = R.to_hexes()
        blocks.append(n3[h3])
        stats["refined_zones"].append({"z": [float(zs[zn["k0"]]), float(zs[zn["k1"]])],
                                       "level": zn["level"], **inf3})
    X = np.concatenate(blocks)                                       # (ne,8,3)
    scale = np.ptp(X.reshape(-1, 3), axis=0).max()
    key = np.round(X.reshape(-1, 3) / (scale * 1e-9)).astype(np.int64)
    uniq, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    nodes = X.reshape(-1, 3)[idx]
    hexes = inv.reshape(-1, 8)
    info.update(stats)
    return nodes, hexes, info

