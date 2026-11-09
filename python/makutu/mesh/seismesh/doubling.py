r"""Maillage hexa conforme avec couches de doublement 2:1 (briques type SPECFEM).

Principe
--------
Grille tensorielle fine (x,y) issue de la segmentation optimale, puis, en descendant, le pas
horizontal double à chaque niveau dès que le critère h_req l'autorise sur toute la profondeur
restante. Chaque doublement = 2 sous-couches conformes :
  - 'tx' : gabarit 2D (x,z) « 4 fines → 2 grossières » extrudé en y (y encore fin),
  - 'ty' : même gabarit en (y,z), extrudé en x (x déjà grossier).

Gabarit 2D (u = index fin dans le bloc de 4, z vers le bas, H = épaisseur de sous-couche) :

    T0----T1----T2----T3----T4      z = 0
    | q1  | q3  |  q4 |  q2 |
    |     A-----C-----D     |       z = H/2
    |   /    q5 |  q6   \   |
    B0----------B2----------B4      z = H

6 quadrangles convexes : q1=T0 T1 A B0, q2=T3 T4 B4 D, q3=T1 T2 C A, q4=T2 T3 D C,
q5=B0 A C B2, q6=B2 C D B4. Les côtés latéraux sont des arêtes uniques → les gabarits
voisins se raccordent sans nœud pendant.
"""
from __future__ import annotations

import numpy as np

from .velocity import VelocityGrid
from .sizing import Segment

# (u, z) avec z : 0 = haut, 1 = milieu, 2 = bas
_T = lambda u: (u, 0)
_M = lambda u: (u, 1)
_B = lambda u: (u, 2)
TEMPLATE = [
    [_T(0), _T(1), _M(1), _B(0)],
    [_T(3), _T(4), _B(4), _M(3)],
    [_T(1), _T(2), _M(2), _M(1)],
    [_T(2), _T(3), _M(3), _M(2)],
    [_B(0), _M(1), _M(2), _B(2)],
    [_B(2), _M(2), _M(3), _B(4)],
]


def _pad(segs: list[Segment], mult: int) -> list[Segment]:
    """Ajoute des cellules (raffinement → toujours admissible) pour que n_total % mult == 0."""
    segs = [Segment(s.t0, s.t1, s.n) for s in segs]
    tot = sum(s.n for s in segs)
    while tot % mult:
        max(segs, key=lambda s: s.spacing).n += 1
        tot += 1
    return segs


def _nodes(segs: list[Segment]) -> np.ndarray:
    pts = [np.linspace(s.t0, s.t1, s.n + 1)[:-1] for s in segs]
    return np.concatenate(pts + [[segs[-1].t1]])


def _cell_req(prof, t0, dt, nodes):
    """min du profil admissible sur chaque cellule [nodes[i], nodes[i+1]]."""
    n = len(prof)
    a = np.clip(np.ceil((nodes[:-1] - t0) / dt - 1e-6).astype(int), 0, n - 1)
    b = np.clip(np.floor((nodes[1:] - t0) / dt + 1e-6).astype(int), 0, n - 1)
    b = np.maximum(a, b)
    return np.array([prof[i:j + 1].min() for i, j in zip(a, b)])


def plan_levels(grid: VelocityGrid, h: np.ndarray, segs_xyz, max_levels=3, trans_layers=1,
                allow_refine=True, min_dwell=2, blocked=frozenset()):
    """Retourne (xs, ys, zs, plan, allowed). plan = [(kind, k_top, k_bot, level_fin)].

    kind : 'reg' (couche régulière), 'tx'/'ty' (déraffinement c→c+1 en x puis y),
           'ry'/'rx' (re-raffinement c→c-1 en y puis x, gabarit retourné : fin en bas).
    allow_refine=False reproduit le comportement monotone (min sur toute la profondeur restante).
    """
    mult = 2 ** (max_levels + 1)
    xs = _nodes(_pad(segs_xyz[0], mult))
    ys = _nodes(_pad(segs_xyz[1], mult))
    zs = _nodes(segs_xyz[2])
    nz = len(zs) - 1
    hzx, hzy = h.min(axis=1), h.min(axis=0)           # (nxg,nzg), (nyg,nzg)
    zg = grid.axis(2)
    (ox, oy, _), (dx, dy, dz) = grid.origin, grid.spacing
    allowed = np.zeros(nz, dtype=int)
    for k in range(nz):
        iz = np.where((zg >= zs[k] - 1e-6) & (zg <= zs[k + 1] + 1e-6))[0]
        if iz.size == 0:
            iz = np.array([int(np.clip(round((0.5 * (zs[k] + zs[k + 1]) - zg[0]) / dz), 0, len(zg) - 1))])
        px, py = hzx[:, iz].min(axis=1), hzy[:, iz].min(axis=1)
        for l in range(1, max_levels + 1):
            xl, yl = xs[::2 ** l], ys[::2 ** l]
            okx = np.all(np.diff(xl) <= _cell_req(px, ox, dx, xl) * (1 + 1e-9))
            oky = np.all(np.diff(yl) <= _cell_req(py, oy, dy, yl) * (1 + 1e-9))
            if not (okx and oky):
                break
            allowed[k] = l
    m = trans_layers
    if not allow_refine:
        U = np.minimum.accumulate(allowed[::-1])[::-1]     # monotone : jamais plus fin en dessous
    else:
        # U[k] = niveau max admissible en k tout en pouvant re-raffiner à temps (2m couches
        # par re-raffinement) : U[k] = min_{j>=k} allowed[j] + (j-k)//(2m)
        U = np.array([min(allowed[j] + (j - k) // (2 * m) for j in range(k, nz)) for k in range(nz)])
    U = np.append(np.minimum(U, max_levels), max_levels)  # U[nz] : pas de contrainte
    plan, c, k = [], 0, 0

    def free(k0, k1):              # aucune couche bloquée (anomalie locale) dans [k0-1, k1]
        return not any(q in blocked for q in range(k0 - 1, k1 + 1))
    while k < nz:
        if c > 0 and k + 2 * m <= nz and c > U[k + 2 * m]:
            plan += [("ry", k, k + m, c - 1), ("rx", k + m, k + 2 * m, c - 1)]
            c, k = c - 1, k + 2 * m
        elif c < max_levels and k + 2 * m <= nz and free(k, k + 2 * m) and \
                U[k:min(k + 2 * m + min_dwell, nz) + 1].min() >= c + 1:
            plan += [("tx", k, k + m, c), ("ty", k + m, k + 2 * m, c)]
            c, k = c + 1, k + 2 * m
        else:
            plan.append(("reg", k, k + 1, c))
            k += 1
    return xs, ys, zs, plan, allowed


def build_doubling(grid: VelocityGrid, h: np.ndarray, segs_xyz, max_levels=3, trans_layers=1,
                   allow_refine=True, planned=None, skip_layers=frozenset()):
    xs, ys, zs, plan, allowed = planned or plan_levels(grid, h, segs_xyz, max_levels, trans_layers,
                                                       allow_refine)
    NX, NY = len(xs) - 1, len(ys) - 1
    keys = []                                             # blocs (ne,8,3) : (ix_fin, iy_fin, zcode)

    def zc(k_top, k_bot, z):                              # zcode : 2k plan, 2k+1 milieu de sous-couche
        return (2 * k_top, 2 * k_top + 1, 2 * k_bot)[z]

    for kind, kt, kb, c in plan:
        s = 2 ** c
        if kind == "reg" and kt in skip_layers:
            continue
        if kind == "reg":
            I, J = np.meshgrid(np.arange(0, NX, s), np.arange(0, NY, s), indexing="ij")
            I, J = I.ravel(), J.ravel()
            quad = [(I, J), (I + s, J), (I + s, J + s), (I, J + s)]
            hx = [np.stack([a, b, np.full_like(a, z)], -1) for z in (2 * kt, 2 * kb) for a, b in quad]
            keys.append(np.stack(hx, axis=1))
            continue
        # gabarit le long de l'axe 'a' (x pour tx/rx, y pour ty/ry) ; extrusion le long de l'autre.
        # s = pas du côté fin. rx/ry : gabarit retourné (côté fin en bas).
        flip = kind in ("rx", "ry")
        if kind in ("tx", "rx"):
            base = np.arange(0, NX, 4 * s)                # blocs de 4 cellules fines (niveau c)
            ext = np.arange(0, NY, s); se = s             # y encore au niveau c
        else:
            base = np.arange(0, NY, 4 * s)
            ext = np.arange(0, NX, 2 * s); se = 2 * s     # x déjà au niveau c+1
        Bb, E = np.meshgrid(base, ext, indexing="ij")
        Bb, E = Bb.ravel(), E.ravel()
        for q in TEMPLATE:
            corners = []
            for e_off in (0, se):
                for u, z in q:
                    a = Bb + u * s
                    e = E + e_off
                    zz = np.full_like(a, zc(kt, kb, 2 - z if flip else z))
                    corners.append(np.stack([a, e, zz], -1) if kind in ("tx", "rx")
                                   else np.stack([e, a, zz], -1))
            keys.append(np.stack(corners, axis=1))
    K = np.concatenate(keys)                              # (ne,8,3)
    code = (K[..., 2].astype(np.int64) * (NY + 1) + K[..., 1]) * (NX + 1) + K[..., 0]
    uniq, inv = np.unique(code.ravel(), return_inverse=True)
    hexes = inv.reshape(-1, 8)
    ix = uniq % (NX + 1)
    iy = (uniq // (NX + 1)) % (NY + 1)
    zcode = uniq // ((NX + 1) * (NY + 1))
    kz = zcode // 2
    # profondeur des nœuds « milieu » : milieu de la sous-couche qui démarre en kz
    sub_bot = {kt: kb for kind, kt, kb, _ in plan if kind != "reg"}
    zmid = np.array([0.5 * (zs[k] + zs[sub_bot.get(k, k)]) for k in range(len(zs))])
    z = np.where(zcode % 2 == 0, zs[np.minimum(kz, len(zs) - 1)], zmid[kz])
    nodes = np.stack([xs[ix], ys[iy], z], -1)
    hexes = _orient(nodes, hexes)
    info = {"levels_used": int(max(p[3] + (p[0] != "reg") for p in plan)),
            "transitions": [(round(float(zs[kt]), 1), c, c + 1) if kind == "tx" else
                            (round(float(zs[kt]), 1), c + 1, c) for kind, kt, _, c in plan if kind in ("tx", "ry")],
            "nx_ny_fine": [NX, NY], "nz_layers": len(zs) - 1,
            "allowed_level_per_layer": allowed.tolist()}
    return nodes, hexes, info


def _orient(nodes, hexes):
    X = nodes[hexes]
    det = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 3] - X[:, 0]), X[:, 4] - X[:, 0])
    neg = det < 0
    hexes = hexes.copy()
    hexes[neg] = hexes[neg][:, [4, 5, 6, 7, 0, 1, 2, 3]]
    return hexes
