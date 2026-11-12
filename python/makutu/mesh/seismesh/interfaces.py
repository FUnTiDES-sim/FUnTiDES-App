"""Interfaces principales d'un modèle SEP 2-D : régions par seuils de vitesse et contours partagés.

Chaîne, à partir de la grille régulière (format SEP, différences finies) :

1. **Régions** : la vitesse est découpée par des seuils (donnés, ou déduits des forts contrastes
   entre cellules voisines : :func:`auto_thresholds`) ; chaque classe est séparée en composantes
   connexes (4-connexité) et les composantes trop petites sont fusionnées dans leur voisine
   majoritaire.
2. **Interfaces** : les bords des régions sont suivis sur les bords des cellules (marches
   d'escalier), découpés aux points triples et au bord du domaine en morceaux partagés par les deux
   régions qu'ils séparent, puis simplifiés (Douglas-Peucker, tolérance de l'ordre d'une cellule)
   à extrémités fixes : les deux régions voient exactement la même ligne.
Ces outils ont servi aux essais de maillages qui suivent les interfaces (gmsh par régions, nœuds
collés sur un quadtree, 10 novembre 2026) : ni l'un ni l'autre n'a battu les stratégies structurées
(voir README, « Essais sans suite ») ; ils restent disponibles pour décrire les interfaces d'un modèle.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .velocity import VelocityGrid


# ------------------------------------------------------------------------------- régions
def auto_thresholds(vel2d: np.ndarray, rel_jump: float = 0.25, min_share: float = 0.05) -> list[float]:
    """Seuils de vitesse tirés des forts contrastes : pour chaque paire de cellules voisines dont le
    saut relatif dépasse ``rel_jump``, la vitesse moyenne géométrique de la paire ; ces valeurs sont
    regroupées (écart relatif > 10 % entre groupes) et chaque groupe qui rassemble au moins
    ``min_share`` des paires donne un seuil (sa médiane)."""
    mids = []
    for ax in (0, 1):
        a = np.moveaxis(vel2d, ax, 0)
        lo, hi = np.minimum(a[1:], a[:-1]), np.maximum(a[1:], a[:-1])
        j = (hi - lo) / lo > rel_jump
        mids.append(np.sqrt(lo[j] * hi[j]))
    m = np.sort(np.concatenate(mids))
    if m.size == 0:
        return []
    cut = np.where(np.diff(np.log(m)) > np.log(1.1))[0] + 1
    groups = np.split(m, cut)
    return [float(np.median(g)) for g in groups if g.size >= min_share * m.size]


def region_labels(vel2d: np.ndarray, thresholds, min_cells: int = 16):
    """Composantes connexes (4-connexité) des classes de vitesse définies par ``thresholds`` ; les
    composantes de moins de ``min_cells`` cellules sont fusionnées dans leur voisine majoritaire.
    Retourne (labels (nx, nz) numérotés 0..n-1, classe de chaque label)."""
    cls = np.digitize(vel2d, sorted(thresholds))
    while True:
        lab = np.full(vel2d.shape, -1)
        n = 0
        for c in np.unique(cls):
            comp, k = ndimage.label(cls == c)
            lab[comp > 0] = comp[comp > 0] - 1 + n
            n += k
        sizes = np.bincount(lab.ravel(), minlength=n)
        small = [i for i in np.argsort(sizes) if sizes[i] < min_cells]
        if not small or len(sizes) == 1:
            break
        i = small[0]
        mask = lab == i
        ring = ndimage.binary_dilation(mask) & ~mask
        cls[mask] = np.bincount(cls[ring]).argmax()
    classes = np.array([cls[lab == i][0] for i in range(n)])
    return lab, classes


# ------------------------------------------------------------------------------ contours
def _loops(lab: np.ndarray):
    """Contours orientés (région à gauche dans le repère (i, k)) de chaque label, en nœuds du réseau
    des coins de cellules : {label: [boucle (liste de (i, k)), ...]}."""
    nx, nz = lab.shape
    pad = np.pad(lab, 1, constant_values=-1)
    out_edges = {}                                              # (label, start) -> list of ends
    for i in range(nx):
        for k in range(nz):
            r = lab[i, k]
            # cell corners counter-clockwise in (i, k); neighbour across each edge
            for (a, b, nb) in (((i, k), (i + 1, k), pad[i + 1, k]), ((i + 1, k), (i + 1, k + 1), pad[i + 2, k + 1]),
                               ((i + 1, k + 1), (i, k + 1), pad[i + 1, k + 2]), ((i, k + 1), (i, k), pad[i, k + 1])):
                if nb != r:
                    out_edges.setdefault((r, a), []).append(b)
    loops = {}
    used = set()
    for (r, a0), ends in list(out_edges.items()):
        for b0 in ends:
            if (r, a0, b0) in used:
                continue
            loop, a, b = [a0], a0, b0
            while True:
                used.add((r, a, b))
                loop.append(b)
                if b == a0 and len(loop) > 2:
                    loop.pop()
                    break
                cand = [c for c in out_edges[(r, b)] if (r, b, c) not in used]
                if len(cand) > 1:                             # pinch point: turn left (stay on the same cell)
                    d = (b[0] - a[0], b[1] - a[1])
                    left = (b[0] - d[1], b[1] + d[0])
                    cand = [left] if left in cand else cand
                a, b = b, cand[0]
            loops.setdefault(r, []).append(loop)
    return loops


def _junctions(lab: np.ndarray) -> set:
    """Nœuds du réseau où se rencontrent au moins trois régions (le dehors compte), ou deux en
    diagonale, plus les coins du domaine."""
    nx, nz = lab.shape
    pad = np.pad(lab, 1, constant_values=-1)
    q = np.stack([pad[:-1, :-1], pad[1:, :-1], pad[1:, 1:], pad[:-1, 1:]], -1)   # cells around node (i, k)
    s = np.sort(q, axis=-1)
    distinct = 1 + (np.diff(s, axis=-1) != 0).sum(-1)
    diag = (q[..., 0] == q[..., 2]) & (q[..., 1] == q[..., 3]) & (q[..., 0] != q[..., 1])
    J = set(map(tuple, np.argwhere((distinct >= 3) | diag)))
    return J | {(0, 0), (nx, 0), (nx, nz), (0, nz)}


def _douglas_peucker(P: np.ndarray, tol: float) -> np.ndarray:
    """Indices conservés d'une polyligne ouverte (extrémités gardées)."""
    keep = np.zeros(len(P), bool)
    keep[[0, -1]] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = P[b] - P[a]
        n = np.hypot(*d)
        seg = P[a + 1:b] - P[a]
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / n if n > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return np.where(keep)[0]


def region_geometry(grid: VelocityGrid, lab: np.ndarray, tol_cells: float = 1.0):
    """Contours simplifiés et partagés des régions. Retourne (pieces, loops) : ``pieces`` liste de
    polylignes (n, 2) en coordonnées (x, profondeur) avec leurs régions voisines, ``loops`` {label:
    [[(numéro de pièce, sens), ...], ...]} (première boucle : contour extérieur)."""
    (x0, _), _, (z0, _) = grid.extent
    dx, dz = grid.spacing[0], grid.spacing[2]
    J = _junctions(lab)
    loops = _loops(lab)
    pieces, index, region_loops = [], {}, {}
    for r, ls in loops.items():
        for loop in ls:
            js = [t for t, p in enumerate(loop) if p in J]
            if not js:                                        # closed loop without junction: two artificial ones
                js = [0, len(loop) // 2]
            if len(js) == 1:
                js.append((js[0] + len(loop) // 2) % len(loop))
                js.sort()
            seq = []
            for a, b in zip(js, js[1:] + [js[0] + len(loop)]):
                path = tuple(loop[t % len(loop)] for t in range(a, b + 1))
                key = min(path, path[::-1])
                if key not in index:
                    index[key] = len(pieces)
                    pieces.append({"lattice": key, "regions": set()})
                pid = index[key]
                pieces[pid]["regions"].add(int(r))
                seq.append((pid, 1 if path == key else -1))
            area = 0.0
            pts = np.array(loop, float)
            area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
            region_loops.setdefault(int(r), []).append((abs(area), seq))
    tol = tol_cells * min(dx, dz)
    for p in pieces:
        L = np.array(p["lattice"], float)
        P = np.stack([x0 + L[:, 0] * dx, z0 + L[:, 1] * dz], -1)
        p["xy"] = P[_douglas_peucker(P, tol)]
    out = {r: [seq for _, seq in sorted(v, key=lambda t: -t[0])] for r, v in region_loops.items()}
    return pieces, out


# ------------------------------------------------------------------------- échantillonnage
def quad_samples(grid: VelocityGrid, nodes, quads, values=None, n=None):
    """Valeurs de la grille (défaut : vitesse) échantillonnées en n x n points de chaque quadrangle
    (application bilinéaire), cellule la plus proche : (ne, n*n)."""
    v = (grid.vel if values is None else values)[:, 0, :]
    X = np.asarray(nodes)[np.asarray(quads)]                      # (ne, 4, 2)
    d = min(grid.spacing[0], grid.spacing[2])
    if n is None:
        size = np.linalg.norm(X - np.roll(X, -1, 1), axis=2).max()
        n = int(np.clip(np.ceil(2 * size / d), 2, 16))
    t = (np.arange(n) + 0.5) / n
    u, w = np.meshgrid(t, t, indexing="ij")
    u, w = u.ravel(), w.ravel()
    W = np.stack([(1 - u) * (1 - w), u * (1 - w), u * w, (1 - u) * w], -1)   # (n*n, 4)
    P = np.einsum("sc,ecd->esd", W, X)
    (x0, _), _, (z0, _) = grid.extent
    i = np.clip(((P[..., 0] - x0) // grid.spacing[0]).astype(int), 0, v.shape[0] - 1)
    k = np.clip(((P[..., 1] - z0) // grid.spacing[2]).astype(int), 0, v.shape[1] - 1)
    return v[i, k]
