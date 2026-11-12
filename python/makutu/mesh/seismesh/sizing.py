"""Critère de taille d'élément et segmentation des axes.

Taille requise (élément d'ordre N, ppw points GLL par longueur d'onde minimale) :
    h_req(x) = N * v_min(x) / (f_max * ppw)
Pour un schéma bas ordre / DF équivalent, N=1 et ppw ~ 8-12.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .velocity import VelocityGrid


# Tolérance numérique de l'échantillonnage : un élément est sous-échantillonné si ppw_effectif <
# ppw_min x (1 - PPW_TOLERANCE). Elle n'absorbe que les arrondis (bords des cellules SEP, positions des
# nœuds) ; un maillage n'est valide que si *aucun* élément n'est sous-échantillonné.
PPW_TOLERANCE = 0.01


@dataclass
class SizingParams:
    fmax: float = 15.0          # Hz
    order: int = 4              # ordre polynomial SEM (1 => critère type DF)
    ppw: float = 5.0            # points GLL par longueur d'onde minimale
    safety: float = 1.0         # facteur <1 pour marge
    hmin: float = 0.0           # bornes optionnelles
    hmax: float = np.inf
    ppw_min: float | None = None  # échantillonnage minimal accepté (validité ; défaut : ppw)

    @property
    def ppw_accept(self) -> float:
        return self.ppw if self.ppw_min is None else min(self.ppw, self.ppw_min)

    def to_dict(self):
        d = asdict(self)
        d["hmax"] = None if np.isinf(self.hmax) else self.hmax
        return d


def h_required(grid: VelocityGrid, p: SizingParams) -> np.ndarray:
    h = p.safety * p.order * grid.vel / (p.fmax * p.ppw)
    return np.clip(h, p.hmin, p.hmax)


def ideal_element_count(grid: VelocityGrid, h: np.ndarray) -> float:
    """Borne inférieure : ∫ dV / h_req^3 (maillage parfaitement adapté)."""
    cell = np.prod(grid.spacing)
    return float(np.sum(cell / h.astype(np.float64) ** 3))


def axis_profile(h: np.ndarray, k: int) -> np.ndarray:
    """Espacement admissible le long de l'axe k : min sur le plan orthogonal."""
    other = tuple(a for a in range(3) if a != k)
    return h.min(axis=other)


# Tolérance numérique de l'échantillonnage : un élément est sous-échantillonné si ppw_effectif <
# ppw_min x (1 - PPW_TOLERANCE). Elle n'absorbe que les arrondis (bords des cellules SEP, positions des
# nœuds) ; un maillage n'est valide que si *aucun* élément n'est sous-échantillonné.
PPW_TOLERANCE = 0.01


@dataclass
class Segment:
    t0: float
    t1: float
    n: int               # nombre d'éléments dans le segment

    @property
    def spacing(self):
        return (self.t1 - self.t0) / self.n


def segment_axis(t: np.ndarray, s: np.ndarray, max_segments: int,
                 min_elems_per_segment: int = 2, max_grading: float = 2.0,
                 max_candidates: int = 160) -> list[Segment]:
    """Découpe optimale (programmation dynamique) de [t0,tN] en ≤ K segments à pas constant.

    Minimise le nombre total d'éléments sous contrainte : pas du segment ≤ min(s) sur le
    segment. `max_grading` borne le rapport de pas entre segments voisins (qualité / CFL) —
    appliqué a posteriori en raffinant le segment le plus grossier.
    """
    n = len(t)
    cand = np.unique(np.linspace(0, n - 1, min(n, max_candidates)).round().astype(int))
    m = len(cand)
    # min de s sur [cand[i], cand[j]] : table O(m^2)
    smin = np.full((m, m), np.inf)
    for i in range(m):
        run = np.minimum.accumulate(s[cand[i]:])
        smin[i, i + 1:] = run[cand[i + 1:] - cand[i]]
    L = t[cand][None, :] - t[cand][:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        cost = np.maximum(np.ceil(L / smin - 1e-9), min_elems_per_segment)
    cost[np.tril_indices(m)] = np.inf
    K = max(1, max_segments)
    INF = np.inf
    best = np.full((K + 1, m), INF)
    arg = np.zeros((K + 1, m), dtype=int)
    best[0, 0] = 0
    for k in range(1, K + 1):
        tot = best[k - 1][:, None] + cost          # (i, j)
        arg[k] = np.argmin(tot, axis=0)
        best[k] = tot[arg[k], np.arange(m)]
    k_best = int(np.argmin(best[:, m - 1]))
    # backtrack
    idx, j = [m - 1], m - 1
    for k in range(k_best, 0, -1):
        j = arg[k, j]
        idx.append(j)
    idx = idx[::-1]
    segs = [Segment(float(t[cand[a]]), float(t[cand[b]]), int(cost[a, b]))
            for a, b in zip(idx[:-1], idx[1:])]
    return _enforce_grading(segs, max_grading)


def _enforce_grading(segs: list[Segment], r: float) -> list[Segment]:
    changed = True
    while changed:
        changed = False
        for a, b in zip(segs[:-1], segs[1:]):
            for big, small in ((a, b), (b, a)):
                if big.spacing > r * small.spacing * (1 + 1e-9):
                    big.n = int(np.ceil((big.t1 - big.t0) / (r * small.spacing)))
                    changed = True
    return segs


def horizontal_interfaces(grid: VelocityGrid, rel_jump: float = 0.1, coverage: float = 0.5) -> list[float]:
    """Profondeurs des interfaces horizontales nettes : saut relatif de vitesse > ``rel_jump`` entre
    deux échantillons voisins en z sur au moins ``coverage`` du plan horizontal. L'interface est
    placée à mi-chemin des deux échantillons (aucun échantillon de part et d'autre n'est mélangé)."""
    v = grid.vel
    jump = np.abs(np.diff(v, axis=2)) / np.minimum(v[:, :, 1:], v[:, :, :-1])
    frac = (jump > rel_jump).mean(axis=(0, 1))
    z = grid.axis(2)
    return [float(0.5 * (z[k] + z[k + 1])) for k in np.where(frac >= coverage)[0]]


def segment_axis_with_breaks(t: np.ndarray, s: np.ndarray, max_segments: int, breaks=(), max_grading=2.0,
                             **kw) -> list[Segment]:
    """:func:`segment_axis` avec des limites de segment imposées (``breaks``, coordonnées) : chaque
    morceau entre deux limites est segmenté séparément, puis la gradation est imposée sur l'ensemble."""
    cuts = [b for b in sorted(breaks) if t[0] < b < t[-1]]
    if not cuts:
        return segment_axis(t, s, max_segments, max_grading=max_grading, **kw)
    bounds = [t[0]] + cuts + [t[-1]]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        inside = np.where((t >= a - 1e-9) & (t <= b + 1e-9))[0]
        smin = float(s[inside].min()) if inside.size else float(s[np.argmin(np.abs(t - 0.5 * (a + b)))])
        if inside.size >= 3:
            part = segment_axis(t[inside], s[inside], max_segments, max_grading=max_grading, **kw)
            part[0].t0, part[-1].t1 = a, b                  # exact interface position
            for g in (part[0], part[-1]):
                lo, hi = np.searchsorted(t, g.t0 - 1e-9), np.searchsorted(t, g.t1 + 1e-9)
                req = float(s[lo:hi].min()) if hi > lo else smin
                g.n = max(g.n, int(np.ceil((g.t1 - g.t0) / req - 1e-9)), 1)
        else:
            part = [Segment(a, b, max(1, int(np.ceil((b - a) / smin - 1e-9))))]
        segs += part
    return _enforce_grading(segs, max_grading)


def segment_all_axes(grid: VelocityGrid, h: np.ndarray, max_segments=(6, 6, 8), honor_interfaces=True, **kw):
    """Segmentation des trois axes ; avec ``honor_interfaces``, des faces d'éléments passent par les
    interfaces horizontales nettes du modèle (:func:`horizontal_interfaces`)."""
    breaks = horizontal_interfaces(grid) if honor_interfaces else ()
    return [segment_axis(grid.edge_axis(k), axis_profile(h, k), max_segments[k], **kw) for k in range(2)] + \
        [segment_axis_with_breaks(grid.edge_axis(2), axis_profile(h, 2), max_segments[2], breaks, **kw)]


def plane_size_map(grid: VelocityGrid, h: np.ndarray, z_range=None) -> np.ndarray:
    """Carte 2D (nx,ny) de taille horizontale admissible : min sur z (dans z_range)."""
    if z_range is None:
        return h.min(axis=2)
    z = grid.axis(2)
    sel = (z >= z_range[0]) & (z <= z_range[1])
    return h[:, :, sel].min(axis=2)
