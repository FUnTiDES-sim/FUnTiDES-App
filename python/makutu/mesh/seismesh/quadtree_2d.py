r"""Maillage 2-D par arbre quaternaire 2:1 conforme, en partant de la vitesse la plus rapide.

La grille de base est faite de *blocs* dimensionnés pour la plus grande taille requise (zone la
plus rapide) ; un bloc est découpé en 4 tant que ses éléments dépassent la taille requise par la
vitesse locale. Un bloc de niveau l contient 2 x 2 éléments réguliers ; deux blocs voisins par un
côté diffèrent d'au plus un niveau. Les zones rapides gardent donc de grands éléments à
l'intérieur de zones plus lentes (grossissement local), les zones lentes de petits.

Conformité : un côté de bloc voisin de blocs plus fins porte 4 segments au lieu de 2 ; le bloc
reçoit un gabarit selon l'ensemble de ces côtés (coordonnées en quarts de bloc, u vers x, v vers
la profondeur) :

    aucun            4 éléments
    un côté          gabarit de doublement (4 fines -> 2 grossières, :data:`doubling.TEMPLATE`) dans
                     la moitié du côté fin, 2 éléments dans l'autre
    deux opposés     deux gabarits de doublement
    deux adjacents   le quart du coin raffiné (2 x 2), deux quarts en gabarit de coin (3 quadrangles,
                     nœud au centre), un quart régulier
    trois côtés      les deux quarts du côté central raffinés, deux quarts en gabarit de coin
    quatre côtés     le bloc est découpé

Chaque côté porte un nombre pair de segments (condition d'un maillage tout quadrangle) ; les
nœuds sont sur un réseau entier de 4 * 2^lmax unités par bloc de base (fusion exacte).
"""
from __future__ import annotations

import itertools

import numpy as np

from .doubling import TEMPLATE

SIDES = ("top", "right", "bottom", "left")          # v = 0, u = 4, v = 4, u = 0
_MID = {"top": (2, 0), "right": (4, 2), "bottom": (2, 4), "left": (0, 2)}


def _box(u0, v0, u1, v1):
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


def _refined_quadrant(u0, v0):
    return [_box(u0 + a, v0 + b, u0 + a + 1, v0 + b + 1) for a in (0, 1) for b in (0, 1)]


def _corner(u0, v0, ku, kv):
    """Quart [u0, u0+2] x [v0, v0+2] avec un nœud au milieu des deux côtés qui se rejoignent au coin
    (u0 + 2 ku, v0 + 2 kv) : 3 quadrangles autour du centre."""
    def p(a, b):                                   # local (a, b) in [0, 2], corner K at (2, 0)
        a, b = (a if ku else 2 - a), (b if kv == 0 else 2 - b)
        return (u0 + a, v0 + b)
    K, A, C, O = p(2, 0), p(0, 0), p(2, 2), p(0, 2)
    H1, H2, P = p(1, 0), p(2, 1), p(1, 1)
    return [[H1, K, H2, P], [A, H1, P, O], [P, H2, C, O]]


def _top_doubling(flip=False):
    """Gabarit de doublement dans la moitié haute (côté fin en v = 0) ; ``flip`` : moitié basse."""
    return [[(u, (4 - z) if flip else z) for u, z in q] for q in TEMPLATE]


_CANON = {
    frozenset(): [_box(0, 0, 2, 2), _box(2, 0, 4, 2), _box(0, 2, 2, 4), _box(2, 2, 4, 4)],
    frozenset({"top"}): _top_doubling() + [_box(0, 2, 2, 4), _box(2, 2, 4, 4)],
    frozenset({"top", "bottom"}): _top_doubling() + _top_doubling(flip=True),
    frozenset({"top", "right"}): _refined_quadrant(2, 0) + _corner(0, 0, 1, 0) + _corner(2, 2, 1, 0)
    + [_box(0, 2, 2, 4)],
    frozenset({"top", "right", "left"}): _refined_quadrant(0, 0) + _refined_quadrant(2, 0)
    + _corner(0, 2, 0, 0) + _corner(2, 2, 1, 0),
}


def _transforms():
    for swap, fu, fv in itertools.product((0, 1), (0, 1), (0, 1)):
        yield swap, fu, fv


def _apply(p, t):
    swap, fu, fv = t
    u, v = (p[1], p[0]) if swap else p
    return (4 - u if fu else u, 4 - v if fv else v)


def _lookup():
    side_of = {v: k for k, v in _MID.items()}
    table = {}
    for sides, quads in _CANON.items():
        for t in _transforms():
            key = frozenset(side_of[_apply(_MID[s], t)] for s in sides)
            if key not in table:
                table[key] = [[_apply(p, t) for p in q] for q in quads]
    return table


LOOKUP = _lookup()


class Quadtree2D:
    """Blocs sur la grille de base (xs, zs) (z profondeur), découpés jusqu'à ``lmax`` niveaux."""

    def __init__(self, xs, zs, lmax: int):
        self.axes = [np.asarray(xs, float), np.asarray(zs, float)]
        self.n0 = [len(a) - 1 for a in self.axes]
        self.lmax = lmax
        self.R = 4 * 2 ** lmax                     # lattice units per base block
        self.leaves = {(0, i, k) for i in range(self.n0[0]) for k in range(self.n0[1])}

    def size(self, level):
        return self.R // 2 ** level

    def phys(self, keys):
        out = np.empty(keys.shape, float)
        for a in range(2):
            q = keys[..., a]
            b = np.minimum(q // self.R, self.n0[a] - 1)
            f = (q - b * self.R) / self.R
            ax = self.axes[a]
            out[..., a] = ax[b] + f * (ax[b + 1] - ax[b])
        return out

    def box(self, c):
        level, i, k = c
        s = self.size(level)
        lo = self.phys(np.array([[i * s, k * s]]))[0]
        hi = self.phys(np.array([[(i + 1) * s, (k + 1) * s]]))[0]
        return lo, hi

    def split(self, c):
        level, i, k = c
        self.leaves.discard(c)
        for a in (0, 1):
            for b in (0, 1):
                self.leaves.add((level + 1, 2 * i + a, 2 * k + b))

    def _inside(self, level, i, k):
        n = 2 ** level
        return 0 <= i < self.n0[0] * n and 0 <= k < self.n0[1] * n

    def _subdivided(self, level, i, k):
        """La région du bloc (level, i, k) est-elle découpée en blocs plus fins ?"""
        if not self._inside(level, i, k) or (level, i, k) in self.leaves:
            return False
        lv, ii, kk = level, i, k
        while lv > 0:                              # covered by a coarser leaf?
            lv, ii, kk = lv - 1, ii // 2, kk // 2
            if (lv, ii, kk) in self.leaves:
                return False
        return True

    _STEP = {"top": (0, -1), "right": (1, 0), "bottom": (0, 1), "left": (-1, 0)}

    def finer_sides(self, c):
        level, i, k = c
        return frozenset(s for s, (di, dk) in self._STEP.items() if self._subdivided(level, i + di, k + dk))

    def refine_where(self, need) -> int:
        """need(lo, hi, level) -> bool (boîte (x, profondeur) du bloc) ; découpe jusqu'à lmax."""
        n, todo = 0, [c for c in self.leaves if c[0] < self.lmax]
        while todo:
            nxt = []
            for c in todo:
                if c in self.leaves and need(*self.box(c), c[0]):
                    self.split(c)
                    n += 1
                    if c[0] + 1 < self.lmax:
                        nxt += [(c[0] + 1, 2 * c[1] + a, 2 * c[2] + b) for a in (0, 1) for b in (0, 1)]
            todo = nxt
        return n

    def balance(self, max_iter=1000) -> int:
        """Équilibre 2:1 par les côtés (un voisin découpé de plus d'un niveau impose de découper)
        et découpe des blocs dont les quatre côtés sont plus fins ; renvoie le nombre de découpes."""
        forced = 0
        for _ in range(max_iter):
            to_split = []
            for c in self.leaves:
                level, i, k = c
                fine = self.finer_sides(c)
                if len(fine) == 4:
                    to_split.append(c)
                    continue
                for s in fine:                     # a neighbour child adjacent to c subdivided again?
                    di, dk = self._STEP[s]
                    ni, nk = 2 * (i + di), 2 * (k + dk)
                    kids = {"top": [(ni, nk + 1), (ni + 1, nk + 1)], "bottom": [(ni, nk), (ni + 1, nk)],
                            "left": [(ni + 1, nk), (ni + 1, nk + 1)], "right": [(ni, nk), (ni, nk + 1)]}[s]
                    if any(self._subdivided(level + 1, a, b) for a, b in kids):
                        to_split.append(c)
                        break
            if not to_split:
                return forced
            for c in to_split:
                if c[0] >= self.lmax:
                    raise RuntimeError("quadtree : équilibrage impossible sans dépasser lmax")
                self.split(c)
            forced += len(to_split)
        raise RuntimeError("quadtree : équilibrage non convergé")

    def to_quads(self):
        """(nœuds (x, profondeur), quadrangles, niveau du bloc de chaque quadrangle)."""
        blocks, levels = [], []
        for c in self.leaves:
            level, i, k = c
            s = self.size(level)
            q = np.array(LOOKUP[self.finer_sides(c)], dtype=np.int64)        # (nq, 4, 2) quarter units
            blocks.append(np.array([i * s, k * s]) + q * (s // 4))
            levels += [level] * len(q)
        K = np.concatenate(blocks)
        M = self.n0[0] * self.R + 1
        code = K[..., 1] * M + K[..., 0]
        uniq, inv = np.unique(code.ravel(), return_inverse=True)
        keys = np.stack([uniq % M, uniq // M], -1)
        return self.phys(keys), inv.reshape(-1, 4), np.array(levels)
