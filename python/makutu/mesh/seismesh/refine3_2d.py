r"""Raffinement local conforme 1:3 en 2-D (arbre 9) avec gabarits de transition de Schneiders.

Version 2-D de :mod:`refine3` : une cellule raffinée devient 3 x 3 sous-cellules ; une cellule
voisine non raffinée reçoit un gabarit d'après ses sommets « marqués » (touchés par une cellule plus
fine) :

    0 sommet          -> quadrangle inchangé
    1 sommet          -> gabarit SOMMET (3 quadrangles)
    2 sommets voisins -> gabarit ARÊTE  (motif E, 7 quadrangles : 3 petits, puis L, M, R, T)
    autre             -> la cellule est raffinée (9), puis on itère

Ce sont les motifs de face des gabarits 3-D : une arête voisine d'une cellule raffinée porte des
nœuds à 1/3 et 2/3, l'arête d'un gabarit SOMMET un nœud à 1/3 côté sommet marqué ; les cellules
voisines se raccordent donc sans nœud pendant. Coordonnées des gabarits en 1/18 de cellule ; nœuds
sur un réseau entier de résolution 18 * 3^lmax par cellule de base (fusion exacte).
"""
from __future__ import annotations

import itertools

import numpy as np

N, A, MU = 18, 6, 14                        # 1/3 et 7/9 en 18es (comme refine3)
_CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]


def _vertex():
    """Sommet marqué (0, 0)."""
    return [[(0, 0), (A, 0), (A, A), (0, A)],
            [(A, 0), (N, 0), (N, N), (A, A)],
            [(0, A), (A, A), (N, N), (0, N)]]


def _edge():
    """Arête marquée (0, 0)-(N, 0) : 3 petits quadrangles puis la partie haute du motif E."""
    small = [[(A * i, 0), (A * i + A, 0), (A * i + A, A), (A * i, A)] for i in range(3)]
    m = MU                                   # nœuds intermédiaires à 7/9 de hauteur
    up = [[(0, A), (A, A), (A, m), (0, N)],
          [(A, A), (2 * A, A), (2 * A, m), (A, m)],
          [(2 * A, A), (N, A), (N, N), (2 * A, m)],
          [(A, m), (2 * A, m), (N, N), (0, N)]]
    return small + up


def _transforms():
    for swap, fx, fy in itertools.product((0, 1), (0, 1), (0, 1)):
        yield swap, fx, fy


def _apply(p, t, n):
    swap, fx, fy = t
    x, y = (p[1], p[0]) if swap else p
    return (n - x if fx else x, n - y if fy else y)


def _lookup():
    table = {0: None}
    index = {c: k for k, c in enumerate(_CORNERS)}
    for name, marked, quads in (("vertex", {0}, _vertex()), ("edge", {0, 1}, _edge())):
        for t in _transforms():
            code = sum(1 << index[_apply(_CORNERS[c], t, 1)] for c in marked)
            if code not in table:
                table[code] = (name, [[_apply(p, t, N) for p in q] for q in quads])
    return table


LOOKUP = _lookup()


class Refine3Mesh2D:
    """Grille de base (xs, zs) raffinée localement d'un facteur 3, jusqu'à ``lmax`` niveaux."""

    def __init__(self, xs, zs, lmax: int):
        self.axes = [np.asarray(xs, float), np.asarray(zs, float)]
        self.n0 = [len(a) - 1 for a in self.axes]
        self.lmax = lmax
        self.R = N * 3 ** lmax
        self.leaves = {(0, i, k) for i in range(self.n0[0]) for k in range(self.n0[1])}

    def size(self, level):
        return self.R // 3 ** level

    def corners(self, c):
        level, i, k = c
        s = self.size(level)
        return [((i + a) * s, (k + b) * s) for a, b in _CORNERS]

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
        cs = np.array(self.corners(c))
        return self.phys(cs[0][None])[0], self.phys(cs[2][None])[0]

    def split(self, c):
        level, i, k = c
        self.leaves.discard(c)
        for a in range(3):
            for b in range(3):
                self.leaves.add((level + 1, 3 * i + a, 3 * k + b))

    def refine_where(self, need) -> int:
        """need(lo, hi, level) -> bool, boîtes (x, z) ; raffine jusqu'à lmax."""
        n, todo = 0, [c for c in self.leaves if c[0] < self.lmax]
        while todo:
            nxt = []
            for c in todo:
                if c in self.leaves and need(*self.box(c), c[0]):
                    self.split(c)
                    n += 1
                    if c[0] + 1 < self.lmax:
                        nxt += [(c[0] + 1, 3 * c[1] + a, 3 * c[2] + b) for a in range(3) for b in range(3)]
            todo = nxt
        return n

    def _levels(self):
        m = {}
        for c in self.leaves:
            for key in self.corners(c):
                if m.get(key, -1) < c[0]:
                    m[key] = c[0]
        return m

    def balance(self, max_iter=500) -> int:
        """Équilibrage (deux feuilles partageant un sommet diffèrent d'au plus un niveau) et
        réduction aux motifs admissibles ; renvoie le nombre de raffinements forcés."""
        forced = 0
        for _ in range(max_iter):
            m = self._levels()
            to_split = []
            for c in self.leaves:
                lev = [m[key] for key in self.corners(c)]
                code = sum(1 << q for q, v in enumerate(lev) if v == c[0] + 1)
                if max(lev) >= c[0] + 2 or (code and code not in LOOKUP):
                    to_split.append(c)
            if not to_split:
                return forced
            for c in to_split:
                if c[0] >= self.lmax:
                    raise RuntimeError("équilibrage impossible sans dépasser lmax")
                self.split(c)
            forced += len(to_split)
        raise RuntimeError("équilibrage non convergé")

    def touches(self, side: str) -> bool:
        """Une feuille raffinée touche-t-elle le bord haut ('top', z = zs[0]) ou bas ('bottom') ?"""
        m = self._levels()
        kz = 0 if side == "top" else self.n0[1] * self.R
        return any(lev > 0 for (x, z), lev in m.items() if z == kz)

    def to_quads(self):
        """(nœuds (x, z), quadrangles)."""
        m = self._levels()
        blocks = []
        for c in self.leaves:
            cs = self.corners(c)
            code = sum(1 << q for q, key in enumerate(cs) if m[key] == c[0] + 1)
            if code == 0:
                blocks.append(np.array([cs]))
                continue
            _, tmpl = LOOKUP[code]
            u = self.size(c[0]) // N
            blocks.append(np.array(cs[0]) + u * np.array(tmpl))
        K = np.concatenate(blocks).astype(np.int64)                 # (ne, 4, 2)
        M = self.n0[0] * self.R + 1
        code = K[..., 1] * M + K[..., 0]
        uniq, inv = np.unique(code.ravel(), return_inverse=True)
        keys = np.stack([uniq % M, uniq // M], -1)
        return self.phys(keys), inv.reshape(-1, 4)
