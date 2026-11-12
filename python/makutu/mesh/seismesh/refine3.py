r"""Raffinement local conforme par 3-raffinement (arbre 27) avec gabarits de transition.

Principe (Schneiders) : on raffine localement les cellules d'une grille de base en 27 sous-cubes
(facteur 3 par direction). Les cellules non raffinées voisines reçoivent un gabarit de transition
choisi d'après leurs sommets « marqués » (sommets touchés par une cellule plus fine) :

    0 sommet          → hexaèdre inchangé
    1 sommet          → gabarit SOMMET  (4 hexas)
    2 sommets (arête) → gabarit ARÊTE   (11 hexas)
    4 sommets (face)  → gabarit FACE    (22 hexas)
    autre             → la cellule est elle-même raffinée (27), puis on itère

Les motifs de face sont canoniques (ne dépendent que des sommets marqués de la face) :
  face à 4 sommets marqués : grille 3×3 ; à 2 sommets (arête) : motif E à 7 quads
  (3 petits + L, M, R, T, nœuds intérieurs à 1/3 et 2/3 de hauteur 2/3) ;
  à 1 sommet : 3 quads ; à 0 : 1 quad. D'où la conformité entre cellules voisines.

Construction : les paires de feuillets de raffinement (2 par direction) se referment en « U »
à l'intérieur de la cellule de transition — c'est ce qui rend le 3-raffinement local possible
(le 2-raffinement n'a qu'un feuillet par direction, qui ne peut pas se refermer seul).

Coordonnées des gabarits en 1/18 de cellule (entiers 0..18). Hauteurs choisies pour égaliser
les arêtes courtes (meilleur pas de temps CFL) : nœuds m des motifs E à 7/9, nœuds intérieurs
du gabarit face à 5/9 → arête minimale 2/9 de cellule (au lieu de 1/6). Les nœuds sont identifiés par
des clés entières sur un réseau de résolution 2·3^Lmax par cellule de base → fusion exacte.
L'équilibrage est « fort » : deux feuilles partageant un sommet diffèrent d'au plus 1 niveau.
"""
from __future__ import annotations

import itertools

import numpy as np

# ------------------------------------------------------------ gabarits canoniques
N = 18          # résolution d'un gabarit
A, MU, TAU = 6, 14, 10   # 1/3, 7/9, 5/9 en 18es


def _box(x0, x1, y0, y1, z0, z1):
    return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]


def _vertex_template():
    """Sommet marqué (0,0,0)."""
    a, n = A, N
    return [
        _box(0, a, 0, a, 0, a),
        [(a, 0, 0), (n, 0, 0), (n, n, 0), (a, a, 0), (a, 0, a), (n, 0, n), (n, n, n), (a, a, a)],
        [(0, a, 0), (a, a, 0), (n, n, 0), (0, n, 0), (0, a, a), (a, a, a), (n, n, n), (0, n, n)],
        [(0, 0, a), (a, 0, a), (a, a, a), (0, a, a), (0, 0, n), (n, 0, n), (n, n, n), (0, n, n)],
    ]


# motif E (partie haute) dans (x, s) ; s = (hauteur - 1/3)/(2/3) ; nœuds m à s_m ↔ hauteur 7/9
_SM = (MU - A) / (N - A)
_E_UP = [[(0, 0), (6, 0), (6, _SM), (0, 1)], [(6, 0), (12, 0), (12, _SM), (6, _SM)],
         [(12, 0), (18, 0), (18, 1), (12, _SM)], [(6, _SM), (12, _SM), (18, 1), (0, 1)]]


def _edge_template():
    """Arête marquée (0,0,0)-(6,0,0) (le long de x, en y=z=0)."""
    hexes = [_box(6 * i, 6 * i + 6, 0, 6, 0, 6) for i in range(3)]

    def bil(c00, c10, c11, c01, s, t):
        return tuple((1 - s) * (1 - t) * a + s * (1 - t) * b + s * t * c + (1 - s) * t * d
                     for a, b, c, d in zip(c00, c10, c11, c01))
    # Qy : de la face y=1/3 (s=0) à y=1 (s=1) ; t : de z=0 (t=0) à la diagonale (t=1). (y,z)
    Qy = ((6, 0), (18, 0), (18, 18), (6, 6))
    Qz = ((0, 6), (0, 18), (18, 18), (6, 6))
    for Q in (Qy, Qz):
        for quad in _E_UP:
            h = []
            for t in (0, 1):
                for x, s in quad:
                    y, z = bil(*Q, s, t)
                    h.append((x, y, z))
            hexes.append(h)
    return [[tuple(int(round(v)) for v in p) for p in h] for h in hexes]


def _face_template():
    """Face marquée z=0."""
    hexes = [_box(6 * i, 6 * i + 6, 6 * j, 6 * j + 6, 0, 6) for i in range(3) for j in range(3)]
    b = {(i, j): (6 * i, 6 * j, 6) for i in range(4) for j in range(4)}
    t = {}
    for i in range(4):
        for j in range(4):
            on_x, on_y = i in (0, 3), j in (0, 3)
            if on_x and on_y:
                t[i, j] = (6 * i, 6 * j, N)          # coins du haut
            elif on_x or on_y:
                t[i, j] = (6 * i, 6 * j, MU)         # nœuds m des motifs E latéraux
            else:
                t[i, j] = (6 * i, 6 * j, TAU)        # nœuds intérieurs
    for i in range(3):
        for j in range(3):
            hexes.append([b[i, j], b[i + 1, j], b[i + 1, j + 1], b[i, j + 1],
                          t[i, j], t[i + 1, j], t[i + 1, j + 1], t[i, j + 1]])
    T = t
    hexes += [
        [T[1, 0], T[2, 0], T[2, 3], T[1, 3], T[0, 0], T[3, 0], T[3, 3], T[0, 3]],   # chapeau
        [T[0, 0], T[0, 1], T[0, 2], T[0, 3], T[1, 0], T[1, 1], T[1, 2], T[1, 3]],   # côté x=0
        [T[3, 0], T[3, 1], T[3, 2], T[3, 3], T[2, 0], T[2, 1], T[2, 2], T[2, 3]],   # côté x=1
        [T[1, 1], T[2, 1], T[2, 2], T[1, 2], T[1, 0], T[2, 0], T[2, 3], T[1, 3]],   # centre
    ]
    return hexes


_CANON = {"vertex": ({0}, _vertex_template()),
          "edge": ({0, 1}, _edge_template()),
          "face": ({0, 1, 2, 3}, _face_template())}
_CORNER_BITS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def _transforms():
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((0, 1), repeat=3):
            yield perm, flips


def _apply(p, perm, flips, n):
    q = [p[perm[a]] for a in range(3)]
    return tuple(n - q[a] if flips[a] else q[a] for a in range(3))


def _build_lookup():
    """motif (bits des 8 coins) -> liste de hexas (coords 0..6) déjà orientés pour la cellule."""
    corner_index = {c: k for k, c in enumerate(_CORNER_BITS)}
    table = {0: None}
    for name, (marked, hexes) in _CANON.items():
        for perm, flips in _transforms():
            mk = {corner_index[_apply(_CORNER_BITS[c], perm, flips, 1)] for c in marked}
            code = sum(1 << c for c in mk)
            if code not in table:
                table[code] = (name, [[_apply(p, perm, flips, N) for p in h] for h in hexes])
    return table


LOOKUP = _build_lookup()


# ------------------------------------------------------------------ arbre 27
class Refine3Mesh:
    def __init__(self, xs, ys, zs, lmax: int):
        self.axes = [np.asarray(xs, float), np.asarray(ys, float), np.asarray(zs, float)]
        self.n0 = [len(a) - 1 for a in self.axes]
        self.lmax = lmax
        self.R = 6 * 3 ** lmax                         # résolution du réseau par cellule de base
        self.leaves = {(0, i, j, k) for i in range(self.n0[0]) for j in range(self.n0[1])
                       for k in range(self.n0[2])}

    def size(self, l):
        return self.R // 3 ** l

    def corners(self, c):
        l, i, j, k = c
        s = self.size(l)
        return [((i + a) * s, (j + b) * s, (k + d) * s) for a, b, d in _CORNER_BITS]

    def phys(self, keys: np.ndarray) -> np.ndarray:
        out = np.empty(keys.shape, float)
        for a in range(3):
            q = keys[..., a]
            b = np.minimum(q // self.R, self.n0[a] - 1)
            f = (q - b * self.R) / self.R
            ax = self.axes[a]
            out[..., a] = ax[b] + f * (ax[b + 1] - ax[b])
        return out

    def cell_box(self, c):
        k0 = np.array(self.corners(c)[0])
        k1 = np.array(self.corners(c)[6])
        return self.phys(k0[None])[0], self.phys(k1[None])[0]

    def split(self, c):
        l, i, j, k = c
        self.leaves.discard(c)
        for a in range(3):
            for b in range(3):
                for d in range(3):
                    self.leaves.add((l + 1, 3 * i + a, 3 * j + b, 3 * k + d))

    def refine_where(self, need) -> int:
        """need(lo, hi, level) -> bool. Raffine itérativement jusqu'à lmax."""
        n = 0
        todo = [c for c in self.leaves if c[0] < self.lmax]
        while todo:
            nxt = []
            for c in todo:
                if c in self.leaves and need(*self.cell_box(c), c[0]):
                    self.split(c)
                    n += 1
                    l, i, j, k = c
                    if l + 1 < self.lmax:
                        nxt += [(l + 1, 3 * i + a, 3 * j + b, 3 * k + d)
                                for a in range(3) for b in range(3) for d in range(3)]
            todo = nxt
        return n

    def _node_maxlevel(self):
        m = {}
        for c in self.leaves:
            for key in self.corners(c):
                if m.get(key, -1) < c[0]:
                    m[key] = c[0]
        return m

    def balance(self, max_iter=500) -> dict:
        """Équilibrage fort + réduction aux motifs admissibles (sinon raffinement 27)."""
        stats = {"iterations": 0, "forced_splits": 0}
        for it in range(max_iter):
            m = self._node_maxlevel()
            unbalanced, invalid = [], []
            for c in self.leaves:
                l = c[0]
                lev = [m[key] for key in self.corners(c)]
                if max(lev) >= l + 2:
                    unbalanced.append(c)
                    continue
                n_mk = sum(v == l + 1 for v in lev)
                code = sum(1 << q for q, v in enumerate(lev) if v == l + 1)
                if code and code not in LOOKUP:
                    invalid.append((n_mk, c))
            stats["iterations"] = it + 1
            if not unbalanced and not invalid:
                return stats
            # on ne raffine que les cellules les plus marquées (coins concaves) : raffiner
            # leurs voisines en même temps provoquerait une cascade inutile
            to_split = unbalanced
            if invalid:
                top = max(n for n, _ in invalid)
                to_split = to_split + [c for n, c in invalid if n == top]
            for c in to_split:
                if c[0] >= self.lmax:
                    raise RuntimeError("équilibrage impossible sans dépasser lmax")
                self.split(c)
            stats["forced_splits"] += len(to_split)
        raise RuntimeError("équilibrage non convergé")

    def to_hexes(self):
        m = self._node_maxlevel()
        blocks, counts = [], {"plain": 0, "vertex": 0, "edge": 0, "face": 0}
        for c in self.leaves:
            l, i, j, k = c
            cs = self.corners(c)
            code = sum(1 << q for q, key in enumerate(cs) if m[key] == l + 1)
            if code == 0:
                blocks.append(np.array([cs]))
                counts["plain"] += 1
                continue
            name, tmpl = LOOKUP[code]
            u = self.size(l) // N
            o = np.array(cs[0])
            blocks.append(o + u * np.array(tmpl))
            counts[name] += 1
        K = np.concatenate(blocks).astype(np.int64)            # (ne,8,3) clés réseau
        M = np.array([n * self.R + 1 for n in self.n0], dtype=np.int64)
        code = (K[..., 2] * M[1] + K[..., 1]) * M[0] + K[..., 0]
        uniq, inv = np.unique(code.ravel(), return_inverse=True)
        keys = np.stack([uniq % M[0], (uniq // M[0]) % M[1], uniq // (M[0] * M[1])], -1)
        nodes = self.phys(keys)
        hexes = inv.reshape(-1, 8)
        levels = {}
        for c in self.leaves:
            levels[c[0]] = levels.get(c[0], 0) + 1
        return nodes, _orient(nodes, hexes), {"templates": counts, "leaves_per_level": levels}


    # ------------------------------------------- gabarits de Tong & Zhang (2026)
    def _leaf_ancestor(self, c):
        """Feuille de niveau <= c[0] qui contient la cellule c (ou None si c est subdivisée)."""
        l, i, j, k = c
        for m in range(l, -1, -1):
            f = 3 ** (l - m)
            a = (m, i // f, j // f, k // f)
            if a in self.leaves:
                return a
        return None

    def balance_touch(self, max_iter=500) -> dict:
        """Équilibrage par contact : deux feuilles qui se touchent (même par un coin) diffèrent d'au
        plus un niveau. C'est la condition des gabarits de Tong & Zhang : le voisinage de chaque
        sommet noir est alors fait de cellules d'un seul niveau plus fin. Tous les codes de sommets
        sont admissibles : pas de raffinement forcé pour un motif interdit."""
        stats = {"iterations": 0, "forced_splits": 0}
        for it in range(max_iter):
            to_split = set()
            for (l, i, j, k) in self.leaves:
                if l < 2:
                    continue
                p = (l - 1, i // 3, j // 3, k // 3)
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        for d in (-1, 0, 1):
                            q = (l - 1, p[1] + a, p[2] + b, p[3] + d)
                            n = 3 ** (l - 1)
                            if not all(0 <= q[1 + e] < self.n0[e] * n for e in range(3)):
                                continue
                            anc = self._leaf_ancestor(q)
                            if anc is not None and anc[0] < l - 1:
                                to_split.add(anc)
            stats["iterations"] = it + 1
            if not to_split:
                return stats
            for c in to_split:
                self.split(c)
            stats["forced_splits"] += len(to_split)
        raise RuntimeError("équilibrage non convergé")

    def to_hexes_tz(self, decimals=6):
        """Hexaèdres avec les gabarits de Tong & Zhang (module ``templates3_es``) : chaque feuille
        reçoit le gabarit de son code de sommets (sommet noir = coin d'une feuille plus fine). Les
        gabarits sont évalués en coordonnées du réseau (réels), puis envoyés dans l'espace physique
        (affine dans chaque cellule de base) ; nœuds fusionnés sur coordonnées arrondies."""
        from . import templates3_es as tz
        m = self._node_maxlevel()
        blocks, cls_count, hex_cls = [], {}, []
        for c in self.leaves:
            l = c[0]
            cs = self.corners(c)
            black = [m[key] > l for key in cs]
            code = tz.cell_code(black)
            lo = np.array(cs[0], float)
            size = np.array(cs[6], float) - lo
            hx = np.array(tz.cell_hexes(lo, size, code))
            blocks.append(hx)
            cls = tz.ROTATIONS[code][0]
            hex_cls.append(np.full(len(hx), cls))
            cls_count[cls] = cls_count.get(cls, 0) + 1
        K = np.concatenate(blocks)                               # (ne, 8, 3) lattice coordinates
        key = np.round(K.reshape(-1, 3), decimals)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        nodes = self._phys_float(uniq)
        hexes = inv.reshape(-1, 8)
        levels = {}
        for c in self.leaves:
            levels[c[0]] = levels.get(c[0], 0) + 1
        return nodes, _orient(nodes, hexes), {"templates": "tong-zhang-2026",
                                               "classes": dict(sorted(cls_count.items())),
                                               "leaves_per_level": levels,
                                               "hex_class": np.concatenate(hex_cls)}

    def _phys_float(self, keys):
        out = np.empty(keys.shape, float)
        for a in range(3):
            q = keys[:, a]
            b = np.clip(np.floor(q / self.R + 1e-12).astype(int), 0, self.n0[a] - 1)
            f = (q - b * self.R) / self.R
            ax = self.axes[a]
            out[:, a] = ax[b] + f * (ax[b + 1] - ax[b])
        return out


def _orient(nodes, hexes):
    X = nodes[hexes]
    det = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 3] - X[:, 0]), X[:, 4] - X[:, 0])
    hexes = hexes.copy()
    neg = det < 0
    hexes[neg] = hexes[neg][:, [4, 5, 6, 7, 0, 1, 2, 3]]
    return hexes
