"""2-D : bandes de profondeur régulières à pas horizontal propre, raccordées par des zones déformées (stratégie ``bands``).

La taille des éléments est fixée par l'ordre du schéma et la fréquence de la source (taille requise
h). Chaque *bande* de profondeur est une grille régulière dont le nombre de colonnes est celui que
demande la vitesse la plus lente de la bande ; le pas vertical suit la vitesse (segmentation optimale
de ``tensor``, faces sur les interfaces horizontales nettes). Borne sans raccords :
L ∫ dz / h_min(z)^2, soit 1,2 à 1,4 fois le nombre idéal d'éléments sur les modèles SEG, Marmousi et
Sigsbee, contre environ 2,16 pour des paliers x 2 dans les deux directions (quadtree).

Raccord de deux bandes de n_f et n_c colonnes (n_f > n_c, même parité, n_f <= 3 n_c), entièrement dans
la bande grossière (taille requise h_c >= largeur B = L / n_c de ses colonnes) :

- *rangée de gabarits* (hauteur h_c) : chaque cellule grossière reçoit 1 ou 3 arêtes fines en parts
  égales (gabarit 3 -> 1 de Schneiders, 4 quadrangles) ; les (n_f - n_c) / 2 gabarits sont répartis
  régulièrement, avec le décalage de départ qui minimise l'écart Delta entre ces nœuds et la grille
  fine uniforme (|Delta| <= environ une cellule fine) ;
- *rangées de déformation* : m = ceil(Delta / (0,4 h_c)) rangées où les colonnes glissent des positions
  fines uniformes vers celles de la rangée de gabarits (glissement d = Delta / m par rangée) ; leur
  hauteur sqrt(h_c^2 - d^2) garde les arêtes obliques sous la taille requise.

Une bande grossière trop mince pour ses raccords reprend le nombre de colonnes de sa voisine plus fine.
"""
from __future__ import annotations

import numpy as np

SHIFT = 0.4          # largest sideways shift per morphing row, in units of the required size
MERGE = 1.05         # neighbour bands closer than this ratio of columns are merged (a transition would not pay)


def _template_positions(x0, x1, nc, nf, phases=24):
    """Positions des nœuds fins d'une rangée de gabarits (chaque cellule grossière reçoit 1 ou 3 arêtes
    en parts égales) et multiplicités, pour le décalage de départ qui minimise l'écart à la grille fine
    uniforme. Renvoie (positions, multiplicités, écart maximal)."""
    xc = np.linspace(x0, x1, nc + 1)
    uf = np.linspace(x0, x1, nf + 1)
    t = (nf - nc) // 2
    best = None
    for ph in np.arange(phases) / phases:
        mult = [3 if np.floor((j + 1) * t / nc + ph) - np.floor(j * t / nc + ph) == 1 else 1 for j in range(nc)]
        if sum(mult) != nf:
            continue
        P = [xc[0]]
        for j, k in enumerate(mult):
            P += [xc[j] + (xc[j + 1] - xc[j]) * i / k for i in range(1, k + 1)]
        P = np.array(P)
        dev = float(np.abs(P - uf).max())
        if best is None or dev < best[2]:
            best = (P, mult, dev)
    return best


def _template_row(P, mult, xc, zf, zc):
    """Quadrangles (listes de 4 points) d'une rangée de gabarits : nœuds fins P en zf, grossiers xc en zc."""
    quads, c = [], 0
    zm = 0.5 * (zf + zc)
    for j, k in enumerate(mult):
        C0, C1 = (xc[j], zc), (xc[j + 1], zc)
        if k == 1:
            quads.append([C0, C1, (P[c + 1], zf), (P[c], zf)])
        else:
            F = [(P[c + i], zf) for i in range(4)]
            G = ((2 * xc[j] + xc[j + 1]) / 3, zm)
            H = ((xc[j] + 2 * xc[j + 1]) / 3, zm)
            quads += [[C0, C1, H, G], [C0, G, F[1], F[0]], [G, H, F[2], F[1]], [H, C1, F[3], F[2]]]
        c += k
    return quads


def _plan(need, z, hmin, coarsen, dwell_m, quantum=2):
    """Bandes [k0, k1) d'échantillons de profondeur et leur nombre de colonnes (voir le module)."""
    n = len(need)
    bands, start, cur, k = [], 0, int(need[0]), 0
    while k < n:
        span = np.searchsorted(z, z[k] + dwell_m * hmin[k])           # samples over the dwell depth
        look = int(need[k:max(span, k + 1)].max())
        if need[k] > cur or (look * coarsen <= cur and k > start):
            if k > start:
                bands.append([start, k, cur])
            start, cur = k, max(look, int(need[k]))
        k += 1
    bands.append([start, n, cur])
    changed = True
    while changed:                                                    # ratio at most 3 between neighbours
        changed = False
        for i in range(len(bands) - 1):
            a, b = bands[i][2], bands[i + 1][2]
            if a > 3 * b:
                bands[i + 1][2], changed = -(-a // 3), True
            elif b > 3 * a:
                bands[i][2], changed = -(-b // 3), True
    changed = True
    while changed and len(bands) > 1:                                 # transitions only where they pay
        changed = False
        for i in range(len(bands) - 1):
            a, b = bands[i][2], bands[i + 1][2]
            if max(a, b) < MERGE * min(a, b):
                bands[i][2] = bands[i + 1][2] = max(a, b)
                bands = _merge(bands)
                changed = True
                break
    if quantum == 4:                                                  # multiples of 4: blocks, even parity
        for bd in bands:
            bd[2] = 4 * -(-bd[2] // 4)
        for i in range(len(bands) - 1):                               # ratio 3 may need one more step
            a, b = bands[i][2], bands[i + 1][2]
            if a > 3 * b:
                bands[i + 1][2] = 4 * -(-(-(-a // 3)) // 4)
            elif b > 3 * a:
                bands[i][2] = 4 * -(-(-(-b // 3)) // 4)
    else:
        par = bands[0][2] % 2
        for bd in bands:
            bd[2] += (bd[2] % 2 != par)
    return _merge(bands)


def _merge(bands):
    out = [list(bands[0])]
    for bd in bands[1:]:
        if bd[2] == out[-1][2]:
            out[-1][1] = bd[1]
        else:
            out.append(list(bd))
    return out


def mesh_bands_2d(grid, h, max_segments=8, coarsen=1.15, dwell=4.0, honor_interfaces=True, max_grading=8.0,
                  local=True):
    """Maillage ``bands`` (voir le module). ``max_segments`` : segments de pas vertical constant par
    partie régulière ; ``coarsen`` : gain minimal pour commencer une bande plus grossière ; ``dwell`` :
    épaisseur minimale visée d'une bande, en tailles requises. Retourne un ``QuadMesh``."""
    from .plane import QuadMesh
    from .sizing import horizontal_interfaces, segment_axis_with_breaks
    from .doubling import _nodes
    h2 = h[:, 0, :].astype(float)
    (x0, x1), _, (z0, z1) = grid.extent
    L = x1 - x0
    ze = z0 + grid.spacing[2] * np.arange(h2.shape[1] + 1)          # sample k spans [ze[k], ze[k+1]]
    hmin = h2.min(0)
    need = np.ceil(L / hmin - 1e-9).astype(int)
    bands = _plan(need, ze[:-1], hmin, coarsen, dwell, quantum=4 if local else 2)
    interfaces = horizontal_interfaces(grid) if honor_interfaces else []
    for _ in range(100):                                              # zones of the transitions; thin bands
        zones, thin = [], None
        for i in range(len(bands) - 1):
            a, b = bands[i], bands[i + 1]
            fine, coarse, down = (a, b, True) if a[2] > b[2] else (b, a, False)
            P, mult, dev = _template_positions(x0, x1, coarse[2], fine[2])
            hc = float(hmin[coarse[0]:coarse[1]].min())                # start: over the whole coarse band
            for _ in range(5):                                          # then over the zone depth only
                m = int(np.ceil(dev / (SHIFT * hc) - 1e-9)) if dev > 1e-9 * L else 0
                hm = np.sqrt(max(hc ** 2 - (dev / max(m, 1)) ** 2, 0.0)) * 0.98
                depth = m * hm + 0.98 * hc
                zb = ze[b[0]]
                za_, zb_ = (zb, zb + depth) if down else (zb - depth, zb)
                k0 = int(np.clip(np.floor((za_ - z0) / grid.spacing[2]), 0, len(hmin) - 1))
                k1 = int(np.clip(np.ceil((zb_ - z0) / grid.spacing[2]), k0 + 1, len(hmin)))
                hz = float(hmin[k0:k1].min())
                if abs(hz - hc) < 1e-6 * hc:
                    break
                hc = hz
            zones.append({"z": ze[b[0]], "down": down, "P": P, "mult": mult, "nf": fine[2], "nc": coarse[2],
                          "m": m, "hm": hm, "ht": 0.98 * hc, "depth": m * hm + 0.98 * hc, "band": coarse})
        for zn in zones:                                              # each coarse band must hold its zones
            cb = zn["band"]
            used = sum(z2["depth"] for z2 in zones if z2["band"] is cb)
            if used > 0.9 * (ze[cb[1]] - ze[cb[0]]):
                thin = zn
                break
        # a zone must not contain a sharp interface (its lines replace those of the zone): move the band
        # boundary to the interface, the finer band grows up to it
        moved = False
        if honor_interfaces and thin is None:
            for i, zn in enumerate(zones):
                a, b = (zn["z"], zn["z"] + zn["depth"]) if zn["down"] else (zn["z"] - zn["depth"], zn["z"])
                inside = [zi for zi in interfaces if a + 1e-6 < zi < b - 1e-6]
                if inside:
                    zi = min(inside) if zn["down"] else max(inside)
                    k = int(round((zi - z0) / grid.spacing[2]))
                    bands[i][1], bands[i + 1][0] = k, k
                    bands = _merge([bd for bd in bands if bd[1] > bd[0]])
                    moved = True
                    break
        if thin is None and not moved:
            break
        if thin is not None:
            thin["band"][2] = thin["nf"]                              # cancel this coarsening
            bands = _merge(bands)
    else:
        raise RuntimeError("bands : plan des bandes non convergé")
    # depth lines: regular parts segmented like tensor, then the zone lines
    cuts = {ze[b[0]] for b in bands[1:]}
    cuts |= {zn["z"] + (zn["depth"] if zn["down"] else -zn["depth"]) for zn in zones}
    cuts |= set(interfaces)
    # grading between rows is not needed here (rectangles stay rectangles): a thin layer kept as a thin
    # row must not shrink the rows around it (cascade of the grading limit)
    segs = segment_axis_with_breaks(grid.edge_axis(2), hmin, max_segments, sorted(cuts), max_grading,
                                    min_elems_per_segment=1)
    zl = list(_nodes(segs))
    for zn in zones:                                                  # replace the lines inside each zone
        a, b = (zn["z"], zn["z"] + zn["depth"]) if zn["down"] else (zn["z"] - zn["depth"], zn["z"])
        zl = [v for v in zl if not (a + 1e-6 < v < b - 1e-6)]
        acc = zn["z"]
        for _ in range(zn["m"]):
            acc = acc + zn["hm"] if zn["down"] else acc - zn["hm"]
            zl.append(acc)
    zl = np.array(sorted(set(np.round(zl, 6))))
    U = lambda n: np.linspace(x0, x1, n + 1)
    lines = [U(max(b[2] for b in bands if ze[b[0]] - 1e-6 <= z <= ze[b[1]] + 1e-6)) for z in zl]
    template = {}
    for zn in zones:
        k0 = int(np.argmin(np.abs(zl - zn["z"])))
        step = 1 if zn["down"] else -1
        uf = U(zn["nf"])
        for r in range(1, zn["m"] + 1):
            lines[k0 + step * r] = uf + (zn["P"] - uf) * r / zn["m"]
        kt = k0 + step * zn["m"]                                      # line holding the template's fine nodes
        lines[kt + step] = U(zn["nc"])
        row = kt if step == 1 else kt - 1
        template[row] = (zn["P"] if zn["m"] > 0 else uf, zn["mult"], U(zn["nc"]), "top" if step == 1 else "bottom")
    quads = []
    in_tree = set()
    n_coarse = 0
    if local:
        from .quadtree_2d import Quadtree2D
        from .plane import _box_min
        regular = [r for r in range(len(zl) - 1) if r not in template and len(lines[r]) == len(lines[r + 1])
                   and np.allclose(lines[r], U(len(lines[r]) - 1)) and np.allclose(lines[r + 1], lines[r])]
        runs, cur = [], []
        for r in regular:                     # runs of consecutive regular rows of equal height and columns
            on_interface = any(abs(zl[r] - zi) < 1e-6 * L for zi in interfaces)   # never across an interface
            if cur and r == cur[-1] + 1 and len(lines[r]) == len(lines[cur[-1]]) and not on_interface and \
                    abs((zl[r + 1] - zl[r]) - (zl[cur[-1] + 1] - zl[cur[-1]])) < 1e-6 * L:
                cur.append(r)
            else:
                if cur:
                    runs.append(cur)
                cur = [r]
        if cur:
            runs.append(cur)
        for run in runs:
            nb = len(run) // 4
            ncol = len(lines[run[0]]) - 1
            if nb < 1 or ncol % 4:
                continue
            rows = run[:4 * nb]
            zs = np.array([zl[rows[0] + 4 * j] for j in range(nb + 1)])
            hb = zs[1] - zs[0]
            # one row of phantom blocks above and below, fully refined: the blocks of the run next to them
            # get transition templates with fine edges on that side (conforming with the fine rows around)
            T = Quadtree2D(np.linspace(x0, x1, ncol // 4 + 1), np.concatenate([[zs[0] - hb], zs, [zs[-1] + hb]]), 1)
            top, bot, tol = zs[0], zs[-1], 1e-6 * L
            T.refine_where(lambda lo, hi, lev: bool(hi[1] <= top + tol or lo[1] >= bot - tol
                                                    or (0.5 * (hi - lo)).max() > _box_min(grid, h2, lo, hi) * (1 + 1e-9)))
            T.balance()
            tn, tq, tl = T.to_quads()
            zc = tn[tq][:, :, 1].mean(1)
            keep = (zc > top + tol) & (zc < bot - tol)                    # drop the phantom blocks
            quads += [[tuple(tn[k]) for k in q] for q in tq[keep]]
            n_coarse += int(np.sum((tl == 0) & keep))
            in_tree.update(rows)
    for r in range(len(zl) - 1):
        if r in in_tree:
            continue
        za, zb = zl[r], zl[r + 1]
        if r in template:
            P, mult, xc, side = template[r]
            quads += _template_row(P, mult, xc, za, zb) if side == "top" else _template_row(P, mult, xc, zb, za)
            continue
        top, bot = lines[r], lines[r + 1]
        if len(top) != len(bot):
            raise RuntimeError(f"bands : rangée à {za:.0f} m sans gabarit ({len(top) - 1} / {len(bot) - 1} colonnes)")
        quads += [[(top[i], za), (top[i + 1], za), (bot[i + 1], zb), (bot[i], zb)] for i in range(len(top) - 1)]
    X = np.array(quads, float)
    key = np.round(X.reshape(-1, 2) / (1e-6 * L), 0)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    first = np.zeros(len(uniq), int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    nodes = X.reshape(-1, 2)[first]
    q = inv.reshape(-1, 4)
    Y = nodes[q]
    area = np.sum(Y[:, :, 0] * np.roll(Y[:, :, 1], -1, 1) - np.roll(Y[:, :, 0], -1, 1) * Y[:, :, 1], 1)
    q[area < 0] = q[area < 0][:, ::-1]
    info = {"bands": [(float(ze[b[0]]), float(ze[b[1]]), int(b[2])) for b in bands],
            "transitions": [{"z": float(zn["z"]), "columns": (int(zn["nf"]), int(zn["nc"])), "morph_rows": int(zn["m"])}
                            for zn in zones],
            "coarsen": coarsen, "dwell": dwell, "max_segments": max_segments, "local": local,
            "coarse_elements": n_coarse}
    return QuadMesh(nodes, q, "bands", info)
