"""2-D : zones de taille requise maillées en grilles régulières, raccordées localement (stratégie ``zones``).

La taille des éléments est fixée par l'ordre du schéma et la fréquence de la source (taille requise h).
Méthode (automatisation, à partir d'un modèle SEP, du maillage bicouche fait à la main pour la
validation du pas de temps local de FUnTiDES) :

1. *zones* : classes de h en progression géométrique (rapport ``ratio``), composantes connexes ; les
   interfaces horizontales nettes séparent les zones ; une zone trop petite pour porter un bloc régulier
   de ``min_block`` x ``min_block`` de ses éléments est fusionnée dans sa voisine la plus fine (jamais
   de sous-échantillonnage ; un petit corps lent ne raffine pas un grand corps rapide voisin) ;
2. *blocs réguliers* : chaque zone porte une grille régulière à sa propre taille h_z (minimum de h dans
   la zone), calée au mieux, réduite de ``gap`` éléments depuis la frontière de la zone ; aucun lien
   de palier entre zones (pas de rapports x 2 imposés) ;
3. *raccords* : le reste du domaine (bandes étroites le long des frontières de zones) est maillé par
   gmsh en triangles, les nœuds du bord des blocs imposés, les interfaces horizontales nettes imposées
   comme lignes ; tout est maillé à deux fois la taille puis subdivisé en quadrangles (bloc : 4
   carrés ; triangle : 3 quadrangles), comme ``Mesh.SubdivisionAlgorithm = 1`` : conforme, sans
   triangle restant ; les nœuds intérieurs des raccords sont optimisés (:mod:`.smoothing`) ;
4. *contrôle* : un élément de raccord plus grand que sa taille requise (géométrie réelle,
   :mod:`.elements2d`) ou trop déformé réduit localement la taille demandée, et seuls les raccords
   sont remaillés.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------------------------------- zones
def zone_labels(h2, ratio, min_cells, interface_rows=()):
    """Zones : (labels (nx, nz), taille requise h_z de chaque zone). ``interface_rows`` : indices k tels
    qu'une interface nette sépare les échantillons k - 1 et k (les zones ne la traversent pas)."""
    lo = float(h2.min())
    ncls = int(np.ceil(np.log(h2.max() / lo) / np.log(ratio) - 1e-9)) + 1
    cls = np.clip(np.floor(np.log(h2 / lo) / np.log(ratio) + 1e-9).astype(int), 0, ncls - 1)
    cuts = [0] + sorted(k for k in interface_rows if 0 < k < h2.shape[1]) + [h2.shape[1]]
    slab = np.zeros(h2.shape, int)
    for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        slab[:, a:b] = j
    lab = np.full(h2.shape, -1)
    n = 0
    for c in np.unique(cls):                                        # components per class and per slab
        for j in range(len(cuts) - 1):
            comp, k = ndimage.label((cls == c) & (slab == j))
            lab[comp > 0] = comp[comp > 0] - 1 + n
            n += k
    zcls = np.array([cls[lab == r][0] for r in range(n)])
    # small zones join their finest neighbour of the same slab (never across an interface)
    for _ in range(10 * n):
        sizes = np.bincount(lab.ravel(), minlength=n)
        small = [r for r in np.argsort(sizes) if 0 < sizes[r] < min_cells]
        merged = False
        for r in small:
            m = lab == r
            sl = ndimage.find_objects(m.astype(int))[0]
            sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
            mm = m[sl]
            ring = ndimage.binary_dilation(mm) & ~mm & (slab[sl] == slab[sl][mm][0])
            nb = np.unique(lab[sl][ring])
            nb = nb[nb != r]
            if nb.size == 0:
                continue
            target = nb[np.argmin(zcls[nb])]
            lab[m] = target
            merged = True
            break
        if not merged:
            break
    _, lab = np.unique(lab, return_inverse=True)
    lab = lab.reshape(h2.shape)
    hz = ndimage.minimum(h2, lab, index=np.arange(lab.max() + 1))
    return lab, np.asarray(hz, float)


# ------------------------------------------------------------------------------- regular blocks
def _orient_ccw(nodes, quads):
    X = nodes[quads]
    a = 0.5 * np.sum(X[:, :, 0] * np.roll(X[:, :, 1], -1, 1) - np.roll(X[:, :, 0], -1, 1) * X[:, :, 1], 1)
    q = quads.copy()
    q[a < 0] = q[a < 0][:, ::-1]
    return q


def _boundary_loops(nodes, quads):
    """Contours orientés (union à gauche) d'un ensemble de quadrangles orientés : listes de nœuds."""
    edges = set()
    for q in quads:
        for k in range(4):
            edges.add((int(q[k]), int(q[(k + 1) % 4])))
    nxt = {}
    for a, b in edges:
        if (b, a) not in edges:
            nxt.setdefault(a, []).append(b)
    loops, used = [], set()
    for a0, ends in list(nxt.items()):
        for b0 in ends:
            if (a0, b0) in used:
                continue
            loop, a, b = [a0], a0, b0
            while True:
                used.add((a, b))
                if b == a0:
                    break
                loop.append(b)
                cand = [c for c in nxt[b] if (b, c) not in used]
                if len(cand) > 1:                                   # pinch: sharpest left turn
                    d = nodes[b] - nodes[a]
                    ang = [np.arctan2(d[0] * (nodes[c] - nodes[b])[1] - d[1] * (nodes[c] - nodes[b])[0],
                                      np.dot(d, nodes[c] - nodes[b])) for c in cand]
                    cand = [cand[int(np.argmax(ang))]]
                a, b = b, cand[0]
            loops.append(loop)
    return loops


def _regular_blocks(grid, lab, hz, gap):
    """Blocs réguliers de chaque zone, en cellules de 2 h_z (moitié après subdivision), à ``gap``
    éléments h_z au moins de la frontière de la zone : (nœuds (n, 2), quadrangles, zone)."""
    (x0, x1), _, (z0, z1) = grid.extent
    dx, dz = grid.spacing[0], grid.spacing[2]
    nodes_all, quads_all, zone_all, off = [], [], [], 0
    for r in range(len(hz)):
        mask = lab == r
        c = 2.0 * hz[r]
        k = int(np.ceil(gap * hz[r] / min(dx, dz)))
        inner = ndimage.binary_erosion(mask, iterations=k, border_value=1)
        if not inner.any():
            continue
        bad = np.pad((~inner).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        best = None
        for fx in (0.0, 0.25, 0.5, 0.75):                         # grid shift giving the most cells
            for fz in (0.0, 0.25, 0.5, 0.75):
                xs = x0 + c * (fx + np.arange(int((x1 - x0 - fx * c) // c) + 1))
                zs = z0 + c * (fz + np.arange(int((z1 - z0 - fz * c) // c) + 1))
                if len(xs) < 2 or len(zs) < 2:
                    continue
                i0 = np.clip(np.floor((xs[:-1] - x0) / dx + 1e-9).astype(int), 0, mask.shape[0] - 1)
                i1 = np.clip(np.ceil((xs[1:] - x0) / dx - 1e-9).astype(int) - 1, 0, mask.shape[0] - 1)
                k0 = np.clip(np.floor((zs[:-1] - z0) / dz + 1e-9).astype(int), 0, mask.shape[1] - 1)
                k1 = np.clip(np.ceil((zs[1:] - z0) / dz - 1e-9).astype(int) - 1, 0, mask.shape[1] - 1)
                I0, K0 = np.meshgrid(i0, k0, indexing="ij")
                I1, K1 = np.meshgrid(i1, k1, indexing="ij")
                keep = (bad[I1 + 1, K1 + 1] - bad[I0, K1 + 1] - bad[I1 + 1, K0] + bad[I0, K0]) == 0
                if best is None or keep.sum() > best[0]:
                    best = (keep.sum(), keep, xs, zs)
        if best is None or best[0] == 0:
            continue
        _, keep, xs, zs = best
        ii, kk = np.nonzero(keep)
        ids = {}

        def nid(i, j):
            if (i, j) not in ids:
                ids[(i, j)] = len(ids)
            return ids[(i, j)]
        q = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)] for i, j in zip(ii, kk)])
        P = np.array([[xs[i], zs[j]] for (i, j) in sorted(ids, key=ids.get)])
        nodes_all.append(P)
        quads_all.append(q + off)
        zone_all.append(np.full(len(q), r))
        off += len(P)
    if not nodes_all:
        return np.zeros((0, 2)), np.zeros((0, 4), int), np.zeros(0, int)
    return np.vstack(nodes_all), np.vstack(quads_all), np.concatenate(zone_all)


# ----------------------------------------------------------------------------------- transitions
def _mesh_transitions(grid, rn, loops, hs, interfaces):
    """Domaine moins les blocs : triangles gmsh (taille ``hs`` x 2, grille SEP), nœuds du bord des blocs
    imposés (un segment par arête de bloc), interfaces horizontales imposées. Renvoie (nœuds, triangles)."""
    import gmsh
    (x0, x1), _, (z0, z1) = grid.extent
    dx, dz = grid.spacing[0], grid.spacing[2]
    span = max(x1 - x0, z1 - z0)

    def hloc(x, z):
        return 2.0 * float(hs[int(np.clip((x - x0) // dx, 0, hs.shape[0] - 1)), int(np.clip((z - z0) // dz, 0, hs.shape[1] - 1))])

    gmsh.initialize(["gmsh", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("zones")
        occ = gmsh.model.occ
        rect = occ.addRectangle(x0, z0, 0, x1 - x0, z1 - z0)
        holes = []
        for lp in loops:
            pts = [occ.addPoint(rn[n, 0], rn[n, 1], 0) for n in lp]
            ls = [occ.addLine(a, b) for a, b in zip(pts, pts[1:] + pts[:1])]
            holes.append((2, occ.addPlaneSurface([occ.addCurveLoop(ls)])))
        zone = occ.cut([(2, rect)], holes, removeObject=True, removeTool=True)[0] if holes else [(2, rect)]
        lines = [(1, occ.addLine(occ.addPoint(x0, zi, 0), occ.addPoint(x1, zi, 0))) for zi in interfaces]
        if lines:
            out, _ = occ.fragment(zone, lines)
            zone = [e for e in out if e[0] == 2]
        occ.synchronize()
        surfaces = [t for d, t in zone]
        # the part of an interface line lying inside a block is outside the transition zone: removed
        for d, t in gmsh.model.getEntities(1):
            if not gmsh.model.getAdjacencies(1, t)[0].size:
                gmsh.model.removeEntities([(1, t)], recursive=False)
        tol = 1e-9 * span
        for s in surfaces:
            for d, c in gmsh.model.getBoundary([(2, s)], oriented=False):
                xa, ya, _, xb, yb, _ = gmsh.model.getBoundingBox(1, abs(c))
                on_box = any(abs(a - v) < tol and abs(b - v) < tol for a, b, v in
                             ((xa, xb, x0), (xa, xb, x1), (ya, yb, z0), (ya, yb, z1)))
                on_interface = abs(ya - yb) < tol and any(abs(ya - zi) < tol for zi in interfaces)
                if not on_box and not on_interface:
                    gmsh.model.mesh.setTransfiniteCurve(abs(c), 2)    # block edge: its two block nodes
        for key, value in {"Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeFromCurvature": 0,
                           "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.Algorithm": 6, "Mesh.Smoothing": 10}.items():
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: hloc(x, y))
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        types, _, conn = gmsh.model.mesh.getElements(2)
        remap = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        remap[tags.astype(np.int64)] = np.arange(len(tags))
        cells = []
        for t, c in zip(types, conn):
            k = 3 if t == 2 else 4
            cells += [list(row) for row in remap[c.astype(np.int64)].reshape(-1, k)]
        return coords.reshape(-1, 3)[:, :2], cells
    finally:
        gmsh.finalize()


def subdivide(nodes, cells):
    """Subdivision en quadrangles : quadrangle -> 4, triangle -> 3 (centre et milieux des arêtes,
    partagés : conforme). Renvoie (nœuds, quadrangles, cellule d'origine de chaque quadrangle)."""
    pts = [tuple(p) for p in np.asarray(nodes, float)]
    mid = {}

    def m(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in mid:
            mid[key] = len(pts)
            pts.append(tuple(0.5 * (np.array(pts[a]) + np.array(pts[b]))))
        return mid[key]
    quads, origin = [], []
    for i, c in enumerate(cells):
        n = len(c)
        g = len(pts)
        pts.append(tuple(np.mean([pts[v] for v in c], axis=0)))
        for k in range(n):
            quads.append([c[k], m(c[k], c[(k + 1) % n]), g, m(c[k - 1], c[k])])
            origin.append(i)
    return np.array(pts), np.array(quads), np.array(origin)


# ------------------------------------------------------------------------ relaxation of the transitions
def relax_transitions(nodes, quads, free, req, weight=200.0, margin=0.95, maxiter=2000):
    """Déplace les nœuds ``free`` des raccords : qualité (conditionnement des jacobiens, Knupp) plus une
    pénalité sur toute arête plus longue que ``margin`` fois la taille requise ``req`` de son élément
    (les raccords se déforment pour tenir la taille, sans ajouter d'éléments). Renvoie (nœuds, valide)."""
    from scipy.optimize import minimize
    from .smoothing import _objective as knupp
    nodes = np.asarray(nodes, float)
    act = free[quads].any(1)
    Q, R = quads[act], np.asarray(req, float)[act]
    used = np.unique(Q)
    loc = np.full(len(nodes), -1)
    loc[used] = np.arange(len(used))
    sub, Qs = nodes[used], loc[Q]
    fidx = np.where(free[used])[0]
    flat = np.stack([2 * fidx, 2 * fidx + 1], 1).ravel()
    delta = 1e-3 * float(np.median(R)) ** 2

    def f(xf):
        k, gk = knupp(xf, sub, Qs, fidx, 2.0, delta)
        x = sub.copy()
        x[fidx] = xf.reshape(-1, 2)
        g = np.zeros_like(x)
        pen = 0.0
        for a in range(4):
            ia, ib = Qs[:, a], Qs[:, (a + 1) % 4]
            d = x[ib] - x[ia]
            L = np.linalg.norm(d, axis=1)
            t = L / (margin * R) - 1.0
            on = t > 0
            pen += weight * np.sum(t[on] ** 2)
            gd = (weight * 2 * t[on] / (margin * R[on] * np.maximum(L[on], 1e-300)))[:, None] * d[on]
            np.add.at(g, ib[on], gd)
            np.add.at(g, ia[on], -gd)
        return k + pen, gk + g.ravel()[flat]

    if not len(fidx):
        return nodes, True
    res = minimize(f, sub[fidx].ravel(), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    x = sub.copy()
    x[fidx] = res.x.reshape(-1, 2)
    Y = x[Qs]
    E1, E2 = np.roll(Y, -1, 1) - Y, np.roll(Y, 1, 1) - Y
    ok = bool(np.all(E1[..., 0] * E2[..., 1] - E1[..., 1] * E2[..., 0] > 0))
    out = nodes.copy()
    if ok:
        out[used] = x
    return out, ok


# -------------------------------------------------------------------------------------- strategy
def mesh_zones_2d(grid, h, ratio=1.25, gap=1.0, size_factor=0.9, min_block=3, honor_interfaces=True,
                  optimize=True, max_rounds=3):
    """Maillage ``zones`` (voir le module). Retourne un ``QuadMesh``."""
    from .plane import QuadMesh
    from .sizing import horizontal_interfaces, PPW_TOLERANCE
    from .elements2d import element_stats_2d, axis_aligned
    h2 = h[:, 0, :].astype(float)
    (x0, x1), _, (z0, z1) = grid.extent
    dx, dz = grid.spacing[0], grid.spacing[2]
    span = max(x1 - x0, z1 - z0)
    interfaces = horizontal_interfaces(grid) if honor_interfaces else []
    rows = [int(round((zi - z0) / dz)) for zi in interfaces]
    lab, hz = zone_labels(h2, ratio, max(4, int((min_block * h2.min() / min(dx, dz)) ** 2)), rows)
    rn, rq, rzone = _regular_blocks(grid, lab, hz, gap)
    rq = _orient_ccw(rn, rq) if len(rq) else rq
    loops = []
    for r in np.unique(rzone):
        loops += _boundary_loops(rn, rq[rzone == r])
    # size of the transitions: required size, minimum over half an element around each sample
    reach = int(np.ceil(0.5 * float(np.median(h2)) / min(dx, dz))) * 2 + 1
    hs = size_factor * ndimage.minimum_filter(h2, size=reach, mode="nearest")
    info = {"ratio": ratio, "gap": gap, "size_factor": size_factor, "n_zones": int(len(hz)),
            "zone_sizes": sorted({round(float(v), 1) for v in hz}), "zones_with_block": int(len(np.unique(rzone)))}
    for rnd in range(max_rounds):
        gn, gcells = _mesh_transitions(grid, rn, loops, hs, interfaces)
        allnodes = np.vstack([rn, gn])
        cells = [list(q) for q in rq] + [[v + len(rn) for v in c] for c in gcells]
        key = np.round(allnodes / (1e-6 * span))
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        cells = [[int(inv[v]) for v in c] for c in cells]
        nodes, quads, origin = subdivide(allnodes[first], cells)
        quads = _orient_ccw(nodes, quads)
        regular = origin < len(rq)                                 # quadrangles of the blocks
        req = element_stats_2d(grid, nodes, quads, values=h)["min"]
        if optimize:
            fixed = np.zeros(len(nodes), bool)
            fixed[quads[regular].ravel()] = True
            fixed |= (np.isclose(nodes[:, 0], x0, atol=1e-9 * span) | np.isclose(nodes[:, 0], x1, atol=1e-9 * span)
                      | np.isclose(nodes[:, 1], z0, atol=1e-9 * span) | np.isclose(nodes[:, 1], z1, atol=1e-9 * span))
            for zi in interfaces:
                fixed |= np.isclose(nodes[:, 1], zi, atol=1e-9 * span)
            nodes, _ = relax_transitions(nodes, quads, ~fixed, req)
            req = element_stats_2d(grid, nodes, quads, values=h)["min"]
        X = nodes[quads]
        size = np.linalg.norm(np.roll(X, -1, 1) - X, axis=2).max(1)
        E1, E2 = np.roll(X, -1, 1) - X, np.roll(X, 1, 1) - X
        cr = E1[..., 0] * E2[..., 1] - E1[..., 1] * E2[..., 0]
        sj = (cr / np.maximum(np.linalg.norm(E1, axis=2) * np.linalg.norm(E2, axis=2), 1e-300)).min(1)
        bad = ((size * (1.0 - PPW_TOLERANCE) > req) | (sj < 0.2)) & ~regular     # same rule as the evaluation
        info.update({"rounds": rnd + 1, "n_regular": int(regular.sum()), "n_transition": int((~regular).sum()),
                     "n_bad": int(bad.sum())})
        if not bad.any():
            break
        for c, sz in zip(X[bad].mean(1), size[bad]):              # shrink the requested size around them
            i0, i1 = int((c[0] - sz - x0) // dx), int((c[0] + sz - x0) // dx) + 1
            k0, k1 = int((c[1] - sz - z0) // dz), int((c[1] + sz - z0) // dz) + 1
            hs[max(i0, 0):i1, max(k0, 0):k1] *= 0.9
    info["frac_regular"] = float(axis_aligned(nodes[quads]).mean())
    m = QuadMesh(nodes, quads, "zones", info)
    m.zone_labels = lab
    return m
