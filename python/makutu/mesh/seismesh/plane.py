r"""Maillages 2-D (coupe (x, z), z profondeur vers le bas) en quadrangles pour la SEM 2-D.

Grille 2-D = :class:`VelocityGrid` de forme (nx, 1, nz). Deux stratégies :

- ``tensor`` : grille à pas constant par morceaux (segments optimaux en x et en z) ;
- ``doubling`` : le pas horizontal double en profondeur par couches de transition conformes
  (gabarit « 4 fines -> 2 grossières » de :mod:`doubling`, 6 quadrangles convexes, sans nœud
  pendant), et peut re-raffiner sous une couche lente (gabarit retourné).

Les quadrangles sont rendus comme :class:`QuadMesh` (nœuds (x, z) en profondeur) et évalués par
:func:`evaluate_2d` (jacobien, points par longueur d'onde, pas de temps CFL, coût).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .doubling import TEMPLATE, _cell_req, _nodes, _pad
from .quality import gll_min_fraction
from .sizing import SizingParams, h_required, horizontal_interfaces, segment_axis, segment_axis_with_breaks
from .velocity import VelocityGrid


@dataclass
class QuadMesh:
    nodes: np.ndarray                 # (nn, 2) : x, profondeur z
    quads: np.ndarray                 # (ne, 4)
    strategy: str
    params: dict = field(default_factory=dict)


def segments_2d(grid: VelocityGrid, h: np.ndarray, max_segments=(6, 8), max_grading=2.0,
                min_elems_per_segment=2, honor_interfaces=True):
    """Segmentation optimale des axes x et z (min de h_req sur l'autre axe) ; avec
    ``honor_interfaces``, des faces d'éléments passent par les interfaces horizontales nettes."""
    h2 = h[:, 0, :]
    breaks = horizontal_interfaces(grid) if honor_interfaces else ()
    return (segment_axis(grid.edge_axis(0), h2.min(axis=1), max_segments[0], min_elems_per_segment, max_grading),
            segment_axis_with_breaks(grid.edge_axis(2), h2.min(axis=0), max_segments[1], breaks, max_grading,
                                     min_elems_per_segment=min_elems_per_segment))


def mesh_tensor_2d(grid, h, segs) -> QuadMesh:
    xs, zs = _nodes(segs[0]), _nodes(segs[1])
    nx, nz = len(xs), len(zs)
    k, i = np.mgrid[0:nz - 1, 0:nx - 1]
    n0 = (i + k * nx).ravel()
    quads = np.stack([n0, n0 + 1, n0 + 1 + nx, n0 + nx], axis=1)
    X, Z = np.meshgrid(xs, zs)
    return QuadMesh(np.stack([X.ravel(), Z.ravel()], -1), quads, "tensor",
                    {"segments": [[(s.t0, s.t1, s.n) for s in a] for a in segs]})


def plan_levels_2d(grid, h, segs, max_levels=3, allow_refine=True, min_dwell=2, blocked=frozenset()):
    """(xs, zs, plan, allowed) ; plan = [(kind, k, level)] avec kind 'reg', 'tx' (c -> c+1
    vers le bas) ou 'rx' (c -> c-1, gabarit retourné). Une transition occupe une couche z."""
    mult = 2 ** (max_levels + 1)
    xs = _nodes(_pad(segs[0], mult))
    zs = _nodes(segs[1])
    nz = len(zs) - 1
    hzx = h[:, 0, :]
    zg = grid.axis(2)
    ox, dx, dz = grid.origin[0], grid.spacing[0], grid.spacing[2]
    allowed = np.zeros(nz, dtype=int)
    for k in range(nz):
        iz = np.where((zg >= zs[k] - 1e-6) & (zg <= zs[k + 1] + 1e-6))[0]
        if iz.size == 0:
            iz = np.array([int(np.clip(round((0.5 * (zs[k] + zs[k + 1]) - zg[0]) / dz), 0, len(zg) - 1))])
        px = hzx[:, iz].min(axis=1)
        for level in range(1, max_levels + 1):
            xl = xs[::2 ** level]
            if not np.all(np.diff(xl) <= _cell_req(px, ox, dx, xl) * (1 + 1e-9)):
                break
            allowed[k] = level
    if allow_refine:
        U = np.array([min(allowed[j] + (j - k) for j in range(k, nz)) for k in range(nz)])
    else:
        U = np.minimum.accumulate(allowed[::-1])[::-1]
    U = np.append(np.minimum(U, max_levels), max_levels)
    plan, c = [], 0
    for k in range(nz):
        if c > 0 and c > U[k + 1] and k + 1 < nz:
            plan.append(("rx", k, c))
            c -= 1
        elif c < max_levels and k + 1 < nz and U[k:min(k + 1 + min_dwell, nz) + 1].min() >= c + 1 \
                and not ({k - 1, k, k + 1} & blocked):
            plan.append(("tx", k, c))
            c += 1
        else:
            plan.append(("reg", k, c))
    return xs, zs, plan, allowed


def _doubling_quads(xs, zs, plan, skip=frozenset()):
    """Nœuds (x, z) et quadrangles d'un plan de doublement ; les couches de ``skip`` sont omises."""
    NX = len(xs) - 1
    keys = []                                     # (ne, 4, 2) : (ix fin, zcode) ; zcode 2k plan, 2k+1 milieu
    for kind, k, c in plan:
        s = 2 ** c
        if kind == "reg" and k in skip:
            continue
        if kind == "reg":
            I = np.arange(0, NX, s)
            corners = [(I, 2 * k), (I + s, 2 * k), (I + s, 2 * k + 2), (I, 2 * k + 2)]
            keys.append(np.stack([np.stack([a, np.full_like(a, z)], -1) for a, z in corners], 1))
            continue
        flip = kind == "rx"
        fine = s // 2 if flip else s                  # pas du côté fin du gabarit
        base = np.arange(0, NX, 4 * fine)
        quads = []
        for q in TEMPLATE:
            pts = [np.stack([base + u * fine, np.full_like(base, 2 * k + (2 - z if flip else z))], -1) for u, z in q]
            quads.append(np.stack(pts, 1))
        keys.append(np.concatenate(quads))
    if not keys:                                  # every layer skipped (refined zones only)
        return np.zeros((0, 2)), np.zeros((0, 4), dtype=np.int64)
    K = np.concatenate(keys)
    code = K[..., 1].astype(np.int64) * (NX + 1) + K[..., 0]
    uniq, inv = np.unique(code.ravel(), return_inverse=True)
    ix, zc = uniq % (NX + 1), uniq // (NX + 1)
    zmid = 0.5 * (zs[:-1] + zs[1:])
    z = np.where(zc % 2 == 0, zs[np.minimum(zc // 2, len(zs) - 1)], zmid[np.minimum(zc // 2, len(zmid) - 1)])
    return np.stack([xs[ix], z], -1), inv.reshape(-1, 4)


def mesh_doubling_2d(grid, h, segs, max_levels=3, allow_refine=True) -> QuadMesh:
    xs, zs, plan, allowed = plan_levels_2d(grid, h, segs, max_levels, allow_refine)
    nodes, quads = _doubling_quads(xs, zs, plan)
    info = {"plan": [(kind, round(float(zs[k]), 1), c) for kind, k, c in plan if kind != "reg"],
            "nx_fine": len(xs) - 1, "nz_layers": len(zs) - 1, "allowed_level_per_layer": allowed.tolist()}
    return QuadMesh(nodes, quads, "doubling", info)


def _need_2d(grid, h2):
    """need(lo, hi, level) : la cellule (boîte (x, z)) dépasse-t-elle la taille requise ?"""
    o = np.array([grid.origin[0], grid.origin[2]])
    d = np.array([grid.spacing[0], grid.spacing[2]])
    n = np.array(h2.shape)

    def need(lo, hi, level=0):
        a = np.clip(np.ceil((lo - o) / d - 1e-6).astype(int), 0, n - 1)
        b = np.maximum(a, np.clip(np.floor((hi - o) / d + 1e-6).astype(int), 0, n - 1))
        return bool((hi - lo).max() > h2[a[0]:b[0] + 1, a[1]:b[1] + 1].min() * (1 + 1e-9))
    return need


def mesh_hybrid_2d(grid, h, segs=None, max_levels=2, q_bg=0.1, lmax3=1, max_segments=(4, 8),
                   max_replan=30) -> QuadMesh:
    """Doublements 2:1 en profondeur sur un fond (quantile latéral ``q_bg`` de h_req à chaque
    profondeur) et 3-raffinement local conforme des anomalies lentes dans les zones de niveau
    constant ; une transition ne doit pas toucher une zone raffinée (sinon couches bloquées et
    nouveau plan)."""
    from .refine3_2d import Refine3Mesh2D
    h2 = h[:, 0, :]
    h_bg = np.ascontiguousarray(np.broadcast_to(np.quantile(h2, q_bg, axis=0)[None, None, :], h.shape))
    segs = segs or segments_2d(grid, h_bg, max_segments)
    need = _need_2d(grid, h2)
    blocked: set[int] = set()
    for _ in range(max_replan):
        xs, zs, plan, allowed = plan_levels_2d(grid, h_bg, segs, max_levels, True, blocked=frozenset(blocked))
        nz = len(zs) - 1
        new_block = set()
        for kind, k, c in plan:                      # anomaly inside a transition layer
            if kind != "reg":
                xl = xs[::2 ** (c + 1 if kind == "tx" else c)]
                if any(need(np.array([a, zs[k]]), np.array([b, zs[k + 1]])) for a, b in zip(xl[:-1], xl[1:])):
                    new_block.add(k)
        zones, cur = [], None
        for kind, k, c in plan:
            if kind == "reg" and cur and cur[0] == c and cur[2] == k:
                cur[2] = k + 1
            elif kind == "reg":
                cur = [c, k, k + 1]
                zones.append(cur)
            else:
                cur = None
        refined = []
        for c, k0, k1 in zones:
            R = Refine3Mesh2D(xs[::2 ** c], zs[k0:k1 + 1], lmax3)
            if R.refine_where(need) == 0:
                continue
            R.balance()
            if k0 > 0 and R.touches("top"):
                new_block |= {k0 - 1, k0}
            if k1 < nz and R.touches("bottom"):
                new_block |= {k1 - 1, k1}
            refined.append((k0, k1, R))
        if not (new_block - blocked):
            break
        blocked |= new_block
    else:
        raise RuntimeError("hybride 2-D : impossible d'isoler les anomalies des transitions")
    skip = {k for k0, k1, _ in refined for k in range(k0, k1)}
    parts = [_doubling_quads(xs, zs, plan, skip)] + [R.to_quads() for _, _, R in refined]
    X = np.concatenate([nodes[q] for nodes, q in parts])               # (ne, 4, 2)
    scale = np.ptp(X.reshape(-1, 2), axis=0).max()
    key = np.round(X.reshape(-1, 2) / (scale * 1e-9)).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    info = {"plan": [(kind, round(float(zs[k]), 1), c) for kind, k, c in plan if kind != "reg"],
            "refined_zones": [(round(float(zs[k0]), 1), round(float(zs[k1]), 1)) for k0, k1, _ in refined],
            "blocked_layers": sorted(blocked), "nx_fine": len(xs) - 1, "nz_layers": nz}
    return QuadMesh(X.reshape(-1, 2)[idx], inv.reshape(-1, 4), "hybrid", info)


def mesh_adaptive_2d(grid, h, size_factor=0.8, honor_interfaces=True) -> QuadMesh:
    """Quadrangles non structurés (gmsh : frontal-Delaunay pour quadrangles + recombinaison blossom full-quad)
    dont la taille suit ``size_factor * h_req(x, z)`` (minimum sur un voisinage de la taille
    locale, pour ne pas sous-échantillonner au bord d'une anomalie lente) ; le rectangle est
    découpé aux interfaces horizontales nettes, qui deviennent des lignes de faces."""
    import gmsh
    from scipy.ndimage import minimum_filter

    h2 = h[:, 0, :]
    (x0, x1), _, (z0, z1) = grid.extent
    dx, dz = grid.spacing[0], grid.spacing[2]
    cuts = [z0] + (horizontal_interfaces(grid) if honor_interfaces else []) + [z1]
    # minimum over half an element around each sample, within each layer (the interfaces are face
    # lines: the size of one layer does not leak into the next one)
    zg = grid.axis(2)
    hs = np.empty_like(h2)
    for a, b in zip(cuts[:-1], cuts[1:]):
        k = np.where((zg >= a) & (zg <= b))[0]
        slab = h2[:, k]
        reach = int(np.ceil(0.5 * np.median(slab) / min(dx, dz))) * 2 + 1
        hs[:, k] = minimum_filter(slab, size=(min(reach, slab.shape[0]), min(reach, slab.shape[1])), mode="nearest")
    hs *= size_factor

    def size(dim, tag, x, y, z, lc):
        i = int(np.clip(round((x - x0) / dx), 0, hs.shape[0] - 1))
        k = int(np.clip(round((y - z0) / dz), 0, hs.shape[1] - 1))
        return float(hs[i, k])

    gmsh.initialize(["gmsh", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("adaptive_2d")
        geo = gmsh.model.geo
        left = [geo.addPoint(x0, z, 0) for z in cuts]
        right = [geo.addPoint(x1, z, 0) for z in cuts]
        horiz = [geo.addLine(a, b) for a, b in zip(left, right)]
        surfaces = []
        for k in range(len(cuts) - 1):
            ll = geo.addLine(left[k + 1], left[k])
            rr = geo.addLine(right[k], right[k + 1])
            surfaces.append(geo.addPlaneSurface([geo.addCurveLoop([horiz[k], rr, -horiz[k + 1], ll])]))
        geo.synchronize()
        for sf in surfaces:
            gmsh.model.mesh.setRecombine(2, sf)
        for key, value in {"Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeFromCurvature": 0,
                           "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.Algorithm": 8,
                           "Mesh.RecombinationAlgorithm": 3, "Mesh.Smoothing": 5}.items():   # blossom full-quad
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.setSizeCallback(size)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        types, _, conn = gmsh.model.mesh.getElements(2)
        if list(types) != [3]:
            raise RuntimeError(f"adaptive_2d : éléments non quadrangulaires {list(types)}")
        remap = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        remap[tags.astype(np.int64)] = np.arange(len(tags))
        nodes = coords.reshape(-1, 3)[:, :2]
        quads = remap[conn[0].astype(np.int64)].reshape(-1, 4)
    finally:
        gmsh.finalize()
    return QuadMesh(nodes, quads, "adaptive", {"size_factor": size_factor, "interfaces": cuts[1:-1]})


def _offset_edges(a, b, block, offset):
    """Bords de blocs de taille ``block`` sur [a, b], décalés de ``offset`` (fraction de bloc) : un
    bloc partiel au début ; un bloc final de moins d'un demi-bloc est fusionné avec le précédent."""
    e = [a] + list(np.arange(a + offset * block, b - 1e-9, block)[1 if offset == 0 else 0:]) + [b]
    e = sorted(set(round(v, 9) for v in e))
    if len(e) > 2 and e[1] - e[0] < 0.5 * block:
        e.pop(1)
    if len(e) > 2 and e[-1] - e[-2] < 0.5 * block:
        e.pop(-2)
    return e


def mesh_quadtree_2d(grid, h, max_block_elements=None, honor_interfaces=True, offset=(0.0, 0.0)) -> QuadMesh:
    """Arbre quaternaire 2:1 conforme en partant de la vitesse la plus rapide
    (:mod:`quadtree_2d`) : blocs de base dimensionnés pour la plus grande taille requise (2 x 2
    éléments de cette taille), découpés tant que leurs éléments dépassent la taille requise ;
    zones rapides à grands éléments, zones lentes à petits, transitions 2:1 conformes.
    Les rangées de blocs de base passent par les interfaces horizontales nettes.
    """
    from .quadtree_2d import Quadtree2D
    h2 = h[:, 0, :]
    s0 = float(h2.max()) if max_block_elements is None else float(max_block_elements)
    lmax = max(0, int(np.ceil(np.log2(s0 / float(h2.min())) - 1e-9)))
    (x0, x1), _, (z0, z1) = grid.extent
    block = 2.0 * s0
    xs = np.array(_offset_edges(x0, x1, block, offset[0]))
    cuts, ignored = [z0], []
    for b in (horizontal_interfaces(grid) if honor_interfaces else []):
        if b - cuts[-1] >= 0.5 * block and z1 - b >= 0.5 * block:     # no thin row of blocks
            cuts.append(b)
        else:
            ignored.append(float(b))                                # reported; the evaluation rejects a crossing
    cuts.append(z1)
    zs = [z0]
    for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        zs += _offset_edges(a, b, block, offset[1] if j == 0 else 0.0)[1:]
    T = Quadtree2D(xs, np.array(zs), lmax)
    # split a block while its elements (half the block) exceed the required size in the block
    T.refine_where(lambda lo, hi, level: (0.5 * (hi - lo)).max() > _box_min(grid, h2, lo, hi) * (1 + 1e-9))
    forced = T.balance()
    nodes, quads, levels = T.to_quads()
    counts = np.bincount(levels, minlength=lmax + 1)
    info = {"base_element_size": s0, "lmax": lmax, "base_blocks": [len(xs) - 1, len(zs) - 1], "offset": list(offset),
            "elements_per_level": counts.tolist(), "forced_splits": forced, "interfaces": cuts[1:-1],
            "ignored_interfaces": ignored}
    return QuadMesh(nodes, quads, "quadtree", info)


def _box_min(grid, h2, lo, hi):
    """Taille requise minimale sur les échantillons de la boîte (x, profondeur)."""
    y0 = grid.origin[1]
    a, b = grid.box_index(np.array([lo[0], y0, lo[1]]), np.array([hi[0], y0, hi[1]]))
    return float(h2[a[0, 0]:b[0, 0] + 1, a[0, 2]:b[0, 2] + 1].min())


def evaluate_2d(m: QuadMesh, grid: VelocityGrid, p: SizingParams, cfl=0.5, t_sim=4.0) -> dict:
    """Rapport de qualité et de coût d'un maillage 2-D (mêmes champs que :func:`quality.evaluate`)."""
    X = m.nodes[m.quads]                                       # (ne, 4, 2)
    e_next = np.roll(X, -1, axis=1) - X
    e_prev = np.roll(X, 1, axis=1) - X
    cross = e_next[..., 0] * e_prev[..., 1] - e_next[..., 1] * e_prev[..., 0]
    sj = np.abs(cross / (np.linalg.norm(e_next, axis=2) * np.linalg.norm(e_prev, axis=2)))
    convex = np.all(np.sign(cross) == np.sign(cross[:, :1]), axis=1)
    sj = np.where(convex, sj.min(axis=1), -1.0)
    edges = np.linalg.norm(e_next, axis=2)
    hmin, hmax = edges.min(axis=1), edges.max(axis=1)
    from .elements2d import element_stats_2d, element_box_stats_2d
    from .sizing import PPW_TOLERANCE
    st = element_stats_2d(grid, m.nodes, m.quads)              # on the real element (Codex, point 3)
    vmin, vmax, vmean = st["min"], st["max"], st["mean"]
    ppw = p.order * vmin / (p.fmax * hmax)
    under = ppw < p.ppw_accept * (1.0 - PPW_TOLERANCE)        # every element (Codex, point 2)
    ppw_box = p.order * element_box_stats_2d(grid, m.nodes, m.quads)[0] / (p.fmax * hmax)
    # sharp horizontal interfaces must be element faces (Codex, point 5)
    zmin_e, zmax_e = X[:, :, 1].min(1), X[:, :, 1].max(1)
    tol_z = 1e-6 * float(np.ptp(m.nodes[:, 1]))
    cross = {float(zi): int(np.sum((zmin_e < zi - tol_z) & (zmax_e > zi + tol_z))) for zi in horizontal_interfaces(grid)}
    n_cross = sum(cross.values())
    dt_e = cfl * hmin * gll_min_fraction(p.order) / vmax
    dt = float(dt_e.min())
    ne = len(m.quads)
    levels = np.floor(np.log2(dt_e / dt))
    rep = {
        "ok": bool(sj.min() > 0.2 and not under.any() and n_cross == 0),
        "strategy": m.strategy, "dim": 2, "n_elements": int(ne), "n_nodes": int(len(m.nodes)),
        "scaled_jacobian": {"min": float(sj.min()), "mean": float(sj.mean())},
        "frac_irregular": float((sj < 0.99).mean()),          # not a right-angled rectangle (transitions)
        "ppw_effective": {"min": float(ppw.min()), "median": float(np.median(ppw)), "target": p.ppw},
        "frac_undersampled": float(under.mean()), "n_undersampled": int(under.sum()),
        "interfaces_crossed": {z: n for z, n in cross.items() if n},
        "ppw_box_min": float(ppw_box.min()),                   # conservative estimate on bounding boxes
        "dt_cfl": dt, "cfl": cfl, "n_steps": int(np.ceil(t_sim / dt)),
        "cost_index": float(ne * p.order ** 2 * t_sim / dt / 1e9),
        "cost_index_lts": float(p.order ** 2 * t_sim * np.sum(1.0 / (dt * 2.0 ** levels)) / 1e9),
        "params": m.params,
    }
    rep["_per_element"] = {"vmin": vmin, "vmax": vmax, "vmean": vmean, "ppw": ppw, "sj": sj, "dt": dt_e}
    return rep


REGULARITY_WEIGHT = 0.5


def mesh_score(rep, rank="elements"):
    """Score de classement d'un candidat valide (plus petit = meilleur).

    ``"elements"`` (défaut) : la taille des éléments est fixée par l'ordre du schéma et la fréquence de la
    source (taille requise) ; on cherche le maillage qui la respecte avec le moins d'éléments, en
    préférant les mailles régulières : nombre d'éléments x (1 + 0,5 x part d'éléments non rectangulaires).
    ``"cost"`` / ``"lts"`` : coût de calcul (éléments / pas de temps global, ou pas local idéal)."""
    if rank == "cost":
        return rep["cost_index"]
    if rank == "lts":
        return rep["cost_index_lts"]
    return rep["n_elements"] * (1 + REGULARITY_WEIGHT * rep.get("frac_irregular", 0.0))


def autopilot_2d(grid: VelocityGrid, p: SizingParams, t_sim=4.0, lts=False,
                 strategies=("tensor", "doubling", "quadtree"), rank="elements", cfl=0.5):
    """Recherche déterministe parmi les stratégies ``strategies`` ; classement par :func:`mesh_score`
    (``rank``, ``lts=True`` équivaut à ``rank="lts"``). Renvoie (meilleur (maillage, rapport) valide,
    liste des rapports)."""
    h = h_required(grid, p)
    tried = []
    for ms in [(2, 3), (3, 5), (4, 6), (6, 8), (8, 12)]:
        segs = segments_2d(grid, h, ms)
        if "tensor" in strategies:
            m = mesh_tensor_2d(grid, h, segs)
            tried.append((m, evaluate_2d(m, grid, p, cfl=cfl, t_sim=t_sim)))
        if "doubling" in strategies:
            for levels in (1, 2, 3):
                for refine in (False, True):
                    m = mesh_doubling_2d(grid, h, segs, levels, refine)
                    rep = evaluate_2d(m, grid, p, cfl=cfl, t_sim=t_sim)
                    rep["params"] = {**rep["params"], "max_levels": levels, "allow_refine": refine,
                                     "max_segments": list(ms)}
                    tried.append((m, rep))
    if "quadtree" in strategies:
        # base element size: around the largest required size (larger: the levels base / 2^l may fall
        # just under the required sizes, the coarse body is then split once and kept below them; smaller:
        # the levels fall closer to the required sizes of the model)
        for k in (1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5):
            for off in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)):   # base grid shift (block fraction)
                m = mesh_quadtree_2d(grid, h, max_block_elements=k * float(h.max()), offset=off)
                rep = evaluate_2d(m, grid, p, cfl=cfl, t_sim=t_sim)
                rep["params"] = {**rep["params"], "base_factor": k, "offset": off}
                tried.append((m, rep))
    if "bands" in strategies:
        from .bands2d import mesh_bands_2d
        for ms in (4, 8, 12):                     # vertical segments per regular part
            for coarsen in (1.1, 1.2):            # least gain to start a coarser band
                try:
                    m = mesh_bands_2d(grid, h, max_segments=ms, coarsen=coarsen)
                except RuntimeError:
                    continue
                rep = evaluate_2d(m, grid, p, cfl=cfl, t_sim=t_sim)
                rep["params"] = {**rep["params"], "max_segments": ms, "coarsen": coarsen}
                tried.append((m, rep))
    if "hybrid" in strategies:
        for levels in (1, 2, 3):
            for q_bg in (0.1, 0.3, 0.5):          # background: lateral quantile of h_req per depth
                m = mesh_hybrid_2d(grid, h, max_levels=levels, q_bg=q_bg)
                rep = evaluate_2d(m, grid, p, cfl=cfl, t_sim=t_sim)
                rep["params"] = {**rep["params"], "max_levels": levels, "q_bg": q_bg}
                tried.append((m, rep))
    if "adaptive" in strategies:
        for sf in (0.9, 0.8, 0.7):
            m = mesh_adaptive_2d(grid, h, sf)
            rep = evaluate_2d(m, grid, p, cfl=cfl, t_sim=t_sim)
            tried.append((m, rep))
            if rep["ok"]:
                break
    rank = "lts" if lts else rank
    for _, r in tried:
        r["score"] = mesh_score(r, rank) if r.get("ok") else None
    tried.sort(key=lambda t: (not t[1]["ok"], mesh_score(t[1], rank)))
    return (tried[0] if tried and tried[0][1]["ok"] else None), [r for _, r in tried]
