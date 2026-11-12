"""Zones du maillage hybride à partir d'un modèle SEP 2-D (:mod:`makutu.mortar.hybrid2d`).

Règle : ordre et points par longueur d'onde fixés ; les mailles suivent la vitesse et la fréquence.
1. Corps à bord net (sel) : masque ``body`` (défaut : vitesse proche du maximum). Son contour est lissé
   (indicatrice filtrée, niveau 1/2) : c'est l'interface que suivront les arêtes des triangles.
2. Bande de triangles : tuiles d'une grille grossière situées à moins de ``clearance`` tailles de triangle
   du contour. Triangles DG, taille tri_factor p v / (fmax ppw) du côté où ils se trouvent.
3. Reste du domaine : rectangles de tuiles, pavés en blocs SEM réguliers par :func:`plan_blocks`
   (h = N v / (fmax ppw)), avec un ordre propre aux zones du corps si ``order_body`` est donné.
Chaque zone lit le modèle de son côté du contour (modèle prolongé par plus proche voisin de l'autre
côté) : le saut est porté par les arêtes des triangles, jamais par l'intérieur d'un élément.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .blocks2d import grid_sampler, plan_blocks
from .hybrid2d import QuadZone, TriZone


def default_body(vel, level=0.95):
    """Masque des corps rapides à bord net : vitesse > level x maximum, sans les couches minces qui
    traversent tout le modèle (interfaces planes, traitées par les coupes des blocs)."""
    m = vel > level * vel.max()
    lab, n = ndimage.label(m)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        wide = (sl[0].stop - sl[0].start) >= 0.9 * vel.shape[0]
        thin = (sl[1].stop - sl[1].start) <= 0.05 * vel.shape[1]
        if (wide and thin) or (lab[sl] == k).sum() < 25:
            m[lab == k] = False
    return m


def _largest_rectangles(free):
    """Décomposition gloutonne du masque ``free`` (tuiles libres) en rectangles : le plus grand rectangle
    entièrement libre d'abord (histogrammes par ligne), puis on recommence sur le reste.
    Retourne [(i0, i1, k0, k1)]."""
    free = free.copy()
    out = []
    while free.any():
        best, h = (0, None), np.zeros(free.shape[0], int)
        for k in range(free.shape[1]):
            h = np.where(free[:, k], h + 1, 0)
            stack = []                                   # (start index, height)
            for i, hv in enumerate(list(h) + [0]):
                start = i
                while stack and stack[-1][1] >= hv:
                    st, sh = stack.pop()
                    if sh * (i - st) > best[0]:
                        best = (sh * (i - st), (st, i, k + 1 - sh, k + 1))
                    start = st
                stack.append((start, hv))
        i0, i1, k0, k1 = best[1]
        out.append((i0, i1, k0, k1))
        free[i0:i1, k0:k1] = False
    return out


def _resample(line, spacing):
    """Points le long de la polyligne (k, 2), espacés d'environ spacing(x, z) ; extrémités conservées."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    out, pos = [0.0], 0.0
    while True:
        x = np.interp(pos, s, line[:, 0]); z = np.interp(pos, s, line[:, 1])
        pos += float(spacing(x, z))
        if pos >= s[-1]:
            break
        out.append(pos)
    n = max(len(out), 3 if np.allclose(line[0], line[-1]) else 1)
    t = np.linspace(0, s[-1], n + 1)                       # equalised
    return np.stack([np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])], 1)


def _triangulate(poly, curves, size, xt, zt):
    """Triangles du polygone rectiligne ``poly`` (shapely, trous permis) avec les polylignes ``curves``
    imposées comme arêtes ; bords divisés uniformément entre les lignes de tuiles xt, zt."""
    import gmsh
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("band")
    g = gmsh.model.geo
    pts, lines_n = {}, []

    def point(p):
        key = (round(float(p[0]), 4), round(float(p[1]), 4))
        if key not in pts:
            pts[key] = g.addPoint(key[0], key[1], 0.0)
        return pts[key]

    ends = [c[j] for c in curves if not np.allclose(c[0], c[-1]) for j in (0, -1)]

    def ring(coords):
        coords = np.asarray(coords)[:-1]
        ls = []
        for a, b in zip(coords, np.roll(coords, -1, axis=0)):
            ax = 0 if abs(a[0] - b[0]) > abs(a[1] - b[1]) else 1          # running axis
            cuts = xt if ax == 0 else zt
            lo, hi = sorted((a[ax], b[ax]))
            ts = [c for c in cuts if lo + 1e-6 < c < hi - 1e-6]
            ts += [e[ax] for e in ends if abs(e[1 - ax] - a[1 - ax]) < 1e-6 and lo + 1e-6 < e[ax] < hi - 1e-6]
            ts = sorted(set([a[ax], b[ax]] + ts), reverse=a[ax] > b[ax])
            for t0, t1 in zip(ts[:-1], ts[1:]):
                p0, p1 = a.copy(), a.copy()
                p0[ax], p1[ax] = t0, t1
                l = g.addLine(point(p0), point(p1))
                mid = 0.5 * (p0 + p1)
                lines_n.append((l, max(1, int(round(abs(t1 - t0) / float(size(mid[0], mid[1]))))) + 1))
                ls.append(l)
        return g.addCurveLoop(ls)

    loops = [ring(poly.exterior.coords)] + [ring(r.coords) for r in poly.interiors]
    surf = g.addPlaneSurface(loops)
    emb = []
    for c in curves:
        closed = np.allclose(c[0], c[-1])
        ids = [point(q) for q in (c[:-1] if closed else c)]
        if closed:
            ids.append(ids[0])
        for i0, i1 in zip(ids[:-1], ids[1:]):
            if i0 != i1:
                l = g.addLine(i0, i1)
                lines_n.append((l, 2))
                emb.append(l)
    g.synchronize()
    if emb:
        gmsh.model.mesh.embed(1, emb, 2, surf)
    for l, n in lines_n:
        gmsh.model.mesh.setTransfiniteCurve(l, n)
    gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: float(size(x, y)))
    for name in ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature", "Mesh.MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber(name, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    et, _, conn = gmsh.model.mesh.getElements(2)
    tri = np.asarray(conn[list(et).index(2)], int).reshape(-1, 3)
    pos = np.zeros((int(tags.max()) + 1, 2))
    pos[np.asarray(tags, int)] = np.asarray(xyz).reshape(-1, 3)[:, :2]
    used = np.unique(tri)
    remap = -np.ones(int(tags.max()) + 1, int)
    remap[used] = np.arange(len(used))
    gmsh.model.remove()
    if own:
        gmsh.finalize()
    return pos[used], remap[tri]


def plan_hybrid(vel, origin, spacing, f0, ppw=5.0, order=4, order_body=None, p_tri=3, tri_factor=4.0 / 3.0,
                body=None, tile=None, clearance=0.75, smooth=1.5, refine=1.0, rho=1000.0, **plan_kwargs):
    """Zones hybrides d'une grille ``vel`` (nx, nz) à cellules centrées. Retourne (zones, info).

    order : ordre des zones SEM ; order_body : ordre des zones SEM à l'intérieur des corps (défaut : order) ;
    p_tri : ordre des triangles ; body : masque (nx, nz) des corps à bord net (défaut :func:`default_body`) ;
    tile : côté des tuiles en cellules (défaut : une taille de triangle du côté lent) ; clearance : distance
    minimale du contour au bord de la bande, en tailles de triangle ; smooth : lissage du contour (cellules) ;
    refine : divise toutes les tailles (maillage de référence)."""
    import shapely
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union
    from skimage.measure import find_contours

    vel = np.asarray(vel, float)
    nx, nz = vel.shape
    x0, z0 = origin
    dx, dz = spacing
    fmax = 2.5 * f0
    order_body = order if order_body is None else order_body
    body = default_body(vel) if body is None else np.asarray(body, bool)
    # smoothed contour (level 1/2 of the filtered indicator), in physical coordinates
    ind = ndimage.gaussian_filter(np.pad(body.astype(float), 1, mode="edge"), smooth)
    side = ind[1:-1, 1:-1] > 0.5
    dom = box(x0, z0, x0 + nx * dx, z0 + nz * dz)
    raw = [np.stack([x0 + (c[:, 0] - 0.5) * dx, z0 + (c[:, 1] - 0.5) * dz], 1) for c in find_contours(ind, 0.5)]
    # each side reads its own model, extended across the contour by nearest neighbour
    def extend(mask):
        if mask.all() or not mask.any():
            return vel.copy()
        idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        return vel[idx[0], idx[1]]
    v_out, v_in = extend(~side), extend(side)
    s_out = grid_sampler(v_out, origin, spacing, rho=rho)
    s_in = grid_sampler(v_in, origin, spacing, rho=rho)
    ind_at = lambda x, z: ndimage.map_coordinates(ind, [np.atleast_1d((np.asarray(x, float) - x0) / dx + 0.5),
                                                        np.atleast_1d((np.asarray(z, float) - z0) / dz + 0.5)], order=1, mode="nearest")

    def tri_size(x, z):
        inside = ind_at(x, z) > 0.5
        v = np.where(inside, s_in(np.atleast_1d(x), np.atleast_1d(z))[0], s_out(np.atleast_1d(x), np.atleast_1d(z))[0])
        return tri_factor * p_tri * v / (fmax * ppw) / refine

    info = dict(body_fraction=float(side.mean()), n_contours=len(raw))
    zones = []
    xt = [x0, x0 + nx * dx]
    zt = [z0, z0 + nz * dz]
    band = np.zeros((1, 1), bool)
    ti = np.array([0, nx]); tk = np.array([0, nz])
    if raw and side.any():
        edge = np.zeros_like(side)
        edge[:-1] |= side[:-1] != side[1:]; edge[1:] |= side[:-1] != side[1:]
        edge[:, :-1] |= side[:, :-1] != side[:, 1:]; edge[:, 1:] |= side[:, :-1] != side[:, 1:]
        dist = ndimage.distance_transform_edt(~edge) * min(dx, dz)
        t_cell = tri_factor * p_tri * np.where(side, v_in, v_out) / (fmax * ppw) / refine
        near = dist < clearance * t_cell
        if tile is None:
            tile = max(2, int(np.ceil(np.median(t_cell[edge & ~side]) / min(dx, dz))))
        def cuts(n):
            c = list(range(0, n, tile)) + [n]
            if len(c) > 2 and c[-1] - c[-2] < tile / 2:
                del c[-2]
            return np.array(c)
        ti, tk = cuts(nx), cuts(nz)
        band = np.add.reduceat(np.add.reduceat(near, ti[:-1], axis=0), tk[:-1], axis=1) > 0
        xt, zt = x0 + ti * dx, z0 + tk * dz
        region = unary_union([box(xt[i], zt[k], xt[i + 1], zt[k + 1]) for i, k in zip(*np.where(band))])
        polys = list(region.geoms) if region.geom_type == "MultiPolygon" else [region]
        n_tri = 0
        for poly in polys:
            curves = []
            for c in raw:
                part = LineString(c).intersection(dom).intersection(poly.buffer(1e-6))
                for gline in (part.geoms if hasattr(part, "geoms") else [part]):
                    if gline.geom_type != "LineString" or gline.length < 1e-6:
                        continue
                    q = np.asarray(gline.coords)
                    if np.linalg.norm(c[0] - c[-1]) < 1e-9 and abs(gline.length - LineString(c).length) < 1e-6:
                        q = np.concatenate([c, c[:1]]) if not np.allclose(c[0], c[-1]) else c
                    q = _resample(q, lambda x, z: min(float(tri_size(x, z)[0]),
                                                      tri_factor * p_tri * float(min(s_in(np.array([x]), np.array([z]))[0][0],
                                                                                     s_out(np.array([x]), np.array([z]))[0][0])) / (fmax * ppw) / refine))
                    # open pieces end on the domain boundary: snap the ends onto it, and onto a tile line if close
                    if not np.allclose(q[0], q[-1]):
                        for j in (0, -1):
                            pb = shapely.ops.nearest_points(poly.exterior, shapely.geometry.Point(q[j]))[0]
                            e = np.array([pb.x, pb.y])
                            h = float(tri_size(e[0], e[1])[0])
                            for ax, cutsv in ((0, xt), (1, zt)):
                                near_cut = cutsv[np.argmin(np.abs(cutsv - e[ax]))]
                                if abs(near_cut - e[ax]) < 0.35 * h and (abs(e[1 - ax] - x0) > -1):
                                    on_edge = min(abs(e[1 - ax] - b) for b in ((x0, x0 + nx * dx) if ax == 1 else (z0, z0 + nz * dz))) < 1e-6
                                    if on_edge:
                                        e[ax] = near_cut
                            q[j] = e
                    curves.append(q)
            nodes, tris = _triangulate(poly, curves, lambda x, z: float(tri_size(x, z)[0]), xt, zt)
            cen = nodes[tris].mean(1)
            ins = ind_at(cen[:, 0], cen[:, 1]) > 0.5

            def tri_model(x, z, e, ins=ins):
                vi, ri = s_in(x, z)
                vo, ro = s_out(x, z)
                return np.where(ins[e], vi, vo), np.where(ins[e], ri, ro)

            zones.append(TriZone(nodes, tris, order=p_tri, rho=rho, model=tri_model))
            n_tri += len(tris)
        info.update(n_triangles=n_tri, tile_m=tile * min(dx, dz), band_fraction=float(
            sum(p.area for p in polys) / (nx * dx * nz * dz)))
    # SEM zones: largest rectangles of free tiles, each tiled by plan_blocks
    free = ~band if band.shape != (1, 1) else np.ones((1, 1), bool)
    rects = _largest_rectangles(free)
    n_quad = 0
    for a, b, k0, k1 in rects:
        i0, i1, c0, c1 = ti[a], ti[b], tk[k0], tk[k1]
        inside = side[i0:i1, c0:c1].mean() > 0.5
        vsub = (v_in if inside else v_out)[i0:i1, c0:c1]
        N = order_body if inside else order
        pl = plan_blocks(vsub, (x0 + i0 * dx, z0 + c0 * dz), spacing, order=N, f0=f0, ppw=ppw * refine, **plan_kwargs)
        for p in pl.patches:
            zones.append(QuadZone(p.x0, p.x1, p.z0, p.z1, p.nx, p.nz, N, s_in if inside else s_out,
                                  inset=0.5 * min(dx, dz)))
            n_quad += p.n_elements
    def model(x, z):
        """Modèle représenté : corps (côté intérieur du contour lissé) et fond prolongés de part et d'autre."""
        x, z = np.asarray(x, float), np.asarray(z, float)
        ins = ind_at(x.ravel(), z.ravel()).reshape(x.shape) > 0.5
        vi, ri = s_in(x, z)
        vo, ro = s_out(x, z)
        return np.where(ins, vi, vo), np.where(ins, ri, ro)

    info.update(n_quad_elements=n_quad, n_zones=len(zones), contours=raw, band=band, xt=np.asarray(xt), zt=np.asarray(zt),
                model=model)
    return zones, info


def _triangulate_rings(poly, curves, size, ring_points):
    """Triangles du polygone ``poly`` (shapely, trous permis) ; chaque côté rectiligne des anneaux reçoit les
    sommets intérieurs ring_points(p0, p1) (nœuds de la grille SEM voisine : sommets communs avec ses cellules) ;
    polylignes ``curves`` imposées comme arêtes."""
    import gmsh
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("band")
    g = gmsh.model.geo
    pts, lines_n = {}, []

    def point(p):
        key = (round(float(p[0]), 6), round(float(p[1]), 6))
        if key not in pts:
            pts[key] = g.addPoint(key[0], key[1], 0.0)
        return pts[key]

    def ring(coords):
        c = np.asarray(coords)[:-1]
        # merge collinear vertices
        keep = []
        for i in range(len(c)):
            a, b, d = c[i - 1], c[i], c[(i + 1) % len(c)]
            if abs((b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0])) > 1e-9:
                keep.append(b)
        c = np.array(keep)
        ls = []
        for a, b in zip(c, np.roll(c, -1, axis=0)):
            q = [a] + [np.asarray(m, float) for m in ring_points(a, b)] + [b]
            for u, v in zip(q[:-1], q[1:]):
                l = g.addLine(point(u), point(v))
                lines_n.append((l, 2))
                ls.append(l)
        return g.addCurveLoop(ls)

    loops = [ring(poly.exterior.coords)] + [ring(r.coords) for r in poly.interiors]
    surf = g.addPlaneSurface(loops)
    emb = []
    for c in curves:
        closed = np.allclose(c[0], c[-1])
        ids = [point(q) for q in (c[:-1] if closed else c)]
        if closed:
            ids.append(ids[0])
        for i0, i1 in zip(ids[:-1], ids[1:]):
            if i0 != i1:
                l = g.addLine(i0, i1)
                lines_n.append((l, 2))
                emb.append(l)
    g.synchronize()
    if emb:
        gmsh.model.mesh.embed(1, emb, 2, surf)
    for l, n in lines_n:
        gmsh.model.mesh.setTransfiniteCurve(l, n)
    gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: float(size(x, y)))
    for name in ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature", "Mesh.MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber(name, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    et, _, conn = gmsh.model.mesh.getElements(2)
    tri = np.asarray(conn[list(et).index(2)], int).reshape(-1, 3)
    pos = np.zeros((int(tags.max()) + 1, 2))
    pos[np.asarray(tags, int)] = np.asarray(xyz).reshape(-1, 3)[:, :2]
    used = np.unique(tri)
    remap = -np.ones(int(tags.max()) + 1, int)
    remap[used] = np.arange(len(used))
    gmsh.model.remove()
    if own:
        gmsh.finalize()
    return pos[used], remap[tri]


def plan_cg_hybrid(vel, origin, spacing, f0, ppw=5.0, order=4, order_body=None, p_tri=3, tri_factor=4.0 / 3.0,
                   body=None, clearance=0.75, clearance_in=0.5, smooth=1.5, refine=1.0, rho=1000.0, n_shift=4, rectilinear=True):
    """Maillage hybride sans interface non conforme entre quadrangles (sans mortar) :
    - une seule grille SEM continue et conforme pour le fond : rectiligne (``rectilinear``), pas de chaque
      ligne et de chaque colonne h = N v_min / (fmax ppw) avec v_min la vitesse la plus faible de la ligne
      (colonne) hors des corps ; sinon pas uniforme fixé par la vitesse la plus faible du fond ;
    - une seule grille SEM continue et conforme pour l'intérieur des corps (pas fixé par leur vitesse
      minimale, ordre ``order_body``) ;
    - une bande de triangles DG entre les deux, qui suit le contour lissé et touche les deux grilles le
      long de leurs bords en escalier (mêmes sommets) : seul couplage DG.
    Retourne (zones, info) comme :func:`plan_hybrid`."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    from skimage.measure import find_contours
    from .hybrid2d import GridZone

    vel = np.asarray(vel, float)
    nx, nz = vel.shape
    x0, z0 = origin
    dx, dz = spacing
    W, H = nx * dx, nz * dz
    fmax = 2.5 * f0
    order_body = order if order_body is None else order_body
    body = default_body(vel) if body is None else np.asarray(body, bool)
    ind = ndimage.gaussian_filter(np.pad(body.astype(float), 1, mode="edge"), smooth)
    side = ind[1:-1, 1:-1] > 0.5
    raw = [np.stack([x0 + (c[:, 0] - 0.5) * dx, z0 + (c[:, 1] - 0.5) * dz], 1) for c in find_contours(ind, 0.5)]

    def extend(mask):
        if mask.all() or not mask.any():
            return vel.copy()
        idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        return vel[idx[0], idx[1]]
    v_out, v_in = extend(~side), extend(side)
    s_out = grid_sampler(v_out, origin, spacing, rho=rho)
    s_in = grid_sampler(v_in, origin, spacing, rho=rho)
    ind_at = lambda x, z: ndimage.map_coordinates(ind, [np.atleast_1d((np.asarray(x, float) - x0) / dx + 0.5),
                                                        np.atleast_1d((np.asarray(z, float) - z0) / dz + 0.5)], order=1, mode="nearest")

    def model(x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        ins = ind_at(x.ravel(), z.ravel()).reshape(x.shape) > 0.5
        vi, ri = s_in(x, z)
        vo, ro = s_out(x, z)
        return np.where(ins, vi, vo), np.where(ins, ri, ro)

    def tri_size(x, z):
        ins = ind_at(x, z) > 0.5
        v = np.where(ins, s_in(np.atleast_1d(x), np.atleast_1d(z))[0], s_out(np.atleast_1d(x), np.atleast_1d(z))[0])
        return tri_factor * p_tri * v / (fmax * ppw) / refine

    # distance to the contour (m), per SEP cell
    edge = np.zeros_like(side)
    edge[:-1] |= side[:-1] != side[1:]; edge[1:] |= side[:-1] != side[1:]
    edge[:, :-1] |= side[:, :-1] != side[:, 1:]; edge[:, 1:] |= side[:, :-1] != side[:, 1:]
    dist = ndimage.distance_transform_edt(~edge) * min(dx, dz)
    t_out = tri_factor * p_tri * v_out / (fmax * ppw) / refine
    t_in = tri_factor * p_tri * v_in / (fmax * ppw) / refine

    def grid_mask(xe, ze, keep):
        """Cellules d'une grille rectiligne de lignes xe, ze ; une cellule est active si ``keep`` est vrai sur
        toutes les cellules SEP qu'elle touche (test conservatif)."""
        ia = np.floor((xe[:-1] - x0) / dx + 1e-9).astype(int); ib = np.ceil((xe[1:] - x0) / dx - 1e-9).astype(int)
        ka = np.floor((ze[:-1] - z0) / dz + 1e-9).astype(int); kb = np.ceil((ze[1:] - z0) / dz - 1e-9).astype(int)
        act = np.zeros((len(xe) - 1, len(ze) - 1), bool)
        for i in range(len(xe) - 1):
            if ia[i] < 0 or ib[i] > nx:
                continue
            col = keep[ia[i]:max(ib[i], ia[i] + 1)].all(0)
            for k in range(len(ze) - 1):
                act[i, k] = ka[k] >= 0 and kb[k] <= nz and bool(col[ka[k]:max(kb[k], ka[k] + 1)].all())
        return act

    def lines_1d(start, n_cells, d, h_req):
        """Lignes de grille sur [start, start + n_cells d] : chaque cellule de taille <= h_req (par échantillon
        SEP, minimum glissant sur la plus grande taille) ; nombre de cellules minimal."""
        w = 2 * int(np.ceil(h_req.max() / d)) + 1
        hf = ndimage.minimum_filter1d(h_req, w, mode="nearest")
        s = np.concatenate([[0.0], np.cumsum(d / hf)])
        n = max(1, int(np.ceil(s[-1] - 1e-9)))
        return start + np.interp(np.linspace(0.0, s[-1], n + 1), s, d * np.arange(n_cells + 1))

    # background grid: one conforming grid; pitch per row (and column) from the slowest velocity of the row
    # (column) outside the bodies when ``rectilinear``, else uniform at the slowest velocity
    keep_out = (~side) & (dist >= clearance * t_out)
    v_bg = np.where(~side, v_out, np.inf)
    if not np.isfinite(v_bg).any():
        v_bg = vel.copy()
    k_bg = order / (fmax * ppw) / refine
    if rectilinear:
        xe = lines_1d(x0, nx, dx, k_bg * v_bg.min(1))
        ze = lines_1d(z0, nz, dz, k_bg * v_bg.min(0))
    else:
        h = k_bg * v_bg.min()
        xe = np.linspace(x0, x0 + W, int(np.ceil(W / h - 1e-9)) + 1)
        ze = np.linspace(z0, z0 + H, int(np.ceil(H / h - 1e-9)) + 1)
    zones = [GridZone(0.0, 0.0, 0.0, 0.0, grid_mask(xe, ze, keep_out), order, s_out, xe=xe, ze=ze)]
    info = dict(n_contours=len(raw), h_out=(float(np.diff(xe).min()), float(np.diff(ze).min())),
                h_out_max=(float(np.diff(xe).max()), float(np.diff(ze).max())))
    # body grid: square cells at the size required by the slowest velocity of the eroded body; origin shifted
    # (fractions of h) to cover the most body cells
    if side.any():
        keep_in = side & (dist >= clearance_in * t_in)
        ii, kk = np.where(keep_in)
        best = None
        if len(ii):
            h_in = order_body * v_in[keep_in].min() / (fmax * ppw) / refine
            bx0, bx1 = x0 + ii.min() * dx, x0 + (ii.max() + 1) * dx
            bz0, bz1 = z0 + kk.min() * dz, z0 + (kk.max() + 1) * dz
            for fx in np.arange(n_shift) / n_shift:
                for fz in np.arange(n_shift) / n_shift:
                    gx0, gz0 = bx0 - fx * h_in, bz0 - fz * h_in
                    xi_ = gx0 + h_in * np.arange(max(1, int(np.ceil((bx1 - gx0) / h_in))) + 1)
                    zi_ = gz0 + h_in * np.arange(max(1, int(np.ceil((bz1 - gz0) / h_in))) + 1)
                    act = grid_mask(xi_, zi_, keep_in)
                    if best is None or act.sum() > best[0].sum():
                        best = (act, xi_, zi_)
        if best is not None and best[0].any():
            act, xi_, zi_ = best
            ai, ak = np.where(act.any(1))[0], np.where(act.any(0))[0]     # crop to the active cells
            act = act[ai.min():ai.max() + 1, ak.min():ak.max() + 1]
            zones.append(GridZone(0.0, 0.0, 0.0, 0.0, act, order_body, s_in,
                                  xe=xi_[ai.min():ai.max() + 2], ze=zi_[ak.min():ak.max() + 2]))
            info["h_in"] = (h_in, h_in)
    # band = domain minus the grids
    def cells_union(zg):
        return unary_union([box(*zg.cell_box(i, k)) for i, k in np.argwhere(zg.active)])
    region = box(x0, z0, x0 + W, z0 + H).difference(unary_union([cells_union(zg) for zg in zones]))
    polys = list(region.geoms) if region.geom_type == "MultiPolygon" else ([region] if not region.is_empty else [])

    def ring_points(a, b):
        """Sommets intérieurs d'un côté rectiligne de la bande : nœuds des grilles qui portent ce côté, sinon
        division uniforme à la taille des triangles."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        ax = 0 if abs(a[1] - b[1]) < 1e-6 else 1                       # running axis
        lo, hi = sorted((a[ax], b[ax]))
        t = 1e-6
        pts = []
        for zg in zones:
            run, fixed = (zg.xe, zg.ze) if ax == 0 else (zg.ze, zg.xe)
            if np.min(np.abs(fixed - a[1 - ax])) < t and run[0] - t <= lo and hi <= run[-1] + t:
                pts += [r for r in run if lo + t < r < hi - t]
        if pts:
            r = np.unique(np.round(pts, 9))
        else:
            mid = 0.5 * (a + b)
            n = max(1, int(round((hi - lo) / float(tri_size(mid[0], mid[1])[0]))))
            r = lo + (hi - lo) * np.arange(1, n) / n
        if a[ax] > b[ax]:
            r = r[::-1]
        return [(v, a[1]) if ax == 0 else (a[0], v) for v in r]

    n_tri = 0
    from shapely.geometry import LineString
    delta = 0.25 * tri_factor * p_tri * vel.min() / (fmax * ppw) / refine
    inner_box = box(x0 + delta, z0 + delta, x0 + W - delta, z0 + H - delta)
    for poly in polys:
        curves = []
        for c in raw:
            # the contour may reach (and run along) the domain boundary: keep it delta away from it
            part = LineString(c).intersection(poly.buffer(-1e-6)).intersection(inner_box)
            for gl in (part.geoms if hasattr(part, "geoms") else [part]):
                if gl.geom_type != "LineString" or gl.length < 1e-6:
                    continue
                q = np.asarray(gl.coords)
                if abs(gl.length - LineString(c).length) < 1e-6:
                    q = c if np.allclose(c[0], c[-1]) else np.concatenate([c, c[:1]])
                q = _resample(q, lambda x, z: float(min(tri_size(x, z)[0],
                                                        tri_factor * p_tri * min(s_in(np.array([x]), np.array([z]))[0][0],
                                                                                 s_out(np.array([x]), np.array([z]))[0][0]) / (fmax * ppw) / refine)))
                curves.append(q)
        nodes, tris = _triangulate_rings(poly, curves, lambda x, z: float(tri_size(x, z)[0]), ring_points)
        cen = nodes[tris].mean(1)
        ins = ind_at(cen[:, 0], cen[:, 1]) > 0.5

        def tri_model(x, z, e, ins=ins):
            vi, ri = s_in(x, z)
            vo, ro = s_out(x, z)
            return np.where(ins[e], vi, vo), np.where(ins[e], ri, ro)
        zones.append(TriZone(nodes, tris, order=p_tri, rho=rho, model=tri_model))
        n_tri += len(tris)
    info.update(n_triangles=n_tri, n_zones=len(zones), contours=raw, model=model,
                n_quad_elements=int(sum(zg.active.sum() for zg in zones if zg.kind == "grid")),
                band_fraction=float(sum(p.area for p in polys) / (W * H)))
    return zones, info
