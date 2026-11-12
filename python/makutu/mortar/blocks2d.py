"""Pavage d'un modèle SEP 2-D en blocs rectangulaires réguliers, chacun à sa propre taille de maille.

Taille requise par cellule SEP : h_req = N v / (fmax ppw) (ordre N du schéma, fmax = 2,5 f0 pour une
source de Ricker de pic f0, ppw points GLL par longueur d'onde minimale). Un bloc R est maillé par
une grille régulière ceil(W / h) x ceil(H / h) avec h = min h_req sur R : tous ses éléments
respectent le critère par construction, sans élément déformé.

Le pavage est choisi par programmation dynamique sur l'arbre des coupes binaires au milieu (selon x
ou selon z, en indices de cellules SEP) : pour chaque rectangle de l'arbre,
    coût(R) = min( feuille(R), coût(R_gauche) + coût(R_droite), coût(R_haut) + coût(R_bas) ),
    feuille(R) = nx nz + interface_weight / 2 x (arêtes de R sur des interfaces intérieures),
le second terme représentant le travail de face (SIPG) des bords du bloc. Le résultat est optimal
parmi ces pavages. Les blocs sont recollés par :class:`makutu.mortar.Mortar2D` (rapports de taille
quelconques). Les coupes au milieu suivent les puissances de deux des indices ; les interfaces nettes
du modèle (fond de mer, bord du sel) sont rendues par les coupes successives, à la taille minimale
des blocs près.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from .solver2d import Patch


@dataclass
class BlockPlan:
    patches: list
    n_elements: int
    ideal: float
    stats: dict


def _overlap(edges_e, edges_c):
    """Matrice (ne, nc) des longueurs de recouvrement des intervalles [e_i, e_i+1] et [c_j, c_j+1]."""
    lo = np.maximum(edges_e[:-1, None], edges_c[None, :-1])
    hi = np.minimum(edges_e[1:, None], edges_c[None, 1:])
    return np.clip(hi - lo, 0.0, None)


_CFL_CACHE = {}


def gll_cfl(order):
    """Constante C du pas stable d'une grille régulière d'éléments carrés : dt_max = C h / vp
    (saute-mouton, valeur propre maximale de M^-1 K calculée sur un bloc 8 x 8 à bords rigides)."""
    if order not in _CFL_CACHE:
        from .solver2d import Mortar2D
        m = Mortar2D([Patch(0.0, 8.0, 0.0, 8.0, 8, 8, 1.0, 1.0)], order=order,
                     boundary={s: "rigid" for s in ("left", "right", "top", "bottom")})
        _CFL_CACHE[order] = m.dt_max()            # h = 1, vp = 1
    return _CFL_CACHE[order]


def graded_size(h, spacing, gamma):
    """Taille requise lipschitzienne : plus grande fonction g <= h telle que |g(x) - g(y)| <= gamma |x - y|
    (passes de chanfrein, métrique L1 sur la grille). Autour d'une zone lente, la taille ne croît que
    progressivement : une couronne d'éléments de transition entoure les corps lents à bord net."""
    g = np.asarray(h, float).copy()
    dx, dz = spacing
    sx, sz = gamma * dx, gamma * dz
    for _ in range(2):
        for i in range(1, g.shape[0]):
            np.minimum(g[i], g[i - 1] + sx, out=g[i])
        for i in range(g.shape[0] - 2, -1, -1):
            np.minimum(g[i], g[i + 1] + sx, out=g[i])
        for k in range(1, g.shape[1]):
            np.minimum(g[:, k], g[:, k - 1] + sz, out=g[:, k])
        for k in range(g.shape[1] - 2, -1, -1):
            np.minimum(g[:, k], g[:, k + 1] + sz, out=g[:, k])
    return g


def sharp_interfaces(vel, rel_jump=0.1, coverage=0.5):
    """Indices k des bords de cellules (entre les rangées k - 1 et k en profondeur) portant une interface
    horizontale nette : saut relatif de vitesse > rel_jump sur au moins ``coverage`` de la largeur."""
    v = np.asarray(vel, float)
    jump = np.abs(np.diff(v, axis=1)) / np.minimum(v[:, 1:], v[:, :-1])
    frac = (jump > rel_jump).mean(axis=0)
    return [int(k + 1) for k in np.where(frac >= coverage)[0]]


def plan_blocks(vel, origin, spacing, order=4, f0=10.0, ppw=5.0, rho=None, min_elements=2,
                min_cells=4, interface_weight=2.0, fmax=None, rank="elements",
                dt_min=None, z_cuts="auto", edge_cuts=2, edge_jump=0.1, edge_coverage=0.5, grading=None,
                force_jump=0.3):
    """Pavage optimal en blocs réguliers d'une grille 2-D ``vel`` (nx, nz), cellules centrées.

    origin, spacing : (x0, z0), (dx, dz) du coin de la grille (bord de la première cellule) ;
    rho : densité (nx, nz) ou scalaire (défaut 1000) ;
    min_elements : un bloc fait au moins ce nombre d'éléments de la plus petite taille requise du
    modèle dans chaque direction (borne la profondeur de l'arbre) ;
    interface_weight : coût d'une arête d'interface entre blocs, en éléments, payé pour moitié par chacun
    des deux blocs (les bords extérieurs ne coûtent rien) ; 2 correspond au coût mesuré sur GPU avec
    FUnTiDES (un segment d'interface coûte de l'ordre de 1,5 élément par pas) ;
    rank : "elements" (nombre d'éléments) ou "lts" (travail avec un pas de temps propre à chaque bloc :
    éléments x dt_ref / dt_bloc, dt_bloc = C min(hx, hz) / vp_max du bloc, dt_ref celui d'un élément de
    taille exactement requise ; pénalise les blocs à éléments minces ou à fort contraste interne).
    z_cuts : coupes horizontales imposées (indices de bords de cellules), "auto" (défaut) : interfaces
    nettes de :func:`sharp_interfaces`, distantes d'au moins une taille d'élément minimale (les blocs
    s'appuient alors sur ces interfaces : aucun élément ne les traverse), None : aucune ;
    edge_cuts : nombre maximal de coupes supplémentaires essayées par direction dans chaque rectangle, sur
    ses bords nets locaux (saut relatif de vitesse > edge_jump entre deux colonnes, ou deux rangées, sur au
    moins edge_coverage du rectangle) : un corps net aligné sur les axes est alors maillé avec des bords de
    blocs sur ses bords (un saut fort au milieu d'un élément coûte plusieurs % sur les traces ; 0 : coupes
    au milieu seulement) ;
    force_jump : si donné, un rectangle qui contient un bord net de saut relatif > force_jump (sur au moins
    edge_coverage de sa hauteur ou de sa largeur, à distance >= taille minimale de ses bords) doit être
    découpé (pas de bloc à travers un saut fort, tant que la taille minimale le permet) ;
    grading : pente maximale gamma de la taille requise (:func:`graded_size`), None : taille locale seule ;
    dt_min : pas de temps global visé (s) ; une feuille dont dt_bloc est plus petit est refusée (le bloc
    est alors découpé), pour un calcul à pas de temps global sans bloc limitant.
    Chaque bloc reçoit ``dt_est`` ; stats : dt_global (min des blocs), work_global (éléments / dt_global)
    et work_lts (somme des éléments / dt_bloc), en éléments-pas par seconde simulée."""
    if rank not in ("elements", "lts"):
        raise ValueError(f"rank inconnu : {rank}")
    vel = np.asarray(vel, float)
    nxc, nzc = vel.shape
    x0, z0 = origin
    dx, dz = spacing
    fmax = 2.5 * f0 if fmax is None else fmax
    hreq = order * vel / (fmax * ppw)
    if grading is not None:
        hreq = graded_size(hreq, spacing, grading)
    hmin = float(hreq.min())
    mcx = max(min_cells, int(np.ceil(min_elements * hmin / dx)))
    mcz = max(min_cells, int(np.ceil(min_elements * hmin / dz)))
    lam = interface_weight
    hv = order / (fmax * ppw)                    # h_req / v (s)
    C = gll_cfl(order)
    memo = {}
    # cumulative counts of sharp jumps: between columns i-1 and i (along z), between rows k-1 and k (along x)
    jx = np.zeros((nxc + 1, nzc))
    jx[1:nxc] = np.abs(np.diff(vel, axis=0)) / np.minimum(vel[1:], vel[:-1]) > edge_jump
    cjx = np.concatenate([np.zeros((nxc + 1, 1)), np.cumsum(jx, axis=1)], axis=1)        # (nxc+1, nzc+1)
    jz = np.zeros((nxc, nzc + 1))
    jz[:, 1:nzc] = np.abs(np.diff(vel, axis=1)) / np.minimum(vel[:, 1:], vel[:, :-1]) > edge_jump
    cjz = np.concatenate([np.zeros((1, nzc + 1)), np.cumsum(jz, axis=0)], axis=0)        # (nxc+1, nzc+1)

    if force_jump is not None:
        fx = np.zeros((nxc + 1, nzc))
        fx[1:nxc] = np.abs(np.diff(vel, axis=0)) / np.minimum(vel[1:], vel[:-1]) > force_jump
        cfx = np.concatenate([np.zeros((nxc + 1, 1)), np.cumsum(fx, axis=1)], axis=1)
        fz = np.zeros((nxc, nzc + 1))
        fz[:, 1:nzc] = np.abs(np.diff(vel, axis=1)) / np.minimum(vel[:, 1:], vel[:, :-1]) > force_jump
        cfz = np.concatenate([np.zeros((1, nzc + 1)), np.cumsum(fz, axis=0)], axis=0)

    def has_strong_edge(i0, i1, k0, k1):
        if force_jump is None:
            return False
        lo, hi = i0 + mcx, i1 - mcx
        if hi > lo and np.any((cfx[lo:hi, k1] - cfx[lo:hi, k0]) / (k1 - k0) >= edge_coverage):
            return True
        lo, hi = k0 + mcz, k1 - mcz
        return bool(hi > lo and np.any((cfz[i1, lo:hi] - cfz[i0, lo:hi]) / (i1 - i0) >= edge_coverage))

    def edge_candidates(axis, i0, i1, k0, k1):
        if edge_cuts <= 0:
            return []
        if axis == 0:
            lo, hi = i0 + mcx, i1 - mcx
            if hi <= lo:
                return []
            frac = (cjx[lo:hi, k1] - cjx[lo:hi, k0]) / (k1 - k0)
        else:
            lo, hi = k0 + mcz, k1 - mcz
            if hi <= lo:
                return []
            frac = (cjz[i1, lo:hi] - cjz[i0, lo:hi]) / (i1 - i0)
        idx = np.where(frac >= edge_coverage)[0]
        if not len(idx):
            return []
        idx = idx[np.argsort(-frac[idx], kind="stable")][:edge_cuts]
        return [int(lo + j) for j in idx]

    def leaf_cost(i0, i1, k0, k1, h, vmax):
        nx = int(np.ceil((i1 - i0) * dx / h - 1e-9))
        nz = int(np.ceil((k1 - k0) * dz / h - 1e-9))
        inner = nx * ((k0 > 0) + (k1 < nzc)) + nz * ((i0 > 0) + (i1 < nxc))      # interface edges
        c = nx * nz + 0.5 * lam * inner
        hs = min((i1 - i0) * dx / nx, (k1 - k0) * dz / nz)
        if rank == "lts":
            c *= hv * vmax / hs
        if dt_min is not None and C * hs / vmax < dt_min:
            c = np.inf
        return c, nx, nz

    def solve(i0, i1, k0, k1):
        key = (i0, i1, k0, k1)
        r = memo.get(key)
        if r is not None:
            return r
        cx = (i1 - i0) >= 2 * mcx
        cz = (k1 - k0) >= 2 * mcz
        best = None
        if cx:
            for im in dict.fromkeys([(i0 + i1) // 2] + edge_candidates(0, i0, i1, k0, k1)):
                a, b = solve(i0, im, k0, k1), solve(im, i1, k0, k1)
                hm, vm = min(a[1], b[1]), max(a[2], b[2])
                if best is None or a[0] + b[0] < best[0]:
                    best = (a[0] + b[0], ("x", im))
        if cz:
            for km in dict.fromkeys([(k0 + k1) // 2] + edge_candidates(1, i0, i1, k0, k1)):
                a, b = solve(i0, i1, k0, km), solve(i0, i1, km, k1)
                hm, vm = min(a[1], b[1]), max(a[2], b[2])
                if best is None or a[0] + b[0] < best[0]:
                    best = (a[0] + b[0], ("z", km))
        if not (cx or cz):
            hm = float(hreq[i0:i1, k0:k1].min())
            vm = float(vel[i0:i1, k0:k1].max())
        lc, _, _ = leaf_cost(i0, i1, k0, k1, hm, vm)
        if best is not None and has_strong_edge(i0, i1, k0, k1):
            lc = np.inf                                   # a strong edge must lie on a block boundary
        if best is None or lc <= best[0]:
            best = (lc, None)
        r = (best[0], hm, vm, best[1])
        memo[key] = r
        return r

    if z_cuts == "auto":
        cand = sharp_interfaces(vel)
        min_gap = int(np.ceil(hmin / dz))
        z_cuts, last = [], 0
        for k in cand:
            if k - last >= min_gap and nzc - k >= min_gap:
                z_cuts.append(k)
                last = k
    slabs = [0] + sorted(int(k) for k in (z_cuts or []) if 0 < k < nzc) + [nzc]
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10000))
    try:
        total = sum(solve(0, nxc, ka, kb)[0] for ka, kb in zip(slabs[:-1], slabs[1:]))
    finally:
        sys.setrecursionlimit(old)
    if not np.isfinite(total):
        raise ValueError(f"dt_min = {dt_min} s inaccessible : aucun pavage n'a tous ses blocs à ce pas "
                         "(le pas d'une grille uniforme au même nombre de points par longueur d'onde l'est)")

    # collect leaves
    leaves, stack = [], [(0, nxc, ka, kb) for ka, kb in zip(slabs[:-1], slabs[1:])]
    while stack:
        i0, i1, k0, k1 = stack.pop()
        _, hm, vm, cut = memo[(i0, i1, k0, k1)]
        if cut is None:
            leaves.append((i0, i1, k0, k1, hm, vm))
        elif cut[0] == "x":
            stack += [(i0, cut[1], k0, k1), (cut[1], i1, k0, k1)]
        else:
            stack += [(i0, i1, k0, cut[1]), (i0, i1, cut[1], k1)]

    rho_a = np.broadcast_to(np.asarray(1000.0 if rho is None else rho, float), vel.shape)
    comp = 1.0 / (rho_a * vel ** 2)          # 1 / (rho v^2), mass integrand
    buoy = 1.0 / rho_a                       # 1 / rho, stiffness integrand
    patches = []
    for i0, i1, k0, k1, hm, vm in sorted(leaves, key=lambda t: (t[2], t[0])):
        _, nx, nz = leaf_cost(i0, i1, k0, k1, hm, vm)
        X0, X1 = x0 + i0 * dx, x0 + i1 * dx
        Z0, Z1 = z0 + k0 * dz, z0 + k1 * dz
        ex = np.linspace(X0, X1, nx + 1)
        ez = np.linspace(Z0, Z1, nz + 1)
        Wx = _overlap(ex, x0 + dx * np.arange(i0, i1 + 1))
        Wz = _overlap(ez, z0 + dz * np.arange(k0, k1 + 1))
        area = np.outer(Wx.sum(1), Wz.sum(1))
        c = Wx @ comp[i0:i1, k0:k1] @ Wz.T / area
        b = Wx @ buoy[i0:i1, k0:k1] @ Wz.T / area
        r_e = 1.0 / b
        v_e = np.sqrt(b / c)
        p = Patch(X0, X1, Z0, Z1, nx, nz, v_e, r_e)
        p.h_target = hm
        p.dt_est = C * min(p.hx, p.hz) / vm
        patches.append(p)
    ne = sum(p.n_elements for p in patches)
    dtg = min(p.dt_est for p in patches)
    ideal = float(np.sum(dx * dz / hreq ** 2))
    hs = np.array([p.h_target for p in patches])
    stats = dict(n_blocks=len(patches), n_elements=ne, ideal=ideal, overhead=ne / ideal,
                 h_min=float(hs.min()), h_max=float(hs.max()), n_tree_nodes=len(memo),
                 min_block_cells=(mcx, mcz), order=order, fmax=fmax, ppw=ppw, rank=rank,
                 z_cuts=[z0 + k * dz for k in slabs[1:-1]],
                 dt_global=dtg, work_global=ne / dtg, work_lts=sum(p.n_elements / p.dt_est for p in patches))
    return BlockPlan(patches, ne, ideal, stats)


def plan_blocks_global(vel, origin, spacing, order=4, f0=10.0, ppw=5.0, rho=None,
                       factors=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7), **kw):
    """Pavage pour un pas de temps global : pour chaque pas visé f x dt de la grille uniforme au même
    échantillonnage, pavage de moindre coût dont tous les blocs tiennent ce pas (``dt_min``) ; retient
    celui de moindre travail éléments / dt_global. stats["dt_factor"] donne le facteur retenu."""
    dt_u = uniform_plan(vel, origin, spacing, order, f0, ppw, rho).stats["dt_global"]
    best = None
    for f in factors:
        try:
            p = plan_blocks(vel, origin, spacing, order, f0, ppw, rho, dt_min=f * dt_u, **kw)
        except ValueError:
            continue
        if best is None or p.stats["work_global"] < best.stats["work_global"]:
            best = p
            best.stats["dt_factor"] = f
    if best is None:
        raise ValueError("aucun pas visé accessible")
    return best


def uniform_plan(vel, origin, spacing, order=4, f0=10.0, ppw=5.0, rho=None, fmax=None):
    """Référence : une seule grille régulière à la plus petite taille requise du modèle."""
    vel = np.asarray(vel, float)
    nxc, nzc = vel.shape
    big = 10 * max(nxc, nzc)
    return plan_blocks(vel, origin, spacing, order, f0, ppw, rho, min_cells=big, fmax=fmax, z_cuts=None)


def grid_sampler(vel, origin, spacing, rho=None, mode="bilinear"):
    """Fonction (x, z) -> (vp, rho) aux nœuds GLL pour :class:`Mortar2D` (argument ``nodal``).

    Grille (nx, nz) à cellules centrées, origin = coin (bord de la première cellule).
    mode "bilinear" (défaut) : interpolation bilinéaire des échantillons placés aux centres des
    cellules (modèle continu) ; "cell" : valeur de la cellule contenant le point (modèle en marches).
    La lecture "cell" crée un petit saut à chaque bord de cellule : des nœuds placés n'importe où dans
    les cellules voient alors un modèle localement décalé, et l'erreur ne converge plus tant que les
    éléments ne sont pas calés sur les cellules (Marmousi : 1,6 à 7,2 % selon le maillage, contre une
    convergence régulière avec "bilinear")."""
    vel = np.asarray(vel, float)
    nx, nz = vel.shape
    x0, z0 = origin
    dx, dz = spacing
    r = np.broadcast_to(np.asarray(1000.0 if rho is None else rho, float), vel.shape)

    def cell(x, z):
        i = np.clip(((x - x0) / dx).astype(int), 0, nx - 1)
        k = np.clip(((z - z0) / dz).astype(int), 0, nz - 1)
        return vel[i, k], r[i, k]

    def bilinear(x, z):
        u = np.clip((x - x0) / dx - 0.5, 0, nx - 1)
        w = np.clip((z - z0) / dz - 0.5, 0, nz - 1)
        i = np.minimum(u.astype(int), nx - 2) if nx > 1 else np.zeros(u.shape, int)
        k = np.minimum(w.astype(int), nz - 2) if nz > 1 else np.zeros(w.shape, int)
        a, b = u - i, w - k
        i1, k1 = np.minimum(i + 1, nx - 1), np.minimum(k + 1, nz - 1)

        def interp(f):
            return ((1 - a) * (1 - b) * f[i, k] + a * (1 - b) * f[i1, k] + (1 - a) * b * f[i, k1]
                    + a * b * f[i1, k1])
        return interp(vel), interp(r)

    return bilinear if mode == "bilinear" else cell
