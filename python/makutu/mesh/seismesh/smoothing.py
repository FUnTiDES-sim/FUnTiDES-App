"""Amélioration de la qualité d'un maillage d'hexaèdres ou de quadrangles par déplacement de nœuds.

Méthode : P. M. Knupp, « Achieving finite element mesh quality via optimization of the Jacobian matrix
norm and associated quantities. Part II », Int. J. Numer. Meth. Eng. 48 (2000) 1165-1185, et
« Algebraic mesh quality metrics », SIAM J. Sci. Comput. 23 (2001) 193-218 : à chaque coin d'un
hexaèdre, la matrice jacobienne A = [e1 e2 e3] (arêtes issues du coin) ; le nombre de conditionnement
kappa(A) = |A|_F |A^-1|_F / d (d = 3 ; en 2-D, quadrangles : A = [e1 e2], d = 2) vaut 1 pour un coin à angles droits et arêtes égales et tend vers
l'infini quand le coin s'écrase : la somme des kappa^2 est une barrière qui empêche de retourner
un élément. Le jacobien est régularisé comme dans J. M. Escobar, E. Rodríguez, R. Montenegro,
G. Montero, J. M. González-Yuste, « Simultaneous untangling and smoothing of tetrahedral meshes »,
Comput. Methods Appl. Mech. Engrg. 192 (2003) 2775-2787, pour que l'objectif reste fini. Voir aussi P. M. Knupp, « Hexahedral and tetrahedral mesh untangling », Engineering with
Computers 17 (2001) 261-268.

Seuls les nœuds ``free`` bougent (par exemple ceux qui n'appartiennent qu'aux hexaèdres de transition
des gabarits de Tong et Zhang) : la connectivité ne change pas, la conformité est conservée.
"""
from __future__ import annotations

import numpy as np

# gmsh hexahedron: for each corner, the three neighbours giving a right-handed frame
# quadrangle: for each corner, the two neighbours (counter-clockwise frame)
_CORNER2 = np.array([(0, 1, 3), (1, 2, 0), (2, 3, 1), (3, 0, 2)])
_CORNER = np.array([(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
                    (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)])


def corner_condition(X):
    """kappa^2 (n, 8) des coins d'hexaèdres X (n, 8, 3), +inf si un coin est retourné."""
    c = X[:, _CORNER]
    A = np.stack([c[:, :, k] - c[:, :, 0] for k in (1, 2, 3)], axis=-1)      # (n, 8, 3, 3), columns = edges
    det = np.linalg.det(A)
    with np.errstate(divide="ignore", invalid="ignore"):
        Ai = np.linalg.inv(np.where((det > 0)[..., None, None], A, np.eye(3)))
    k2 = (A ** 2).sum((-1, -2)) * (Ai ** 2).sum((-1, -2)) / 9.0
    return np.where(det > 0, k2, np.inf)


def _objective(x_free, nodes, hexes, free_idx, p, delta):
    """Somme des kappa_delta^p, kappa_delta^2 = |A|^2 |cof A|^2 / (9 h(det)^2), h(d) = (d + sqrt(d^2 + 4 delta^2)) / 2
    (Escobar et al. 2003) : égal à kappa^2 pour un jacobien positif grand devant delta, fini et dérivable
    partout (la recherche linéaire peut traverser un élément retourné sans arrêter l'optimisation)."""
    dim = nodes.shape[1]
    corner = _CORNER if dim == 3 else _CORNER2
    x = nodes.copy()
    x[free_idx] = x_free.reshape(-1, dim)
    X = x[hexes]
    c = X[:, corner]
    A = np.stack([c[:, :, k] - c[:, :, 0] for k in range(1, dim + 1)], axis=-1)   # columns = edges
    det = np.linalg.det(A)
    AtA = np.swapaxes(A, -1, -2) @ A
    nA = np.trace(AtA, axis1=-2, axis2=-1)                                      # |A|_F^2
    if dim == 3:
        nC = 0.5 * (nA ** 2 - (AtA ** 2).sum((-1, -2)))                         # |cof A|_F^2
        dnC = 2 * (nA[..., None, None] * A - A @ AtA)
    else:
        nC, dnC = nA, 2 * A                                                     # 2-D: |cof A| = |A|
    r = np.sqrt(det ** 2 + 4 * delta ** 2)
    hh = 0.5 * (det + r)
    k2 = nA * nC / (dim ** 2 * hh ** 2)
    f = (k2 ** (p / 2)).sum()
    cof = np.swapaxes(np.linalg.inv(np.where((np.abs(det) > 1e-300)[..., None, None], A, np.eye(dim))), -1, -2) \
        * det[..., None, None]                                                  # cofactor matrix = d det / dA
    dk2 = (2 * A * nC[..., None, None] + nA[..., None, None] * dnC) / (dim ** 2 * hh[..., None, None] ** 2) \
        - (2 * k2 / hh * 0.5 * (1 + det / r))[..., None, None] * cof
    G = dk2 * (0.5 * p * k2 ** (p / 2 - 1))[..., None, None]
    g = np.zeros_like(x)
    nb = hexes[:, corner]                                                        # (n, corners, dim + 1) node ids
    for k in range(dim):
        np.add.at(g, nb[:, :, k + 1].ravel(), G[:, :, :, k].reshape(-1, dim))
        np.add.at(g, nb[:, :, 0].ravel(), -G[:, :, :, k].reshape(-1, dim))
    return f, g[free_idx].ravel()


def optimize_nodes(nodes, hexes, free, active=None, p=2.0, maxiter=500, delta_rel=1e-3, verbose=False):
    """Déplace les nœuds ``free`` (masque booléen) pour minimiser la somme des kappa^p aux coins des
    hexaèdres ``active`` (masque, défaut : ceux qui touchent un nœud libre). L-BFGS (scipy), point de
    départ valide requis (jacobiens positifs) ; retourne (nœuds, informations)."""
    from scipy.optimize import minimize
    nodes = np.asarray(nodes, float)
    hexes = np.asarray(hexes)
    free = np.asarray(free, bool)
    if active is None:
        active = free[hexes].any(1)
    H = hexes[active]
    # compact the problem to the nodes of the active hexahedra
    used = np.unique(H)
    local = np.full(len(nodes), -1)
    local[used] = np.arange(len(used))
    sub, Hs = nodes[used], local[H]
    free_idx = np.where(free[used])[0]
    X = sub[Hs]
    delta = delta_rel * float(np.median(np.linalg.norm(X[:, 1] - X[:, 0], axis=1))) ** nodes.shape[1]
    f0, _ = _objective(sub[free_idx].ravel(), sub, Hs, free_idx, p, delta)
    res = minimize(_objective, sub[free_idx].ravel(), args=(sub, Hs, free_idx, p, delta), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10, "maxls": 60, "maxcor": 30})
    out = nodes.copy()
    x = sub.copy()
    dim = nodes.shape[1]
    x[free_idx] = res.x.reshape(-1, dim)
    out[used] = x
    corner = _CORNER if dim == 3 else _CORNER2
    sj_ok = bool(np.all(np.linalg.det(np.stack([x[Hs][:, corner][:, :, k] - x[Hs][:, corner][:, :, 0]
                                                for k in range(1, dim + 1)], axis=-1)) > 0))
    return out, {"valid": sj_ok, "f0": float(f0), "f": float(res.fun), "iterations": int(res.nit), "message": str(res.message),
                 "n_free": int(len(free_idx)), "n_active": int(active.sum())}
