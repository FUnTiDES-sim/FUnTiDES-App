"""Valeurs d'une grille SEP sur les quadrangles d'un maillage 2-D, sur la géométrie réelle des éléments.

- élément rectangulaire aligné sur les axes : calcul exact sur les cellules recouvertes
  (:meth:`VelocityGrid.box_minmax`, :meth:`VelocityGrid.box_mean`, pondération par le recouvrement) ;
- élément déformé (gabarits de transition, zones de raccord) : quadrature sur l'élément réel, points
  environ un huitième de cellule SEP l'un de l'autre dans l'espace de référence de l'application bilinéaire,
  pondérés par |det J| ; extrêmes sur les cellules contenant ces points.

La boîte englobante d'un élément déformé contient des cellules extérieures à l'élément ; elle ne sert
plus qu'à l'estimation conservative séparée (:func:`element_box_stats_2d`).
"""
from __future__ import annotations

import numpy as np


def _lift(grid, P2):
    """(x, profondeur) -> (x, y0, profondeur)."""
    y0 = grid.origin[1]
    return np.stack([P2[:, 0], np.full(len(P2), y0), P2[:, 1]], -1)


def axis_aligned(X, tol=1e-6):
    """Quadrangles (ne, 4, 2) rectangulaires et alignés sur les axes."""
    E = np.roll(X, -1, 1) - X
    scale = np.abs(E).max(axis=(1, 2)) + 1e-300
    return np.all(np.min(np.abs(E), axis=2) <= tol * scale[:, None], axis=1)


def element_box_stats_2d(grid, nodes, quads, values=None):
    """(min, max) de ``values`` sur la boîte englobante de chaque élément (estimation conservative)."""
    X = np.asarray(nodes)[np.asarray(quads)]
    v = grid.vel if values is None else values
    a, b = grid.box_index(_lift(grid, X.min(1)), _lift(grid, X.max(1)))
    vmin, vmax = np.empty(len(X)), np.empty(len(X))
    for e in range(len(X)):
        blk = v[a[e, 0]:b[e, 0] + 1, a[e, 1]:b[e, 1] + 1, a[e, 2]:b[e, 2] + 1]
        vmin[e], vmax[e] = blk.min(), blk.max()
    return vmin, vmax


def element_stats_2d(grid, nodes, quads, values=None):
    """Statistiques de ``values`` (défaut : la vitesse ; tableau de la forme de ``grid.vel``) sur chaque
    quadrangle : dict ``min``, ``max`` (cellules touchées), ``mean`` (moyenne pondérée par la surface)
    et ``harmonic`` (inverse de la moyenne des inverses : vitesse issue de la moyenne des lenteurs)."""
    nodes, quads = np.asarray(nodes, float), np.asarray(quads)
    v = grid.vel if values is None else values
    X = nodes[quads]
    ne = len(X)
    out = {k: np.empty(ne) for k in ("min", "max", "mean", "harmonic")}
    rect = axis_aligned(X)
    if rect.any():
        lo, hi = _lift(grid, X[rect].min(1)), _lift(grid, X[rect].max(1))
        a, b = grid.box_index(lo, hi)
        vmin, vmax = np.empty(len(lo)), np.empty(len(lo))
        for e in range(len(lo)):
            blk = v[a[e, 0]:b[e, 0] + 1, a[e, 1]:b[e, 1] + 1, a[e, 2]:b[e, 2] + 1]
            vmin[e], vmax[e] = blk.min(), blk.max()
        out["min"][rect], out["max"][rect] = vmin, vmax
        out["mean"][rect] = grid.box_mean(lo, hi, values=v)
        out["harmonic"][rect] = 1.0 / grid.box_mean(lo, hi, values=1.0 / v)
    if (~rect).any():
        v2 = v[:, 0, :]
        dx, dz = grid.spacing[0], grid.spacing[2]
        (x0, _), _, (z0, _) = grid.extent
        idx = np.where(~rect)[0]
        for c0 in range(0, len(idx), 256):                    # chunks of elements (memory)
            ids = idx[c0:c0 + 256]
            Y = X[ids]
            size = np.linalg.norm(np.roll(Y, -1, 1) - Y, axis=2).max()
            n = int(np.clip(np.ceil(8.0 * size / min(dx, dz)), 8, 64))   # points about 1/8 cell apart
            t = (np.arange(n) + 0.5) / n
            u, w = [g.ravel() for g in np.meshgrid(t, t, indexing="ij")]
            N = np.stack([(1 - u) * (1 - w), u * (1 - w), u * w, (1 - u) * w], -1)          # (q, 4)
            dNu = np.stack([-(1 - w), 1 - w, w, -w], -1)
            dNw = np.stack([-(1 - u), -u, u, 1 - u], -1)
            P = np.einsum("qc,ecd->eqd", N, Y)
            Ju, Jw = np.einsum("qc,ecd->eqd", dNu, Y), np.einsum("qc,ecd->eqd", dNw, Y)
            jac = np.abs(Ju[..., 0] * Jw[..., 1] - Ju[..., 1] * Jw[..., 0])                 # (e, q)
            i = np.clip(((P[..., 0] - x0) // dx).astype(int), 0, v2.shape[0] - 1)
            k = np.clip(((P[..., 1] - z0) // dz).astype(int), 0, v2.shape[1] - 1)
            S = v2[i, k]
            W = jac / jac.sum(1, keepdims=True)
            out["min"][ids], out["max"][ids] = S.min(1), S.max(1)
            out["mean"][ids] = (W * S).sum(1)
            out["harmonic"][ids] = 1.0 / (W / S).sum(1)
    return out
