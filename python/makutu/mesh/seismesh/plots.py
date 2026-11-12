"""Figures de contrôle des maillages seismesh et des tirs (utilisées par les tests et les scripts).

- :func:`plot_quad_mesh` : maillage 2-D (x, profondeur) en trois panneaux, vitesse avec le maillage,
  taille des éléments, taille / taille requise (au plus 1 : échantillonnage atteint) ;
- :func:`plot_hex_mesh` : même chose pour un maillage 3-D, sur une coupe y = constante (faces des
  hexaèdres dans le plan s'il en existe, sinon centres des éléments proches du plan) ;
- :func:`plot_traces` : traces comparées (référence / calcul, même normalisation) et écart relatif.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

_QUAD_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
_HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
_HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def element_sizes(nodes, cells):
    """Plus grande arête de chaque élément (quadrangles ou hexaèdres, ordre gmsh)."""
    X = np.asarray(nodes)[np.asarray(cells)]
    edges = _QUAD_EDGES if X.shape[1] == 4 else _HEX_EDGES
    return np.max([np.linalg.norm(X[:, b] - X[:, a], axis=-1) for a, b in edges], axis=0)


def required_sizes(grid, h, nodes, cells):
    """Taille requise de chaque élément : minimum de ``h`` sur les cellules recouvertes par sa boîte."""
    X = np.asarray(nodes)[np.asarray(cells)]
    if X.shape[-1] == 2:                                    # on the real quadrangle (as the evaluation)
        from .elements2d import element_stats_2d
        return element_stats_2d(grid, nodes, cells, values=h)["min"]
    lo, hi = X.min(axis=1), X.max(axis=1)
    if X.shape[-1] == 2:                                    # (x, depth) -> (x, y0, depth)
        y0 = np.full((len(X), 1), grid.origin[1])
        lo, hi = np.hstack([lo[:, :1], y0, lo[:, 1:]]), np.hstack([hi[:, :1], y0, hi[:, 1:]])
    a, b = grid.box_index(lo, hi)
    return np.array([h[a[e, 0]:b[e, 0] + 1, a[e, 1]:b[e, 1] + 1, a[e, 2]:b[e, 2] + 1].min() for e in range(len(X))])


_AXIS_LABELS = {0: ("y (m)", "profondeur (m)"), 1: ("x (m)", "profondeur (m)"), 2: ("x (m)", "y (m)")}


def _section(grid, axis=1, value=None):
    """Coupe de la grille de vitesse par le plan coordonnée[axis] = value (défaut : milieu), image
    (lignes = 2e coordonnée restante) et extension pour imshow (profondeur vers le bas)."""
    o, d = np.asarray(grid.origin), np.asarray(grid.spacing)
    i = grid.shape[axis] // 2 if value is None else int(np.clip((value - grid.extent[axis][0]) // d[axis], 0,
                                                                      grid.shape[axis] - 1))
    v = np.take(grid.vel, i, axis=axis).T
    a, b = [k for k in range(3) if k != axis]
    (a0, a1), (b0, b1) = grid.extent[a], grid.extent[b]
    return v, [a0, a1, b1, b0]


def _three_panels(path, polys, vp_img, extent, size, ratio, title, lines=None,
                  labels=("x (m)", "profondeur (m)")):
    polys = list(polys)
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2))
    im = axs[0].imshow(vp_img, extent=extent, cmap="viridis", aspect="equal")
    axs[0].add_collection(LineCollection(lines if lines is not None else [np.vstack([Q, Q[:1]]) for Q in polys],
                                         colors="w", linewidths=0.3))
    fig.colorbar(im, ax=axs[0], shrink=0.75, label="vp (m/s)")
    axs[0].set_title("modèle et maillage")
    for ax, vals, cmap, lab, clim in [(axs[1], size, "viridis", "taille de l'élément (m)", None),
                                      (axs[2], ratio, "RdYlGn_r", "taille / taille requise (≤ 1)", (0, 1.2))]:
        pc = PolyCollection(polys, array=vals, cmap=cmap, edgecolors="k", linewidths=0.15)
        if clim:
            pc.set_clim(*clim)
        elif np.ptp(vals) == 0:
            pc.set_clim(0.9 * vals.min(), 1.1 * vals.max())
        ax.add_collection(pc)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, shrink=0.75, label=lab)
    axs[1].set_title(f"taille des éléments : {size.min():.0f} à {size.max():.0f} m")
    axs[2].set_title(f"taille / taille requise : max {ratio.max():.2f}")
    for ax in axs:
        ax.set_xlabel(labels[0])
    axs[0].set_ylabel(labels[1])
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_quad_mesh(path, grid, h, nodes, quads, title=""):
    """Maillage 2-D : nœuds (x, profondeur), quadrangles ; ``h`` taille requise de la grille."""
    nodes, quads = np.asarray(nodes), np.asarray(quads)
    size = element_sizes(nodes, quads)
    ratio = size / required_sizes(grid, h, nodes, quads)
    vp, ext = _section(grid)
    _three_panels(path, nodes[quads], vp, ext, size, ratio,
                  f"{title} — {len(quads)} quadrangles, taille / taille requise max {ratio.max():.2f}")
    return size, ratio


def hex_section(nodes, hexes, y, axis=1):
    """Coupe d'un maillage d'hexaèdres par le plan ``coordonnée[axis] = y`` : (polygones des traces des
    éléments coupés dans les deux autres coordonnées, dans l'ordre (x, z), (y, z) ou (x, y) ; indices
    de ces éléments). Le plan est légèrement décalé s'il passe par des nœuds."""
    nodes, hexes = np.asarray(nodes, float), np.asarray(hexes)
    keep = [a for a in range(3) if a != axis]
    span = np.ptp(nodes[:, axis]) if np.ptp(nodes[:, axis]) > 0 else 1.0
    if np.any(np.abs(nodes[:, axis] - y) < 1e-9 * span):
        y = y + 1e-4 * span
    X = nodes[hexes]
    ymin, ymax = X[:, :, axis].min(1), X[:, :, axis].max(1)
    cut = np.where((ymin < y) & (ymax > y))[0]
    E = np.array(_HEX_EDGES)
    A, B = X[cut][:, E[:, 0]], X[cut][:, E[:, 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (y - A[:, :, axis]) / (B[:, :, axis] - A[:, :, axis])
    ok = (t >= 0) & (t <= 1) & np.isfinite(t)
    pts = A + np.where(ok, t, 0.0)[..., None] * (B - A)
    polys, ids = [], []
    for k in range(len(cut)):
        q = np.unique(np.round(pts[k][ok[k]][:, keep], 6), axis=0)
        if len(q) < 3:
            continue
        c = q.mean(0)
        polys.append(q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))])
        ids.append(cut[k])
    return polys, np.array(ids, dtype=int)


def plot_hex_mesh(path, grid, h, nodes, hexes, title="", y=None, axis=1):
    """Maillage 3-D : nœuds (x, y, profondeur), hexaèdres ; coupe exacte par le plan
    ``coordonnée[axis] = y`` (défaut : y au milieu ; ``axis=2`` : coupe horizontale à la profondeur y)."""
    nodes, hexes = np.asarray(nodes), np.asarray(hexes)
    lo_hi = grid.extent[axis]
    y = 0.5 * (lo_hi[0] + lo_hi[1]) if y is None else y
    polys, ids = hex_section(nodes, hexes, y, axis)
    size = element_sizes(nodes, hexes)
    ratio = size / required_sizes(grid, h, nodes, hexes)
    vp, ext = _section(grid, axis, y)
    name = "xyz"[axis] if axis != 2 else "profondeur"
    _three_panels(path, polys, vp, ext, size[ids], ratio[ids],
                  f"{title} — {len(hexes)} hexaèdres ({len(ids)} coupés par {name} = {y:.0f} m), taille / taille "
                  f"requise max {ratio.max():.2f}", labels=_AXIS_LABELS[axis])
    return size, ratio


def plot_traces(path, ref, data, dt, labels=None, title="", names=("référence", "calcul")):
    """Traces (n_t, n_r) comparées, normalisées par trace sur la référence, écart relatif par trace."""
    ref, data = np.asarray(ref, float), np.asarray(data, float)
    t = np.arange(len(ref)) * dt
    diff = np.linalg.norm(data - ref, axis=0) / np.maximum(np.linalg.norm(ref, axis=0), 1e-30)
    n = ref.shape[1]
    fig, ax = plt.subplots(figsize=(10, 1.0 + 0.55 * n))
    for j in range(n):
        s = 0.45 / max(np.abs(ref[:, j]).max(), 1e-30)
        ax.plot(t, j + ref[:, j] * s, "k", lw=1.4, label=names[0] if j == 0 else None)
        ax.plot(t, j + data[:, j] * s, "r--", lw=1.0, label=names[1] if j == 0 else None)
        lab = labels[j] if labels is not None else f"trace {j}"
        ax.text(t[-1], j + 0.2, f"{lab} : {100 * diff[j]:.1f} %", ha="right", fontsize=7.5)
    ax.set_yticks([])
    ax.set_xlabel("t (s)")
    ax.legend(loc="lower left", fontsize=8)
    glob = np.linalg.norm(data - ref) / max(np.linalg.norm(ref), 1e-30)
    ax.set_title(f"{title} — écart global {100 * glob:.1f} %", fontsize=10)
    _save(fig, path)
    return diff


def read_funtides_msh(msh):
    """(nœuds (x, y, hauteur), cellules, numéros de groupe) d'un fichier MSH 2.2 écrit par seismesh."""
    lines = Path(msh).read_text().split("\n")
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    nodes = np.array([[float(v) for v in lines[i + 2 + k].split()[1:]] for k in range(n)])
    j = lines.index("$Elements")
    rows = [[int(v) for v in lines[j + 2 + k].split()] for k in range(int(lines[j + 1]))]
    kind = 5 if any(r[1] == 5 for r in rows) else 3
    cells = np.array([np.array(r[3 + r[2]:]) - 1 for r in rows if r[1] == kind])
    tags = np.array([r[3] for r in rows if r[1] == kind])
    return nodes, cells, tags


def plot_mesh_file(path, msh, grid, h, title="", y=None, axis=1):
    """Figure de contrôle d'un maillage exporté par seismesh (``mesh.msh``), profondeur reconstruite."""
    nodes, cells, _ = read_funtides_msh(msh)
    depth = grid.extent[2][1] - nodes[:, 2]                 # height = max depth - depth
    if cells.shape[1] == 4:
        return plot_quad_mesh(path, grid, h, np.stack([nodes[:, 0], depth], -1), cells, title)
    return plot_hex_mesh(path, grid, h, np.stack([nodes[:, 0], nodes[:, 1], depth], -1), cells, title, y, axis)


def plot_gmsh_mesh(path, mesh, title=""):
    """Maillage d'un :class:`makutu.mesh.GmshMesh` (sans modèle SEP) : 2-D, éléments colorés par taille ;
    3-D, faces des hexaèdres dans le plan y du milieu s'il en existe."""
    C = mesh._corners
    if mesh.dim == 2:
        polys = C[:, [0, 1, 3, 2], :]
    else:                                                  # exact section by the middle y plane
        G = C[:, [0, 1, 3, 2, 4, 5, 7, 6], :]              # vertex bits -> gmsh hexahedron order
        polys, _ = hex_section(G.reshape(-1, 3), np.arange(G.shape[0] * 8).reshape(-1, 8),
                               0.5 * (mesh.lower[1] + mesh.upper[1]))
    size = np.array([np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1).max() for Q in polys])
    fig, ax = plt.subplots(figsize=(9, 5))
    pc = PolyCollection(polys, array=size, cmap="magma", edgecolors="k", linewidths=0.2)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, shrink=0.8, label="taille de l'élément (m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m), vers le haut")
    ax.set_title(f"{title} — {mesh.n_elements} éléments" + ("" if mesh.dim == 2 else ", coupe au milieu en y"), fontsize=10)
    _save(fig, path)


def plot_quad_mesh_detail(path, grid, h, nodes, quads, title="", zoom=None):
    """Maillage 2-D lisible pour les grands modèles : (1) modèle et maillage, (2) zoom (cadre rouge) sur
    les arêtes des éléments, fond de vitesse, (3) taille des éléments en couleurs distinctes (légende :
    taille et nombre d'éléments), (4) taille / taille requise. ``zoom`` = (x0, x1, z0, z1)."""
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.colors import ListedColormap, BoundaryNorm
    nodes, quads = np.asarray(nodes), np.asarray(quads)
    P = nodes[quads]
    size = element_sizes(nodes, quads)
    ratio = size / required_sizes(grid, h, nodes, quads)
    vp, ext = _section(grid)
    (x0, x1), _, (z0, z1) = grid.extent
    if zoom is None:
        cx, cz, w = 0.5 * (x0 + x1), 0.5 * (z0 + z1), 0.25 * (x1 - x0)
        zoom = (cx - w / 2, cx + w / 2, cz - w / 4, cz + w / 4)
    levels = np.unique(np.round(size, -1 if size.max() > 50 else 0))
    discrete = len(levels) <= 8
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1.0], height_ratios=[1, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(vp, extent=ext, cmap="viridis", aspect="equal")
    # edges drawn only when they stay readable at this scale (otherwise a moiré fakes a coarse grid)
    width_px = 16 * 100 * 0.6
    if (x1 - x0) / np.median(size) < 0.25 * width_px:
        ax.add_collection(LineCollection([np.vstack([Q, Q[:1]]) for Q in P], colors="w", linewidths=0.15, alpha=0.6))
        ax.set_title(f"modèle et maillage ({len(quads)} quadrangles) ; cadre rouge : zoom")
    else:
        ax.set_title(f"modèle ({len(quads)} quadrangles, trop nombreux pour être tracés ici : voir le zoom) ; cadre rouge : zoom")
    ax.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False, ec="r", lw=2))
    fig.colorbar(im, ax=ax, shrink=0.8, label="vp (m/s)")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(vp, extent=ext, cmap="viridis", aspect="equal")
    ax.add_collection(LineCollection([np.vstack([Q, Q[:1]]) for Q in P], colors="k", linewidths=0.5))
    ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[3], zoom[2]); ax.set_title("zoom : éléments sur le modèle")
    ax = fig.add_subplot(gs[1, 1])
    if discrete:
        idx = np.argmin(np.abs(np.round(size, -1 if size.max() > 50 else 0)[:, None] - levels[None]), 1)
        cols = plt.cm.tab10(np.arange(len(levels)) % 10)
        pc = PolyCollection(P, facecolors=cols[idx], edgecolors="k", linewidths=0.3)
        ax.add_collection(pc)
        ax.legend(handles=[Patch(color=cols[k], label=f"{levels[k]:.0f} m : {np.sum(idx == k)} éléments")
                           for k in range(len(levels))], loc="lower right", fontsize=9, framealpha=0.9)
    else:
        pc = PolyCollection(P, array=size, cmap="viridis", edgecolors="k", linewidths=0.3)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, shrink=0.8, label="taille de l'élément (m)")
    ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[3], zoom[2]); ax.set_aspect("equal")
    ax.set_title("zoom : taille des éléments")
    ax = fig.add_subplot(gs[2, 0])
    pc = PolyCollection(P, array=size, cmap="viridis", edgecolors="none")
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, shrink=0.8, label="taille de l'élément (m)")
    ax.set_xlim(x0, x1); ax.set_ylim(z1, z0); ax.set_aspect("equal")
    ax.set_title(f"taille des éléments sur tout le modèle (taille / requise max {ratio.max():.2f})")
    ax = fig.add_subplot(gs[2, 1])                                  # depth profile
    zc = P[:, :, 1].mean(1)
    bins = np.linspace(z0, z1, 41)
    mid = 0.5 * (bins[:-1] + bins[1:])
    zg = grid.axis(2)
    def prof(vals, zs, f):
        return [f(vals[(zs >= a) & (zs < b)]) if np.any((zs >= a) & (zs < b)) else np.nan for a, b in zip(bins[:-1], bins[1:])]
    hreq = h[:, 0, :] if h.ndim == 3 else h
    ax.plot(prof(size, zc, np.median), mid, "b-", lw=2, label="taille médiane des éléments")
    ax.plot(prof(size, zc, np.min), mid, "b:", label="taille minimale des éléments")
    ax.plot([np.nanmin(hreq[:, (zg >= a) & (zg < b)]) if np.any((zg >= a) & (zg < b)) else np.nan for a, b in zip(bins[:-1], bins[1:])],
            mid, "k--", label="taille requise minimale")
    ax.plot([np.nanmedian(hreq[:, (zg >= a) & (zg < b)]) if np.any((zg >= a) & (zg < b)) else np.nan for a, b in zip(bins[:-1], bins[1:])],
            mid, "k:", label="taille requise médiane")
    ax.set_ylim(z1, z0); ax.set_xlabel("taille (m)"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("profil en profondeur")
    for a in fig.axes[:-1]:
        if a.get_label() != "<colorbar>" and not a.get_xlabel():
            a.set_xlabel("x (m)")
        if a.get_label() != "<colorbar>":
            a.set_ylabel("profondeur (m)")
    fig.suptitle(f"{title} — taille / taille requise max {ratio.max():.2f}, médiane {np.median(ratio):.2f}", fontsize=12)
    _save(fig, path)
    return size, ratio
