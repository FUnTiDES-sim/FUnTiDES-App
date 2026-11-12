"""Figures de contrôle des pavages en blocs et des tirs du prototype :mod:`makutu.mortar`."""
from __future__ import annotations

import numpy as np


def _segments(patches, max_lines=60000):
    """Arêtes des éléments (traits fins) et bords des blocs (traits épais), en segments (x, z)."""
    fine, thick = [], []
    total = sum(p.nx + p.nz + 2 for p in patches)
    for p in patches:
        thick += [[(p.x0, p.z0), (p.x1, p.z0)], [(p.x1, p.z0), (p.x1, p.z1)],
                  [(p.x1, p.z1), (p.x0, p.z1)], [(p.x0, p.z1), (p.x0, p.z0)]]
        if total > max_lines:
            continue
        for i in range(1, p.nx):
            x = p.x0 + i * p.hx
            fine.append([(x, p.z0), (x, p.z1)])
        for k in range(1, p.nz):
            z = p.z0 + k * p.hz
            fine.append([(p.x0, z), (p.x1, z)])
    return fine, thick


def draw_blocks(ax, patches, vel=None, extent=None, edges=True, cmap="viridis", label="vp (m/s)",
                fine_color="w", fine_width=0.25, block_color="k", block_width=1.0, zoom=None):
    """Modèle de vitesse (si donné) et pavage : bords des blocs et arêtes des éléments."""
    from matplotlib.collections import LineCollection
    im = None
    if vel is not None:
        im = ax.imshow(np.asarray(vel).T, extent=extent, cmap=cmap, aspect="equal", interpolation="nearest")
    fine, thick = _segments(patches)
    if edges and fine:
        ax.add_collection(LineCollection(fine, colors=fine_color, linewidths=fine_width, alpha=0.8))
    ax.add_collection(LineCollection(thick, colors=block_color, linewidths=block_width))
    x0 = min(p.x0 for p in patches); x1 = max(p.x1 for p in patches)
    z0 = min(p.z0 for p in patches); z1 = max(p.z1 for p in patches)
    if zoom is not None:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[3], zoom[2])
    else:
        ax.set_xlim(x0, x1); ax.set_ylim(z1, z0)
    ax.set_xlabel("x (m)"); ax.set_ylabel("profondeur (m)")
    return im


def element_size_image(patches, nx, nz, bbox):
    """Image (nx, nz) de la taille des éléments (max(hx, hz)) aux centres d'une grille régulière."""
    x0, x1, z0, z1 = bbox
    gx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    gz = z0 + (np.arange(nz) + 0.5) * (z1 - z0) / nz
    img = np.full((nx, nz), np.nan)
    for p in patches:
        ix = (gx >= p.x0) & (gx < p.x1)
        iz = (gz >= p.z0) & (gz < p.z1)
        img[np.ix_(ix, iz)] = max(p.hx, p.hz)
    return img


def plot_plan(path, plan, vel, extent, title, zooms=(), hreq=None):
    """Figure de contrôle d'un pavage : vitesse + blocs + éléments, taille des éléments rapportée à la
    taille requise (h / h_req, <= 1 partout), zooms éventuels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x0, x1, z1, z0 = extent
    W, H = x1 - x0, z1 - z0
    nrow = 2 + len(zooms)
    fig_w = 13
    fig, axes = plt.subplots(nrow, 1, figsize=(fig_w, nrow * fig_w * H / W * 1.05 + 1.2), squeeze=False)
    ax = axes[0, 0]
    im = draw_blocks(ax, plan.patches, vel, extent, edges=True)
    fig.colorbar(im, ax=ax, label="vp (m/s)", shrink=0.9)
    st = plan.stats
    ax.set_title(f"{title}\n{st['n_blocks']} blocs réguliers, {st['n_elements']} éléments "
                 f"(idéal {st['ideal']:.0f}, x{st['overhead']:.2f}) ; traits noirs : blocs, blancs : éléments")
    ax = axes[1, 0]
    nxi, nzi = vel.shape
    hs = element_size_image(plan.patches, nxi, nzi, (x0, x1, z0, z1))
    if hreq is not None:
        im2 = ax.imshow((hs / hreq).T, extent=extent, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="equal",
                        interpolation="nearest")
        fig.colorbar(im2, ax=ax, label="h / h_req (<= 1 : critère respecté)", shrink=0.9)
        ax.set_title("taille des éléments rapportée à la taille requise par la vitesse locale "
                     "(1 : maille la plus grande permise)")
    else:
        im2 = ax.imshow(hs.T, extent=extent, cmap="magma_r", aspect="equal", interpolation="nearest")
        fig.colorbar(im2, ax=ax, label="taille des éléments (m)", shrink=0.9)
    draw_blocks(ax, plan.patches, None, extent, edges=False, block_width=0.6)
    for r, z in enumerate(zooms):
        ax = axes[2 + r, 0]
        draw_blocks(ax, plan.patches, vel, extent, edges=True, fine_width=0.5, zoom=z)
        ax.set_title(f"zoom x {z[0]:.0f}-{z[1]:.0f} m, profondeur {z[2]:.0f}-{z[3]:.0f} m")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
