#!/usr/bin/env python3
"""Corps rapide à bord courbe (disque) : maillage hybride ajusté au contour contre blocs en escalier.

Disque de vitesse v_in dans un fond v_bg. Quatre calculs (prototypes NumPy, même pas de temps) :
- hybride : zones SEM régulières (fond d'ordre 4, intérieur du disque d'ordre 6 : p-adaptivité à travers
  l'interface DG) et bande de triangles DG d'ordre 3 dont les arêtes suivent le cercle ;
- blocs : pavage makutu.mortar.plan_blocks du disque échantillonné sur une grille (bord en escalier) ;
- uniforme : une grille à la plus petite taille requise ;
- référence : le maillage hybride, toutes tailles divisées par 2,5.
Tailles : h = N v / (fmax ppw) dans chaque zone SEM ; triangles : 4 v / (fmax ppw).

    python apps/mortar/hybrid_disk_2d.py --out tests/figures/hybrid
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from makutu.mortar import Mortar2D, grid_sampler, plan_blocks_global, ricker, uniform_plan
from makutu.mortar.hybrid2d import Hybrid2D, QuadZone, TriZone, draw_hybrid, triangulate_ring

X, Z, CX, CZ, R = 4000.0, 3000.0, 2000.0, 1500.0, 600.0


def hybrid(v_bg, v_in, fmax, ppw, refine=1.0, p_bg=4, p_in=6, p_tri=3, margin=1.0):
    h_bg = p_bg * v_bg / (fmax * ppw) / refine
    h_in = p_in * v_in / (fmax * ppw) / refine
    t_bg = 4 * v_bg / (fmax * ppw) / refine                  # triangle edge, background side
    t_in = 4 * v_in / (fmax * ppw) / refine
    m_out = margin * 1.5 * t_bg
    m_in = margin * 1.5 * t_in
    o = (CX - R - m_out, CX + R + m_out, CZ - R - m_out, CZ + R + m_out)
    a = R / np.sqrt(2) - m_in
    i = (CX - a, CX + a, CZ - a, CZ + a)
    nc = int(round(2 * np.pi * R / min(t_bg, t_in)))
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    circ = np.stack([CX + R * np.cos(th), CZ + R * np.sin(th)], 1)

    def size(x, z):
        r = np.hypot(x - CX, z - CZ)
        return t_bg if r >= R else min(t_in, t_bg + 0.7 * (R - r))

    nodes, tris = triangulate_ring(o, i, circ, size)
    cen = nodes[tris].mean(1)
    inside = np.hypot(cen[:, 0] - CX, cen[:, 1] - CZ) < R
    n = lambda L, h: max(1, int(np.ceil(L / h - 1e-9)))
    bg, inn = (v_bg, 1000.0), (v_in, 1000.0)
    zones = [QuadZone(0, X, 0, o[2], n(X, h_bg), n(o[2], h_bg), p_bg, bg),
             QuadZone(0, X, o[3], Z, n(X, h_bg), n(Z - o[3], h_bg), p_bg, bg),
             QuadZone(0, o[0], o[2], o[3], n(o[0], h_bg), n(o[3] - o[2], h_bg), p_bg, bg),
             QuadZone(o[1], X, o[2], o[3], n(X - o[1], h_bg), n(o[3] - o[2], h_bg), p_bg, bg),
             QuadZone(i[0], i[1], i[2], i[3], n(2 * a, h_in), n(2 * a, h_in), p_in, inn),
             TriZone(nodes, tris, order=p_tri, vp=np.where(inside, v_in, v_bg), rho=1000.0)]
    return Hybrid2D(zones, sponge=400.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v-bg", type=float, default=2000.0)
    ap.add_argument("--v-in", type=float, default=4000.0)
    ap.add_argument("--f0", type=float, default=8.0)
    ap.add_argument("--ppw", type=float, default=5.0)
    ap.add_argument("--tmax", type=float, default=1.6)
    ap.add_argument("--cell", type=float, default=10.0, help="pas de la grille du modèle pour les blocs (m)")
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args()
    fmax = 2.5 * a.f0
    nx, nz = int(X / a.cell), int(Z / a.cell)
    xc, zc = (np.arange(nx) + 0.5) * a.cell, (np.arange(nz) + 0.5) * a.cell
    XX, ZZ = np.meshgrid(xc, zc, indexing="ij")
    v = np.where(np.hypot(XX - CX, ZZ - CZ) < R, a.v_in, a.v_bg)
    nodal = grid_sampler(v, (0, 0), (a.cell, a.cell))
    kw = dict(order=4, nodal=nodal, nodal_inset=0.5 * a.cell, sponge=400.0)
    cases = {}
    t = time.time(); cases["hybride"] = hybrid(a.v_bg, a.v_in, fmax, a.ppw); print("hybride", time.time() - t, flush=True)
    plan = plan_blocks_global(v, (0, 0), (a.cell, a.cell), order=4, f0=a.f0, ppw=a.ppw)
    cases["blocs (escalier)"] = Mortar2D(plan.patches, **kw)
    cases["uniforme"] = Mortar2D(uniform_plan(v, (0, 0), (a.cell, a.cell), order=4, f0=a.f0, ppw=a.ppw).patches, **kw)
    cases["référence"] = hybrid(a.v_bg, a.v_in, fmax, a.ppw, refine=2.5)
    info = {}
    for k, m in cases.items():
        info[k] = dict(elements=m.n_elements, ndof=m.ndof, dt_max=m.dt_max())
        if k in ("hybride", "référence"):
            info[k]["triangles"] = m.zones[-1].n_elements
            info[k]["dt_zones"] = [m.zone_dt_max(i) for i in range(len(m.zones))]
        info[k]["work"] = m.K.nnz / (0.9 * info[k]["dt_max"])
        print(f"{k:18s} {m.n_elements:6d} éléments {m.ndof:7d} inconnues dt_max {info[k]['dt_max'] * 1e3:.3f} ms", flush=True)
    dt = 0.8 * min(i["dt_max"] for i in info.values())
    nt = int(a.tmax / dt)
    w = ricker(np.arange(nt) * dt, a.f0)
    src = (700.0, 1500.0)
    recs = [(x, 600.0) for x in np.arange(500.0, 3501.0, 50.0)] + [(3300.0, z) for z in np.arange(600.0, 2401.0, 50.0)]
    nrx, nrz = 400, 300
    snap = int(0.62 * nt)
    res = {}
    for k, m in cases.items():
        t = time.time()
        res[k] = m.run(dt, nt, src, w, recs, snapshots=[snap], raster=m.raster(nrx, nrz))
        print(f"   {k:18s} {time.time() - t:.0f} s", flush=True)
    ref = res["référence"]
    for k in cases:
        if k == "référence":
            continue
        info[k]["err_traces"] = float(np.linalg.norm(res[k]["traces"] - ref["traces"]) / np.linalg.norm(ref["traces"]))
        info[k]["err_snapshot"] = float(np.linalg.norm(res[k]["snapshots"][snap] - ref["snapshots"][snap])
                                        / np.linalg.norm(ref["snapshots"][snap]))
        print(f"   {k:18s} écart traces {100 * info[k]['err_traces']:.2f} %  instantané {100 * info[k]['err_snapshot']:.2f} %", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"disque_{a.v_in:.0f}_dans_{a.v_bg:.0f}"
    (a.out / f"{tag}.json").write_text(json.dumps(dict(dt=dt, f0=a.f0, ppw=a.ppw, cases=info), indent=1, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from makutu.mortar.plots2d import draw_blocks
    fig, ax = plt.subplots(3, 3, figsize=(19, 14))
    pc = draw_hybrid(ax[0, 0], cases["hybride"].zones, vmin=min(a.v_bg, a.v_in), vmax=max(a.v_bg, a.v_in))
    h = info["hybride"]
    ax[0, 0].set_title(f"hybride : {h['elements']} éléments dont {h['triangles']} triangles DG (ordre 3) ;\n"
                       "SEM ordre 4 (fond), ordre 6 (intérieur) ; traits noirs : zones", fontsize=9)
    draw_hybrid(ax[0, 1], cases["hybride"].zones, vmin=min(a.v_bg, a.v_in), vmax=max(a.v_bg, a.v_in), lw=0.5)
    ax[0, 1].set_xlim(CX - R - 250, CX + R + 250); ax[0, 1].set_ylim(CZ + R + 250, CZ - R - 250)
    ax[0, 1].set_title("zoom : triangles ajustés au cercle, grilles régulières ailleurs", fontsize=9)
    draw_blocks(ax[0, 2], plan.patches, v, (0, X, Z, 0), edges=True, fine_width=0.3)
    ax[0, 2].set_aspect("equal")
    ax[0, 2].set_title(f"blocs (escalier) : {info['blocs (escalier)']['elements']} éléments, {len(plan.patches)} blocs", fontsize=9)
    ext = (0, X, Z, 0)
    sr = ref["snapshots"][snap].reshape(nrx, nrz)
    c = np.percentile(np.abs(sr), 99.5)
    ax[1, 0].imshow(sr.T, extent=ext, cmap="seismic", vmin=-c, vmax=c, aspect="equal")
    ax[1, 0].add_patch(plt.Circle((CX, CZ), R, fill=False, color="k", lw=0.8))
    ax[1, 0].set_title(f"référence (hybride fin, {info['référence']['elements']} éléments) : instantané t = {snap * dt:.2f} s", fontsize=9)
    for j, k in enumerate(("hybride", "blocs (escalier)")):
        d = (res[k]["snapshots"][snap] - ref["snapshots"][snap]).reshape(nrx, nrz)
        ax[1, 1 + j].imshow(d.T, extent=ext, cmap="seismic", vmin=-c / 5, vmax=c / 5, aspect="equal")
        ax[1, 1 + j].add_patch(plt.Circle((CX, CZ), R, fill=False, color="k", lw=0.8))
        ax[1, 1 + j].set_title(f"{k} - référence (échelle x5) : {100 * info[k]['err_snapshot']:.2f} %", fontsize=9)
    tt = np.arange(nt) * dt
    g = 0.05 * np.abs(ref["traces"]).max()
    ax[2, 0].imshow(ref["traces"], extent=(0, len(recs), tt[-1], 0), cmap="gray_r", vmin=-g, vmax=g, aspect="auto")
    ax[2, 0].set_title("référence : traces (ligne z = 600 m puis colonne x = 3300 m)", fontsize=9)
    for j, k in enumerate(("hybride", "blocs (escalier)")):
        ax[2, 1 + j].imshow(res[k]["traces"] - ref["traces"], extent=(0, len(recs), tt[-1], 0), cmap="gray_r",
                            vmin=-g / 5, vmax=g / 5, aspect="auto")
        ax[2, 1 + j].set_title(f"{k} - référence (échelle x5) : {100 * info[k]['err_traces']:.2f} %", fontsize=9)
    for axx in ax[2]:
        axx.set_xlabel("récepteur"); axx.set_ylabel("temps (s)")
    lines = ["maillage            éléments  inconnues  dt stable   traces  instantané"]
    for k in ("hybride", "blocs (escalier)", "uniforme"):
        i = info[k]
        lines.append(f"{k:18s} {i['elements']:8d} {i['ndof']:9d} {i['dt_max'] * 1e3:8.3f} ms {100 * i['err_traces']:6.2f} % {100 * i['err_snapshot']:6.2f} %")
    fig.suptitle(f"Disque {a.v_in:.0f} m/s dans {a.v_bg:.0f} m/s, f0 = {a.f0:g} Hz, {a.ppw:g} ppw\n" + "\n".join(lines),
                 fontsize=10, family="monospace")
    fig.tight_layout()
    fig.savefig(a.out / f"{tag}.png", dpi=100)


if __name__ == "__main__":
    main()
