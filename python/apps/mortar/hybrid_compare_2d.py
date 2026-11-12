#!/usr/bin/env python3
"""Maillage hybride automatique (SEM régulier + triangles DG le long des contours nets) contre blocs en
escalier et grille uniforme, sur un modèle SEP 2-D. Prototypes NumPy, même pas de temps pour tous.

Modèle commun : celui que représente le maillage hybride (corps délimité par le contour lissé, fond
prolongé), lu aux nœuds par les blocs et la grille uniforme. Référence : maillage hybride, tailles
divisées par ``--ref-refine``.

    python apps/mortar/hybrid_compare_2d.py --model seg2d --tmax 3 --out tests/figures/hybrid
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from makutu.mortar import Mortar2D, plan_blocks_global, ricker, uniform_plan
from makutu.mortar.hybrid2d import Hybrid2D, draw_hybrid, mesh_cmap
from makutu.mortar.hybrid_plan import plan_cg_hybrid, plan_hybrid

DATA = Path(__file__).resolve().parents[3].parent / "FUnTiDES-data" / "SEP"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="seg2d")
    ap.add_argument("--f0", type=float, default=8.0)
    ap.add_argument("--ppw", type=float, default=5.0)
    ap.add_argument("--order-body", type=int, default=None)
    ap.add_argument("--tri-factor", type=float, default=4.0 / 3.0, help="taille des triangles DG (hybride mortar)")
    ap.add_argument("--cg-tri-factor", type=float, default=4.0 / 3.0, help="taille des triangles DG du cas CG + DG")
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--ref-refine", type=float, default=2.5)
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args()
    from makutu.mesh.seismesh import grid_from_sep
    g = grid_from_sep(str(DATA / a.model / "Vp_xz.H"))
    v = g.vel[:, 0, :].astype(float)
    (x0, x1), _, (z0, z1) = g.extent
    sp_ = (g.spacing[0], g.spacing[2])
    t = time.time()
    zones, info = plan_hybrid(v, (x0, z0), sp_, f0=a.f0, ppw=a.ppw, order_body=a.order_body, tri_factor=a.tri_factor)
    zc, info_c = plan_cg_hybrid(v, (x0, z0), sp_, f0=a.f0, ppw=a.ppw, order_body=a.order_body,
                              tri_factor=a.cg_tri_factor)
    zr, info_r = plan_hybrid(v, (x0, z0), sp_, f0=a.f0, ppw=a.ppw, order_body=a.order_body, refine=a.ref_refine)
    print(f"zones : {time.time() - t:.1f} s", flush=True)
    model = info["model"]
    sponge = min(800.0, 0.12 * (x1 - x0))
    kw = dict(sponge=sponge, sponge_sides=("left", "right", "bottom"))
    cases = {"hybride": Hybrid2D(zones, **kw), "CG + DG": Hybrid2D(zc, **kw), "référence": Hybrid2D(zr, **kw)}
    bp = plan_blocks_global(v, (x0, z0), sp_, order=4, f0=a.f0, ppw=a.ppw)
    up = uniform_plan(v, (x0, z0), sp_, order=4, f0=a.f0, ppw=a.ppw)
    cases["blocs (escalier)"] = Mortar2D(bp.patches, order=4, nodal=model, **kw)
    cases["uniforme"] = Mortar2D(up.patches, order=4, nodal=model, **kw)
    out = {}
    for k, m in cases.items():
        out[k] = dict(elements=m.n_elements, ndof=m.ndof, dt_max=m.dt_max(), nnz=m.K.nnz)
        if k in ("hybride", "CG + DG", "référence"):
            out[k]["triangles"] = sum(z.n_elements for z in m.zones if z.kind == "tri")
        print(f"{k:18s} {m.n_elements:7d} éléments {m.ndof:8d} inconnues dt_max {out[k]['dt_max'] * 1e3:.3f} ms", flush=True)
    dt = 0.8 * min(o["dt_max"] for o in out.values())
    nt = int(a.tmax / dt)
    w = ricker(np.arange(nt) * dt, a.f0)
    src = (x0 + 0.5 * (x1 - x0) + 7.0, z0 + 60.0)
    recs = [(x, z0 + 80.0) for x in np.arange(x0 + 100.0, x1 - 100.0, 25.0)]
    far = np.abs(np.array([r[0] for r in recs]) - src[0]) > 300.0
    nrx = min(600, v.shape[0] * 2); nrz = int(round(nrx * (z1 - z0) / (x1 - x0)))
    snap = int(0.55 * nt)
    res = {}
    for k, m in cases.items():
        t = time.time()
        res[k] = m.run(dt, nt, src, w, recs, snapshots=[snap], raster=m.raster(nrx, nrz))
        out[k]["run_s"] = time.time() - t
        print(f"   {k:18s} {out[k]['run_s']:.0f} s", flush=True)
    ref = res["référence"]
    for k in cases:
        if k == "référence":
            continue
        out[k]["err_traces"] = float(np.linalg.norm((res[k]["traces"] - ref["traces"])[:, far]) / np.linalg.norm(ref["traces"][:, far]))
        s, sr = res[k]["snapshots"][snap], ref["snapshots"][snap]
        out[k]["err_snapshot"] = float(np.linalg.norm(s - sr) / np.linalg.norm(sr))
        print(f"   {k:18s} écart traces {100 * out[k]['err_traces']:.2f} %  instantané {100 * out[k]['err_snapshot']:.2f} %", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"compare_{a.model}.json").write_text(json.dumps(dict(dt=dt, cases=out), indent=1, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = (x0, x1, z1, z0)
    W, H = x1 - x0, z1 - z0
    fig, ax = plt.subplots(3, 4, figsize=(26, 3 * 6.6 * H / W + 4))
    draw_hybrid(ax[0, 0], cases["hybride"].zones, lw=0.2)
    ax[0, 0].set_title(f"hybride : {out['hybride']['elements']} éléments dont {out['hybride']['triangles']} triangles", fontsize=9)
    for c in info["contours"]:
        ax[0, 0].plot(c[:, 0], c[:, 1], "r-", lw=0.6)
    from makutu.mortar.plots2d import draw_blocks
    mv = model(*np.meshgrid(x0 + (np.arange(v.shape[0]) + 0.5) * sp_[0], z0 + (np.arange(v.shape[1]) + 0.5) * sp_[1], indexing="ij"))[0]
    draw_blocks(ax[0, 1], bp.patches, mv, (x0, x1, z1, z0), edges=True, fine_width=0.2, cmap=mesh_cmap())
    ax[0, 1].set_title(f"blocs (escalier) : {out['blocs (escalier)']['elements']} éléments, {len(bp.patches)} blocs ; "
                       "modèle commun (contour lissé)", fontsize=9)
    draw_hybrid(ax[0, 2], cases["CG + DG"].zones, lw=0.2)
    ax[0, 2].set_title(f"CG + DG sans mortar : {out['CG + DG']['elements']} éléments dont {out['CG + DG']['triangles']} triangles", fontsize=9)
    for c in info["contours"]:
        ax[0, 2].plot(c[:, 0], c[:, 1], "r-", lw=0.6)
    ax[0, 3].axis("off")
    lines = ["maillage            éléments inconnues  dt stable  traces  instantané"]
    for k in ("hybride", "CG + DG", "blocs (escalier)", "uniforme"):
        o = out[k]
        lines.append(f"{k:17s} {o['elements']:8d} {o['ndof']:9d} {o['dt_max'] * 1e3:7.3f} ms {100 * o['err_traces']:6.2f} % {100 * o['err_snapshot']:6.2f} %")
    lines += ["", f"référence : hybride {a.ref_refine:g} fois plus fin,", f"{out['référence']['elements']} éléments ; dt = {dt * 1e3:.3f} ms pour tous"]
    ax[0, 3].text(0, 0.9, "\n".join(lines), va="top", family="monospace", fontsize=9, transform=ax[0, 3].transAxes)
    sr = ref["snapshots"][snap].reshape(nrx, nrz)
    c = np.percentile(np.abs(sr), 99.5)
    ax[1, 0].imshow(sr.T, extent=ext, cmap="seismic", vmin=-c, vmax=c, aspect="equal")
    ax[1, 0].set_title(f"référence : instantané t = {snap * dt:.2f} s", fontsize=9)
    for j, k in enumerate(("hybride", "CG + DG", "blocs (escalier)")):
        d = (res[k]["snapshots"][snap] - ref["snapshots"][snap]).reshape(nrx, nrz)
        ax[1, 1 + j].imshow(d.T, extent=ext, cmap="seismic", vmin=-c / 5, vmax=c / 5, aspect="equal")
        ax[1, 1 + j].set_title(f"{k} - référence (échelle x5) : {100 * out[k]['err_snapshot']:.2f} %", fontsize=9)
    for axx in ax[1]:
        for cc in info["contours"]:
            axx.plot(cc[:, 0], cc[:, 1], "k-", lw=0.4)
    tt = np.arange(nt) * dt
    gm = 0.05 * np.abs(ref["traces"][:, far]).max()
    xr = (recs[0][0], recs[-1][0], tt[-1], 0)
    ax[2, 0].imshow(ref["traces"], extent=xr, cmap="gray_r", vmin=-gm, vmax=gm, aspect="auto")
    ax[2, 0].set_title("référence : sismogramme (récepteurs à 80 m)", fontsize=9)
    for j, k in enumerate(("hybride", "CG + DG", "blocs (escalier)")):
        ax[2, 1 + j].imshow(res[k]["traces"] - ref["traces"], extent=xr, cmap="gray_r", vmin=-gm / 5, vmax=gm / 5, aspect="auto")
        ax[2, 1 + j].set_title(f"{k} - référence (échelle x5) : {100 * out[k]['err_traces']:.2f} %", fontsize=9)
    for axx in ax[2]:
        axx.set_xlabel("x récepteur (m)"); axx.set_ylabel("temps (s)")
    fig.suptitle(f"{a.model} : hybride (mortar), CG + DG sans mortar, blocs en escalier (f0 = {a.f0:g} Hz, {a.ppw:g} ppw)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out / f"compare_{a.model}.png", dpi=100)

    # seismograms of every mesh and their differences to the reference, all at the same grey scale
    np.savez(a.out / f"traces_{a.model}.npz", t=tt, x=np.array([r[0] for r in recs]),
             **{k.replace(" ", "_").replace("(", "").replace(")", ""): res[k]["traces"] for k in cases})
    order = ["référence", "hybride", "CG + DG", "blocs (escalier)", "uniforme"]
    fig, ax = plt.subplots(2, len(order), figsize=(4.2 * len(order), 10), sharex=True, sharey=True)
    for j, k in enumerate(order):
        ax[0, j].imshow(res[k]["traces"], extent=xr, cmap="gray_r", vmin=-gm, vmax=gm, aspect="auto")
        ax[0, j].set_title(f"{k} ({out[k]['elements']} él.)", fontsize=10)
        if k == "référence":
            ax[1, j].axis("off")
            ax[1, j].text(0.05, 0.6, f"même échelle de gris\npour toutes les vignettes\n(écrêtage ±{gm:.3g})",
                          transform=ax[1, j].transAxes, fontsize=10)
            continue
        ax[1, j].imshow(res[k]["traces"] - ref["traces"], extent=xr, cmap="gray_r", vmin=-gm, vmax=gm, aspect="auto")
        ax[1, j].set_title(f"{k} - référence : {100 * out[k]['err_traces']:.2f} %", fontsize=10)
    for axx in ax.ravel():
        axx.set_xlabel("x récepteur (m)")
    ax[0, 0].set_ylabel("temps (s)"); ax[1, 0].set_ylabel("temps (s)")
    fig.suptitle(f"{a.model} : sismogrammes (haut) et écarts à la référence (bas), même échelle "
                 f"(f0 = {a.f0:g} Hz, {a.ppw:g} ppw)", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out / f"sismos_{a.model}.png", dpi=100)


if __name__ == "__main__":
    main()
