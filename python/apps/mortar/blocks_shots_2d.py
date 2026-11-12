#!/usr/bin/env python3
"""Tirs FUnTiDES (GPU) sur un modèle SEP 2-D : blocs réguliers non conformes contre grille uniforme.

Trois maillages, tous calculés par le solveur SEM 2-D de FUnTiDES avec les mêmes propriétés aux nœuds
(interpolation bilinéaire du modèle SEP) et le même pas de temps :
- blocs : pavage makutu.mortar.plan_blocks (pas global : dt_min = pas de la grille uniforme), recollés
  par l'opérateur d'interface SIPG (Solver.set_interface_operator) ;
- uniforme : une grille à la plus petite taille requise du modèle ;
- référence : grille uniforme calée sur les cellules SEP, au moins deux fois plus fine.

Sorties dans --out : <modèle>.npz (traces, instantané), <modèle>.json (éléments, pas stables, temps,
écarts) et <modèle>.png (instantanés, différences, sismogrammes).

Exemple :
    python apps/mortar/blocks_shots_2d.py --model seg2d --f0 8 --tmax 4 --out runs/mortar
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[3].parent / "FUnTiDES-data" / "SEP"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="nom du modèle dans FUnTiDES-data/SEP ou chemin d'un .H")
    ap.add_argument("--f0", type=float, default=8.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--ppw", type=float, default=5.0)
    ap.add_argument("--tmax", type=float, default=4.0)
    ap.add_argument("--blocks-ppw", type=float, nargs="+", default=[5.0, 6.0],
                    help="points par longueur d'onde des pavages en blocs comparés (précision égale : plus que --ppw)")
    ap.add_argument("--ref-factor", type=float, default=2.0, help="référence au moins ce facteur plus fine")
    ap.add_argument("--src-depth", type=float, default=60.0)
    ap.add_argument("--rec-depth", type=float, default=80.0)
    ap.add_argument("--out", type=Path, default=Path("mortar_shots"))
    a = ap.parse_args()

    import kokkos
    kokkos.initialize()
    try:
        run(a)
    finally:
        import gc
        gc.collect()
        kokkos.finalize()


def run(a):
    from makutu.mesh.seismesh import grid_from_sep
    from makutu.mortar import Mortar2D, grid_sampler, plan_blocks_global, ricker, uniform_plan
    from makutu.mortar.funtides2d import FuntidesBlocks2D

    path = Path(a.model) if a.model.endswith(".H") else DATA / a.model / "Vp_xz.H"
    name = path.parent.name if a.model.endswith(".H") else a.model
    g = grid_from_sep(str(path))
    v = g.vel[:, 0, :].astype(float)
    (x0, x1), _, (z0, z1) = g.extent
    sp_ = (g.spacing[0], g.spacing[2])
    N, f0, ppw = a.order, a.f0, a.ppw
    kw = dict(order=N, f0=f0, ppw=ppw)
    uni = uniform_plan(v, (x0, z0), sp_, **kw)
    plans = {}
    for bp in a.blocks_ppw:
        # global time step: best work over a few targets around the uniform grid's step
        plans[f"blocs {bp:g}"] = plan_blocks_global(v, (x0, z0), sp_, order=N, f0=f0, ppw=bp)
    plans["uniforme"] = uni
    hmin = N * v.min() / (2.5 * f0 * ppw)
    k = max(1, int(np.floor(hmin / a.ref_factor / max(sp_))))        # SEP cells per reference element
    href = k * max(sp_)
    plans["référence"] = uniform_plan(v, (x0, z0), sp_, order=N, f0=f0, ppw=ppw * hmin / href * 0.999)
    nodal = grid_sampler(v, (x0, z0), sp_)
    sponge = min(800.0, 0.12 * (x1 - x0))
    W, H = x1 - x0, z1 - z0
    nrx = min(800, v.shape[0]); nrz = max(2, int(round(nrx * H / W)))
    out = {}
    runs = {}
    for lab, pl in plans.items():
        m = Mortar2D(pl.patches, order=N, nodal=nodal, assemble=False, nodal_inset=0.5 * min(sp_))
        t = time.time()
        f = FuntidesBlocks2D(m, sponge=(sponge, sponge))
        t_init = time.time() - t
        runs[lab] = (m, f)
        out[lab] = dict(elements=f.n_elements, nodes=f.n_nodes, blocks=len(pl.patches),
                        interface_segments=f.n_segments, init_s=t_init, dt_max=f.dt_max())
        print(f"{name} {lab:10s} {f.n_elements:8d} éléments {len(pl.patches):5d} blocs {f.n_segments:6d} segments "
              f"dt_max {out[lab]['dt_max'] * 1e3:.3f} ms", flush=True)
    dt = 0.9 * min(o["dt_max"] for o in out.values())
    nt = int(a.tmax / dt)
    w = ricker(np.arange(nt) * dt, f0)
    src = (x0 + 0.5 * W + 0.37 * sp_[0], z0 + a.src_depth)
    recs = [(x, z0 + a.rec_depth) for x in np.arange(x0 + 100.0, x1 - 100.0, 25.0)]
    snap = int(0.6 * nt)
    res = {}
    nb = min(nt, 1000)
    for lab, (m, f) in runs.items():
        # timing without receivers (reading the field from the host each step dominates otherwise):
        # warm-up (the laptop GPU clocks up slowly), then the best of three
        f.run(dt, nb, src, w[:nb])
        el = min(_timed(f, dt, nb, src, w[:nb]) for _ in range(3))
        o = out[lab]
        o["s_per_step"] = el / nb
        o["s_at_own_dt"] = el / nb * a.tmax / (0.9 * o["dt_max"])          # same mesh at its own stable step
        res[lab] = f.run(dt, nt, src, w, recs, snapshots=[snap], raster=m.raster(nrx, nrz))
        print(f"   {lab:10s} {1e3 * o['s_per_step']:.3f} ms par pas ; {a.tmax:g} s simulées à son propre pas : "
              f"{o['s_at_own_dt']:.1f} s", flush=True)
    ref = res["référence"]["traces"]
    xr = np.array([p[0] for p in recs])
    far = np.abs(xr - src[0]) > 300.0
    for lab in [k for k in res if k != "référence"]:
        d = res[lab]["traces"]
        out[lab]["err_traces"] = float(np.linalg.norm((d - ref)[:, far]) / np.linalg.norm(ref[:, far]))
        s, sr = res[lab]["snapshots"][snap], res["référence"]["snapshots"][snap]
        out[lab]["err_snapshot"] = float(np.linalg.norm(s - sr) / np.linalg.norm(sr))
        print(f"   {lab:10s} écart traces {100 * out[lab]['err_traces']:.2f} %  instantané "
              f"{100 * out[lab]['err_snapshot']:.2f} %", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = dict(model=name, f0=f0, order=N, ppw=ppw, tmax=a.tmax, dt=dt, nt=nt, source=src,
                reference_h=href, sponge=sponge, meshes=out)
    (a.out / f"{name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    np.savez(a.out / f"{name}.npz", dt=dt, recx=xr, extent=np.array([x0, x1, z1, z0]), tsnap=snap * dt,
             **{f"tr_{k}": r["traces"] for k, r in res.items()},
             **{f"snap_{k}": r["snapshots"][snap].reshape(nrx, nrz) for k, r in res.items()})
    figure(a.out / f"{name}.png", name, meta, res, snap, xr, dt, nt, (x0, x1, z1, z0), (nrx, nrz), far)


def _timed(f, dt, n, src, w):
    t = time.time()
    f.run(dt, n, src, w)
    return time.time() - t


def figure(path, name, meta, res, snap, xr, dt, nt, ext, shape, far):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    o = meta["meshes"]
    ref = res["référence"]["traces"]
    sr = res["référence"]["snapshots"][snap].reshape(shape)
    W, H = ext[1] - ext[0], ext[2] - ext[3]
    labs = ["référence"] + [k for k in res if k != "référence"]
    nc = len(labs)
    fig, ax = plt.subplots(3, nc, figsize=(5.7 * nc, 4.0 * 3 * max(0.45, min(1.0, 2.2 * H / W)) + 1.5))
    c = np.percentile(np.abs(sr), 99)
    for j, lab in enumerate(labs):
        s = res[lab]["snapshots"][snap].reshape(shape)
        ax[0, j].imshow(s.T, extent=ext, cmap="seismic", vmin=-c, vmax=c, aspect="auto")
        ax[0, j].set_title(f"{lab} : {o[lab]['elements']} éléments, {o[lab]['blocks']} bloc(s)\n"
                           f"instantané t = {snap * dt:.2f} s", fontsize=9)
        if lab != "référence":
            ax[1, j].imshow((s - sr).T, extent=ext, cmap="seismic", vmin=-c, vmax=c, aspect="auto")
            ax[1, j].set_title(f"{lab} - référence (même échelle) : {100 * o[lab]['err_snapshot']:.1f} %", fontsize=9)
    win = max(3, int(0.25 / dt))
    gain = 1 / (np.sqrt(np.clip(uniform_filter1d(ref ** 2, win, axis=0), 0, None)) + 1e-3 * np.abs(ref).max())
    t = np.arange(nt) * dt
    for j, lab in enumerate(labs):
        D = gain * (ref if lab == "référence" else res[lab]["traces"] - ref)
        ax[2, j].imshow(D, extent=(xr[0], xr[-1], t[-1], 0), cmap="gray_r", vmin=-2, vmax=2, aspect="auto")
        ax[2, j].set_title(("sismogramme" if lab == "référence" else f"{lab} - référence") +
                           " (gain AGC 0,25 s de la référence, même échelle)", fontsize=9)
        ax[2, j].set_xlabel("x récepteur (m)"); ax[2, j].set_ylabel("temps (s)")
    for axx in ax[:2].ravel():
        axx.set_xlabel("x (m)"); axx.set_ylabel("profondeur (m)")
    ax[1, 0].axis("off")
    lines = [f"{name} : Ricker f0 = {meta['f0']:g} Hz, ordre {meta['order']}, {meta['ppw']:g} points GLL / λ",
             f"FUnTiDES SEM 2-D (GPU), même dt = {dt * 1e3:.3f} ms pour tous",
             f"référence : éléments de {meta['reference_h']:.1f} m (cellules SEP)", "",
             "maillage    éléments  dt stable  temps à son dt  écart traces / instantané"]
    for lab in labs:
        e = "" if lab == "référence" else f"{100 * o[lab]['err_traces']:.2f} % / {100 * o[lab]['err_snapshot']:.2f} %"
        lines.append(f"{lab:10s} {o[lab]['elements']:9d} {o[lab]['dt_max'] * 1e3:7.3f} ms {o[lab]['s_at_own_dt']:9.1f} s   {e}")
    ax[1, 0].text(0.0, 0.95, "\n".join(lines), va="top", fontsize=9, family="monospace", transform=ax[1, 0].transAxes)
    fig.suptitle(f"{name} : blocs réguliers non conformes (SIPG, FUnTiDES) contre grille uniforme", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
