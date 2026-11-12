#!/usr/bin/env python3
"""Convergence et coût : blocs non conformes contre grille uniforme, à plusieurs échantillonnages (ppw),
tirs FUnTiDES (GPU) comparés à une référence calée sur les cellules SEP.

Pour chaque maillage : éléments, pas stable, travail (éléments / pas stable, par seconde simulée), écart
L2 des traces (offsets > 300 m) et de l'instantané à 0,6 T. Figure : écart en fonction du travail.

    python apps/mortar/blocks_convergence_2d.py --model marmousi --tmax 3 --out runs/mortar_conv
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[3].parent / "FUnTiDES-data" / "SEP"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--f0", type=float, default=8.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--blocks-ppw", type=float, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--uniform-ppw", type=float, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--out", type=Path, default=Path("mortar_convergence"))
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
    g = grid_from_sep(str(DATA / a.model / "Vp_xz.H"))
    v = g.vel[:, 0, :].astype(float)
    (x0, x1), _, (z0, z1) = g.extent
    sp_ = (g.spacing[0], g.spacing[2])
    N, f0 = a.order, a.f0
    nodal = grid_sampler(v, (x0, z0), sp_)
    inset = 0.5 * min(sp_)
    plans = {}
    for p in a.blocks_ppw:
        plans[f"blocs {p:g}"] = plan_blocks_global(v, (x0, z0), sp_, order=N, f0=f0, ppw=p)
    for p in a.uniform_ppw:
        plans[f"uniforme {p:g}"] = uniform_plan(v, (x0, z0), sp_, order=N, f0=f0, ppw=p)
    hmin = N * v.min() / (2.5 * f0 * 5.0)
    k = max(1, int(np.floor(hmin / 2.0 / max(sp_))))
    plans["référence"] = uniform_plan(v, (x0, z0), sp_, order=N, f0=f0, ppw=5.0 * hmin / (k * max(sp_)) * 0.999)
    sponge = min(800.0, 0.12 * (x1 - x0))
    runs, out = {}, {}
    for lab, pl in plans.items():
        m = Mortar2D(pl.patches, order=N, nodal=nodal, assemble=False, nodal_inset=inset)
        f = FuntidesBlocks2D(m, sponge=(sponge, sponge))
        dtm = f.dt_max()
        runs[lab] = (m, f)
        out[lab] = dict(elements=f.n_elements, segments=f.n_segments, dt_max=dtm, work=f.n_elements / (0.9 * dtm))
        print(f"{a.model} {lab:12s} {f.n_elements:8d} éléments, dt_max {dtm * 1e3:.3f} ms", flush=True)
    dt = 0.9 * min(o["dt_max"] for o in out.values())
    nt = int(a.tmax / dt)
    w = ricker(np.arange(nt) * dt, f0)
    src = (x0 + 0.5 * (x1 - x0) + 0.37 * sp_[0], z0 + 60.0)
    recs = [(x, z0 + 80.0) for x in np.arange(x0 + 100.0, x1 - 100.0, 25.0)]
    xr = np.array([r[0] for r in recs])
    far = np.abs(xr - src[0]) > 300.0
    nrx = min(800, v.shape[0]); nrz = max(2, int(round(nrx * (z1 - z0) / (x1 - x0))))
    snap = int(0.6 * nt)
    res = {}
    for lab, (m, f) in runs.items():
        res[lab] = f.run(dt, nt, src, w, recs, snapshots=[snap], raster=m.raster(nrx, nrz))
    ref = res["référence"]
    for lab in out:
        if lab == "référence":
            continue
        out[lab]["err_traces"] = float(np.linalg.norm((res[lab]["traces"] - ref["traces"])[:, far])
                                       / np.linalg.norm(ref["traces"][:, far]))
        s, sr = res[lab]["snapshots"][snap], ref["snapshots"][snap]
        out[lab]["err_snapshot"] = float(np.linalg.norm(s - sr) / np.linalg.norm(sr))
        print(f"   {lab:12s} travail {out[lab]['work'] / 1e6:7.2f} M ; écart traces {100 * out[lab]['err_traces']:.2f} %"
              f" ; instantané {100 * out[lab]['err_snapshot']:.2f} %", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.model}.json").write_text(json.dumps(dict(model=a.model, f0=f0, order=N, dt=dt, meshes=out),
                                                     indent=1, ensure_ascii=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, key in enumerate(("err_traces", "err_snapshot")):
        for fam, mk in (("blocs", "o-"), ("uniforme", "s--")):
            labs = [k for k in out if k.startswith(fam)]
            W = [out[k]["work"] / 1e6 for k in labs]
            E = [100 * out[k][key] for k in labs]
            ax[j].loglog(W, E, mk, label=f"{fam} ({', '.join(k.split()[1] for k in labs)} ppw)")
            for k, x, y in zip(labs, W, E):
                ax[j].annotate(k.split()[1], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax[j].set_xlabel("travail : éléments / pas stable (millions par seconde simulée)")
        ax[j].set_ylabel("écart à la référence (%)")
        ax[j].set_title(("traces (offsets > 300 m)" if j == 0 else f"instantané à {snap * dt:.2f} s")
                        + f" : {a.model}", fontsize=10)
        ax[j].legend(fontsize=8); ax[j].grid(True, which="both", alpha=0.3)
    fig.suptitle(f"{a.model} : précision contre coût, blocs non conformes et grille uniforme (FUnTiDES, ordre {N}, "
                 f"f0 = {f0:g} Hz ; référence {plans['référence'].n_elements} éléments)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / f"{a.model}.png", dpi=110)


if __name__ == "__main__":
    main()
