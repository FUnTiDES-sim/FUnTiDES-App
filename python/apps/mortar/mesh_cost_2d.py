#!/usr/bin/env python3
"""Maillages et coûts de calcul des quatre approches sur un modèle SEP 2-D (prototypes NumPy, un cœur) :
hybride mortar (blocs SEM + triangles DG), CG + DG sans mortar, blocs en escalier, grille uniforme.

Pour chaque maillage : temps de construction (plan + assemblage), pas stable propre, temps par pas
(moyenne sur ``--steps`` pas), temps estimé de la simulation de ``--tmax`` s au pas 0,8 dt_max propre.
Figure : maillage entier et zoom sur le contour du corps.

    python apps/mortar/mesh_cost_2d.py --model sigsbee2a --f0 4 --tmax 6 --tri-factor 1 --zoom 15000 21000 2500 6000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from makutu.mortar import Mortar2D, plan_blocks_global, ricker, uniform_plan
from makutu.mortar.hybrid2d import Hybrid2D, draw_hybrid, mesh_cmap
from makutu.mortar.hybrid_plan import plan_cg_hybrid, plan_hybrid
from makutu.mortar.plots2d import draw_blocks

DATA = Path(__file__).resolve().parents[3].parent / "FUnTiDES-data" / "SEP"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="seg2d")
    ap.add_argument("--f0", type=float, default=8.0)
    ap.add_argument("--ppw", type=float, default=5.0)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--tri-factor", type=float, default=4.0 / 3.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--zoom", type=float, nargs=4, default=None, metavar=("X0", "X1", "Z0", "Z1"))
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args()
    from makutu.mesh.seismesh import grid_from_sep
    g = grid_from_sep(str(DATA / a.model / "Vp_xz.H"))
    v = g.vel[:, 0, :].astype(float)
    (x0, x1), _, (z0, z1) = g.extent
    sp_ = (g.spacing[0], g.spacing[2])
    sponge = min(800.0, 0.12 * (x1 - x0))
    kw = dict(sponge=sponge, sponge_sides=("left", "right", "bottom"))
    out, meshes = {}, {}

    def build(name, fn):
        t = time.time()
        m, extra = fn()
        out[name] = dict(build_s=time.time() - t)
        meshes[name] = (m, extra)

    def hyb():
        zones, info = plan_hybrid(v, (x0, z0), sp_, f0=a.f0, ppw=a.ppw, tri_factor=a.tri_factor)
        return Hybrid2D(zones, **kw), info

    def cg():
        zones, info = plan_cg_hybrid(v, (x0, z0), sp_, f0=a.f0, ppw=a.ppw, tri_factor=a.tri_factor)
        return Hybrid2D(zones, **kw), info

    build("hybride mortar", hyb)
    model = meshes["hybride mortar"][1]["model"]
    build("CG + DG", cg)
    build("blocs (escalier)", lambda: (lambda bp: (Mortar2D(bp.patches, order=4, nodal=model, **kw), bp))(
        plan_blocks_global(v, (x0, z0), sp_, order=4, f0=a.f0, ppw=a.ppw)))
    build("uniforme", lambda: (lambda up: (Mortar2D(up.patches, order=4, nodal=model, **kw), up))(
        uniform_plan(v, (x0, z0), sp_, order=4, f0=a.f0, ppw=a.ppw)))
    src = (x0 + 0.5 * (x1 - x0) + 7.0, z0 + 60.0)
    recs = [(x, z0 + 80.0) for x in np.arange(x0 + 100.0, x1 - 100.0, 25.0)]
    for k, (m, extra) in meshes.items():
        t = time.time()
        dtm = m.dt_max()
        o = out[k]
        o.update(eig_s=time.time() - t, dt_max=dtm, elements=m.n_elements, ndof=m.ndof)
        if hasattr(m, "zones"):
            o["triangles"] = sum(z.n_elements for z in m.zones if z.kind == "tri")
        dt = 0.8 * dtm
        w = ricker(np.arange(a.steps) * dt, a.f0)
        m.run(dt, 20, src, w[:20], recs)                         # warm-up (factorisations, caches)
        t = time.time()
        m.run(dt, a.steps, src, w, recs)
        o["step_ms"] = 1e3 * (time.time() - t) / a.steps
        o["n_steps"] = int(np.ceil(a.tmax / dt))
        o["run_s"] = o["step_ms"] * 1e-3 * o["n_steps"]
        o["ns_per_dof_step"] = 1e6 * o["step_ms"] / m.ndof
        print(f"{k:18s} {o['elements']:6d} él. {o['ndof']:8d} inc. construction {o['build_s']:5.1f} s "
              f"dt {dtm * 1e3:.3f} ms  {o['step_ms']:6.2f} ms/pas ({o['ns_per_dof_step']:.0f} ns/inc./pas)  "
              f"{o['n_steps']} pas -> {o['run_s']:.0f} s", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"cout_{a.model}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W, H = x1 - x0, z1 - z0
    names = list(meshes)
    fig, ax = plt.subplots(len(names), 2, figsize=(20, len(names) * max(3.2, 9.5 * H / W) + 1),
                           gridspec_kw=dict(width_ratios=[1.25, 1]))
    mv = model(*np.meshgrid(x0 + (np.arange(v.shape[0]) + 0.5) * sp_[0], z0 + (np.arange(v.shape[1]) + 0.5) * sp_[1],
                            indexing="ij"))[0]
    vmin, vmax = mv.min(), mv.max()
    contours = meshes["hybride mortar"][1]["contours"]
    for r, k in enumerate(names):
        m, extra = meshes[k]
        o = out[k]
        for c, lw in ((0, 0.15), (1, 0.4)):
            axx = ax[r, c]
            if hasattr(m, "zones"):
                draw_hybrid(axx, m.zones, lw=lw, vmin=vmin, vmax=vmax)
            else:
                draw_blocks(axx, extra.patches, mv, (x0, x1, z1, z0), edges=True, fine_width=lw, cmap=mesh_cmap())
            for cc in contours:
                axx.plot(cc[:, 0], cc[:, 1], "r-", lw=0.5 if c == 0 else 0.9)
            if c == 1 and a.zoom is not None:
                axx.set_xlim(a.zoom[0], a.zoom[1]); axx.set_ylim(a.zoom[3], a.zoom[2])
        tri = f", dont {o['triangles']} triangles" if "triangles" in o else ""
        ax[r, 0].set_title(f"{k} : {o['elements']} éléments{tri}, {o['ndof']} inconnues", fontsize=10)
        ax[r, 1].set_title(f"zoom ; dt stable {o['dt_max'] * 1e3:.2f} ms, {o['step_ms']:.1f} ms/pas, "
                           f"{o['n_steps']} pas pour {a.tmax:g} s : {o['run_s']:.0f} s (NumPy, 1 cœur)", fontsize=10)
    fig.suptitle(f"{a.model} : maillages (f0 = {a.f0:g} Hz, {a.ppw:g} ppw, triangles {a.tri_factor:.2g} p v/(fmax ppw)) ; "
                 "rouge : contour du corps", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(a.out / f"maillages_{a.model}.png", dpi=100)


if __name__ == "__main__":
    main()
