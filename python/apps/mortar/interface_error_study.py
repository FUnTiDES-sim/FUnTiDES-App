#!/usr/bin/env python3
"""Erreur due aux interfaces non conformes (makutu.mortar), à ordre et points par longueur d'onde fixés.

Onde plane en incidence normale sur une interface plane de contraste c = v2 / v1 ; chaque côté maillé par
la règle h = N v / (fmax ppw) (rapport des tailles = c). Pour chaque c et chaque ppw : erreurs sur les
coefficients de réflexion et de transmission (valeurs exactes) et écarts des impulsions réfléchie et
transmise à une référence fine, pour les blocs et pour un maillage conforme à la plus petite taille.

    python apps/mortar/interface_error_study.py --out tests/figures/mortar
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface_study import plane_wave_case, window_errors  # noqa: E402  (apps/mortar)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--ppw", type=float, nargs="+", default=[4.0, 5.0, 6.0])
    ap.add_argument("--contrasts", type=float, nargs="+", default=[0.25, 0.33, 0.5, 0.67, 0.8, 1.25, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args()
    rows = []
    for ppw in a.ppw:
        for c in a.contrasts:
            fine = plane_wave_case(c, order=a.order, ppw=ppw, mesh="fine")
            dt = fine["dt"]
            r = {"ppw": ppw, "c": c}
            for mesh in ("blocs", "conforme"):
                res = plane_wave_case(c, order=a.order, ppw=ppw, mesh=mesh, dt=dt)
                er, et = window_errors(res, fine)
                r[mesh] = dict(R=res["R"], T=res["T"], dR=abs(res["R"] - res["R_exact"]), dT=abs(res["T"] - res["T_exact"]),
                               wave_R=er, wave_T=et, n_elements=res["n_elements"], ratio=res["ratio"])
            r["fine"] = dict(dR=abs(fine["R"] - fine["R_exact"]), dT=abs(fine["T"] - fine["T_exact"]))
            rows.append(r)
            print(f"ppw {ppw:g} c {c:5.2f} | blocs: dR {r['blocs']['dR']:.1e} dT {r['blocs']['dT']:.1e} onde R {100 * r['blocs']['wave_R']:.2f} % T {100 * r['blocs']['wave_T']:.2f} %"
                  f" | conforme: dR {r['conforme']['dR']:.1e} dT {r['conforme']['dT']:.1e} onde R {100 * r['conforme']['wave_R']:.2f} % T {100 * r['conforme']['wave_T']:.2f} %", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "interface_error_study.json").write_text(json.dumps(rows, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    titles = ["|R - R exact| (en amplitude incidente)", "|T - T exact|", "impulsion réfléchie : écart à la référence fine",
              "impulsion transmise : écart à la référence fine"]
    keys = ["dR", "dT", "wave_R", "wave_T"]
    for j, (k, tt) in enumerate(zip(keys, titles)):
        for i, ppw in enumerate(a.ppw):
            rr = [r for r in rows if r["ppw"] == ppw]
            cs = [r["c"] for r in rr]
            col = f"C{i}"
            ax[j].semilogy(cs, [max(r["blocs"][k], 1e-7) for r in rr], "o-", color=col, label=f"blocs {ppw:g} ppw")
            ax[j].semilogy(cs, [max(r["conforme"][k], 1e-7) for r in rr], "s--", color=col, alpha=0.6,
                           label=f"conforme {ppw:g} ppw")
        ax[j].set_xscale("log"); ax[j].set_xlabel("contraste c = v2 / v1 (= rapport des tailles de mailles des blocs)")
        ax[j].set_title(tt, fontsize=9); ax[j].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Interface non conforme, ordre {a.order}, onde plane en incidence normale : blocs (h = N v / (fmax ppw) de "
                 "chaque côté) contre maillage conforme à la plus petite taille", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / "interface_error_study.png", dpi=110)


if __name__ == "__main__":
    main()
