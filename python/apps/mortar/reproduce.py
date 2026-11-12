#!/usr/bin/env python3
"""Reproduit les modélisations mortar / hybride SEM + DG sur les différents modèles (registre ``cases.py``).

Chaque étude tourne dans ``<out>/<modèle>/<étude>/`` (journal ``log.txt``, JSON, figures) ; les valeurs
obtenues sont comparées aux valeurs attendues du registre, et un résumé ``<out>/RESUME.md`` liste les
études, leurs contrôles et leurs figures. Les études GPU (FUnTiDES) ne tournent qu'avec ``--gpu`` et si
aucun autre processus de calcul n'occupe la carte (``--force`` pour passer outre).

    python apps/mortar/reproduce.py --list
    python apps/mortar/reproduce.py --model seg2d                       # études CPU de seg2d
    python apps/mortar/reproduce.py --model all --study compare cost   # comparaisons et coûts, tous modèles
    python apps/mortar/reproduce.py --model sigsbee2a --gpu            # y compris les tirs FUnTiDES
    python apps/mortar/reproduce.py --model all --check-only           # recontrôle des sorties existantes

À lancer depuis ``FUnTiDES-App/python`` après ``source <FUnTiDES install>/funtides_env.sh``.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES, ERR_TOL  # noqa: E402


def gpu_busy():
    """PIDs des processus de calcul présents sur le GPU (liste vide si nvidia-smi est absent)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def check(study, path):
    """Contrôles d'une étude : liste de (message, ok)."""
    out = []
    jf = path / study.json if study.json else None
    if jf is None or not jf.exists():
        return [(f"{study.json} absent", False)] if study.json else []
    data = json.loads(jf.read_text())
    meshes = data.get("cases", data.get("meshes", data))
    for name, (et, es) in study.expected.get("errors", {}).items():
        if name not in meshes:
            out.append((f"{name} : absent du JSON", False))
            continue
        gt, gs = meshes[name]["err_traces"], meshes[name]["err_snapshot"]
        ok = abs(gt - et) <= ERR_TOL and abs(gs - es) <= ERR_TOL
        out.append((f"{name} : traces {100 * gt:.2f} % (attendu {100 * et:.2f}), "
                    f"instantané {100 * gs:.2f} % (attendu {100 * es:.2f})", ok))
    for name, n in study.expected.get("elements", {}).items():
        got = meshes.get(name, {}).get("elements")
        out.append((f"{name} : {got} éléments (attendu {n})", got == n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="liste les modèles et études")
    ap.add_argument("--model", nargs="+", default=["seg2d"], help="modèles du registre, ou all")
    ap.add_argument("--study", nargs="+", default=["all"], help="études du registre, ou all")
    ap.add_argument("--gpu", action="store_true", help="inclut les études GPU (FUnTiDES)")
    ap.add_argument("--force", action="store_true", help="lance les études GPU même si la carte est occupée")
    ap.add_argument("--check-only", action="store_true", help="ne relance rien, recontrôle les sorties")
    ap.add_argument("--out", type=Path, default=Path("runs/mortar"))
    a = ap.parse_args()

    if a.list:
        for model, case in CASES.items():
            print(f"{model} : {case['description']}")
            for name, st in case["studies"].items():
                tag = "GPU" if st.gpu else "CPU"
                chk = "contrôlée" if st.expected else "sans valeurs attendues"
                print(f"   {name:15s} {tag}  ~{st.runtime:7s} {st.app} {' '.join(st.args)}  ({chk})")
        return 0

    models = list(CASES) if a.model == ["all"] else a.model
    for m in models:
        if m not in CASES:
            ap.error(f"modèle inconnu : {m} (voir --list)")
    rows = []
    for model in models:
        for name, st in CASES[model]["studies"].items():
            if a.study != ["all"] and name not in a.study:
                continue
            path = a.out / model / name
            if st.gpu and not a.gpu:
                rows.append((model, name, st, path, "ignorée (GPU : --gpu)", []))
                continue
            status = "contrôle seul"
            if not a.check_only:
                if st.gpu and not a.force:
                    busy = gpu_busy()
                    if busy:
                        rows.append((model, name, st, path, f"ignorée : GPU occupé ({'; '.join(busy)})", []))
                        continue
                path.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(HERE / st.app), *st.args, "--out", str(path)]
                print(f"[{model}/{name}] {' '.join(cmd)}", flush=True)
                t = time.time()
                with open(path / "log.txt", "w") as log:
                    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                status = f"code {r.returncode}, {time.time() - t:.0f} s"
            checks = check(st, path)
            nok = sum(1 for _, ok in checks if not ok)
            print(f"[{model}/{name}] {status} ; contrôles : {len(checks) - nok}/{len(checks)} conformes", flush=True)
            for msg, ok in checks:
                if not ok:
                    print(f"      ÉCART  {msg}", flush=True)
            rows.append((model, name, st, path, status, checks))

    a.out.mkdir(parents=True, exist_ok=True)
    lines = ["# Reproduction des modélisations mortar / hybride", "",
             f"Tolérance sur les écarts : {100 * ERR_TOL:.1f} point ; éléments : égalité exacte.", ""]
    for model, name, st, path, status, checks in rows:
        lines += [f"## {model} / {name}", "", f"`{st.app} {' '.join(st.args)}` — {status}"]
        if st.note:
            lines.append(f"\n{st.note}")
        lines.append("")
        for msg, ok in checks:
            lines.append(f"- {'conforme' if ok else '**écart**'} : {msg}")
        for fig in st.figures:
            if (path / fig).exists():
                lines.append(f"\n![{model} {name}]({(path / fig).relative_to(a.out)})")
        lines.append("")
    (a.out / "RESUME.md").write_text("\n".join(lines))
    print(f"résumé : {a.out / 'RESUME.md'}")
    bad = sum(1 for *_, checks in rows for _, ok in checks if not ok)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
