"""
Prepare the 2-D Marmousi model for the FUnTiDES 2-D SEM: check (and fix) the SEP
files, then choose the mesh for a source frequency and a polynomial order.

1. SEP files (FUnTiDES-data/SEP/marmousi/Vp_xz.H, Rho_xz.H): the header fields, the
   binary named by ``in=`` and its ``data_format`` are checked. With ``--fix``:
   ``in=`` is set to the binary next to the header when the declared one is missing,
   ``data_format`` is set to the byte order that gives plausible values, and a density
   of 1 is written when Rho_xz.H is missing. Modified headers are saved as ``*.H.orig``.
2. Mesh: element size h such that the shortest wavelength, vmin / (2.5 f0) for a Ricker
   wavelet of peak frequency f0, holds ``--ppw`` GLL points: h = order * lambda_min / ppw.
   The time step follows dt = cfl * (smallest GLL spacing) / vmax.
3. Prints the summary and the matching modelling command (shot_modeling); writes a QC figure
   (model, element grid) unless --no-plot.

Usage
-----
    source <FUnTiDES install>/funtides_env.sh
    python apps/modeling/examples/marmousi_2d/prepare_marmousi_2d.py [--fix] [--f0 8] [--order 4] [--ppw 6]
"""

import argparse
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np

DEFAULT_DATA = Path(os.environ.get("FUNTIDES_DATA", Path(__file__).resolve().parents[6] / "FUnTiDES-data"))
REQUIRED = ("n1", "n2", "d1", "d2", "in")
FORMATS = {"native_float": "<f4", "xdr_float": ">f4"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                   help="FUnTiDES-data directory (default: $FUNTIDES_DATA or next to FUnTiDES-App)")
    p.add_argument("--fix", action="store_true", help="fix the SEP headers / create Rho (backups *.H.orig)")
    p.add_argument("--f0", type=float, default=8.0, help="Ricker peak frequency [Hz] (default 8)")
    p.add_argument("--order", type=int, default=4, help="polynomial order (default 4)")
    p.add_argument("--ppw", type=float, default=6.0,
                   help="GLL points per shortest wavelength (default 6; 5 is a usual minimum)")
    p.add_argument("--cfl", type=float, default=0.3, help="CFL number for dt (default 0.3)")
    p.add_argument("--tmax", type=float, default=4.0, help="recorded time [s], for the cost estimate (default 4)")
    p.add_argument("--qc", default="marmousi_2d_qc.png", help="QC figure (default marmousi_2d_qc.png)")
    p.add_argument("--no-plot", action="store_true", help="do not write the QC figure")
    return p.parse_args()


def read_header(path):
    kv = {}
    for raw in Path(path).read_text().splitlines():
        for m in re.finditer(r'(\w+)\s*=\s*("([^"]*)"|(\S+))', raw.split("#")[0]):
            kv[m.group(1).lower()] = m.group(3) if m.group(3) is not None else m.group(4)
    return kv


def set_header_field(path, key, value):
    """Replace (or append) ``key=value`` in a SEP header; keep a *.H.orig backup once."""
    path = Path(path)
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    lines = path.read_text().splitlines()
    pattern = re.compile(rf"^\s*{key}\s*=")
    new = [f"{key}={value}" if pattern.match(line) else line for line in lines]
    if not any(pattern.match(line) for line in lines):
        new.append(f"{key}={value}")
    path.write_text("\n".join(new) + "\n")


def plausible(values, name):
    """Physical range check: velocities 100-10000 m/s, densities 0.5-5000 (g/cc or kg/m3)."""
    lo, hi = (100.0, 10000.0) if name == "Vp" else (0.5, 5000.0)
    return bool(np.all(np.isfinite(values)) and values.min() >= lo and values.max() <= hi)


def check_sep(header, name, fix, issues):
    """Check one SEP file; return its values (n2, n1) or None."""
    kv = read_header(header)
    missing = [k for k in REQUIRED if k not in kv]
    if missing:
        issues.append(f"{header.name}: missing fields {missing}")
        return None, kv
    n1, n2 = int(kv["n1"]), int(kv["n2"])
    binary = Path(kv["in"].strip('"'))
    if not binary.is_absolute():
        binary = header.parent / binary
    if not binary.exists():
        local = header.parent / (header.name + "@")
        issues.append(f"{header.name}: in={kv['in']} does not exist"
                      + (f"; {local.name} is next to the header" if local.exists() else ""))
        if not local.exists():
            return None, kv
        binary = local
        if fix:
            set_header_field(header, "in", local.name)
            issues[-1] += " -> fixed"
    declared = kv.get("data_format", "native_float").lower()
    if declared not in FORMATS:
        issues.append(f"{header.name}: unknown data_format {declared}")
        return None, kv
    values = np.fromfile(binary, dtype=FORMATS[declared], count=n1 * n2)
    if values.size < n1 * n2:
        issues.append(f"{header.name}: {values.size} values, expected {n1 * n2}")
        return None, kv
    if not plausible(values, name):
        other = "xdr_float" if declared == "native_float" else "native_float"
        alt = np.fromfile(binary, dtype=FORMATS[other], count=n1 * n2)
        if plausible(alt, name):
            issues.append(f"{header.name}: data_format={declared} gives implausible values, {other} fits")
            values = alt
            if fix:
                set_header_field(header, "data_format", other)
                issues[-1] += " -> fixed"
        else:
            issues.append(f"{header.name}: values out of range ({values.min():.3g}-{values.max():.3g})")
    return values.reshape(n2, n1).astype(np.float32), kv


def write_constant_rho(vp_header, rho_header):
    kv = read_header(vp_header)
    n1, n2 = int(kv["n1"]), int(kv["n2"])
    binary = rho_header.with_name(rho_header.name + "@")
    np.ones(n1 * n2, dtype="<f4").tofile(binary)
    lines = [f"{k}={kv[k]}" for k in ("n1", "n2", "d1", "d2", "o1", "o2") if k in kv]
    lines += [f'label1="{kv.get("label1", "CDP")}"', f'label2="{kv.get("label2", "DEPTH")}"',
              "esize=4", f"in={binary.name}", "data_format=native_float"]
    rho_header.write_text("#SEP\n" + "\n".join(lines) + "\n")


def gll_min_gap(order):
    """Smallest gap between GLL points of [-1, 1] (end gap), as a fraction of the element size."""
    from numpy.polynomial import legendre
    inner = legendre.Legendre.basis(order).deriv().roots() if order > 1 else np.array([])
    pts = np.sort(np.concatenate(([-1.0], inner.real, [1.0])))
    return float(np.min(np.diff(pts))) / 2.0


def main():
    args = parse_args()
    sep_dir = args.data_dir / "SEP" / "marmousi"
    vp_h, rho_h = sep_dir / "Vp_xz.H", sep_dir / "Rho_xz.H"
    print(f"Data: {sep_dir}")
    if not vp_h.exists():
        raise SystemExit(f"{vp_h} not found: put the Marmousi SEP files there (see README.md)")

    issues = []
    vp, kv = check_sep(vp_h, "Vp", args.fix, issues)
    if not rho_h.exists():
        issues.append("Rho_xz.H missing: a density of 1 is used")
        if args.fix:
            write_constant_rho(vp_h, rho_h)
            issues[-1] += " -> Rho_xz.H written"
    rho = check_sep(rho_h, "Rho", args.fix, issues)[0] if rho_h.exists() else None
    for line in issues or ["SEP files OK"]:
        print("  " + line)
    if vp is None:
        raise SystemExit("Vp_xz.H unusable" + ("" if args.fix else ": rerun with --fix"))

    labels = kv.get("label1", "?"), kv.get("label2", "?")
    n1, n2, d1, d2 = int(kv["n1"]), int(kv["n2"]), float(kv["d1"]), float(kv["d2"])
    lx, lz = n1 * d1, n2 * d2
    vmin, vmax = float(vp.min()), float(vp.max())
    print(f"Model: {n1} x {n2} samples at {d1:g} x {d2:g} m (axis 1 {labels[0]}, axis 2 {labels[1]}): "
          f"{lx / 1e3:g} km x {lz / 1e3:g} km; vp {vmin:.0f}-{vmax:.0f} m/s"
          + (f"; rho {rho.min():.3g}-{rho.max():.3g}" if rho is not None else ""))

    # Mesh: h = order * lambda_min / ppw, lambda_min = vmin / (2.5 f0)
    lam = vmin / (2.5 * args.f0)
    h = args.order * lam / args.ppw
    ex, ez = math.ceil(lx / h), math.ceil(lz / h)
    hx, hz = lx / ex, lz / ez
    nodes = (ex * args.order + 1) * (ez * args.order + 1)
    dt = args.cfl * gll_min_gap(args.order) * min(hx, hz) / vmax
    steps = int(args.tmax / dt)
    print(f"Mesh for f0 = {args.f0:g} Hz, order {args.order}, {args.ppw:g} points per wavelength "
          f"(lambda_min = {lam:.0f} m): element size <= {h:.1f} m -> {ex} x {ez} elements "
          f"({hx:.1f} x {hz:.1f} m), {nodes} nodes")
    if min(hx, hz) < min(d1, d2):
        print(f"  elements smaller than the model sampling ({d1:g} m): the model is piecewise constant "
              f"per sample, which is fine")
    print(f"Time step (CFL {args.cfl:g}): {dt * 1e3:.3f} ms -> {steps} steps for {args.tmax:g} s")
    # run_shot_modeling.py uses ex = round(lx / elem_size), ez = round(lz / elem_size)
    elem_size = max(lx / ex, lz / ez)
    rho = f" --rho-file {rho_h}" if rho_h.exists() else ""
    print("\nRun:\n  python apps/modeling/shot_modeling/run_shot_modeling.py "
          f"--vp-file {vp_h}{rho} --order {args.order} --f0 {args.f0:g} --elem-size {elem_size:.2f} "
          f"--cfl {args.cfl:g} --tmax {args.tmax:g} --plot")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(vp, extent=[0, lx / 1e3, lz / 1e3, 0], aspect="auto", cmap="viridis")
        for x in np.linspace(0, lx, ex + 1)[::max(1, ex // 50)]:
            ax.axvline(x / 1e3, color="w", lw=0.2, alpha=0.5)
        for z in np.linspace(0, lz, ez + 1)[::max(1, ez // 20)]:
            ax.axhline(z / 1e3, color="w", lw=0.2, alpha=0.5)
        ax.set(title=f"Marmousi Vp, {ex} x {ez} elements of order {args.order} (grid lines thinned)",
               xlabel="x (km)", ylabel="depth (km)")
        fig.colorbar(im, ax=ax, label="m/s")
        fig.tight_layout()
        fig.savefig(args.qc, dpi=110)
        print(f"QC figure: {args.qc}")


if __name__ == "__main__":
    main()
