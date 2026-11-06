"""
One acoustic shot on the 2-D Marmousi model with the FUnTiDES 2-D SEM solver.

Model:   FUnTiDES-data/SEP/marmousi/Vp_xz.H and Rho_xz.H (500 x 174 samples at
         20 m: 10 km x 3.48 km; axis 1 = x, axis 2 = depth), sampled per element.
Mesh:    2-D (x, z) quadrilaterals, z up, free surface on top, absorbing edges
         elsewhere (makutu.mesh.CartesianMesh2D).
Shot:    Ricker point source near the surface; a line of receivers just below it.
Outputs: <output>.h5 (pressure at the receivers, receiver x, time step) and
         <output>.png (model, wavefield snapshot, shot gather) unless --no-plot.

Usage
-----
    source <FUnTiDES install>/funtides_env.sh
    python apps/modeling/examples/acoustic_marmousi_2d.py [--order 4] [--f0 8] [--tmax 4]
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import h5py

import kokkos
kokkos.initialize()

import pyfuntides.solver as FTSolver

from makutu.mesh import CartesianMesh2D
from makutu.solvers import AcousticSolver

# FUnTiDES-data is next to FUnTiDES-App: <dir>/FUnTiDES-App/python/apps/modeling/examples/this.py
DEFAULT_DATA = Path(os.environ.get("FUNTIDES_DATA", Path(__file__).resolve().parents[5] / "FUnTiDES-data"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                   help="FUnTiDES-data directory (default: $FUNTIDES_DATA or next to FUnTiDES-App)")
    p.add_argument("--order", type=int, default=4, help="polynomial order (default 4)")
    p.add_argument("--elem-size", type=float, default=40.0, help="element size [m] (default 40)")
    p.add_argument("--f0", type=float, default=8.0, help="Ricker peak frequency [Hz] (default 8)")
    p.add_argument("--tmax", type=float, default=4.0, help="recorded time [s] (default 4)")
    p.add_argument("--cfl", type=float, default=0.3, help="dt = cfl * min spacing / max vp (default 0.3)")
    p.add_argument("--src-x", type=float, default=5000.0, help="source x [m] (default 5000)")
    p.add_argument("--src-depth", type=float, default=10.0, help="source depth [m] (default 10)")
    p.add_argument("--rcv-depth", type=float, default=20.0, help="receiver depth [m] (default 20)")
    p.add_argument("--rcv-dx", type=float, default=20.0, help="receiver spacing [m] (default 20)")
    p.add_argument("--dt-seismo", type=float, default=0.002, help="trace sampling [s] (default 0.002)")
    p.add_argument("--snap-time", type=float, default=1.5, help="wavefield snapshot time [s] (default 1.5)")
    p.add_argument("--output", default="marmousi_2d_shot", help="output basename (default marmousi_2d_shot)")
    p.add_argument("--no-plot", action="store_true", help="do not write the PNG figure")
    return p.parse_args()


def main():
    args = parse_args()
    sep_dir = args.data_dir / "SEP" / "marmousi"
    lx, lz = 500 * 20.0, 174 * 20.0
    ex, ez = int(round(lx / args.elem_size)), int(round(lz / args.elem_size))

    t0 = time.time()
    mesh = CartesianMesh2D(lx, lz, ex, ez, order=args.order,
                           vp_file=str(sep_dir / "Vp_xz.H"), rho_file=str(sep_dir / "Rho_xz.H"))
    vmax = float(mesh.vp.max())
    dt = args.cfl * mesh.model.get_min_spacing() / vmax
    nx, nz = ex * args.order + 1, ez * args.order + 1
    print(f"Mesh: {ex} x {ez} elements, order {args.order}, {nx * nz} nodes; "
          f"vp {mesh.vp.min():.0f}-{vmax:.0f} m/s; dt {dt * 1e3:.3f} ms")

    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=dt, minTime=0.0, maxTime=args.tmax, dtSeismo=args.dt_seismo,
                            sourceType="ricker2", sourceFreq=args.f0, **mesh.solver_kwargs())
    solver.initialize()
    rcv_x = np.arange(args.rcv_dx / 2, lx, args.rcv_dx)
    receivers = [[x, 0.0, lz - args.rcv_depth] for x in rcv_x]
    solver.updateSourceAndReceiversFromCoords([[args.src_x, 0.0, lz - args.src_depth]], receivers)
    solver.evaluateSource()
    print(f"Setup {time.time() - t0:.1f} s; {solver.n_time_steps} steps, {len(rcv_x)} receivers")

    t0 = time.time()
    snap_step, snapshot = int(round(args.snap_time / dt)), None
    for t in range(solver.n_time_steps):
        solver.execute(t)
        if t == snap_step:
            snapshot = np.array(solver.data.get_current_field(0), copy=True).reshape(nz, nx)
        if t % 2000 == 0:
            print(f"  step {t:6d} / {solver.n_time_steps}  (t = {t * dt:.2f} s)")
    gather = solver.getPressureAtReceivers()
    print(f"Time loop {time.time() - t0:.1f} s; gather {gather.shape}")

    with h5py.File(args.output + ".h5", "w") as f:
        f.create_dataset("pressure", data=gather)
        f.create_dataset("receiver_x", data=rcv_x)
        f.attrs.update(dt_seismo=args.dt_seismo, f0=args.f0, order=args.order, source_x=args.src_x,
                       source_depth=args.src_depth, receiver_depth=args.rcv_depth)
    print(f"Wrote {args.output}.h5")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(10, 11))
        im = ax[0].imshow(mesh.vp[::-1], extent=[0, lx / 1e3, lz / 1e3, 0], aspect="auto", cmap="viridis")
        ax[0].plot(args.src_x / 1e3, args.src_depth / 1e3, "r*", ms=12)
        ax[0].set(title="Marmousi Vp (m/s)", xlabel="x (km)", ylabel="depth (km)")
        fig.colorbar(im, ax=ax[0])
        if snapshot is not None:
            clip = np.percentile(np.abs(snapshot), 99.5) or 1.0
            ax[1].imshow(snapshot[::-1], extent=[0, lx / 1e3, lz / 1e3, 0], aspect="auto", cmap="seismic",
                         vmin=-clip, vmax=clip)
            ax[1].set(title=f"Pressure at t = {args.snap_time} s", xlabel="x (km)", ylabel="depth (km)")
        clip = np.percentile(np.abs(gather), 99) or 1.0
        ax[2].imshow(gather, extent=[0, lx / 1e3, gather.shape[0] * args.dt_seismo, 0], aspect="auto",
                     cmap="gray", vmin=-clip, vmax=clip)
        ax[2].set(title="Shot gather", xlabel="receiver x (km)", ylabel="time (s)")
        fig.tight_layout()
        fig.savefig(args.output + ".png", dpi=110)
        print(f"Wrote {args.output}.png")

    del solver, mesh


if __name__ == "__main__":
    main()
    kokkos.finalize()
