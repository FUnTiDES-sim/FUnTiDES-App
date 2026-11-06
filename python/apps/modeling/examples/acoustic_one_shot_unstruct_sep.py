"""
One-shot acoustic modeling from a SEP velocity model — unstructured mesh.

Workflow
--------
1. Read a SEP Vp file to determine the full domain extent (lx, ly, lz).
2. Derive mesh and time-stepping parameters from the model (CFL).
3. Build a heterogeneous unstructured Cartesian mesh (ModelUnstruct backend)
   with per-element Vp and rho from the SEP files.
4. Place one source near the surface at (lx/2, ly/2, z_src).
5. Place a line of receivers along X at y = ly/2, z = z_rcv.
6. Run the forward simulation and save the seismogram to HDF5.

Usage
-----
    python acoustic_one_shot_unstruct_sep.py --vp-file sep_models/vp_layered.H \\
                             [--rho-file sep_models/vp_layered_rho.H] \\
                             [--elem-per-wavelength 8] \\
                             [--f0 10.0] \\
                             [--tmax 2.0] \\
                             [--order 2] \\
                             [--n-rcv 200] \\
                             [--output-dir seismoTraces]
"""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import h5py

# kokkos MUST be initialized before any pyfuntides constructor is called
import kokkos
kokkos.initialize()

import pyfuntides.solver as FTSolver

from makutu.input.SEPReader import read_sep
from makutu.mesh.InternalMesh import StructuredMesh
from makutu.solvers import AcousticSolver


# ---------------------------------------------------------------------------
# CFL / mesh helpers
# ---------------------------------------------------------------------------

def suggest_mesh(lx, ly, lz, vp_max, f0, elem_per_wavelength, order):
    """
    Suggest element counts (ex, ey, ez) and time step dt.

    Rule of thumb:
      element size  h ≤ λ_min / elem_per_wavelength
                    λ_min = vp_max / (2 * f0)
      CFL           dt ≤ h / (vp_max * order²)
    """
    lambda_min = vp_max / (2.0 * f0)
    h_max = lambda_min / elem_per_wavelength

    ex = max(1, int(np.ceil(lx / h_max)))
    ey = max(1, int(np.ceil(ly / h_max)))
    ez = max(1, int(np.ceil(lz / h_max)))

    hx = lx / ex
    hy = ly / ey
    hz = lz / ez
    h_min = min(hx, hy, hz)

    dt = 0.4 * h_min / (vp_max * order ** 2)

    return ex, ey, ez, dt


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--vp-file",  required=True,
                   help="SEP .H header for Vp model [m/s]")
    p.add_argument("--rho-file", default=None,
                   help="SEP .H header for density (optional, default: 1.0 everywhere)")
    p.add_argument("--elem-per-wavelength", type=float, default=8.0,
                   help="Elements per minimum wavelength (default: 8)")
    p.add_argument("--f0",   type=float, default=10.0,
                   help="Source peak frequency [Hz] (default: 10)")
    p.add_argument("--tmax", type=float, default=None,
                   help="Simulation end time [s] (default: 3 * lz / vp_min)")
    p.add_argument("--order", type=int, default=2, choices=[1, 2, 3],
                   help="Polynomial order (default: 2)")
    p.add_argument("--n-rcv", type=int, default=200,
                   help="Number of receivers along X (default: 200)")
    p.add_argument("--sponge", type=float, default=None,
                   help="Absorbing sponge thickness [m] (default: 5%% of min extent)")
    p.add_argument("--output-dir", default="seismoTraces",
                   help="Output directory for seismogram HDF5 (default: seismoTraces)")
    p.add_argument("--snapshots", action="store_true",
                   help="Save XZ pressure snapshots every --snapshot-interval steps")
    p.add_argument("--snapshot-interval", type=int, default=100,
                   help="Steps between pressure snapshots (default: 100)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_snapshot(solver, mesh, order, t_step, dt, snap_dir):
    """Save an XZ mid-Y pressure cross-section to snap_dir/snapshot_NNNNNN.png."""
    import matplotlib.pyplot as plt

    nx = mesh.ex * order + 1
    ny = mesh.ey * order + 1
    nz = mesh.ez * order + 1

    pn = np.array(solver.kk_pn_curr, copy=False).reshape(nz, ny, nx)
    xz = pn[::-1, ny // 2, :]     # shape (nz, nx), top (z = lz, depth 0) first

    vmax = float(np.abs(xz).max()) or 1.0
    x0, x1 = mesh.origin_x, mesh.origin_x + mesh.lx

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        xz, origin="upper", aspect="auto",
        extent=[x0, x1, mesh.lz, 0.0],
        cmap="seismic", vmin=-vmax, vmax=vmax,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(f"Pressure  —  t = {t_step * dt:.4f} s  (step {t_step})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pressure")
    fig.tight_layout()
    os.makedirs(snap_dir, exist_ok=True)
    fig.savefig(os.path.join(snap_dir, f"snapshot_{t_step:06d}.png"), dpi=100)
    plt.close(fig)


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Read Vp SEP model — derive domain extents and velocity statistics
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"Reading Vp model: {args.vp_file}")
    sep_vp = read_sep(args.vp_file)

    lx = sep_vp.n1 * sep_vp.d1
    ly = sep_vp.n2 * sep_vp.d2
    lz = sep_vp.n3 * sep_vp.d3

    vp_min = float(sep_vp.data.min())
    vp_max = float(sep_vp.data.max())

    print(f"  Grid   : nx={sep_vp.n1}  ny={sep_vp.n2}  nz={sep_vp.n3}")
    print(f"  Spacing: dx={sep_vp.d1:.1f}  dy={sep_vp.d2:.1f}  dz={sep_vp.d3:.1f} m")
    print(f"  Domain : lx={lx:.0f}  ly={ly:.0f}  lz={lz:.0f} m")
    print(f"  Vp     : min={vp_min:.0f}  max={vp_max:.0f} m/s")

    # -----------------------------------------------------------------------
    # 2. Mesh and time-step parameters
    # -----------------------------------------------------------------------
    ex, ey, ez, dt = suggest_mesh(
        lx, ly, lz, vp_max, args.f0, args.elem_per_wavelength, args.order
    )
    hx, hy, hz = lx / ex, ly / ey, lz / ez

    tmax = args.tmax if args.tmax is not None else 3.0 * lz / vp_min
    dtSeismo = dt

    sponge = args.sponge if args.sponge is not None else 0.05 * min(lx, ly, lz)

    print(f"\nMesh parameters (derived):")
    print(f"  Elements : {ex} x {ey} x {ez}  (h ≈ {hx:.1f} x {hy:.1f} x {hz:.1f} m)")
    print(f"  Order    : {args.order}")
    print(f"  dt       : {dt:.6f} s   (CFL-limited)")
    print(f"  tmax     : {tmax:.3f} s")
    print(f"  Steps    : {int(tmax / dt)}")
    print(f"  Sponge   : {sponge:.1f} m on all sides")

    # -----------------------------------------------------------------------
    # 3. Build heterogeneous unstructured mesh
    # -----------------------------------------------------------------------
    print("\nBuilding mesh ...")
    mesh = StructuredMesh(
        lx=lx, ly=ly, lz=lz,
        ex=ex, ey=ey, ez=ez,
        order=args.order,
        on_nodes=False,
        is_elastic=False,
        free_surface_on_top=True,
        vp_file=args.vp_file,
        rho_file=args.rho_file,
    )
    print(f"  Nodes  : {mesh.numberOfPoints}")
    print(f"  Cells  : {mesh.numberOfCells}")
    print(f"  Backend: {'ModelUnstruct (heterogeneous)' if mesh.heterogeneous else 'ModelStruct'}")

    # -----------------------------------------------------------------------
    # 3b. Display the Vp (and rho) model on the unstructured grid
    # -----------------------------------------------------------------------
    if mesh.heterogeneous:
        print("\nDisplaying Vp model ...")
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend for headless runs
        vp_fig = mesh.plot_model(show=False)
        vp_fig_path = os.path.join(args.output_dir, "vp_model_slices.png")
        os.makedirs(args.output_dir, exist_ok=True)
        vp_fig.savefig(vp_fig_path, dpi=150, bbox_inches="tight")
        print(f"  → {vp_fig_path}")
        import matplotlib.pyplot as plt
        plt.close(vp_fig)

    # -----------------------------------------------------------------------
    # 4. Create acoustic solver
    # -----------------------------------------------------------------------
    print("\nCreating solver ...")
    solver = AcousticSolver(
        model=mesh.model,
        order=args.order,
        implem_type=FTSolver.ImplemType.MAKUTU,
        mesh_type=FTSolver.MeshType.UNSTRUCT if mesh.heterogeneous else FTSolver.MeshType.STRUCT,
        model_location=FTSolver.ModelLocationType.ONELEMENTS,
        sponge_size=[sponge, sponge, sponge],
        surface_sponge=False,
        taper_delta=0.015,
        dt=dt,
        minTime=0.0,
        maxTime=tmax,
        dtSeismo=dtSeismo,
        sourceType="ricker2",
        sourceFreq=args.f0,
        lx=mesh.lx, ly=mesh.ly, lz=mesh.lz,
        ex=mesh.ex, ey=mesh.ey, ez=mesh.ez,
        origin_x=mesh.origin_x,
        kk_model_vp=mesh.kk_model_vp,
        kk_model_rho=mesh.kk_model_rho,
    )
    solver.initialize()
    print("  Solver initialized")

    # -----------------------------------------------------------------------
    # 5. Acquisition geometry
    # -----------------------------------------------------------------------
    z_src = min(2.0 * hz, 0.02 * lz)
    z_rcv = min(1.0 * hz, 0.01 * lz)

    src_coords = [[lx / 2.0, ly / 2.0, z_src]]

    rcv_margin = 2.0 * hx
    rcv_x = np.linspace(rcv_margin, lx - rcv_margin, args.n_rcv)
    rcv_coords = [[x, ly / 2.0, z_rcv] for x in rcv_x]

    print(f"\nAcquisition:")
    print(f"  Source   : x={src_coords[0][0]:.1f}  y={src_coords[0][1]:.1f}"
          f"  z={src_coords[0][2]:.1f} m")
    print(f"  Receivers: {len(rcv_coords)} along X  "
          f"x=[{rcv_x[0]:.1f} … {rcv_x[-1]:.1f}]  "
          f"y={ly/2.0:.1f}  z={z_rcv:.1f} m")

    # -----------------------------------------------------------------------
    # 6. Forward simulation
    # -----------------------------------------------------------------------
    print("\nRunning forward simulation ...")
    solver.resetWaveField()
    solver.updateSourceAndReceiversFromCoords(src_coords, rcv_coords)
    solver.evaluateSource()

    snap_dir = os.path.join(args.output_dir, "snapshots")
    n_steps = solver.n_time_steps
    for t in range(n_steps):
        if t % max(1, n_steps // 20) == 0:
            print(f"  step {t:6d} / {n_steps}  (t = {t * dt:.3f} s)")
        solver.execute(t)
        if args.snapshots and t % args.snapshot_interval == 0:
            _save_snapshot(solver, mesh, args.order, t, dt, snap_dir)

    seismogram = solver.getPressureAtReceivers()
    print(f"  Seismogram shape: {seismogram.shape}  "
          f"(n_seismo_steps × n_receivers)")

    # -----------------------------------------------------------------------
    # 7. Save seismogram
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, "seismo_unstruct_from_sep.hdf5")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("seismogram",      data=seismogram)
        f.create_dataset("receiver_coords", data=np.array(rcv_coords))
        f.create_dataset("source_coords",   data=np.array(src_coords))
        f.attrs["dt"]      = dtSeismo
        f.attrs["f0"]      = args.f0
        f.attrs["lx"]      = lx
        f.attrs["ly"]      = ly
        f.attrs["lz"]      = lz
        f.attrs["vp_file"] = args.vp_file
        f.attrs["mesh"]    = "unstruct"
    print(f"\n→ {outfile}")
    print("Done.")


if __name__ == "__main__":
    main()
    kokkos.finalize()
