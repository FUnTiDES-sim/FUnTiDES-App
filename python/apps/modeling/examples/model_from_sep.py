"""
Acoustic modeling from a SEP velocity model.

Workflow
--------
1. Read a SEP Vp file to determine the full domain extent (lx, ly, lz).
2. Derive mesh and time-stepping parameters from the model.
3. Build the heterogeneous FUnTiDES mesh (ModelUnstruct backend).
4. Place one source near the surface at (lx/2, ly/2, z_src).
5. Place a line of receivers along X at y = ly/2, z = z_rcv.
6. Run the forward simulation and save the seismogram to HDF5.

Usage
-----
    python model_from_sep.py --vp-file sep_models/vp_layered.H \\
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

# CUDA_VISIBLE_DEVICES must be set BEFORE kokkos/CUDA initializes.
# On multi-GPU systems, leaving it unset causes UVM allocation failures
# (cudaErrorIllegalAddress inside CudaUVMSpace::impl_allocate).
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
                   help="Absorbing sponge thickness [m] (default: 5 %% of lx)")
    p.add_argument("--output-dir", default="seismoTraces",
                   help="Output directory for seismogram HDF5 (default: seismoTraces)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Read Vp SEP model — derive domain extents and velocity statistics
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"Reading Vp model: {args.vp_file}")
    sep_vp = read_sep(args.vp_file)

    # SEP axes: n1=X, n2=Y, n3=Z  →  data shape (nz, ny, nx)
    lx = sep_vp.n1 * sep_vp.d1   # total extent X [m]
    ly = sep_vp.n2 * sep_vp.d2   # total extent Y [m]
    lz = sep_vp.n3 * sep_vp.d3   # total extent Z [m]

    vp_min = float(sep_vp.data.min())
    vp_max = float(sep_vp.data.max())

    print(f"  Grid   : nx={sep_vp.n1}  ny={sep_vp.n2}  nz={sep_vp.n3}")
    print(f"  Spacing: dx={sep_vp.d1:.1f}  dy={sep_vp.d2:.1f}  dz={sep_vp.d3:.1f} m")
    print(f"  Domain : lx={lx:.0f}  ly={ly:.0f}  lz={lz:.0f} m")
    print(f"  Vp     : min={vp_min:.0f}  max={vp_max:.0f} m/s")

    # -----------------------------------------------------------------------
    # 2. Mesh and time-step parameters
    # -----------------------------------------------------------------------
    ex, ey, ez, dt_cfl = suggest_mesh(
        lx, ly, lz, vp_max, args.f0, args.elem_per_wavelength, args.order
    )
    dt = dt_cfl
    hx, hy, hz = lx / ex, ly / ey, lz / ez

    # Simulation duration: default = 3 * lz / vp_min (enough for reflections)
    tmax = args.tmax if args.tmax is not None else 3.0 * lz / vp_min
    dtSeismo = dt   # record every time step (can be coarsened if needed)

    # Sponge: default 5 % of the smallest horizontal extent
    sponge = args.sponge if args.sponge is not None else 0.05 * min(lx, ly, lz)

    print(f"\nMesh parameters (derived):")
    print(f"  Elements : {ex} x {ey} x {ez}  (h ≈ {hx:.1f} x {hy:.1f} x {hz:.1f} m)")
    print(f"  Order    : {args.order}")
    print(f"  dt       : {dt:.6f} s   (CFL-limited)")
    print(f"  tmax     : {tmax:.3f} s")
    print(f"  Steps    : {int(tmax / dt)}")
    print(f"  Sponge   : {sponge:.1f} m on all sides")

    # -----------------------------------------------------------------------
    # 3. Build heterogeneous mesh (StructuredMesh with SEP → ModelUnstruct)
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
    z_src = min(2.0 * hz, 0.02 * lz)   # near surface: 2 elements or 2% of depth
    z_rcv = min(1.0 * hz, 0.01 * lz)   # receivers shallower than source

    src_coords = [[lx / 2.0, ly / 2.0, z_src]]

    rcv_margin = 2.0 * hx                        # keep receivers off the boundary
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

    n_steps = solver.n_time_steps
    for t in range(n_steps):
        if t % max(1, n_steps // 20) == 0:
            print(f"  step {t:6d} / {n_steps}  (t = {t * dt:.3f} s)")
        solver.execute(t)

    seismogram = solver.getPressureAtReceivers()
    print(f"  Seismogram shape: {seismogram.shape}  "
          f"(n_seismo_steps × n_receivers)")

    # -----------------------------------------------------------------------
    # 7. Save seismogram
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, "seismo_from_sep.hdf5")
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
    print(f"\n→ {outfile}")
    print("Done.")


if __name__ == "__main__":
    main()
    kokkos.finalize()
