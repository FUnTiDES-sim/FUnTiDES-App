"""
Acoustic seismic modeling workflow with FUnTiDES SEM.

Reads acquisition geometry from a native decision file,
runs one forward shot per source, and exports seismograms as SEG-Y
(requires segyio) or HDF5 as fallback.

Usage
-----
    python acoustic_modeling.py \\
        --decision  acquisition.dec  \\
        --database  coordinates.db   \\
        --lx 5000 --ly 5000 --lz 2000 \\
        --ex 50   --ey 50   --ez 20   \\
        --vp 1500 --dt 0.001 --tmax 2.0 --f0 5 \\
        --output-dir seismoTraces
"""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import h5py

# MPI must be initialized before any MPI-dependent output (e.g. SEGYTraceOutput)
from mpi4py import MPI

# kokkos MUST be initialized before any pyfuntides constructor is called
import kokkos
kokkos.initialize()

import pyfuntides.model as FTModel
import pyfuntides.solver as FTSolver

from utilities.solvers import AcousticSolver

from utilities.acquisition.decision_reader import read_decision_file

try:
    from utilities.output import SeismicTraceOutput
    _HAS_SEGYIO = True
except ImportError:
    _HAS_SEGYIO = False


# ---------------------------------------------------------------------------
# Acquisition reading
# ---------------------------------------------------------------------------

def read_acquisition(args):
    """
    Read source and receiver coordinates from the decision file.

    Returns
    -------
        shots : list of dict with keys 'id', 'src_coords', 'rcv_coords'
    """

    print(f"Reading decision file: {args.decision}")
    d_src, d_rcv, n_sources = read_decision_file(
        args.decision,
        database_file=args.database,
        utmxy2xy=(args.database != "identity"),
        scalco=args.scalco,
        scalel=args.scalel,
        zopp=args.zopp,
        swords=args.swords,
        rwords=args.rwords,
    )
    print(f"  {n_sources} shot(s) found")

    shots = []
    for i in range(1, n_sources + 1):
        src_key = f"src{i}"
        rcv_key = f"rcv_{i}"
        if src_key not in d_src or rcv_key not in d_rcv:
            continue
        shots.append({
            "id":         f"{i:05d}",
            "src_coords": [d_src[src_key]],
            "rcv_coords": d_rcv[rcv_key],
        })
    return shots


# ---------------------------------------------------------------------------
# Per-shot processing
# ---------------------------------------------------------------------------

def process_shot(shot, solver, output_dir, output_format, scalco, scalel):
    """
    Run one forward shot and save the seismogram.

    Parameters
    ----------
        shot : dict   keys: id, src_coords, rcv_coords
        solver : AcousticSolver  (must have lx, ly, lz, ex, ey, ez set)
        output_dir : str
        output_format : str  'segy' or 'hdf5'
        scalco, scalel : int  SEG-Y coordinate scalers
    """
    src_coords = shot["src_coords"]
    rcv_coords = shot["rcv_coords"]

    if not rcv_coords:
        print(f"  SHOT {shot['id']}: no receivers — skipped")
        return

    print(f"  SHOT {shot['id']}: {len(src_coords)} src, {len(rcv_coords)} rcv")

    solver.resetWaveField()
    solver.updateSourceAndReceiversFromCoords(src_coords, rcv_coords)
    solver.evaluateSource()

    n_steps = solver.n_time_steps
    for t in range(n_steps):
        if t % 100 == 0:
            print(f"    step {t:5d} / {n_steps}  (t = {t * solver.dt:.3f} s)")
        solver.execute(t)

    seismogram = solver.getPressureAtReceivers()

    os.makedirs(output_dir, exist_ok=True)
    rootname = f"seismo_{shot['id']}"

    if output_format == "segy" and _HAS_SEGYIO:
        SeismicTraceOutput(seismogram, format="SEGY").export(
            directory=output_dir,
            rootname=rootname,
            receiverCoords=rcv_coords,
            sourceCoords=src_coords[0],
            dt=solver.dtSeismo,
            scalco=scalco,
            scalel=scalel,
        )
    else:
        if output_format == "segy" and not _HAS_SEGYIO:
            print("  WARNING: segyio not available, falling back to HDF5")
        outfile = os.path.join(output_dir, rootname + ".hdf5")
        with h5py.File(outfile, "w") as f:
            f.create_dataset("seismogram",      data=seismogram)
            f.create_dataset("receiver_coords", data=np.array(rcv_coords))
            f.create_dataset("source_coords",   data=np.array(src_coords))
            f.attrs["dt"]      = solver.dtSeismo
            f.attrs["shot_id"] = shot["id"]
        print(f"  → {outfile}  shape={seismogram.shape}")

    print(f"  SHOT {shot['id']} DONE")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Acoustic modeling workflow — FUnTiDES SEM")

    # Acquisition
    p.add_argument("--decision",  type=str, required=True,
                   help="Decision file")
    p.add_argument("--database",  type=str, default="identity",
                   help="Database file for coordinate conversion (default: identity)")
    p.add_argument("--scalco",    type=int, default=1,
                   help="Coordinate scaling factor (default: 1)")
    p.add_argument("--scalel",    type=int, default=1,
                   help="Elevation scaling factor (default: 1)")
    p.add_argument("--zopp",      action="store_true", default=False,
                   help="Flip Z coordinate sign")
    p.add_argument("--swords",    nargs="+", type=int, default=None,
                   help="Source word positions in decision file (default: 22 23 14)")
    p.add_argument("--rwords",    nargs="+", type=int, default=None,
                   help="Receiver word positions in decision file (default: 24 25 13)")

    # Mesh
    p.add_argument("--lx",    type=float, required=True, help="Domain length X [m]")
    p.add_argument("--ly",    type=float, required=True, help="Domain length Y [m]")
    p.add_argument("--lz",    type=float, required=True, help="Domain length Z [m]")
    p.add_argument("--ex",    type=int,   required=True, help="Elements in X direction")
    p.add_argument("--ey",    type=int,   required=True, help="Elements in Y direction")
    p.add_argument("--ez",    type=int,   required=True, help="Elements in Z direction")
    p.add_argument("--order", type=int,   default=2,     help="Polynomial order (default: 2)")

    # Simulation
    p.add_argument("--vp",   type=float, default=1500.0, help="P-wave velocity [m/s] (default: 1500)")
    p.add_argument("--dt",   type=float, required=True,  help="Time step [s]")
    p.add_argument("--tmax", type=float, required=True,  help="End time [s]")
    p.add_argument("--f0",   type=float, default=5.0,    help="Source peak frequency [Hz] (default: 5)")

    # Output
    p.add_argument("--output-dir",    default="seismoTraces",
                   help="Output directory (default: seismoTraces)")
    p.add_argument("--output-format", choices=["segy", "hdf5"], default="segy",
                   help="Output format (default: segy)")

    args, _ = p.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Mesh / domain parameters
    # ------------------------------------------------------------------
    order = args.order
    ex, ey, ez = args.ex, args.ey, args.ez
    lx, ly, lz = args.lx, args.ly, args.lz
    hx, hy, hz = lx / ex, ly / ey, lz / ez

    print(f"Mesh: {ex}x{ey}x{ez} elements, order {order}, "
          f"domain {lx:.0f}x{ly:.0f}x{lz:.0f} m")

    # ------------------------------------------------------------------
    # Build FUnTiDES model
    # ------------------------------------------------------------------
    builder = FTModel.CartesianStructBuilder_f32_i32_O2(
        ex, hx, ey, hy, ez, hz, False, False
    )
    model = builder.get_model()
    print("Model built")

    # ------------------------------------------------------------------
    # Create solver (shared across all shots)
    # ------------------------------------------------------------------
    solver = AcousticSolver(
        model=model,
        order=order,
        implem_type=FTSolver.ImplemType.MAKUTU,
        mesh_type=FTSolver.MeshType.STRUCT,
        model_location=FTSolver.ModelLocationType.ONELEMENTS,
        dt=args.dt,
        minTime=0.0,
        maxTime=args.tmax,
        dtSeismo=args.dt,
        sourceType="ricker2",
        sourceFreq=args.f0,
        lx=lx, ly=ly, lz=lz,
        ex=ex, ey=ey, ez=ez,
    )
    solver.initialize()
    print("Solver initialized")

    # ------------------------------------------------------------------
    # Read acquisition
    # ------------------------------------------------------------------
    shots = read_acquisition(args)
    print(f"Acquisition: {len(shots)} shot(s)")

    # ------------------------------------------------------------------
    # Main shot loop
    # ------------------------------------------------------------------
    for ishot, shot in enumerate(shots):
        print(f"\nShot {ishot + 1}/{len(shots)}  (id={shot['id']})")
        process_shot(
            shot, solver,
            args.output_dir, args.output_format,
            args.scalco, args.scalel,
        )

    print("\nAll shots done.")


if __name__ == "__main__":
    main()
    kokkos.finalize()
