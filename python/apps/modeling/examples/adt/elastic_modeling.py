"""
Elastic seismic modeling workflow with FUnTiDES SEM.

Reads acquisition geometry from a native decision file,
runs one forward shot per source, and exports three-component (X, Y, Z)
displacement seismograms in SEP format (requires mpi4py) or HDF5 as fallback.

Usage
-----
    python elastic_modeling.py \
        --decision  acquisition.dec  \
        --database  coordinates.db   \
        --lx 5000 --ly 5000 --lz 2000 \
        --ex 50   --ey 50   --ez 20   \
        --vp 3000 --vs 1700 --dt 0.001 --tmax 2.0 --f0 5 \
        --source-components z \
        --output-dir seismoTraces
"""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import h5py

# kokkos MUST be initialized before any pyfuntides constructor is called
import kokkos
kokkos.initialize()

import pyfuntides.model as FTModel
import pyfuntides.solver as FTSolver

from utilities.solvers import ElasticSolver

try:
    from utilities.acquisition.decision_reader import read_decision_file
    _HAS_DECISION_READER = True
except ImportError:
    _HAS_DECISION_READER = False

try:
    from utilities.output import SeismicTraceOutput
    _HAS_SEP = True
except ImportError:
    _HAS_SEP = False


# ---------------------------------------------------------------------------
# Helpers: coordinate -> mesh index conversions (structured Cartesian)
# ---------------------------------------------------------------------------

def coord_to_element(x, y, z, ex, ey, ez, lx, ly, lz):
    """Return the element flat index containing physical coordinate (x, y, z)."""
    ix = min(int(x / (lx / ex)), ex - 1)
    iy = min(int(y / (ly / ey)), ey - 1)
    iz = min(int(z / (lz / ez)), ez - 1)
    return ix + iy * ex + iz * ex * ey


def coord_to_node(x, y, z, ex, ey, ez, lx, ly, lz, order):
    """Return the global node index nearest to physical coordinate (x, y, z)."""
    nx = ex * order + 1
    ny = ey * order + 1
    nz = ez * order + 1
    ix = int(round(x / (lx / (nx - 1))))
    iy = int(round(y / (ly / (ny - 1))))
    iz = int(round(z / (lz / (nz - 1))))
    ix = max(0, min(ix, nx - 1))
    iy = max(0, min(iy, ny - 1))
    iz = max(0, min(iz, nz - 1))
    return ix + iy * nx + iz * nx * ny


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
    if not _HAS_DECISION_READER:
        raise RuntimeError(
            "utilities.acquisition.decision_reader could not be imported "
            "(check that makutu 'utilities' is on PYTHONPATH, e.g. via env.sh)."
        )

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
# Per-shot output helpers
# ---------------------------------------------------------------------------

def _export_sep(seismo, directory, rootname, dt):
    """Export one seismogram component as a SEP (.H) file."""
    seismoOut = SeismicTraceOutput(seismo, format="SEP")
    seismoOut.export(
        directory=directory,
        rootname=rootname,
        dt=dt,
        tIncluded=False,
        verbose=True,
    )


def _export_hdf5(seis_x, seis_y, seis_z, rcv_coords, src_coords, dt, output_dir, rootname):
    """Export all three displacement components to a single HDF5 file."""
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, rootname + ".hdf5")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("displacement_x",  data=seis_x)
        f.create_dataset("displacement_y",  data=seis_y)
        f.create_dataset("displacement_z",  data=seis_z)
        f.create_dataset("receiver_coords", data=np.array(rcv_coords))
        f.create_dataset("source_coords",   data=np.array(src_coords))
        f.attrs["dt"] = dt
    print(f"  -> {outfile}  shape={seis_x.shape} (per component)")


# ---------------------------------------------------------------------------
# Per-shot processing
# ---------------------------------------------------------------------------

def process_shot(shot, solver, mesh_params, output_dir, output_format):
    """
    Run one forward elastic shot and save the three-component seismogram.

    Parameters
    ----------
        shot : dict   keys: id, src_coords, rcv_coords
        solver : ElasticSolver
        mesh_params : dict  ex, ey, ez, lx, ly, lz, order
        output_dir : str
        output_format : str  'sep' or 'hdf5'
    """
    ex, ey, ez = mesh_params["ex"], mesh_params["ey"], mesh_params["ez"]
    lx, ly, lz = mesh_params["lx"], mesh_params["ly"], mesh_params["lz"]
    order      = mesh_params["order"]

    src_coords = shot["src_coords"]
    rcv_coords = shot["rcv_coords"]

    if not rcv_coords:
        print(f"  SHOT {shot['id']}: no receivers -- skipped")
        return

    src_elements = [
        coord_to_element(x, y, max(z, lz / ez * 0.01), ex, ey, ez, lx, ly, lz)
        for x, y, z in src_coords
    ]
    rcv_nodes = [
        coord_to_node(x, y, z, ex, ey, ez, lx, ly, lz, order)
        for x, y, z in rcv_coords
    ]

    print(f"  SHOT {shot['id']}: {len(src_elements)} src elem(s), {len(rcv_nodes)} rcv node(s)")

    solver.resetWaveField()
    solver.updateSourceAndReceivers(src_elements, rcv_nodes)
    solver.evaluateSource()

    n_steps = solver.n_time_steps
    for t in range(n_steps):
        if t % 100 == 0:
            print(f"    step {t:5d} / {n_steps}  (t = {t * solver.dt:.3f} s)")
        solver.execute(t)

    seis_x, seis_y, seis_z = solver.getAllDisplacementAtReceivers()

    rootname = f"seismo_{shot['id']}_U"

    if output_format == "sep" and _HAS_SEP:
        os.makedirs(output_dir, exist_ok=True)
        for comp, seis in (("X", seis_x), ("Y", seis_y), ("Z", seis_z)):
            _export_sep(seis, output_dir, rootname + comp, solver.dtSeismo)
    else:
        if output_format == "sep" and not _HAS_SEP:
            print("  WARNING: SEP output not available, falling back to HDF5")
        _export_hdf5(seis_x, seis_y, seis_z,
                     rcv_coords, src_coords, solver.dtSeismo,
                     output_dir, rootname)

    print(f"  SHOT {shot['id']} DONE")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Elastic modeling workflow -- FUnTiDES SEM")

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
    p.add_argument("--vp",   type=float, default=3000.0, help="P-wave velocity [m/s] (default: 3000)")
    p.add_argument("--vs",   type=float, default=1700.0, help="S-wave velocity [m/s] (default: 1700)")
    p.add_argument("--dt",   type=float, required=True,  help="Time step [s]")
    p.add_argument("--tmax", type=float, required=True,  help="End time [s]")
    p.add_argument("--f0",   type=float, default=5.0,    help="Source peak frequency [Hz] (default: 5)")
    p.add_argument("--source-components", nargs="+", default=["z"],
                   metavar="COMP",
                   help="Displacement components driven by the source: x y z (default: z)")

    # Output
    p.add_argument("--output-dir",    default="seismoTraces",
                   help="Output directory (default: seismoTraces)")
    p.add_argument("--output-format", choices=["sep", "hdf5"], default="sep",
                   help="Output format (default: sep)")

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

    mesh_params = dict(ex=ex, ey=ey, ez=ez, lx=lx, ly=ly, lz=lz, order=order)

    print(f"Mesh: {ex}x{ey}x{ez} elements, order {order}, "
          f"domain {lx:.0f}x{ly:.0f}x{lz:.0f} m")

    # ------------------------------------------------------------------
    # Build FUnTiDES model  (is_elastic=True)
    # ------------------------------------------------------------------
    builder = FTModel.CartesianStructBuilder_f32_i32_O2(
        ex, hx, ey, hy, ez, hz, False, True
    )
    model = builder.get_model()
    print("Model built")

    # ------------------------------------------------------------------
    # Create solver (shared across all shots)
    # ------------------------------------------------------------------
    solver = ElasticSolver(
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
        sourceComponents=args.source_components,
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
        process_shot(shot, solver, mesh_params, args.output_dir, args.output_format)

    print("\nAll shots done.")


if __name__ == "__main__":
    main()
    kokkos.finalize()
