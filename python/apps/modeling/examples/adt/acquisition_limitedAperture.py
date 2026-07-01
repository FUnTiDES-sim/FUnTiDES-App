"""
Multi-shot acoustic modeling with optional limited aperture, using FUnTiDES SEM.

Domain
------
    Homogeneous box  1000 x 1000 x 200 m
    20 x 20 x 4 elements, polynomial order 2
    Velocity : 1500 m/s,  density : 1000 kg/m3

Acquisition
-----------
    ACQ 1 : 3 sources at X=[299, 499] m, Y=499 m, Z=20 m
    ACQ 2 : 3 sources at X=[599, 799] m, Y=499 m, Z=20 m
    Receivers : 10 lines x 30 rcv, X=[20,960] m, Y=[399,699] m, Z=30 m

Limited aperture (optional)
----------------------------
    Receivers are kept only within (l1, l2, l3) half-widths of each source.

Usage
-----
    python acquisition_limitedAperture.py [--LA] [--l1 L1] [--l2 L2] [--l3 L3]
                                          [--output-dir DIR]
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

from utilities.acquisition.Shot import Shot, Source, SourceSet, Receiver, ReceiverSet
from utilities.solvers import AcousticSolver


# ---------------------------------------------------------------------------
# Helpers: coordinate → mesh index conversions (structured Cartesian)
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
# Acquisition helpers
# ---------------------------------------------------------------------------

def make_shots(source_coords, receiver_coords):
    """Build a list of Shot objects: one source per shot, shared full receiver set."""
    full_rcv_list = [Receiver(*r) for r in receiver_coords]
    shots = []
    for i, src in enumerate(source_coords):
        src_set = SourceSet([Source(*src)])
        rcv_set = ReceiverSet(list(full_rcv_list))   # copy so per-shot filtering is independent
        shots.append(Shot(src_set, rcv_set, shotId=f"{i + 1:05d}"))
    return shots


def apply_limited_aperture(shot, l1, l2, l3):
    """
    Filter receivers to those within (l1, l2, l3) half-widths of the source centre.

    Parameters
    ----------
        shot : Shot
        l1 : float   half-width in X [m]
        l2 : float   half-width in Y [m]
        l3 : float   half-width in Z [m]
    """
    src_coords = shot.getSourceCoords()
    cx, cy, cz = np.mean(src_coords, axis=0)
    bounds = (cx - l1, cx + l1, cy - l2, cy + l2, cz - l3, cz + l3)
    shot.receivers.keepReceiversWithinBounds(bounds)


# ---------------------------------------------------------------------------
# Per-shot processing
# ---------------------------------------------------------------------------

def process_shot(shot, acq_id, solver, mesh_params, output_dir):
    """
    Run one forward shot with FUnTiDES and save the seismogram.

    Parameters
    ----------
        shot : Shot
        acq_id : str
        solver : AcousticSolver
        mesh_params : dict
            Keys: ex, ey, ez, lx, ly, lz, order
        output_dir : str
    """
    ex, ey, ez = mesh_params["ex"], mesh_params["ey"], mesh_params["ez"]
    lx, ly, lz = mesh_params["lx"], mesh_params["ly"], mesh_params["lz"]
    order = mesh_params["order"]

    src_coords = shot.getSourceCoords()
    rcv_coords = shot.getReceiverCoords()

    if not rcv_coords:
        print(f"  ACQ {acq_id} SHOT {shot.id}: no receivers after aperture filter — skipped")
        return

    # Map physical coordinates to FUnTiDES element / node indices
    src_elements = [
        coord_to_element(x, y, max(z, lz / ez * 0.01), ex, ey, ez, lx, ly, lz)
        for x, y, z in src_coords
    ]
    rcv_nodes = [
        coord_to_node(x, y, z, ex, ey, ez, lx, ly, lz, order)
        for x, y, z in rcv_coords
    ]

    print(f"  ACQ {acq_id} SHOT {shot.id}: "
          f"{len(src_elements)} src elem(s), {len(rcv_nodes)} rcv node(s)")

    # Reset wavefield from previous shot, then configure for this shot
    solver.resetWaveField()
    solver.updateSourceAndReceivers(src_elements, rcv_nodes)
    solver.evaluateSource()

    n_steps = solver.n_time_steps
    for t in range(n_steps):
        if t % 100 == 0:
            print(f"    step {t:5d} / {n_steps}  (t = {t * solver.dt:.3f} s)")
        solver.execute(t)

    seismogram = solver.getPressureAtReceivers()

    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"seismos_ACQ{acq_id}_SHOT{shot.id}.hdf5")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("seismogram",      data=seismogram)
        f.create_dataset("receiver_coords", data=np.array(rcv_coords))
        f.create_dataset("source_coords",   data=np.array(src_coords))
        f.attrs["dt"]      = solver.dtSeismo
        f.attrs["acq_id"]  = acq_id
        f.attrs["shot_id"] = shot.id
    print(f"  → {outfile}  shape={seismogram.shape}")

    print(f"  ACQ {acq_id} SHOT {shot.id} DONE")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-shot limited-aperture FUnTiDES acoustic workflow"
    )
    p.add_argument("--LA", "--limited_aperture", dest="flagLA",
                   action="store_true", default=False,
                   help="Enable limited aperture acquisition")
    p.add_argument("--l1", type=float, default=None,
                   help="Half-width in X for limited aperture [m]")
    p.add_argument("--l2", type=float, default=None,
                   help="Half-width in Y for limited aperture [m]")
    p.add_argument("--l3", type=float, default=None,
                   help="Half-width in Z for limited aperture [m]")
    p.add_argument("--output-dir", default="seismoTraces",
                   help="Output directory for seismograms (default: seismoTraces)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    flagLA = args.flagLA
    if flagLA and not (args.l1 and args.l2 and args.l3):
        print("WARNING: --LA set but --l1/--l2/--l3 not fully provided. "
              "Switching to full aperture.")
        flagLA = False

    # ------------------------------------------------------------------
    # Mesh / domain parameters
    # ------------------------------------------------------------------
    order = 2
    ex, ey, ez = 20, 20, 4
    lx, ly, lz = 1000.0, 1000.0, 200.0
    hx, hy, hz = lx / ex, ly / ey, lz / ez

    mesh_params = dict(ex=ex, ey=ey, ez=ez, lx=lx, ly=ly, lz=lz, order=order)

    print(f"Mesh: {ex}x{ey}x{ez} elements, order {order}")

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    f0   = 5.0
    dt   = 0.001
    tmax = 0.5

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
        dt=dt,
        minTime=0.0,
        maxTime=tmax,
        dtSeismo=dt,
        sourceType="ricker2",
        sourceFreq=f0,
    )
    solver.initialize()
    print("Solver initialized")

    # ------------------------------------------------------------------
    # Acquisition geometry
    # ------------------------------------------------------------------
    # Receiver grid: 10 lines x 30 receivers
    rcv_x_positions = np.linspace(20.0, 960.0, 10)
    rcv_y_positions = np.linspace(399.0, 699.0, 30)
    rcv_z = 30.0
    receiver_coords = [
        [x, y, rcv_z]
        for x in rcv_x_positions
        for y in rcv_y_positions
    ]

    # ACQ 1: 3 sources at X=[299, 499], Y=499, Z=20
    src_coords_acq1 = [[x, 499.0, 20.0] for x in np.linspace(299.0, 499.0, 3)]

    # ACQ 2: 3 sources at X=[599, 799], Y=499, Z=20
    src_coords_acq2 = [[x, 499.0, 20.0] for x in np.linspace(599.0, 799.0, 3)]

    acquisitions = [
        ("00001", src_coords_acq1),
        ("00002", src_coords_acq2),
    ]

    # ------------------------------------------------------------------
    # Main loop: acquisitions → shots
    # ------------------------------------------------------------------
    for acq_id, src_coords in acquisitions:
        shots = make_shots(src_coords, receiver_coords)
        print(f"\nACQ {acq_id}: {len(shots)} shots")

        if flagLA:
            for shot in shots:
                apply_limited_aperture(shot, args.l1, args.l2, args.l3)
                n_rcv = len(shot.getReceiverCoords())
                print(f"  SHOT {shot.id}: {n_rcv} receivers after limited aperture")

        for shot in shots:
            process_shot(shot, acq_id, solver, mesh_params, args.output_dir)

    print("\nAll shots done.")


if __name__ == "__main__":
    main()
    kokkos.finalize()
