"""
One-shot acoustic profile modeling with FUnTiDES SEM.

Domain
------
    Homogeneous box  1500 x 1500 x 1500 m
    30 x 30 x 30 elements, polynomial order 2
    Velocity : 1500 m/s,  density : 1000 kg/m3

Acquisition
-----------
    Source  : 1 point source at the top surface, domain centre
    Profile : 31 receivers along X at constant Y = domain centre, z = 30 m depth

Usage
-----
    python acoustic_one_shot.py [--output seismos.hdf5]
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
    ix = int(round(x / (lx / (nx - 1))))
    iy = int(round(y / (ly / (ny - 1))))
    iz = int(round(z / (lz / (ex * order))))   # nz = ez*order+1 same formula
    ix = max(0, min(ix, nx - 1))
    iy = max(0, min(iy, ny - 1))
    nz = ez * order + 1
    iz = int(round(z / (lz / (nz - 1))))
    iz = max(0, min(iz, nz - 1))
    return ix + iy * nx + iz * nx * ny


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="One-shot acoustic FUnTiDES SEM example")
    parser.add_argument("--output", default="seismos_one_shot.hdf5",
                        help="Output HDF5 file for the seismogram (default: seismos_one_shot.hdf5)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Mesh / domain parameters
    # ------------------------------------------------------------------
    order = 2
    ex = ey = ez = 100          # elements per direction
    lx = ly = lz = 1500.0      # domain size [m]
    hx = lx / ex
    hy = ly / ey
    hz = lz / ez

    nx = ex * order + 1        # nodes per direction
    ny = ey * order + 1
    nz = ez * order + 1
    n_dof = nx * ny * nz
    print(f"Mesh: {ex}x{ey}x{ez} elements, order {order} → {n_dof} nodes")

    # ------------------------------------------------------------------
    # Build FUnTiDES model  (structured, acoustic, model on elements)
    # ------------------------------------------------------------------
    on_nodes = False
    is_elastic = False
    builder = FTModel.CartesianStructBuilder_f32_i32_O2(
        ex, hx, ey, hy, ez, hz, on_nodes, is_elastic
    )
    model = builder.get_model()
    print("Model built")

    # ------------------------------------------------------------------
    # Simulation time parameters
    # ------------------------------------------------------------------
    vp       = 1500.0      # homogeneous velocity [m/s]
    f0       = 5.0         # source peak frequency [Hz]
    dt       = 0.001       # time step [s]
    maxTime  = 1.5         # end time [s]
    dtSeismo = dt          # sample every time step

    # ------------------------------------------------------------------
    # Create solver
    # ------------------------------------------------------------------
    solver = AcousticSolver(
        model=model,
        order=order,
        implem_type=FTSolver.ImplemType.MAKUTU,
        mesh_type=FTSolver.MeshType.STRUCT,
        model_location=FTSolver.ModelLocationType.ONELEMENTS,
        dt=dt,
        minTime=0.0,
        maxTime=maxTime,
        dtSeismo=dtSeismo,
        sourceType="ricker2",
        sourceFreq=f0,
    )

    # ------------------------------------------------------------------
    # Source: 1 point source at the top surface, domain centre
    # ------------------------------------------------------------------
    src_x = lx / 2.0
    src_y = ly / 2.0
    src_z = 0.0   # top surface (z = 0)

    src_elem = coord_to_element(src_x, src_y, src_z + hz / 2,
                                 ex, ey, ez, lx, ly, lz)
    print(f"Source  : ({src_x:.1f}, {src_y:.1f}, {src_z:.1f}) → element {src_elem}")

    # ------------------------------------------------------------------
    # Receivers: profile of 31 receivers along X at Y = ly/2, z = 30 m
    # ------------------------------------------------------------------
    n_receivers = 31
    rcv_z = 30.0
    rcv_y = ly / 2.0
    rcv_x_positions = np.linspace(0.0, lx, n_receivers)

    receiver_node_indices = [
        coord_to_node(x, rcv_y, rcv_z, ex, ey, ez, lx, ly, lz, order)
        for x in rcv_x_positions
    ]
    print(f"Receivers: {n_receivers} along X at y={rcv_y:.0f} m, z={rcv_z:.0f} m")

    # ------------------------------------------------------------------
    # Initialize solver and set source / receivers
    # ------------------------------------------------------------------
    solver.initialize()
    print("Solver initialized")

    solver.updateSourceAndReceivers(
        sourceElementIndices=[src_elem],
        receiverNodeIndices=receiver_node_indices,
    )

    # Evaluate the Ricker source waveform and populate kk_RHSTerm
    solver.evaluateSource()
    print("Source evaluated")

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    n_steps = solver.n_time_steps
    print(f"Time loop: {n_steps} steps, dt={dt} s, maxTime={maxTime} s")

    for t in range(n_steps):
        if t % 500 == 0:
            print(f"  step {t:5d} / {n_steps}  (t = {t * dt:.3f} s)")
        solver.execute(t)

    print("Simulation done")

    # ------------------------------------------------------------------
    # Retrieve and save seismogram
    # ------------------------------------------------------------------
    seismogram = solver.getPressureAtReceivers()  # [n_seismo, n_receivers]
    print(f"Seismogram shape: {seismogram.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("seismogram", data=seismogram)
        f.create_dataset("receiver_x", data=rcv_x_positions)
        f.attrs["dt"]        = dtSeismo
        f.attrs["maxTime"]   = maxTime
        f.attrs["f0"]        = f0
        f.attrs["vp"]        = vp
        f.attrs["source_x"]  = src_x
        f.attrs["source_y"]  = src_y
        f.attrs["source_z"]  = src_z
    print(f"Seismogram saved to {args.output}")

if __name__ == "__main__":
    main()
    kokkos.finalize()
