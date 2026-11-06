"""
Small forward shots shared by the integrated tests (and by mpi_shot.py).

Homogeneous box 1500 m x 1500 m x 1500 m, 30^3 elements of order 2,
vp = 1500 m/s, Ricker order 2 at 5 Hz (peak at t = 1 / f0 = 0.2 s).
Kokkos must be initialized by the caller.
"""

import numpy as np

import pyfuntides.model as FTModel
import pyfuntides.solver as FTSolver

from makutu.solvers import AcousticSolver, ElasticSolver

L = 1500.0          # domain size [m]
NE = 30             # elements per direction
ORDER = 2
VP = 1500.0         # default vp of the FUnTiDES builders [m/s]
F0 = 5.0            # source peak frequency [Hz]
T_PEAK = 1.0 / F0   # Ricker delay [s]

# Source at the centre of an element: the default source weights are uniform
# over the element, so its effective position is the element centre.
# Coordinates are [x, y, z], z downwards: z = 775 m is the FUnTiDES mesh height 725 m.
SOURCE = (775.0, 775.0, 775.0)


def build_model(mesh="struct", elastic=False):
    """Homogeneous FUnTiDES model on the whole box."""
    if mesh == "struct":
        builder = FTModel.CartesianStructBuilder_f32_i32_O2(
            NE, L, NE, L, NE, L, False, elastic)
        return builder.get_model(), FTSolver.MeshType.STRUCT
    params = FTModel.CartesianParams_f32_i32(ORDER, NE, NE, NE, L, L, L, False, elastic)
    return (FTModel.CartesianUnstructBuilder_f32_i32(params).get_model(),
            FTSolver.MeshType.UNSTRUCT)


def solver_kwargs(dt, tmax):
    return dict(order=ORDER, implem_type=FTSolver.ImplemType.MAKUTU,
                model_location=FTSolver.ModelLocationType.ONELEMENTS,
                dt=dt, minTime=0.0, maxTime=tmax, dtSeismo=dt,
                sourceType="ricker2", sourceFreq=F0,
                lx=L, ly=L, lz=L, ex=NE, ey=NE, ez=NE)


def run(solver):
    """Initialize-free time loop: evaluate the source and step to the end."""
    solver.evaluateSource()
    for t in range(solver.n_time_steps):
        solver.execute(t)


def acoustic_shot(receivers, mesh="struct", dt=0.001, tmax=1.0):
    """Pressure at the receivers (coordinates), shape (n_samples, n_receivers)."""
    model, mesh_type = build_model(mesh)
    solver = AcousticSolver(model=model, mesh_type=mesh_type, **solver_kwargs(dt, tmax))
    solver.initialize()
    solver.updateSourceAndReceiversFromCoords([list(SOURCE)], receivers)
    run(solver)
    return solver.getPressureAtReceivers().astype(np.float64)


def elastic_shot(receivers, dt=0.0005, tmax=0.8):
    """Displacement (ux, uy, uz) at the receivers, vertical source."""
    model, mesh_type = build_model("struct", elastic=True)
    solver = ElasticSolver(model=model, mesh_type=mesh_type, **solver_kwargs(dt, tmax))
    solver.initialize()
    solver.updateSourceAndReceiversFromCoords([list(SOURCE)], receivers)
    run(solver)
    return tuple(np.asarray(u, dtype=np.float64)
                 for u in solver.getAllDisplacementAtReceivers())


def peak_times(traces, dt):
    """Time of the absolute maximum of each trace (column)."""
    return np.argmax(np.abs(traces), axis=0) * dt


def expected_times(receivers, velocity=VP):
    """Direct arrival time of the Ricker peak at each receiver."""
    r = np.asarray(receivers, dtype=np.float64) - np.asarray(SOURCE)
    return np.linalg.norm(r, axis=1) / velocity + T_PEAK
