"""2-D acoustic modelling through makutu (CartesianMesh2D + AcousticSolver) on the
FUnTiDES 2-D SEM solver, against the exact 2-D solution."""

import numpy as np
import pytest

import pyfuntides.model as FTModel
import pyfuntides.solver as FTSolver

pytestmark = pytest.mark.skipif(
    not hasattr(FTModel, "CartesianUnstruct2DBuilder_f32_i32"),
    reason="FUnTiDES built without COMPILE_SEM_2D",
)

from makutu.input.SEPReader import SEPModel  # noqa: E402
from makutu.mesh import CartesianMesh2D, sample_sep_2d  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402

L, VP, F0, DT, TMAX = 2000.0, 1500.0, 5.0, 0.001, 0.7
SOURCE, RECEIVER = (1010.0, 0.0, 990.0), (1310.0, 0.0, 690.0)    # [x, y, z], z downwards


def exact_2d(t, r, f, dt, c=VP, rho=1.0, n=2000):
    """(rho c / 2 pi) int_0^(t - r/c) f(tau) / sqrt(c^2 (t - tau)^2 - r^2) dtau, with
    tau = t - r/c - w^2 (Simpson); f sampled at k * dt, zero outside."""
    t0 = t - r / c
    if t0 <= 0:
        return 0.0
    w = np.linspace(0.0, np.sqrt(t0), n + 1)
    g = 2.0 * np.interp(t0 - w * w, np.arange(len(f)) * dt, f, left=0.0, right=0.0) \
        / np.sqrt(c * c * w * w + 2.0 * c * r)
    simpson = np.ones(n + 1)
    simpson[1:-1:2], simpson[2:-1:2] = 4.0, 2.0
    return rho * c / (2.0 * np.pi) * (simpson * g).sum() * (w[1] - w[0]) / 3.0


@pytest.mark.parametrize("order, tol", [(2, 2e-3), (3, 2e-3)])
def test_point_source_matches_exact_2d_solution(order, tol):
    ne = 400 // (order + 1)
    mesh = CartesianMesh2D(L, L, ne, ne, order=order, free_surface_on_top=False)
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=DT, minTime=0.0, maxTime=TMAX, dtSeismo=DT,
                            sourceType="ricker2", sourceFreq=F0, **mesh.solver_kwargs())
    assert solver.dim == 2
    solver.initialize()
    solver.updateSourceAndReceiversFromCoords([list(SOURCE)], [list(RECEIVER)])
    solver.evaluateSource()
    for t in range(solver.n_time_steps):
        solver.execute(t)
    trace = solver.getPressureAtReceivers()[:, 0].astype(np.float64)   # p(n dt)
    f = solver.sourceValue[:, 0]
    r = np.hypot(RECEIVER[0] - SOURCE[0], RECEIVER[2] - SOURCE[2])
    ref = np.array([exact_2d(n * DT, r, f, DT) for n in range(len(trace))])
    err = np.linalg.norm(trace - ref) / np.linalg.norm(ref)
    print(f"[ 2-D makutu ] order {order}: relative L2 error {err:.3e}")
    assert err < tol


def test_free_surface_and_absorbing_edges():
    """Free surface: the top edge stays at zero. Receivers symmetric about the
    source record the same trace; the y coordinate is ignored in 2-D."""
    mesh = CartesianMesh2D(1000.0, 500.0, 20, 10, order=2, free_surface_on_top=True)
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.STRUCT,
                            dt=0.001, minTime=0.0, maxTime=0.3, dtSeismo=0.001,
                            sourceType="ricker2", sourceFreq=10.0, **mesh.solver_kwargs())
    assert solver.mesh_type == FTSolver.MeshType.UNSTRUCT   # forced for a 2-D model
    solver.initialize()
    rcv = [[x, 123.0, 50.0] for x in (300.0, 700.0)]            # symmetric about x = 500
    solver.updateSourceAndReceiversFromCoords([[500.0, 999.0, 100.0]], rcv)   # depth 100 m
    # z downwards: receivers (z = 50 m) in the surface element row (FUnTiDES row 9), source in the next one
    assert [e // 20 for e in solver.receiverElementIndices] == [9, 9]
    assert list(solver.sourceElementIndices) == [10 + 8 * 20]          # x = 500 m: ix = 10, iz = 8
    solver.evaluateSource()
    for t in range(solver.n_time_steps):
        solver.execute(t)
    d = solver.getPressureAtReceivers()
    assert np.isfinite(d).all() and np.abs(d).max() > 0
    np.testing.assert_allclose(d[:, 0], d[:, 1], atol=1e-5 * np.abs(d).max())
    p = np.array(solver.data.get_current_field(0), copy=False)
    top = [i for i in range(mesh.model.get_number_of_nodes()) if mesh.model.is_free_surface(i)]
    assert len(top) == 20 * 2 + 1 and np.all(p[top] == 0.0)
    grid = solver.getPressureField()                            # z down: row 0 = free surface
    assert grid.shape == (10 * 2 + 1, 20 * 2 + 1)
    assert np.all(grid[0] == 0.0) and np.abs(grid[-1]).max() > 0


def test_sep_sampling_orientation():
    """SEP axis 1 = x, axis 2 = depth (0 at the top), as the makutu z."""
    n1, n2 = 10, 5
    depth_index = np.arange(n2, dtype=np.float32)[:, None] * np.ones((1, n1), np.float32)
    sep = SEPModel(depth_index[None, :, :], n1, n2, 1, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0)
    v = sample_sep_2d(sep, ex=10, ez=5, lx=1000.0, lz=500.0)
    assert v.shape == (5, 10)
    np.testing.assert_array_equal(v[:, 0], [0, 1, 2, 3, 4])   # row 0 = surface


def test_model_order():
    """Model arrays in makutu: row 0 at the surface; FUnTiDES stores the bottom row first."""
    vp = np.repeat(np.linspace(1500.0, 3000.0, 4, dtype=np.float32)[:, None], 6, axis=1)   # (ez, ex)
    mesh = CartesianMesh2D(600.0, 400.0, 6, 4, order=2, vp=vp)
    np.testing.assert_array_equal(mesh.vp, vp)
    np.testing.assert_array_equal(np.array(mesh.kk_model_vp, copy=False).reshape(4, 6), vp[::-1])
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=0.001, minTime=0.0, maxTime=0.01, dtSeismo=0.001,
                            sourceType="ricker2", sourceFreq=10.0, **mesh.solver_kwargs())
    solver.initialize()
    np.testing.assert_array_equal(solver.getVelocityModel().reshape(4, 6), vp)
    new = vp.copy()
    new[0] = 1000.0                                             # surface row
    solver.updateVelocityModel(new.ravel())
    np.testing.assert_array_equal(solver.getVelocityModel().reshape(4, 6), new)
    np.testing.assert_array_equal(np.array(mesh.kk_model_vp, copy=False).reshape(4, 6)[-1], 1000.0)
    # coordinates: z = 50 m is in the surface row (FUnTiDES row ez - 1)
    assert solver.coordToElementIndex(350.0, 0.0, 50.0) == 3 + 3 * 6
