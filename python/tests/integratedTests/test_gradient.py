"""Adjoint-state gradient of makutu (solver.computeGradient, imaging.gradf) against
finite differences of the cost function, in 2-D and 3-D."""

import os

import numpy as np
import pytest
import segyio

import pyfuntides.model as FTModel
import pyfuntides.solver as FTSolver

from makutu.acquisition import Acquisition
from makutu.forwardBackward import directSimulation
from makutu.imaging import FWIProblem
from makutu.imaging import gradf
from makutu.mesh import CartesianMesh2D
from makutu.mesh.CartesianUnstructModel import build_cartesian_unstruct_model
from makutu.solvers import AcousticSolver

HAS_2D = hasattr(FTModel, "CartesianUnstruct2DBuilder_f32_i32")
PARAM = {"c": (lambda v: v, lambda m: m),
         "1/c": (lambda v: 1 / v, lambda m: 1 / m),
         "1/c2": (lambda v: v ** -2, lambda m: m ** -0.5)}


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    """Run in tmp_path (the solvers are built in the tests: fixture values would outlive
    the Kokkos runtime)."""
    monkeypatch.chdir(tmp_path)
    yield


def solver_2d(order, vp):
    ez, ex = vp.shape
    mesh = CartesianMesh2D(1000.0, 500.0, ex, ez, order=order, vp=vp, free_surface_on_top=False)
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=4e-4, minTime=0.0, maxTime=0.6, dtSeismo=2e-3, sourceType="ricker2",
                            sourceFreq=10.0, **mesh.solver_kwargs())
    solver.initialize()
    return solver


def grid_2d(ex=40, ez=20):
    X, Z = np.meshgrid((np.arange(ex) + 0.5) * 1000.0 / ex, (np.arange(ez) + 0.5) * 500.0 / ez)
    return X, Z                                                   # (ez, ex), z down


def gradient_check(solver, v0, vtrue, delta_shape, param, src, rcv, eps=0.5):
    """(adjoint, finite-difference) derivatives of J along delta_shape * 1 % of the model."""
    def shoot(v, step=None):
        solver.updateVelocityModel(np.ravel(v))
        solver.updateSourceAndReceiversFromCoords(src, rcv)
        solver.evaluateSource()
        solver.resetWaveField()
        solver.runForward(step)
        return solver.getPressureAtReceivers().astype(np.float64)

    obs = shoot(vtrue)
    cost = lambda v: solver.dtSeismoRecorded / 2 * np.sum((obs - shoot(v)) ** 2)
    sim = shoot(v0, solver.snapshotStepFor())
    grad = solver.computeGradient(obs - sim, param).reshape(v0.shape)
    to_m, to_v = PARAM[param]
    m0 = to_m(v0)
    delta = delta_shape * 0.01 * m0
    fd = (cost(to_v(m0 + eps * delta)) - cost(to_v(m0 - eps * delta))) / (2 * eps)
    return np.sum(grad * delta), fd


@pytest.mark.skipif(not HAS_2D, reason="FUnTiDES built without COMPILE_SEM_2D")
@pytest.mark.parametrize("order, param", [(1, "c"), (2, "c"), (3, "c"), (2, "1/c"), (2, "1/c2")])
def test_gradient_2d_matches_finite_differences(order, param):
    X, Z = grid_2d()
    v0 = np.full(X.shape, 2000.0)
    vtrue = v0 + 300.0 * np.exp(-((X - 600) ** 2 + (Z - 300) ** 2) / 80.0 ** 2)
    bump = np.exp(-((X - 500) ** 2 + (Z - 250) ** 2) / 100.0 ** 2)
    solver = solver_2d(order, v0)
    adjoint, fd = gradient_check(solver, v0, vtrue, bump, param, [[300.0, 0.0, 100.0]],
                                 [[x, 0.0, 60.0] for x in np.linspace(100, 900, 17)])
    # float32 GPU sums are not reproducible: a few % of noise on the finite differences
    assert adjoint == pytest.approx(fd, rel=5e-2)


@pytest.mark.skipif(not HAS_2D, reason="FUnTiDES built without COMPILE_SEM_2D")
def test_snapshot_compression():
    """int16 snapshots: half the memory, the same gradient within the run-to-run noise."""
    X, Z = grid_2d()
    v0 = np.full(X.shape, 2000.0)
    vtrue = v0 + 300.0 * np.exp(-((X - 600) ** 2 + (Z - 300) ** 2) / 80.0 ** 2)
    solver = solver_2d(2, v0)
    src, rcv = [[300.0, 0.0, 100.0]], [[x, 0.0, 60.0] for x in np.linspace(100, 900, 17)]
    step = solver.snapshotStepFor()

    def gradient():
        solver.updateVelocityModel(vtrue.ravel())
        solver.updateSourceAndReceiversFromCoords(src, rcv)
        solver.evaluateSource()
        solver.resetWaveField()
        solver.runForward()
        obs = solver.getPressureAtReceivers().astype(np.float64)
        solver.updateVelocityModel(v0.ravel())
        solver.updateSourceAndReceiversFromCoords(src, rcv)
        solver.evaluateSource()
        solver.resetWaveField()
        solver.runForward(step)
        return solver.computeGradient(obs - solver.getPressureAtReceivers(), "c")

    full = solver.snapshotMemory(step)
    g32, g32b = gradient(), gradient()
    solver.snapshotCompression = "int16"
    assert solver.snapshotMemory(step) * 2 == full
    g16 = gradient()
    noise = np.linalg.norm(g32b - g32) / np.linalg.norm(g32)        # float32 GPU sums between runs
    print(f"run-to-run {noise:.2e}, int16 {np.linalg.norm(g16 - g32) / np.linalg.norm(g32):.2e}")
    assert np.linalg.norm(g16 - g32) < max(3 * noise, 1e-4) * np.linalg.norm(g32)


def test_gradient_3d_matches_finite_differences():
    ex, ey, ez, order = 20, 20, 10, 2
    c = lambda n, length: (np.arange(n) + 0.5) * length / n
    Z, Y, X = np.meshgrid(c(ez, 500.0), c(ey, 1000.0), c(ex, 1000.0), indexing="ij")
    v0 = np.full(X.shape, 2000.0)
    vtrue = v0 + 300.0 * np.exp(-((X - 600) ** 2 + (Y - 450) ** 2 + (Z - 300) ** 2) / 100.0 ** 2)
    model, kvp, krho = build_cartesian_unstruct_model(1000.0, 1000.0, 500.0, ex, ey, ez, order,
                                                      v0.astype(np.float32), free_surface_on_top=False)
    solver = AcousticSolver(model=model, order=order, implem_type=FTSolver.ImplemType.MAKUTU,
                            mesh_type=FTSolver.MeshType.UNSTRUCT, dt=5e-4, minTime=0.0, maxTime=0.5,
                            dtSeismo=2e-3, sourceType="ricker2", sourceFreq=10.0,
                            lx=1000.0, ly=1000.0, lz=500.0, ex=ex, ey=ey, ez=ez, kk_model_vp=kvp, kk_model_rho=krho)
    solver.initialize()
    bump = np.exp(-((X - 500) ** 2 + (Y - 500) ** 2 + (Z - 250) ** 2) / 120.0 ** 2)
    # receivers off the nodes: checks the adjoint source weights
    adjoint, fd = gradient_check(solver, v0, vtrue, bump, "c", [[325.0, 525.0, 125.0]],
                                 [[x, 437.3, 63.7] for x in np.linspace(113.0, 887.0, 13)])
    assert adjoint == pytest.approx(fd, rel=5e-2)


@pytest.mark.skipif(not HAS_2D, reason="FUnTiDES built without COMPILE_SEM_2D")
def test_gradf_descent():
    """gradf on the shots of an acquisition: J = 0 on the true model; a small step
    along -gradient lowers J."""
    X, Z = grid_2d()
    v0 = np.full(X.shape, 2000.0)
    vtrue = v0 + 300.0 * np.exp(-((X - 600) ** 2 + (Z - 300) ** 2) / 80.0 ** 2)
    solver = solver_2d(2, vtrue)
    acq = Acquisition(sources=[[300.0, 0.0, 100.0], [700.0, 0.0, 100.0]],
                      receivers=[[x, 0.0, 60.0] for x in np.linspace(100, 900, 17)])
    problem = FWIProblem(solver, acq, minModel=1000.0, maxModel=4000.0, freqList=["all"])
    directSimulation(problem, 0, exportSeismicTrace=True, traceDirectory="obs")           # observed data
    for i in (1, 2):
        os.replace(f"obs/seismoTrace_shot{i:05d}_10.sgy", f"obs/seismoTrace_shot{i:05d}.sgy")

    energy = 0.0
    for i in (1, 2):
        with segyio.open(f"obs/seismoTrace_shot{i:05d}.sgy", ignore_geometry=True) as f:
            energy += segyio.tools.dt(f) * 1e-6 / 2 * np.sum(f.trace.raw[:].astype(float) ** 2)

    g_true, J_true = gradf(problem, vtrue.ravel(), "obs", exportSeismicTrace=False, computeFullCF=True)
    g0, J0 = gradf(problem, v0.ravel(), "obs", exportSeismicTrace=False, computeFullCF=True)
    H = problem.illumination                                           # summed over the 2 shots
    assert H.shape == g0.shape and np.all(np.isfinite(H)) and H.min() >= 0.0
    top = H.reshape(v0.shape)[:4].max()                           # near the sources (100 m)
    assert top > 10 * H.reshape(v0.shape)[-2:].mean()             # decreases with depth
    assert J_true < 1e-8 * energy                                 # float32 GPU noise
    assert np.abs(g_true).max() < 1e-2 * np.abs(g0).max()
    step = 10.0 / np.abs(g0).max()                                # at most 10 m/s
    _, J1 = gradf(problem, v0.ravel() - step * g0, "obs", exportSeismicTrace=False, computeFullCF=True)
    assert J1 < J0


@pytest.mark.skipif(not HAS_2D, reason="FUnTiDES built without COMPILE_SEM_2D")
def test_filtered_stage_vanishes_on_the_true_model():
    """Multi-scale stage (filter frequency 6 Hz): the source (solver.filterSource) and the
    observed data (ComputeResidual) are filtered consistently, so that J is close to 0 on
    the true model, and the filtered gradient still lowers J."""
    X, Z = grid_2d()
    v0 = np.full(X.shape, 2000.0)
    vtrue = v0 + 300.0 * np.exp(-((X - 600) ** 2 + (Z - 300) ** 2) / 80.0 ** 2)
    solver = solver_2d(2, vtrue)
    acq = Acquisition(sources=[[300.0, 0.0, 100.0], [700.0, 0.0, 100.0]],
                      receivers=[[x, 0.0, 60.0] for x in np.linspace(100, 900, 17)])
    problem = FWIProblem(solver, acq, minModel=1000.0, maxModel=4000.0, freqList=["all"])
    directSimulation(problem, 0, exportSeismicTrace=True, traceDirectory="obs")          # unfiltered data
    for i in (1, 2):
        os.replace(f"obs/seismoTrace_shot{i:05d}_10.sgy", f"obs/seismoTrace_shot{i:05d}.sgy")
    problem.freqList = [6.0]

    kw = dict(exportSeismicTrace=False, computeFullCF=True, ifreq=0)
    _, J_true = gradf(problem, vtrue.ravel(), "obs", **kw)
    g0, J0 = gradf(problem, v0.ravel(), "obs", **kw)
    assert J_true < 1e-2 * J0                     # 1.5e-3 measured (0.99 with the former FFT filter)
    _, J1 = gradf(problem, v0.ravel() - 10.0 / np.abs(g0).max() * g0, "obs", **kw)
    assert J1 < J0
