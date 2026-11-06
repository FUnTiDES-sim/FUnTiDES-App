"""Forward modelling chain of makutu.forwardBackward on a small 2-D model:
seismogram export, residual and cost function (func), source filtering."""

import os

import numpy as np
import pytest
import segyio

import pyfuntides.model as FTModel
import pyfuntides.solver as FTSolver

pytestmark = pytest.mark.skipif(
    not hasattr(FTModel, "CartesianUnstruct2DBuilder_f32_i32"),
    reason="FUnTiDES built without COMPILE_SEM_2D",
)

from makutu.acquisition import Acquisition  # noqa: E402
from makutu.forwardBackward import directSimulation, func  # noqa: E402
from makutu.imaging import globals as gb  # noqa: E402
from makutu.mesh import CartesianMesh2D  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402

DT, DT_SEISMO, TMAX, F0 = 0.0005, 0.002, 0.5, 10.0


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    """Run in tmp_path; drop the global solver afterwards. (The solver is built in the
    tests: a fixture value would outlive the Kokkos runtime, finalized by the session
    fixture before pytest releases the values of the last test.)"""
    monkeypatch.chdir(tmp_path)
    yield
    gb.solver = None


def make_setup():
    """Homogeneous 1 km x 0.5 km model (vp 2000 m/s), one shot, 9 receivers."""
    mesh = CartesianMesh2D(1000.0, 500.0, 20, 10, order=2, vp=np.full((10, 20), 2000.0))
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=DT, minTime=0.0, maxTime=TMAX, dtSeismo=DT_SEISMO,
                            sourceType="ricker2", sourceFreq=F0, **mesh.solver_kwargs())
    solver.initialize()
    acq = Acquisition(sources=[[500.0, 0.0, 100.0]],
                      receivers=[[x, 0.0, 50.0] for x in np.linspace(100.0, 900.0, 9)])
    gb.initializeGlobalVariables(solver, acq, [acq], 1000.0, 4000.0, 0.0, ["all"])
    return solver


def observe(solver):
    """Run the shot, export it and keep it as the observed data obs/seismoTrace_shot00001.sgy."""
    directSimulation(0, exportSeismicTrace=True, traceDirectory="obs")
    os.replace(f"obs/seismoTrace_shot00001_{int(F0)}.sgy", "obs/seismoTrace_shot00001.sgy")
    with segyio.open("obs/seismoTrace_shot00001.sgy", ignore_geometry=True) as f:
        return f.trace.raw[:].T.astype(np.float64), segyio.tools.dt(f) * 1e-6


def test_exported_seismograms():
    solver = make_setup()
    data, dt = observe(solver)
    n = int(round(TMAX / DT))
    assert solver.dtSeismoRecorded == pytest.approx(DT_SEISMO)
    assert dt == pytest.approx(DT_SEISMO)
    assert data.shape == (n // 4 + 1, 9)                     # one sample every 4 steps
    assert np.isfinite(data).all() and np.abs(data).max() > 0
    np.testing.assert_allclose(data, solver.getWaveField(), rtol=1e-6, atol=1e-6 * np.abs(data).max())


def test_cost_function_vanishes_on_the_true_model():
    solver = make_setup()
    data, dt = observe(solver)
    energy = dt * np.sum(data ** 2) / 2
    vp = solver.getVelocityModel()
    J_true = func(vp, "obs")
    J_slow = func(0.95 * vp, "obs")
    assert J_true < 1e-8 * energy                             # float32 GPU noise
    assert J_slow > 1e-2 * energy
    np.testing.assert_allclose(solver.getVelocityModel(), 0.95 * vp, rtol=1e-6)   # model updated by func


def test_source_filter_keeps_the_time_axis():
    solver = make_setup()
    solver.evaluateSource()
    raw = solver.sourceValue[:, 0].copy()
    solver.filterSource(40.0)                                 # above the Ricker band: nearly unchanged
    wide = solver.sourceValue[:, 0].copy()
    assert wide.shape == raw.shape
    assert np.argmax(np.abs(wide)) == np.argmax(np.abs(raw))
    assert np.corrcoef(raw, wide)[0, 1] > 0.999
    solver.evaluateSource()
    solver.filterSource(5.0)                                  # below the peak frequency: smaller
    assert np.abs(solver.sourceValue).max() < 0.8 * np.abs(raw).max()
