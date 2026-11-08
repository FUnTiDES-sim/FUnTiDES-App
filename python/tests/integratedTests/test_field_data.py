"""Tools for field data: data selection in the cost and the gradient, estimation of the
source wavelet (makutu.forwardBackward.dataprocessing), on 2-D synthetic data."""

import os

import numpy as np
import pytest

import pyfuntides.model as FTModel

from makutu.acquisition import Acquisition
from makutu.forwardBackward import DataSelection, directSimulation, func
from makutu.imaging import FWIProblem, gradf

from test_gradient import grid_2d, solver_2d

HAS_2D = hasattr(FTModel, "CartesianUnstruct2DBuilder_f32_i32")
pytestmark = pytest.mark.skipif(not HAS_2D, reason="FUnTiDES built without COMPILE_SEM_2D")


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


def setup(source_filter=None):
    """True model with an anomaly, 2 shots, 17 receivers; observed data in obs/, computed
    with the source filtered by ``source_filter`` = (frequencies, c) if given."""
    X, Z = grid_2d()
    v0 = np.full(X.shape, 2000.0)
    vtrue = v0 + 300.0 * np.exp(-((X - 600) ** 2 + (Z - 300) ** 2) / 80.0 ** 2)
    solver = solver_2d(2, vtrue)
    acq = Acquisition(sources=[[300.0, 0.0, 100.0], [700.0, 0.0, 100.0]],
                      receivers=[[x, 0.0, 60.0] for x in np.linspace(100, 900, 17)])
    problem = FWIProblem(solver, acq, minModel=1000.0, maxModel=4000.0, freqList=["all"])
    if source_filter is not None:
        problem.sourceCorrection = {0: source_filter}
    directSimulation(problem, 0, exportSeismicTrace=True, traceDirectory="obs")
    for i in (1, 2):
        os.replace(f"obs/seismoTrace_shot{i:05d}_10.sgy", f"obs/seismoTrace_shot{i:05d}.sgy")
    problem.sourceCorrection = None
    return problem, v0, vtrue


def test_selected_gradient_matches_finite_differences():
    """With offsets, mute and window, the gradient is the derivative of the selected cost."""
    problem, v0, vtrue = setup()
    problem.dataSelection = DataSelection(offsetMax=500.0, muteVelocity=2500.0, muteTime=0.05, window=0.3,
                                          taper=0.05)
    X, Z = grid_2d()
    bump = np.exp(-((X - 500) ** 2 + (Z - 250) ** 2) / 100.0 ** 2).ravel() * 20.0     # +- 20 m/s
    g, J = gradf(problem, v0.ravel(), "obs", exportSeismicTrace=False, computeFullCF=True)
    J_all = FWIProblem(problem.solver, problem.acquisition, minModel=1000.0, maxModel=4000.0).func(v0.ravel(), "obs",
                                                                                                  exportSeismicTrace=False)
    assert 0 < J < 0.8 * J_all                                     # part of the data only (0.5)
    fd = (func(problem, v0.ravel() + 0.5 * bump, "obs", exportSeismicTrace=False)
          - func(problem, v0.ravel() - 0.5 * bump, "obs", exportSeismicTrace=False))
    assert g @ bump == pytest.approx(fd, rel=5e-2)


def test_source_estimation():
    """Observed data made with a source 2 times stronger and delayed by 10 ms: the estimated
    filter brings J on the true model close to 0."""
    f = np.linspace(0.0, 1250.0, 5001)
    problem, v0, vtrue = setup((f, 2.0 * np.exp(-2j * np.pi * f * 0.010)))
    J0 = problem.func(vtrue.ravel(), "obs", exportSeismicTrace=False)
    freqs, c = problem.estimateSource("obs")
    near = (freqs > 5.0) & (freqs < 15.0)                          # main band of the 10 Hz Ricker
    assert np.abs(c[near]) == pytest.approx(2.0, rel=2e-2)
    J1 = problem.func(vtrue.ravel(), "obs", exportSeismicTrace=False)
    assert J1 < 1e-3 * J0
