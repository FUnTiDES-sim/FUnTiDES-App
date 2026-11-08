"""makutu.imaging.FWIProblem: state of an inversion, model files and counters (no solver run)."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from makutu.imaging import FWIProblem, GradientConditioner
from makutu.imaging.Minimizer import CGPROptimizer

SOLVER_2D = SimpleNamespace(dim=2, lx=1000.0, lz=500.0, ex=40, ez=20)
ACQ = SimpleNamespace(shots=[1, 2, 3])


def test_defaults_and_mode():
    p = FWIProblem(SOLVER_2D, ACQ)
    assert p.listOfAcquisition == [ACQ] and p.totalShots == 3
    assert p.freqList == ["all"] and not p.debug and p.iteration == 1
    assert FWIProblem(SOLVER_2D, ACQ, mode="DBG").debug
    with pytest.raises(ValueError):
        FWIProblem(SOLVER_2D, ACQ, mode="debug")
    p.release()
    assert p.solver is None


def test_callback_numbers_the_model_files(tmp_path, monkeypatch):
    """Restart: the model files are numbered from ``iteration``, with the grid attributes."""
    monkeypatch.chdir(tmp_path)
    p = FWIProblem(SOLVER_2D, ACQ, iteration=24)
    x = np.arange(800.0)
    p.callback(x)
    p.callback(x + 1)
    assert p.iteration == 26
    with h5py.File("velocityModel25.hdf5") as f:
        d = f["velocity"]
        assert np.array_equal(d[:], x + 1) and tuple(d.attrs["shape"]) == (20, 40)
    assert not (tmp_path / "velocityModel1.hdf5").exists()


def test_two_problems_are_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a, b = FWIProblem(SOLVER_2D, ACQ), FWIProblem(SOLVER_2D, ACQ, iteration=5)
    a.callback(np.zeros(800))
    assert (a.iteration, b.iteration) == (2, 5)
    a.illumination = np.ones(800)
    assert b.illumination is None


def test_minimizer_cache_restores_the_illumination_of_its_problem():
    """The Wolfe optimisers reuse cached gradients: the illumination of the preconditioner
    must be the one of the cached model."""
    problem = FWIProblem(SOLVER_2D, ACQ)
    precond = GradientConditioner(SOLVER_2D, illumination=True, problem=problem)

    def jac(x, computeFullCF=False):
        problem.illumination = np.full(800, 1.0 + x[0])            # depends on the model
        g = 2 * x
        return (g, float(x @ x)) if computeFullCF else g

    o = CGPROptimizer(lambda x: float(x @ x), jac=jac, precond=precond)
    x0, x1 = np.full(800, 1.0), np.full(800, 2.0)
    o._fg(x0)
    o._fg(x1)
    assert problem.illumination[0] == 3.0
    o._fg(x0)                                                       # from the cache
    assert problem.illumination[0] == 2.0
