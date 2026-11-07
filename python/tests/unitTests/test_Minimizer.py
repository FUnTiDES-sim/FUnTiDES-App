"""Optimisers of makutu.imaging.Minimizer on a known quadratic cost (no solver)."""

import numpy as np
import pytest

from makutu.imaging import Minimizer

N = 20
W = np.linspace(1.0, 10.0, N) * 1e-4                 # curvature: condition number 10
TARGET = 2000.0 + 150.0 * np.sin(np.arange(N))        # "true" velocities
X0 = np.full(N, 2000.0)


def func(x, *args):
    return 0.5 * np.sum(W * (x - TARGET) ** 2)


def jac(x, *args, computeFullCF=False):
    """Same interface as makutu.imaging.gradf."""
    g = W * (x - TARGET)
    return (g, func(x)) if computeFullCF else g


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # the optimisers write fullCostFunction.txt


@pytest.mark.parametrize("method, options, ratio", [
    ("steepestDescent", {"maxiter": 5, "minls": 2}, 0.05),
    ("CGPR2", {"maxiter": 5, "minls": 2}, 0.2),
    ("SD2", {"maxiter": 5}, 0.5),
    ("CGPR", {"maxiter": 5}, 0.5),
    ("L-BFGS-B", {"maxiter": 5}, 0.05),
])
def test_cost_decreases(method, options, ratio):
    x = Minimizer(func, method=method, jac=jac, bounds=[[1000.0, 4000.0]] * N, options=dict(options)).optimize(X0, ())
    assert isinstance(x, np.ndarray) and x.shape == X0.shape
    assert func(x) < ratio * func(X0)


@pytest.mark.parametrize("method", ["steepestDescent", "SD2"])
def test_bounds_are_respected(method):
    x = Minimizer(func, method=method, jac=jac, bounds=[[1990.0, 2010.0]],
                  options={"maxiter": 3, "minls": 2}).optimize(X0, ())
    assert x.min() >= 1990.0 and x.max() <= 2010.0


def test_starting_step_is_relative_to_the_model():
    # default first step: 1 % of max|x| along -gradient / max|gradient|
    m = Minimizer(func, method="steepestDescent", jac=jac)
    assert m._initialStep(X0) == pytest.approx(20.0)
    assert Minimizer(func, options={"alpha": 5.0})._initialStep(X0) == 5.0
