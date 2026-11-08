"""makutu.imaging.GradientConditioner: masks and illumination (no solver run)."""

from types import SimpleNamespace

import numpy as np
import pytest

from makutu.imaging import GradientConditioner
from makutu.imaging import FWIProblem

SOLVER_2D = SimpleNamespace(dim=2, lx=1000.0, lz=500.0, ex=40, ez=20)          # 25 m elements


def test_no_conditioning_is_identity():
    g = np.random.default_rng(0).normal(size=800)
    assert np.array_equal(GradientConditioner(SOLVER_2D)(g), g)


def test_depth_mask_and_ramp():
    P = GradientConditioner(SOLVER_2D, minDepth=50.0, taper=50.0)
    m = P.mask.reshape(20, 40)                       # seismic order: row 0 at the surface
    assert np.all(m[:2] == 0.0)                      # centres 12.5 and 37.5 m < 50 m
    assert np.all((m[2:4] > 0) & (m[2:4] < 1))       # 62.5 and 87.5 m: ramp
    assert np.all(m[4:] == 1.0)


def test_edge_mask_sides_and_bottom_not_top():
    m = GradientConditioner(SOLVER_2D, edge=50.0).mask.reshape(20, 40)
    assert np.all(m[:, :2] == 0) and np.all(m[:, -2:] == 0) and np.all(m[-2:] == 0)
    assert np.all(m[:-2, 2:-2] == 1)                 # the surface is not masked


def test_edge_mask_3d():
    s = SimpleNamespace(dim=3, lx=500.0, ly=250.0, lz=250.0, ex=20, ey=10, ez=10)
    m = GradientConditioner(s, edge=25.0).mask.reshape(10, 10, 20)
    assert np.all(m[:, 0] == 0) and np.all(m[:, :, -1] == 0) and np.all(m[-1] == 0)
    assert m[0, 5, 10] == 1.0


def test_illumination_division():
    H = np.linspace(1.0, 100.0, 800)
    problem = FWIProblem(SOLVER_2D, SimpleNamespace(shots=[]))
    problem.illumination = H
    g = np.ones(800)
    p = GradientConditioner(SOLVER_2D, illumination=True, eps=0.01, problem=problem)(g)
    assert p == pytest.approx(1.0 / (H + 1.0))


def test_illumination_missing():
    with pytest.raises(ValueError):                                  # no problem given
        GradientConditioner(SOLVER_2D, illumination=True)
    problem = FWIProblem(SOLVER_2D, SimpleNamespace(shots=[]))      # gradf not run yet
    with pytest.raises(RuntimeError):
        GradientConditioner(SOLVER_2D, illumination=True, problem=problem)(np.ones(800))
