"""Data selection and source filter of makutu.forwardBackward.dataprocessing (no solver)."""

from types import SimpleNamespace

import numpy as np
import pytest

from makutu.forwardBackward import DataSelection
from makutu.forwardBackward.dataprocessing import applySourceCorrection, sourceCorrection, sourceCorrectionTerms

SRC = [500.0, 0.0, 10.0]
RCV = [[x, 0.0, 20.0] for x in (500.0, 700.0, 1100.0, 1500.0)]          # offsets 0, 200, 600, 1000


def test_offset_range():
    w = DataSelection(offsetMin=100.0, offsetMax=800.0).weights(50, 4e-3, SRC, RCV)
    assert w.shape == (50, 4)
    assert np.all(w[:, [1, 2]] == 1.0) and np.all(w[:, [0, 3]] == 0.0)


def test_mute_and_window():
    dt = 1e-3
    sel = DataSelection(muteVelocity=2000.0, muteTime=0.1, window=0.2, taper=0.02)
    w = sel.weights(1000, dt, SRC, RCV)
    t = np.arange(1000) * dt
    for j, offset in enumerate((0.0, 200.0, 600.0, 1000.0)):
        tm = 0.1 + offset / 2000.0
        assert np.all(w[t < tm - 0.02 - 1e-9, j] == 0.0)                  # muted before the line
        assert np.all(w[(t >= tm) & (t <= tm + 0.18), j] == 1.0)          # kept in the window
        assert np.all(w[t > tm + 0.2 + 1e-9, j] == 0.0)                   # after the window
    with pytest.raises(ValueError):
        DataSelection(window=0.2)                                         # window without mute line


def test_apply_is_the_weighting():
    sel = DataSelection(offsetMax=650.0, muteVelocity=3000.0)
    r = np.random.default_rng(0).standard_normal((300, 4))
    np.testing.assert_allclose(sel.apply(r, 2e-3, SRC, RCV), sel.weights(300, 2e-3, SRC, RCV) * r)


def ricker(t, f0=10.0, t0=0.1):
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def test_source_filter_recovers_scale_and_delay():
    """Observed = 2 x simulated delayed by 8 ms: the filter applied to the source wavelet
    gives the same transformation of the wavelet."""
    dt, nt = 2e-3, 500
    t = np.arange(nt) * dt
    rng = np.random.default_rng(1)
    green = [np.roll(ricker(t, t0=0.1 + 0.05 * k), 0) * rng.uniform(0.5, 1.5) for k in range(6)]
    sim = np.stack(green, axis=1)
    obs = 2.0 * np.stack([np.interp(t - 0.008, t, g, left=0.0) for g in green], axis=1)
    freqs, cross, power = sourceCorrectionTerms(sim, obs, dt)
    correction = sourceCorrection(freqs, cross, power)

    src_dt = 5e-4                                                         # source at the time step
    ts = np.arange(4 * nt) * src_dt
    solver = SimpleNamespace(dt=src_dt, sourceValue=ricker(ts)[:, None])
    solver.updateSourceValue = lambda v: setattr(solver, "sourceValue", v)
    applySourceCorrection(solver, correction)
    expected = 2.0 * ricker(ts - 0.008)
    got = solver.sourceValue[:, 0]
    assert np.linalg.norm(got - expected) < 0.03 * np.linalg.norm(expected)
