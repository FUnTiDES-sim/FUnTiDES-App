"""Residual and cost functions of makutu.forwardBackward (no solver needed)."""

import numpy as np
import pytest
import segyio

from makutu.forwardBackward import ComputeResidual, ComputeCostFunction, computeFullCostFunction


def write_segy(path, data, dt):
    """Traces (n_samples, n_traces) at sampling dt [s]."""
    spec = segyio.spec()
    spec.format, spec.sorting = 5, 2
    spec.samples = np.arange(data.shape[0]) * dt * 1e3
    spec.tracecount = data.shape[1]
    with segyio.create(str(path), spec) as f:
        for i in range(data.shape[1]):
            f.header[i] = {segyio.su.dt: int(round(dt * 1e6)), segyio.su.ns: data.shape[0]}
            f.trace[i] = data[:, i].astype(np.float32)
        f.bin.update(hdt=int(round(dt * 1e6)), hns=data.shape[0])


def test_residual_resamples_the_observed_data(tmp_path):
    # observed at 1 ms, simulated at 2 ms: residual on the simulation times
    t = np.arange(500) * 1e-3
    obs = np.stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t)], axis=1)
    write_segy(tmp_path / "obs.sgy", obs, 1e-3)
    sim = np.zeros((300, 2))                                  # 0.6 s at 2 ms, beyond the data
    r = ComputeResidual(str(tmp_path / "obs.sgy"), dt=2e-3).computeResidual(sim)
    assert r.shape == sim.shape
    np.testing.assert_allclose(r[:250], obs[::2], atol=1e-6)
    assert np.all(r[250:] == 0.0)                             # no data after 0.499 s


def test_residual_is_observed_minus_simulated(tmp_path):
    obs = np.random.default_rng(0).standard_normal((100, 3))
    write_segy(tmp_path / "obs.sgy", obs, 2e-3)
    sim = 0.5 * obs
    r = ComputeResidual(str(tmp_path / "obs.sgy"), dt=2e-3).computeResidual(sim)
    np.testing.assert_allclose(r, 0.5 * obs, atol=1e-6)


def test_residual_filters_the_observed_data(tmp_path):
    t = np.arange(1000) * 1e-3
    obs = (np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 100 * t))[:, None]
    write_segy(tmp_path / "obs.sgy", obs, 1e-3)
    r = ComputeResidual(str(tmp_path / "obs.sgy"), dt=1e-3, fmax=20).computeResidual(np.zeros((1000, 1)))
    # the 100 Hz component is removed, the 5 Hz one kept (away from the edges)
    np.testing.assert_allclose(r[200:800, 0], np.sin(2 * np.pi * 5 * t[200:800]), atol=2e-2)


def test_residual_checks_the_number_of_traces(tmp_path):
    write_segy(tmp_path / "obs.sgy", np.zeros((10, 3)), 1e-3)
    with pytest.raises(ValueError):
        ComputeResidual(str(tmp_path / "obs.sgy"), dt=1e-3).computeResidual(np.zeros((10, 2)))


def test_partial_and_full_cost(tmp_path):
    d = tmp_path / "pcost"
    r1, r2 = np.ones((10, 2)), 2 * np.ones((5, 1))
    j1 = ComputeCostFunction(str(d), dt=0.1, rootname="partialCost_00001").computePartialCostFunction(r1)
    j2 = ComputeCostFunction(str(d), dt=0.1, rootname="partialCost_00002").computePartialCostFunction(r2)
    assert j1 == pytest.approx(0.1 * 20 / 2) and j2 == pytest.approx(0.1 * 20 / 2)
    J = computeFullCostFunction(2, str(d), save=True, sfile=str(tmp_path / "J.txt"))
    assert J == pytest.approx(j1 + j2)
    assert not d.exists()                                     # removed after reading (not DBG)
    assert float((tmp_path / "J.txt").read_text()) == pytest.approx(J)


def test_full_cost_reports_missing_shots(tmp_path):
    d = tmp_path / "pcost"
    ComputeCostFunction(str(d), dt=1.0, rootname="partialCost_00001").computePartialCostFunction(np.ones((2, 1)))
    with pytest.raises(FileNotFoundError, match=r"\[2, 3\]"):
        computeFullCostFunction(3, str(d), save=False)
