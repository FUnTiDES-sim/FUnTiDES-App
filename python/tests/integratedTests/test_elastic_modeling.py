"""Elastic forward modelling through makutu and FUnTiDES: P-wave arrivals."""

import numpy as np

import shots

DT = 0.0005

# Receivers below the vertical source, on its axis
RECEIVERS = [[shots.SOURCE[0], shots.SOURCE[1], shots.SOURCE[2] + dz] for dz in (200.0, 400.0, 600.0)]


def test_vertical_source_p_wave():
    ux, uy, uz = shots.elastic_shot(RECEIVERS, dt=DT)
    for u in (ux, uy, uz):
        assert np.isfinite(u).all()
    assert np.abs(uz).max() > 0
    # On the source axis the horizontal components vanish by symmetry
    assert np.abs(ux).max() < 1e-3 * np.abs(uz).max()
    assert np.abs(uy).max() < 1e-3 * np.abs(uz).max()
    # P wave (vp = 1500 m/s) on the vertical component; the near-field term
    # delays the peak at short offsets
    err = shots.peak_times(uz, DT) - shots.expected_times(RECEIVERS)
    assert np.all(err > -0.005) and np.all(err < 0.03), err
