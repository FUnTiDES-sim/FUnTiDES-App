"""Acoustic forward modelling through makutu and FUnTiDES: direct arrivals."""

import numpy as np
import pytest

import shots

DT = 0.001

# Receivers on a line through the source (off the GLL nodes), symmetric
# about it: offsets from 63 m to 713 m.
RECEIVERS = [[shots.SOURCE[0] + dx, shots.SOURCE[1] + 1.3, shots.SOURCE[2] + 33.7]
             for dx in np.linspace(-713.0, 713.0, 21)]


@pytest.fixture(scope="module")
def traces():
    return {mesh: shots.acoustic_shot(RECEIVERS, mesh=mesh, dt=DT) for mesh in ("struct", "unstruct")}


@pytest.mark.parametrize("mesh", ["struct", "unstruct"])
def test_direct_arrival_times(traces, mesh):
    d = traces[mesh]
    assert np.isfinite(d).all()
    assert np.abs(d).max() > 0
    err = shots.peak_times(d, DT) - shots.expected_times(RECEIVERS)
    # Peak picking on a 1 ms grid plus numerical dispersion: a few ms
    assert np.abs(err).max() < 0.01, err
    # Symmetric receivers see the same arrival
    assert np.abs(err - err[::-1]).max() < 0.005, err


def test_struct_and_unstruct_meshes_agree(traces):
    a, b = traces["struct"], traces["unstruct"]
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-4
