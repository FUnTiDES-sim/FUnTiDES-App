"""makutu.display (HDF5 fields, SEG-Y traces) and the grid attributes of the HDF5 files."""

import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import segyio

import matplotlib
matplotlib.use("Agg")

from makutu.display import Hdf5Field, SEGYSeismicTrace, VelocityModel  # noqa: E402
from makutu.forwardBackward.modelio import gridAttributes, writeFields  # noqa: E402

SOLVER_2D = SimpleNamespace(dim=2, lx=1000.0, lz=500.0, ex=40, ez=20)
SOLVER_3D = SimpleNamespace(dim=3, lx=400.0, ly=300.0, lz=200.0, ex=4, ey=3, ez=2)


def seismic(nx, ny, nz):
    """Field value = 100 iz + 10 iy + ix, flat in seismic order (x fastest)."""
    iz, iy, ix = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return (100 * iz + 10 * iy + ix).ravel().astype(float)


def test_grid_attributes():
    assert gridAttributes(SOLVER_2D) == {"shape": (20, 40), "lengths": (500.0, 1000.0)}
    assert gridAttributes(SOLVER_3D) == {"shape": (2, 3, 4), "lengths": (200.0, 300.0, 400.0)}
    assert gridAttributes(None) == {}
    decomposed = SimpleNamespace(**vars(SOLVER_3D), comm=SimpleNamespace(Get_size=lambda: 2))
    assert gridAttributes(decomposed) == {}                 # local sizes only


def test_2d_field_from_attributes(tmp_path):
    f = tmp_path / "m.hdf5"
    writeFields(f, {"velocity": seismic(40, 1, 20), "unused": None}, SOLVER_2D)
    with h5py.File(f) as h:
        assert list(h) == ["velocity"]
    m = Hdf5Field(str(f), "velocity")
    assert m.read().shape == (20, 40) and m.dim == 2
    assert m.model[3, 5] == 305                             # z row 3, x column 5
    assert m.lengths == (500.0, 1000.0)
    assert m._axes("y") == ("x [m]", 1000.0, "z [m]", 500.0)
    assert np.array_equal(m.getSlice(axis="x"), m.model)    # 2-D: whole field


def test_3d_slices_and_axes(tmp_path):
    f = tmp_path / "m.hdf5"
    writeFields(f, {"velocity": seismic(4, 3, 2)}, SOLVER_3D)
    m = Hdf5Field(str(f), "velocity")
    assert m.read().shape == (2, 3, 4)
    np.testing.assert_array_equal(m.getSlice("x", 2), [[2, 12, 22], [102, 112, 122]])   # (z, y)
    np.testing.assert_array_equal(m.getSlice("y", 1)[:, 0], [10, 110])                   # (z, x)
    np.testing.assert_array_equal(m.getSlice("z", 1)[2], [120, 121, 122, 123])           # (y, x)
    assert m.getSlice("y").shape == (2, 4)                  # middle y index 1
    assert m._axes("x")[0::2] == ("y [m]", "z [m]")
    assert m._axes("z")[0::2] == ("x [m]", "y [m]")
    with pytest.raises(ValueError):
        m.getSlice("t")


def test_shape_given_without_attributes(tmp_path):
    f = tmp_path / "old.hdf5"
    with h5py.File(f, "w") as h:
        h.create_dataset("fullGradient", data=seismic(4, 3, 2))
    with pytest.raises(ValueError, match="give n"):
        Hdf5Field(str(f), "fullGradient").read()
    m = Hdf5Field(str(f), "fullGradient", n=(4, 3, 2))
    assert m.read()[1, 2, 3] == 123 and m.lengths is None
    assert m._axes("y") == ("x [elements]", 4, "z [elements]", 2)
    with pytest.raises(ValueError, match="does not match"):
        Hdf5Field(str(f), "fullGradient", n=(4, 4, 2)).read()


def test_velocity_model_conversion_and_png(tmp_path):
    f = tmp_path / "m.hdf5"
    v = np.full(800, 2000.0)
    writeFields(f, {"velocity": 1.0 / v ** 2}, SOLVER_2D)
    m = VelocityModel(str(f), type="1/c2")
    np.testing.assert_allclose(m.read(), 2000.0)
    np.testing.assert_allclose(m.read(), 2000.0)            # read twice: converted once each time
    png = tmp_path / "m.png"
    data = m.display(savefig=str(png))
    assert data.shape == (20, 40) and png.stat().st_size > 0
    with pytest.raises(ValueError):
        VelocityModel(str(f), type="c3")


def test_segy_traces(tmp_path):
    data = np.random.default_rng(0).standard_normal((250, 5))
    spec = segyio.spec()
    spec.format, spec.sorting = 5, 2
    spec.samples = np.arange(250) * 2.0
    spec.tracecount = 5
    path = tmp_path / "seismoTrace_shot0.sgy"
    with segyio.create(str(path), spec) as f:
        for i in range(5):
            f.header[i] = {segyio.su.dt: 2000, segyio.su.ns: 250}
            f.trace[i] = data[:, i].astype(np.float32)
        f.bin.update(hdt=2000, hns=250)
    t = SEGYSeismicTrace("seismoTrace_shot0.sgy", directory=str(tmp_path))
    np.testing.assert_allclose(t.read(), data, atol=1e-6)
    assert t.dt == pytest.approx(2e-3)
    assert t.getMinAndMaxValues() == (t.data.min(), t.data.max())
    png = tmp_path / "traces.png"
    t.display(savefig=str(png))
    assert os.path.getsize(png) > 0
