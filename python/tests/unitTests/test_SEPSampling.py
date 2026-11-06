"""Orientation of the SEP sampling on the 3-D mesh: SEP axis 3 = depth (0 at the top), mesh z up."""

import numpy as np

from makutu.input.SEPReader import SEPModel, sample_on_elements


def test_depth_axis_is_flipped():
    n1, n2, n3 = 4, 3, 5
    depth_index = np.arange(n3, dtype=np.float32)[:, None, None] * np.ones((1, n2, n1), np.float32)
    sep = SEPModel(depth_index, n1, n2, n3, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0)
    v = sample_on_elements(sep, ex=4, ey=3, ez=5, hx=100.0, hy=100.0, hz=100.0)
    assert v.shape == (5, 3, 4)                      # (ez, ey, ex), iz = 0 at the bottom
    np.testing.assert_array_equal(v[-1], 0.0)        # top layer (z near lz) = depth 0
    np.testing.assert_array_equal(v[0], 4.0)         # bottom layer = deepest SEP sample
    np.testing.assert_array_equal(v[:, 0, 0], [4.0, 3.0, 2.0, 1.0, 0.0])


def test_finer_mesh():
    # 2 elements per SEP sample in depth: element centres at depths 25, 75, ..., 475 m
    n3 = 5
    data = np.arange(n3, dtype=np.float32)[:, None, None] * np.ones((1, 1, 1), np.float32)
    sep = SEPModel(data, 1, 1, n3, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0)
    v = sample_on_elements(sep, ex=1, ey=1, ez=10, hx=100.0, hy=100.0, hz=50.0)
    np.testing.assert_array_equal(v[::-1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
