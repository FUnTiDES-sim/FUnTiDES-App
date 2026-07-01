"""
SEP (Stanford Exploration Project) file reader.

File format
-----------
A SEP dataset consists of two files:

* **Header** (``.H``): plain-text, ``key=value`` pairs, one or more per line.
  Key fields used here:

  ============  =====================================================
  ``n1/n2/n3``  Number of samples on axis 1/2/3 (n1 is fastest)
  ``d1/d2/d3``  Sampling interval [m or s] on each axis
  ``o1/o2/o3``  Origin of each axis
  ``in``        Path to the binary data file (relative to header dir)
  ``data_format`` ``"native_float"`` (default), ``"xdr_float"``,
                ``"native_double"``, ``"native_int"``
  ``esize``     Element size in bytes (4 = float32, 8 = float64)
  ============  =====================================================

* **Binary** (``.@``): raw binary array, C-order ``[n3][n2][n1]``,
  i.e. n1 varies fastest.

Axis convention assumed here (can be overridden at call site):
  * axis 1 → X (inline)
  * axis 2 → Y (crossline)
  * axis 3 → Z (depth)

So ``data[iz, iy, ix]`` after loading.
"""

import os
import re
import struct

import numpy as np


_FORMAT_MAP = {
    "native_float":  (np.float32,  "=f"),
    "xdr_float":     (np.float32,  ">f"),
    "native_double": (np.float64,  "=d"),
    "native_int":    (np.int32,    "=i"),
}


class SEPModel:
    """
    Container for a loaded SEP dataset.

    Attributes
    ----------
    data : np.ndarray, shape (n3, n2, n1)
        The physical values, indexed ``[iz, iy, ix]`` under the default
        axis convention (n1=X, n2=Y, n3=Z).
    n1, n2, n3 : int
        Number of samples on each axis.
    d1, d2, d3 : float
        Sampling interval on each axis [m].
    o1, o2, o3 : float
        Origin on each axis [m].
    """

    def __init__(self, data, n1, n2, n3, d1, d2, d3, o1, o2, o3):
        self.data = data
        self.n1, self.n2, self.n3 = int(n1), int(n2), int(n3)
        self.d1, self.d2, self.d3 = float(d1), float(d2), float(d3)
        self.o1, self.o2, self.o3 = float(o1), float(o2), float(o3)

    def __repr__(self):
        return (
            f"SEPModel(n=({self.n1},{self.n2},{self.n3}) "
            f"d=({self.d1},{self.d2},{self.d3}) "
            f"o=({self.o1},{self.o2},{self.o3}) "
            f"dtype={self.data.dtype})"
        )


def read_sep(header_file):
    """
    Read a SEP dataset from its header file.

    Parameters
    ----------
    header_file : str
        Path to the ``.H`` header file.

    Returns
    -------
    SEPModel

    Raises
    ------
    FileNotFoundError
        If the header or binary file does not exist.
    KeyError
        If required fields (``n1``, ``n2``, ``n3``, ``in``) are absent.
    ValueError
        If ``data_format`` is not recognised.
    """
    header_dir = os.path.dirname(os.path.abspath(header_file))

    # ---------- parse header ----------
    kv = {}
    with open(header_file, "r") as fh:
        for raw in fh:
            line = raw.split("#")[0]
            for m in re.finditer(r'(\w+)\s*=\s*("([^"]*)"|([\S]+))', line):
                key = m.group(1).lower()
                val = m.group(3) if m.group(3) is not None else m.group(4)
                kv[key] = val

    # ---------- required fields ----------
    try:
        n1 = int(kv["n1"])
        n2 = int(kv.get("n2", 1))
        n3 = int(kv.get("n3", 1))
    except KeyError as exc:
        raise KeyError(f"SEP header '{header_file}' missing field: {exc}") from exc

    d1 = float(kv.get("d1", 1.0))
    d2 = float(kv.get("d2", 1.0))
    d3 = float(kv.get("d3", 1.0))
    o1 = float(kv.get("o1", 0.0))
    o2 = float(kv.get("o2", 0.0))
    o3 = float(kv.get("o3", 0.0))

    data_format = kv.get("data_format", "native_float").strip().lower()
    if data_format not in _FORMAT_MAP:
        raise ValueError(
            f"Unsupported data_format '{data_format}'. "
            f"Choose from: {list(_FORMAT_MAP)}"
        )
    dtype, _ = _FORMAT_MAP[data_format]

    # ---------- locate binary file ----------
    binary_path = kv.get("in", "").strip().strip('"')
    if not binary_path:
        raise KeyError(f"SEP header '{header_file}' has no 'in=' field")

    if not os.path.isabs(binary_path):
        binary_path = os.path.join(header_dir, binary_path)

    if not os.path.exists(binary_path):
        raise FileNotFoundError(
            f"SEP binary file not found: '{binary_path}' "
            f"(referenced from '{header_file}')"
        )

    # ---------- read binary ----------
    raw = np.fromfile(binary_path, dtype=dtype)
    expected = n1 * n2 * n3
    if raw.size < expected:
        raise ValueError(
            f"SEP binary '{binary_path}' contains {raw.size} values, "
            f"expected at least {expected} ({n1}×{n2}×{n3})"
        )

    data = raw[:expected].reshape(n3, n2, n1)   # [iz, iy, ix] order
    return SEPModel(data, n1, n2, n3, d1, d2, d3, o1, o2, o3)


def sample_on_elements(sep, ex, ey, ez, hx, hy, hz, origin_x=0.0):
    """
    Sample a :class:`SEPModel` at element-centre coordinates of a Cartesian
    mesh, using nearest-neighbour interpolation.

    The mesh axes are assumed to align with SEP axes as follows:
    ``X ↔ axis-1``, ``Y ↔ axis-2``, ``Z ↔ axis-3``.

    Parameters
    ----------
    sep : SEPModel
        Loaded SEP dataset.
    ex, ey, ez : int
        Number of elements in X, Y, Z.
    hx, hy, hz : float
        Element sizes [m].
    origin_x : float, optional
        X-coordinate of the local subdomain origin (MPI). Default 0.

    Returns
    -------
    np.ndarray, shape (ez, ey, ex), dtype float32
        Values sampled at element centres ``(origin_x+(ix+0.5)*hx,
        (iy+0.5)*hy, (iz+0.5)*hz)``.
    """
    # Element centre coordinates
    xc = origin_x + (np.arange(ex) + 0.5) * hx   # (ex,)
    yc =            (np.arange(ey) + 0.5) * hy   # (ey,)
    zc =            (np.arange(ez) + 0.5) * hz   # (ez,)

    # Nearest-neighbour indices in SEP grid
    def _nn(coord, origin, delta, n):
        idx = np.round((coord - origin) / delta).astype(int)
        return np.clip(idx, 0, n - 1)

    i1 = _nn(xc, sep.o1, sep.d1, sep.n1)   # (ex,)
    i2 = _nn(yc, sep.o2, sep.d2, sep.n2)   # (ey,)
    i3 = _nn(zc, sep.o3, sep.d3, sep.n3)   # (ez,)

    # Broadcast sample: result[iz, iy, ix] = sep.data[i3[iz], i2[iy], i1[ix]]
    result = sep.data[
        i3[:, None, None],   # (ez, 1, 1)
        i2[None, :, None],   # (1, ey, 1)
        i1[None, None, :],   # (1, 1, ex)
    ].astype(np.float32)     # (ez, ey, ex)

    return result
