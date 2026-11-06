"""
Synthetic model builders that write SEP datasets.

Two model types are provided:

* :func:`layered_model`   -- horizontally layered velocity / density model.
* :func:`inclusion_model` -- layered background with an ellipsoidal inclusion
                             (analogous to a salt body).

Both functions write a SEP ``.H`` header and a ``.@`` binary file and return
a :class:`~utilities.input.SEPReader.SEPModel` for immediate use.

SEP axis convention used throughout
------------------------------------
  axis 1 (n1, d1, o1) → X  (fastest in binary)
  axis 2 (n2, d2, o2) → Y
  axis 3 (n3, d3, o3) → Z  (depth, slowest in binary)

The on-disk binary layout is therefore ``data[iz, iy, ix]`` in C order,
which matches the convention expected by :func:`~utilities.input.SEPReader.read_sep`.
"""

import os

import numpy as np

from utilities.input.SEPReader import SEPModel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_sep(header_path, data, dx, dy, dz, ox=0.0, oy=0.0, oz=0.0):
    """
    Write *data* (shape ``(nz, ny, nx)``) to a SEP dataset.

    Creates ``<header_path>`` (the ``.H`` file) and
    ``<header_path>.@`` (the binary).

    Parameters
    ----------
    header_path : str
        Path for the ``.H`` header file.
    data : np.ndarray, shape (nz, ny, nx), dtype float32
        Values to store.  Converted to float32 if necessary.
    dx, dy, dz : float
        Sample intervals [m].
    ox, oy, oz : float
        Origins [m].

    Returns
    -------
    SEPModel
    """
    data = data.astype(np.float32)
    nz, ny, nx = data.shape
   
    binary_path = os.path.abspath(header_path + ".@")

    header_lines = [
        f"n1={nx} d1={dx:.6g} o1={ox:.6g}",
        f"n2={ny} d2={dy:.6g} o2={oy:.6g}",
        f"n3={nz} d3={dz:.6g} o3={oz:.6g}",
        'data_format="native_float"',
        f'in="{binary_path}"',
        "",   # trailing newline
    ]
    with open(header_path, "w") as fh:
        fh.write("\n".join(header_lines))

    data.tofile(binary_path)

    return SEPModel(data, nx, ny, nz, dx, dy, dz, ox, oy, oz)


def _make_grid(nx, ny, nz, dx, dy, dz, ox, oy, oz):
    """Return 3-D coordinate arrays ``(X, Y, Z)`` of shape ``(nz, ny, nx)``."""
    x = ox + (np.arange(nx) + 0.5) * dx
    y = oy + (np.arange(ny) + 0.5) * dy
    z = oz + (np.arange(nz) + 0.5) * dz
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")   # (nz, ny, nx)
    return X, Y, Z


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def layered_model(
        header_path,
        lx, ly, lz,
        nx, ny, nz,
        layers,
        default_vp=1500.0,
        default_rho=1.0,
        ox=0.0, oy=0.0, oz=0.0):
    """
    Create a horizontally layered Vp (and optionally rho) SEP model.

    Layers are defined from top (z = oz) to bottom (z = oz + lz).  Each layer
    spans from its ``z_top`` down to the next layer's ``z_top`` (or to ``lz``
    for the deepest layer).

    Parameters
    ----------
    header_path : str
        Path for the ``.H`` file (e.g. ``"models/vp_layered.H"``).
        The binary is written alongside as ``<header_path>.@``.
    lx, ly, lz : float
        Domain extents [m].
    nx, ny, nz : int
        Grid dimensions (number of samples).
    layers : list of dict
        Each dict describes one layer.  Keys:

        =========  ======================================================
        ``z_top``  Depth of the layer top [m] (from oz).
        ``vp``     P-wave velocity in this layer [m/s].
        ``rho``    Density (optional; uses *default_rho* if absent).
        =========  ======================================================

        Layers must be sorted by ``z_top`` in ascending order.
    default_vp : float, optional
        Velocity for cells above the first layer boundary.  Default 1500.
    default_rho : float, optional
        Density for cells above the first layer boundary.  Default 1.0.
    ox, oy, oz : float, optional
        Grid origins [m].  Default 0.

    Returns
    -------
    vp_model : SEPModel
        Vp SEP dataset (also written to disk).
    rho_model : SEPModel
        Rho SEP dataset (written to ``<stem>_rho.H``).

    Examples
    --------
    >>> layers = [
    ...     {"z_top": 0.0,   "vp": 1500.0, "rho": 1.0},
    ...     {"z_top": 500.0, "vp": 2000.0, "rho": 1.5},
    ...     {"z_top": 800.0, "vp": 2500.0, "rho": 2.0},
    ... ]
    >>> vp, rho = layered_model("models/vp.H", 1000, 1000, 1000,
    ...                          50, 50, 50, layers)
    """
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    _, _, Z = _make_grid(nx, ny, nz, dx, dy, dz, ox, oy, oz)

    vp_data  = np.full((nz, ny, nx), default_vp,  dtype=np.float32)
    rho_data = np.full((nz, ny, nx), default_rho, dtype=np.float32)

    for layer in sorted(layers, key=lambda l: l["z_top"]):
        mask = Z >= (oz + layer["z_top"])
        vp_data[mask]  = float(layer["vp"])
        rho_data[mask] = float(layer.get("rho", default_rho))

    vp_model  = _write_sep(header_path, vp_data, dx, dy, dz, ox, oy, oz)

    stem      = header_path[:-2] if header_path.endswith(".H") else header_path
    rho_path  = stem + "_rho.H"
    rho_model = _write_sep(rho_path, rho_data, dx, dy, dz, ox, oy, oz)

    return vp_model, rho_model


def inclusion_model(
        header_path,
        lx, ly, lz,
        nx, ny, nz,
        layers,
        inclusion,
        default_vp=1500.0,
        default_rho=1.0,
        ox=0.0, oy=0.0, oz=0.0):
    """
    Create a layered model with an ellipsoidal high-velocity inclusion
    (analogous to a salt body).

    The background is built identically to :func:`layered_model`.  The
    inclusion is then stamped on top: any grid cell whose centre falls inside
    the ellipsoid takes the inclusion velocity / density.

    Parameters
    ----------
    header_path : str
        Path for the Vp ``.H`` file.
    lx, ly, lz : float
        Domain extents [m].
    nx, ny, nz : int
        Grid dimensions.
    layers : list of dict
        Background layering (same format as :func:`layered_model`).
    inclusion : dict
        Ellipsoidal inclusion geometry and properties.  Keys:

        ============  =====================================================
        ``cx``        X-centre [m] (default: lx/2).
        ``cy``        Y-centre [m] (default: ly/2).
        ``cz``        Z-centre [m] (default: lz/2).
        ``rx``        Semi-axis in X [m].
        ``ry``        Semi-axis in Y [m] (default: rx).
        ``rz``        Semi-axis in Z [m] (default: rx).
        ``vp``        Inclusion velocity [m/s].
        ``rho``       Inclusion density (optional).
        ============  =====================================================

    default_vp : float, optional
        Velocity above the first layer.
    default_rho : float, optional
        Density above the first layer.
    ox, oy, oz : float, optional
        Grid origins [m].

    Returns
    -------
    vp_model : SEPModel
    rho_model : SEPModel

    Examples
    --------
    >>> layers = [
    ...     {"z_top": 0.0,    "vp": 1500.0, "rho": 1.0},
    ...     {"z_top": 400.0,  "vp": 2000.0, "rho": 1.5},
    ...     {"z_top": 700.0,  "vp": 2500.0, "rho": 2.0},
    ... ]
    >>> salt = {
    ...     "cx": 500.0, "cy": 500.0, "cz": 600.0,
    ...     "rx": 200.0, "ry": 150.0, "rz": 120.0,
    ...     "vp": 4480.0, "rho": 2.2,
    ... }
    >>> vp, rho = inclusion_model("models/vp_salt.H", 1000, 1000, 1000,
    ...                            50, 50, 50, layers, salt)
    """
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    X, Y, Z = _make_grid(nx, ny, nz, dx, dy, dz, ox, oy, oz)

    # --- Background ---
    vp_data  = np.full((nz, ny, nx), default_vp,  dtype=np.float32)
    rho_data = np.full((nz, ny, nx), default_rho, dtype=np.float32)

    for layer in sorted(layers, key=lambda l: l["z_top"]):
        mask = Z >= (oz + layer["z_top"])
        vp_data[mask]  = float(layer["vp"])
        rho_data[mask] = float(layer.get("rho", default_rho))

    # --- Inclusion ---
    cx = float(inclusion.get("cx", ox + lx * 0.5))
    cy = float(inclusion.get("cy", oy + ly * 0.5))
    cz = float(inclusion.get("cz", oz + lz * 0.5))
    rx = float(inclusion["rx"])
    ry = float(inclusion.get("ry", rx))
    rz = float(inclusion.get("rz", rx))

    ellipsoid = (
        ((X - cx) / rx) ** 2 +
        ((Y - cy) / ry) ** 2 +
        ((Z - cz) / rz) ** 2
    ) <= 1.0

    vp_data[ellipsoid]  = float(inclusion["vp"])
    rho_data[ellipsoid] = float(inclusion.get("rho", default_rho))

    vp_model  = _write_sep(header_path, vp_data, dx, dy, dz, ox, oy, oz)

    stem      = header_path[:-2] if header_path.endswith(".H") else header_path
    rho_path  = stem + "_rho.H"
    rho_model = _write_sep(rho_path, rho_data, dx, dy, dz, ox, oy, oz)

    return vp_model, rho_model
