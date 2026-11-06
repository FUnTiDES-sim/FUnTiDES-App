"""
2-D Cartesian quadrilateral meshes for the FUnTiDES 2-D SEM solver.

The mesh is the rectangle [0, lx] x [0, lz] of the (x, z) plane, z pointing up:
the free surface (when enabled) is the top edge z = lz, the other edges are
absorbing. It is a FUnTiDES ``ModelUnstruct`` of dimension 2 built from explicit
arrays, so heterogeneous velocity and density models are supported.

Numbering (FUnTiDES 2-D convention): element ``ix + iz * ex``; local node
``a + c * (order + 1)`` (a along x, c along z); global node ``gi + gk * nx`` with
``nx = ex * order + 1``.
"""

import numpy as np
from numpy.polynomial import legendre

import kokkos
import pyfuntides.model as FTModel

from makutu.tools.kokkosarrays import funtides_array

_BOUNDARY_INTERIOR, _BOUNDARY_DAMPING, _BOUNDARY_SURFACE = 0, 1, 3


def gll_points(order):
    """Gauss-Lobatto-Legendre points of [-1, 1] for ``order``: -1, roots of P'_order, 1."""
    if order < 1:
        raise ValueError("order must be >= 1")
    inner = legendre.Legendre.basis(order).deriv().roots() if order > 1 else np.array([])
    return np.concatenate(([-1.0], np.sort(inner.real), [1.0]))


def _kk(arr, dtype=kokkos.float32):
    kk = funtides_array(arr.shape, dtype=dtype)
    np.array(kk, copy=False)[...] = arr
    return kk


def build_cartesian_2d_model(lx, lz, ex, ez, order, vp_elem, rho_elem=None,
                             on_nodes=False, free_surface_on_top=True):
    """
    Build a 2-D ``ModelUnstruct_f32_i32`` on an ex x ez grid of [0, lx] x [0, lz].

    Parameters
    ----------
    lx, lz : float
        Domain size along x and z [m].
    ex, ez : int
        Number of elements along x and z.
    order : int
        Polynomial order.
    vp_elem : np.ndarray, shape (ez, ex)
        P-wave velocity per element [m/s], row iz from the bottom (z up).
    rho_elem : np.ndarray, shape (ez, ex), optional
        Density per element (default 1).
    on_nodes : bool
        Store the model on nodes (value of the owning element) instead of elements.
    free_surface_on_top : bool
        Free surface on z = lz; otherwise all four edges are absorbing.

    Returns
    -------
    (model, arrays) : the FUnTiDES model and the dict of its Kokkos arrays
        (``"vp_element"``, ``"rho_element"``, ``"vp_node"``, ``"rho_node"``, ...),
        which must stay alive as long as the model is used.
    """
    vp_elem = np.asarray(vp_elem, dtype=np.float32)
    if vp_elem.shape != (ez, ex):
        raise ValueError(f"vp_elem must have shape (ez, ex) = {(ez, ex)}, got {vp_elem.shape}")
    rho_elem = (np.ones((ez, ex), np.float32) if rho_elem is None
                else np.asarray(rho_elem, dtype=np.float32))
    if rho_elem.shape != (ez, ex):
        raise ValueError(f"rho_elem must have shape (ez, ex) = {(ez, ex)}, got {rho_elem.shape}")

    n1d = order + 1
    nx, nz = ex * order + 1, ez * order + 1
    n_node, n_elem = nx * nz, ex * ez
    hx, hz = lx / ex, lz / ez
    gll = gll_points(order)

    # Node coordinates: line g lies in element g // order at local node g % order
    def coords_1d(n, h):
        g = np.arange(n)
        return ((g // order) * h + (gll[g % order] + 1.0) * 0.5 * h).astype(np.float32)

    gk, gi = np.mgrid[0:nz, 0:nx]
    x = coords_1d(nx, hx)[gi].ravel()
    z = coords_1d(nz, hz)[gk].ravel()

    # Element-node table
    ke, ie = np.mgrid[0:ez, 0:ex]
    c, a = np.mgrid[0:n1d, 0:n1d]
    gni = ((ie.ravel()[:, None] * order + a.ravel()[None, :])
           + (ke.ravel()[:, None] * order + c.ravel()[None, :]) * nx).astype(np.int32)

    # Boundary flags
    tol = 1e-4 * min(hx, hz)
    boundaries = np.full(n_node, _BOUNDARY_INTERIOR, np.int32)
    top = z > lz - tol
    boundaries[(x < tol) | (x > lx - tol) | (z < tol) | top] = _BOUNDARY_DAMPING
    if free_surface_on_top:
        boundaries[top] = _BOUNDARY_SURFACE

    # Material: per element, and per node from the owning element
    vp_e, rho_e = vp_elem.ravel(), rho_elem.ravel()
    e_of_node = (np.minimum(gi.ravel() // order, ex - 1)
                 + np.minimum(gk.ravel() // order, ez - 1) * ex)
    zeros_n, zeros_e = np.zeros(n_node, np.float32), np.zeros(n_elem, np.float32)

    arrays = {
        "gni": _kk(gni, kokkos.int32),
        "x": _kk(x), "y": _kk(zeros_n), "z": _kk(z),
        "vp_node": _kk(vp_e[e_of_node]), "vp_element": _kk(vp_e),
        "rho_node": _kk(rho_e[e_of_node]), "rho_element": _kk(rho_e),
        "zero_node": _kk(zeros_n), "zero_element": _kk(zeros_e),
        "C": _kk(np.zeros((n_elem, 6, 6), np.float32)),
        "boundaries": _kk(boundaries, kokkos.int32),
    }
    zn, ze = arrays["zero_node"], arrays["zero_element"]
    data = FTModel.ModelUnstructData_f32_i32(
        order, n_elem, n_node, float(lx), 0.0, float(lz), on_nodes, False,
        arrays["gni"], arrays["x"], arrays["y"], arrays["z"],
        arrays["vp_node"], arrays["vp_element"], arrays["rho_node"], arrays["rho_element"],
        zn, ze,          # vs
        zn, ze, zn, ze,  # delta, epsilon
        zn, ze, zn, ze,  # gamma, theta
        zn, ze,          # phi
        arrays["C"], arrays["boundaries"],
    )
    data.dim = 2
    return FTModel.ModelUnstruct_f32_i32(data), arrays


def sample_sep_2d(sep, ex, ez, lx, lz, depth_axis=2):
    """
    Sample a 2-D SEP model at the element centres of an ex x ez mesh of
    [0, lx] x [0, lz] (z up), nearest neighbour.

    The SEP horizontal axis is axis 1 and its depth axis ``depth_axis`` (2 or 3),
    depth 0 at the top: the element centre (x, z) reads depth ``lz - z``.

    Returns
    -------
    np.ndarray, shape (ez, ex), float32 — row iz from the bottom.
    """
    xc = (np.arange(ex) + 0.5) * lx / ex
    depth = lz - (np.arange(ez) + 0.5) * lz / ez

    def nn(coord, origin, delta, n):
        return np.clip(np.round((coord - origin) / delta).astype(int), 0, n - 1)

    i1 = nn(xc, sep.o1, sep.d1, sep.n1)
    if depth_axis == 2:
        i2 = nn(depth, sep.o2, sep.d2, sep.n2)
        return sep.data[0][i2[:, None], i1[None, :]].astype(np.float32)
    if depth_axis == 3:
        i3 = nn(depth, sep.o3, sep.d3, sep.n3)
        return sep.data[i3[:, None], 0, i1[None, :]].astype(np.float32)
    raise ValueError("depth_axis must be 2 or 3")


class CartesianMesh2D:
    """
    2-D (x, z) Cartesian mesh and model for the FUnTiDES 2-D SEM solver.

    Parameters
    ----------
    lx, lz : float
        Domain size [m]; z points up, the free surface is z = lz.
    ex, ez : int
        Number of elements along x and z.
    order : int
        Polynomial order (at most the order compiled in FUnTiDES).
    vp : float or np.ndarray (ez, ex), optional
        P-wave velocity [m/s]; default 1500. Ignored if ``vp_file`` is given.
    rho : float or np.ndarray (ez, ex), optional
        Density; default 1. Ignored if ``rho_file`` is given.
    vp_file, rho_file : str, optional
        2-D SEP headers (axis 1 horizontal, axis 2 depth), sampled per element.
    on_nodes : bool
        Model on nodes instead of elements.
    free_surface_on_top : bool
        Free surface on the top edge (default True).

    Attributes
    ----------
    model : the FUnTiDES ModelUnstruct (dimension 2), for the makutu solvers
    kk_model_vp, kk_model_rho : Kokkos arrays of the model (per element or per node)
    lx, lz, ex, ez, order, dim (= 2), heterogeneous
    """

    dim = 2

    def __init__(self, lx, lz, ex, ez, order=2, vp=1500.0, rho=1.0, vp_file=None,
                 rho_file=None, on_nodes=False, free_surface_on_top=True):
        self.lx, self.lz, self.ex, self.ez, self.order = float(lx), float(lz), int(ex), int(ez), int(order)
        self.on_nodes = on_nodes
        if vp_file is not None or rho_file is not None:
            from makutu.input.SEPReader import read_sep
        if vp_file is not None:
            vp = sample_sep_2d(read_sep(vp_file), ex, ez, lx, lz)
        if rho_file is not None:
            rho = sample_sep_2d(read_sep(rho_file), ex, ez, lx, lz)
        self.vp = np.broadcast_to(np.asarray(vp, np.float32), (ez, ex)).copy()
        self.rho = np.broadcast_to(np.asarray(rho, np.float32), (ez, ex)).copy()
        self.heterogeneous = bool(np.ptp(self.vp) > 0 or np.ptp(self.rho) > 0)
        self.model, self._arrays = build_cartesian_2d_model(
            lx, lz, ex, ez, order, self.vp, self.rho, on_nodes, free_surface_on_top)
        loc = "node" if on_nodes else "element"
        self.kk_model_vp = self._arrays["vp_" + loc]
        self.kk_model_rho = self._arrays["rho_" + loc]

    def solver_kwargs(self):
        """Mesh keyword arguments of the makutu solvers (coordinate placement)."""
        return dict(model=self.model, order=self.order, lx=self.lx, lz=self.lz, ex=self.ex, ez=self.ez,
                    kk_model_vp=self.kk_model_vp, kk_model_rho=self.kk_model_rho)
