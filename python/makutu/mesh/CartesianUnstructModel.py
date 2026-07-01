"""
Build a FUnTiDES ``ModelUnstruct`` from a Cartesian structured grid with
explicit (heterogeneous) vp and rho Kokkos arrays.

This is used by :class:`~utilities.mesh.InternalMesh.StructuredMesh` when
a SEP velocity (or density) file is provided.  The Cartesian geometry is
reproduced exactly — GLL node coordinates, element-node connectivity, and
boundary flags are all generated analytically — so the solver physics are
identical to the structured-mesh case, but with user-supplied material
properties instead of the hardcoded defaults.

The returned model is a ``ModelUnstruct_f32_i32`` which supports mutable
vp/rho arrays for FWI.

Supported polynomial orders: 1, 2, 3.
"""

import numpy as np

import kokkos
import pyfuntides.model as FTModel


# ---------------------------------------------------------------------------
# Gauss-Lobatto-Legendre quadrature points on [-1, 1] per order
# (must match GLLPoints::get in FUnTiDES C++)
# ---------------------------------------------------------------------------
_GLL = {
    1: np.array([-1.0, 1.0], dtype=np.float64),
    2: np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    3: np.array([-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 1.0],
                dtype=np.float64),
}


def _kk(arr, dtype=kokkos.float32):
    """Allocate a Kokkos array (CudaUVMSpace, LayoutLeft) and fill from numpy.

    CudaUVMSpace + LayoutLeft is what ModelUnstructData_f32_i32 requires on
    CUDA builds (confirmed from the pybind11 constructor signature).

    For 2D arrays (e.g. global_node_index [n_elem, n_pts]) the numpy view of
    a LayoutLeft Kokkos array uses Fortran (column-major) strides.  Numpy's
    element-wise broadcast assignment correctly maps arr[i,j] → kk[i,j]
    regardless of the stride difference between source and destination.
    """
    kk = kokkos.array(list(arr.shape), dtype=dtype,
                      space=kokkos.CudaUVMSpace,
                      layout=kokkos.LayoutRight)
    np.array(kk, copy=False)[:] = arr
    return kk


def build_cartesian_unstruct_model(
        lx, ly, lz, ex, ey, ez, order,
        vp_elem,               # np.ndarray shape (ez, ey, ex), float32 [m/s]
        rho_elem=None,         # np.ndarray shape (ez, ey, ex), float32, or None
        on_nodes=False,
        is_elastic=False,
        origin_x=0.0,
        global_x_max=None,     # global domain xmax for boundary classification
        free_surface_on_top=True):
    """
    Build a ``ModelUnstruct_f32_i32`` from a Cartesian hex mesh with explicit
    material properties.

    Parameters
    ----------
    lx, ly, lz : float
        **Local** domain extents [m].
    ex, ey, ez : int
        **Local** element counts per direction.
    order : int
        Polynomial order (1, 2 or 3).
    vp_elem : np.ndarray, shape (ez, ey, ex), float32
        P-wave velocity at element centres [m/s].
    rho_elem : np.ndarray, shape (ez, ey, ex), float32, optional
        Density at element centres.  Defaults to 1.0 everywhere.
    on_nodes : bool
        Whether the model is defined on nodes (True) or elements (False).
    is_elastic : bool
        Elastic model flag (False for acoustic).
    origin_x : float
        X-coordinate of this rank's subdomain origin [m].
    global_x_max : float, optional
        X-coordinate of the global domain right boundary [m].
        Used for boundary classification in MPI runs.
        Defaults to ``origin_x + lx`` (single-process).
    free_surface_on_top : bool
        Apply free-surface BC on the top (z = lz) face.

    Returns
    -------
    model : ModelUnstruct_f32_i32
        FUnTiDES unstructured model object.
    kk_model_vp_element : kokkos array, shape [n_elem], float32
        Mutable Kokkos array backing vp (element-based).  Pass to
        ``WaveSolver`` as ``kk_model_vp`` for in-place FWI updates.
    kk_model_rho_element : kokkos array, shape [n_elem], float32
        Mutable Kokkos array backing rho (element-based).
    """
    if order not in _GLL:
        raise ValueError(f"Unsupported order {order}; available: {sorted(_GLL)}")

    if global_x_max is None:
        global_x_max = origin_x + lx

    gll   = _GLL[order]
    p1    = order + 1         # nodes per element edge
    n_pts = p1 ** 3           # nodes per element
    n_elem = ex * ey * ez
    nx = ex * order + 1
    ny = ey * order + 1
    nz = ez * order + 1
    n_nodes = nx * ny * nz
    hx = lx / ex
    hy = ly / ey
    hz = lz / ez

    # ------------------------------------------------------------------
    # 1. Node coordinates (GLL-based)
    # ------------------------------------------------------------------
    def _1d_coords(n_global, n_elem_1d, h, origin):
        gi = np.arange(n_global)
        ie = np.minimum(gi // order, n_elem_1d - 1)
        il = gi - ie * order
        return (origin + ie * h + (gll[il] + 1.0) * 0.5 * h).astype(np.float32)

    cx = _1d_coords(nx, ex, hx, origin_x)   # (nx,)
    cy = _1d_coords(ny, ey, hy, 0.0)         # (ny,)
    cz = _1d_coords(nz, ez, hz, 0.0)         # (nz,)

    # Full 3-D coordinates — gn = gix + giy*nx + giz*nx*ny  (x fastest)
    giz, giy, gix = np.mgrid[0:nz, 0:ny, 0:nx]   # (nz, ny, nx)
    coords_x = cx[gix].ravel().astype(np.float32)
    coords_y = cy[giy].ravel().astype(np.float32)
    coords_z = cz[giz].ravel().astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Element-node connectivity  global_node_index[e, pt] = global node
    # ------------------------------------------------------------------
    # Element 3-D indices (x fastest to match e = ix + iy*ex + iz*ex*ey)
    iz_e, iy_e, ix_e = np.mgrid[0:ez, 0:ey, 0:ex]
    ix_e = ix_e.ravel()    # (n_elem,)
    iy_e = iy_e.ravel()
    iz_e = iz_e.ravel()

    # Local node 3-D indices within an element
    k_l, j_l, i_l = np.mgrid[0:p1, 0:p1, 0:p1]
    i_l = i_l.ravel()      # (n_pts,)
    j_l = j_l.ravel()
    k_l = k_l.ravel()

    gx = ix_e[:, None] * order + i_l[None, :]   # (n_elem, n_pts)
    gy = iy_e[:, None] * order + j_l[None, :]
    gz = iz_e[:, None] * order + k_l[None, :]
    gni = (gx + gy * nx + gz * (nx * ny)).astype(np.int32)   # (n_elem, n_pts)

    # ------------------------------------------------------------------
    # 3. Boundary flags  (0=Interior, 1=Damping, 3=FreeSurface)
    # ------------------------------------------------------------------
    tol = min(hx, hy, hz) * 1e-4
    boundaries = np.zeros(n_nodes, dtype=np.int32)   # Interior by default

    on_xmin = coords_x < (origin_x + tol)
    on_xmax = coords_x > (global_x_max - tol)
    on_ymin = coords_y < tol
    on_ymax = coords_y > (ly - tol)
    on_zmin = coords_z < tol
    on_zmax = coords_z > (lz - tol)

    on_boundary = on_xmin | on_xmax | on_ymin | on_ymax | on_zmin | on_zmax
    boundaries[on_boundary] = 1   # Damping
    if free_surface_on_top:
        boundaries[on_zmax] = 3   # FreeSurface

    # ------------------------------------------------------------------
    # 4. Material property arrays
    # ------------------------------------------------------------------
    vp_flat = vp_elem.ravel().astype(np.float32)     # (n_elem,)

    if rho_elem is None:
        rho_flat = np.ones(n_elem, dtype=np.float32)
    else:
        rho_flat = rho_elem.ravel().astype(np.float32)

    # Node-based values: nearest-element lookup (used only when on_nodes=True)
    # Map each node to its "owning" element: e = ix_e + iy_e*ex + iz_e*ex*ey
    # Node global index → element index via floor division
    gix_nodes = gix.ravel()
    giy_nodes = giy.ravel()
    giz_nodes = giz.ravel()
    ie_nodes = np.minimum(gix_nodes // order, ex - 1)
    je_nodes = np.minimum(giy_nodes // order, ey - 1)
    ke_nodes = np.minimum(giz_nodes // order, ez - 1)
    e_of_node = ie_nodes + je_nodes * ex + ke_nodes * ex * ey   # (n_nodes,)

    vp_node_flat  = vp_flat[e_of_node].astype(np.float32)
    rho_node_flat = rho_flat[e_of_node].astype(np.float32)

    zeros_elem  = np.zeros(n_elem,  dtype=np.float32)
    zeros_nodes = np.zeros(n_nodes, dtype=np.float32)
    zeros_C     = np.zeros((n_elem, 6, 6), dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Allocate Kokkos arrays
    # ------------------------------------------------------------------
    kk_coords_x      = _kk(coords_x)
    kk_coords_y      = _kk(coords_y)
    kk_coords_z      = _kk(coords_z)
    kk_gni           = _kk(gni,  dtype=kokkos.int32)
    # FUn-Libs ModelUnstructData stores boundaries_t as a real (float32) view and
    # casts to BoundaryFlag in boundaryType(); allocate float32 to match.
    kk_boundaries    = _kk(boundaries.astype(np.float32), dtype=kokkos.float32)

    kk_vp_node       = _kk(vp_node_flat)
    kk_vp_element    = _kk(vp_flat)
    kk_rho_node      = _kk(rho_node_flat)
    kk_rho_element   = _kk(rho_flat)

    kk_vs_node       = _kk(zeros_nodes)
    kk_vs_element    = _kk(zeros_elem)
    kk_delta_node    = _kk(zeros_nodes)
    kk_delta_element = _kk(zeros_elem)
    kk_eps_node      = _kk(zeros_nodes)
    kk_eps_element   = _kk(zeros_elem)
    kk_gamma_node    = _kk(zeros_nodes)
    kk_gamma_element = _kk(zeros_elem)
    kk_theta_node    = _kk(zeros_nodes)
    kk_theta_element = _kk(zeros_elem)
    kk_phi_node      = _kk(zeros_nodes)
    kk_phi_element   = _kk(zeros_elem)
    kk_C_tensor      = _kk(zeros_C)

    # ------------------------------------------------------------------
    # 6. Build ModelUnstructData and ModelUnstruct
    # ------------------------------------------------------------------
    params = FTModel.ModelUnstructData_f32_i32(
        order, n_elem, n_nodes,
        float(lx), float(ly), float(lz),
        on_nodes, is_elastic,
        kk_gni,
        kk_coords_x, kk_coords_y, kk_coords_z,
        kk_vp_node,    kk_vp_element,
        kk_rho_node,   kk_rho_element,
        kk_vs_node,    kk_vs_element,
        kk_delta_node, kk_delta_element,
        kk_eps_node,   kk_eps_element,
        kk_gamma_node, kk_gamma_element,
        kk_theta_node, kk_theta_element,
        kk_phi_node,   kk_phi_element,
        kk_C_tensor,
        kk_boundaries,
    )

    model = FTModel.ModelUnstruct_f32_i32(params)
    model.build_face_connectivity()

    return model, kk_vp_element, kk_rho_element
