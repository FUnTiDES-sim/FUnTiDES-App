"""Cartesian mesh and model built from SEP velocity (and density) files."""

from types import SimpleNamespace

from .sizing import suggest_mesh


def mesh_from_sep(vp_file, f0, order, rho_file=None, ppw=6.0, cfl=0.3, elem_size=None,
                  free_surface_on_top=True, verbose=True):
    """
    Mesh of the domain of a SEP model, sized for a source frequency.

    The SEP file sets the dimension: n3 = 1 gives a 2-D model (axis 1 = x, axis 2 = z,
    CartesianMesh2D), otherwise 3-D (axes x, y, z, StructuredMesh); z is the depth
    (0 at the top). The element size and the time step come from
    :func:`makutu.mesh.suggest_mesh` (``ppw``, ``cfl``, or ``elem_size``).

    Returns
    -------
    SimpleNamespace with
        mesh : CartesianMesh2D or StructuredMesh (``mesh.solver_kwargs()`` for the solvers)
        dim : 2 or 3
        lengths : (lx, lz) or (lx, ly, lz) [m]
        counts : element counts along the same axes
        dt : time step [s]
        vp_min, vp_max : velocity range [m/s]
        lambda_min : shortest wavelength [m]
    """
    from ..input.SEPReader import read_sep
    from .CartesianModel2D import CartesianMesh2D
    from .InternalMesh import StructuredMesh

    sep = read_sep(vp_file)
    dim = 2 if sep.n3 == 1 else 3
    vp_min, vp_max = float(sep.data.min()), float(sep.data.max())
    if dim == 2:
        lengths = (sep.n1 * sep.d1, sep.n2 * sep.d2)
    else:
        lengths = (sep.n1 * sep.d1, sep.n2 * sep.d2, sep.n3 * sep.d3)
    counts, dt, lam = suggest_mesh(lengths, vp_min, vp_max, f0, order, ppw=ppw, cfl=cfl, elem_size=elem_size)
    if dim == 2:
        mesh = CartesianMesh2D(*lengths, *counts, order=order, vp_file=vp_file, rho_file=rho_file,
                               free_surface_on_top=free_surface_on_top)
        nodes = (counts[0] * order + 1) * (counts[1] * order + 1)
    else:
        mesh = StructuredMesh(*lengths, *counts, order=order, on_nodes=False, is_elastic=False,
                              free_surface_on_top=free_surface_on_top, vp_file=vp_file, rho_file=rho_file)
        nodes = mesh.numberOfPoints
    if verbose:
        sizes = " x ".join(f"{length / n:.1f}" for length, n in zip(lengths, counts))
        print(f"Model : {dim}-D, {' x '.join(f'{v / 1e3:g}' for v in lengths)} km, "
              f"vp {vp_min:.0f}-{vp_max:.0f} m/s")
        print(f"Mesh  : {' x '.join(map(str, counts))} elements of order {order} ({sizes} m), {nodes} nodes; "
              f"lambda_min {lam:.0f} m")
    return SimpleNamespace(mesh=mesh, dim=dim, lengths=lengths, counts=counts, dt=dt,
                           vp_min=vp_min, vp_max=vp_max, lambda_min=lam)
