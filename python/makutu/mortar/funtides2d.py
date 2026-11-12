"""Blocs non conformes dans le solveur SEM 2-D de FUnTiDES (pyfuntides, GPU).

Le maillage de :class:`makutu.mortar.Mortar2D` (blocs sans nœud commun) devient un ``ModelUnstruct`` de
dimension 2 ; l'opérateur SIPG de ses segments d'interface, sous forme structurée (nœuds de face,
lignes normales, interpolation aux points de Gauss : ``Mortar2D.interface_faces``), est passé au solveur
par ``Solver.set_interface_operator`` (FUnTiDES SEMsolver2D, branche dev-2d). La numérotation globale des
nœuds est celle de Mortar2D : les matrices, la source et les récepteurs s'y appliquent tels quels.

Conventions FUnTiDES : z vers le haut (z_FUnTiDES = z_max - profondeur), nœud local a + c (N + 1)
(a selon x, c selon z vers le haut). Bords extérieurs : Damping (paraxial), ou Surface en haut avec
``free_surface`` ; nœuds des interfaces entre blocs : intérieurs (condition naturelle, le couplage
vient de l'opérateur d'interface).
"""
from __future__ import annotations

import numpy as np

_INTERIOR, _DAMPING, _SURFACE = 0, 1, 3


def _kk(arr, dtype=None):
    import kokkos
    from makutu.tools.kokkosarrays import funtides_array
    kk = funtides_array(arr.shape, dtype=kokkos.float32 if dtype is None else dtype)
    np.array(kk, copy=False)[...] = arr
    return kk


def funtides_model(m, free_surface=False):
    """``ModelUnstruct`` 2-D de FUnTiDES pour le maillage en blocs ``m`` (:class:`Mortar2D`), avec sa
    numérotation de nœuds. Retourne (modèle, tableaux Kokkos à garder vivants, propriétés aux nœuds,
    ordre local Mortar2D des nœuds FUnTiDES)."""
    import kokkos
    import pyfuntides.model as FTModel
    N = m.order
    x0, x1, z0, z1 = m.bbox
    n_node = m.ndof
    # element-node table in FUnTiDES order: local a + c (N+1), c along z up = N - b (b along depth)
    a, c = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="xy")      # (c, a) grids
    loc_mortar = (a * (N + 1) + (N - c)).ravel()                               # index l = a (N+1) + b
    gni = np.concatenate([d.reshape(-1, (N + 1) ** 2)[:, loc_mortar] for d in m.elem_dofs]).astype(np.int32)
    n_elem = gni.shape[0]
    xs, zs = m.x.astype(np.float32), (z1 - m.z).astype(np.float32)
    if m.nodal is not None:
        vp_n, rho_n = (np.asarray(v, float) for v in m.nodal(m.x_model, m.z_model))
    else:
        vp_n, rho_n = np.zeros(n_node), np.zeros(n_node)
        for q, d in zip(m.patches, m.elem_dofs):
            vp_n[d.reshape(q.nx, q.nz, -1)] = q.vp[:, :, None]
            rho_n[d.reshape(q.nx, q.nz, -1)] = q.rho[:, :, None]
    vp_e = np.concatenate([q.vp.ravel() for q in m.patches])
    rho_e = np.concatenate([q.rho.ravel() for q in m.patches])
    tol = 1e-6 * max(x1 - x0, z1 - z0)
    outer = (m.x < x0 + tol) | (m.x > x1 - tol) | (m.z < z0 + tol) | (m.z > z1 - tol)
    flags = np.where(outer, _DAMPING, _INTERIOR).astype(np.int32)
    if free_surface:
        flags[m.z < z0 + tol] = _SURFACE
    zn, ze = np.zeros(n_node, np.float32), np.zeros(n_elem, np.float32)
    on_nodes = m.nodal is not None
    A = {
        "gni": _kk(gni, kokkos.int32), "x": _kk(xs), "y": _kk(zn), "z": _kk(zs),
        "vp_node": _kk(vp_n.astype(np.float32)), "vp_element": _kk(vp_e.astype(np.float32)),
        "rho_node": _kk(rho_n.astype(np.float32)), "rho_element": _kk(rho_e.astype(np.float32)),
        "zero_node": _kk(zn), "zero_element": _kk(ze),
        "C": _kk(np.zeros((n_elem, 6, 6), np.float32)), "boundaries": _kk(flags, kokkos.int32)}
    data = FTModel.ModelUnstructData_f32_i32(
        N, n_elem, n_node, float(x1 - x0), 0.0, float(z1 - z0), on_nodes, False,
        A["gni"], A["x"], A["y"], A["z"], A["vp_node"], A["vp_element"], A["rho_node"], A["rho_element"],
        A["zero_node"], A["zero_element"], A["zero_node"], A["zero_element"], A["zero_node"], A["zero_element"],
        A["zero_node"], A["zero_element"], A["zero_node"], A["zero_element"], A["zero_node"], A["zero_element"],
        A["C"], A["boundaries"])
    data.dim = 2
    return FTModel.ModelUnstruct_f32_i32(data), A, on_nodes, loc_mortar


def set_interface_operator(fun_solver, m):
    """Donne l'opérateur d'interface de ``m`` au solveur FUnTiDES ; retourne le nombre de segments."""
    F = m.interface_faces()
    if F is None:
        return 0
    i32 = lambda k: np.ascontiguousarray(F[k], dtype=np.int32).ravel()
    f32 = lambda k: np.ascontiguousarray(F[k], dtype=np.float32).ravel()
    fun_solver.set_interface_operator(i32("face"), i32("line"), f32("dn"), f32("lag"), f32("half_inv_rho"),
                                      f32("weight"), f32("gamma"), F["lag"].shape[2])
    return len(F["gamma"])


class FuntidesBlocks2D:
    """Modèle et solveur FUnTiDES d'un maillage en blocs :class:`Mortar2D`.

    m : Mortar2D (son argument ``nodal`` fournit vp, rho aux nœuds ; sinon valeurs par élément) ;
    free_surface : surface libre en haut (sinon les quatre bords absorbent) ;
    sponge : largeurs d'éponge (x, z) en m, passées à compute_fe_init ; surface_sponge : True, pas
    d'éponge le long du haut (tir et récepteurs près de la surface)."""

    def __init__(self, m, free_surface=False, sponge=(0.0, 0.0), taper_delta=0.0, with_interfaces=True,
                 surface_sponge=True):
        import kokkos
        import pyfuntides.model as FTModel
        import pyfuntides.solver as FTSolver
        self.m = m
        self.model, self._arrays, on_nodes, self.loc_mortar = funtides_model(m, free_surface)
        self.n_elements, self.n_nodes = self.model.get_number_of_elements(), m.ndof
        loc = FTSolver.ModelLocationType.ONNODES if on_nodes else FTSolver.ModelLocationType.ONELEMENTS
        self.solver = FTSolver.create_solver(FTSolver.MethodType.SEM, FTSolver.ImplemType.MAKUTU,
                                             FTSolver.MeshType.UNSTRUCT, loc, FTSolver.PhysicType.ACOUSTIC, m.order, dim=2)
        self.solver.compute_fe_init(self.model, [float(sponge[0]), 0.0, float(sponge[1])], bool(surface_sponge),
                                     float(taper_delta))
        self.n_segments = set_interface_operator(self.solver, m) if with_interfaces else 0

    def mass(self):
        return np.array(self.solver.get_mass_matrix(), copy=False).copy()

    def _data(self, cur=None):
        import kokkos
        import pyfuntides.solver as FTSolver
        npe = (self.m.order + 1) ** 2
        arrs = (_kk(np.zeros((1, 2), np.float32)), _kk(np.array([0], np.int32), kokkos.int32),
                _kk(np.zeros((1, npe), np.float32)), _kk(np.zeros(self.n_nodes, np.float32)),
                _kk(np.zeros(self.n_nodes, np.float32) if cur is None else cur.astype(np.float32)))
        data = FTSolver.SEMsolverDataAcoustic(FTSolver.WavefieldAcoustic(arrs[3], arrs[4]),
                                              FTSolver.RhsAcoustic(arrs[0], arrs[1], arrs[2]))
        return data, arrs

    def stiffness_product(self, u):
        """K u (raideur des blocs + opérateur d'interface), calculé par FUnTiDES."""
        data, arrs = self._data(u)
        self.solver.compute_forces(1e-3, 0, data)
        return np.array(self.solver.get_force_vector(0), copy=False).astype(float)

    def dt_max(self, iterations=60, seed=0):
        """Pas stable du saute-mouton, 2 / sqrt(lambda_max(M^-1 K)), par itération de puissance sur
        M^-1/2 K M^-1/2 (forces calculées par FUnTiDES ; borne inférieure de lambda, d'où une
        marge : utiliser 0,9 dt_max au plus)."""
        M = self.mass()
        ok = M > 0
        s = np.zeros_like(M, dtype=float)
        s[ok] = 1.0 / np.sqrt(M[ok])
        v = np.random.default_rng(seed).standard_normal(len(M)) * ok
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(iterations):
            w = s * self.stiffness_product(s * v)
            lam = float(v @ w)
            v = w / np.linalg.norm(w)
        return 2.0 / np.sqrt(lam)

    def run(self, dt, nt, source, wavelet, receivers=None, snapshots=(), raster=None):
        """Tir : source ponctuelle (x, profondeur), ondelette (nt,), récepteurs [(x, profondeur)].
        Retourne traces (nt, nrec) et instantanés (dict pas -> image si raster)."""
        import kokkos
        import pyfuntides.solver as FTSolver
        m, N = self.m, self.m.order
        npe = (N + 1) ** 2
        ip, ex, ez, xi, eta = m.locate(*source)
        from .solver2d import lagrange
        w_mortar = np.outer(lagrange(m.xi, xi)[0], lagrange(m.xi, eta)[0]).ravel()     # l = a (N+1) + b
        elem = sum(q.n_elements for q in m.patches[:ip]) + ex * m.patches[ip].nz + ez
        term = _kk(np.asarray(wavelet, np.float32)[None, :])
        el = _kk(np.array([elem], np.int32), kokkos.int32)
        wts = _kk(w_mortar[self.loc_mortar][None, :].astype(np.float32))
        prev, cur = _kk(np.zeros(self.n_nodes, np.float32)), _kk(np.zeros(self.n_nodes, np.float32))
        self._run_arrays = (term, el, wts, prev, cur)
        data = FTSolver.SEMsolverDataAcoustic(FTSolver.WavefieldAcoustic(prev, cur),
                                              FTSolver.RhsAcoustic(term, el, wts))
        R = m.point_matrix(receivers) if receivers else None
        sampler = None
        if R is not None and hasattr(FTSolver, "FieldSampler"):
            # receivers sampled on the device: reading the field from the host at every step moves the
            # (unified-memory) pages back and forth and dominates the step time
            R = R.tocsr()
            npp = int(np.diff(R.indptr).max())
            nodes = np.zeros((R.shape[0], npp), np.int32)
            wts = np.zeros((R.shape[0], npp), np.float32)
            for i in range(R.shape[0]):
                a, b = R.indptr[i], R.indptr[i + 1]
                nodes[i, :b - a], wts[i, :b - a] = R.indices[a:b], R.data[a:b]
            sampler = FTSolver.FieldSampler(nodes.ravel(), wts.ravel(), npp)
        traces = np.zeros((nt, len(receivers))) if receivers else None
        snaps, snapset = {}, set(snapshots)
        for n in range(nt):
            self.solver.compute_one_step(dt, n, data)
            data.swap_wavefields()
            if sampler is not None:
                traces[n] = sampler.sample(data, 0)
            if (R is not None and sampler is None) or n in snapset:
                p = np.array(data.get_current_field(0), copy=False)
                if R is not None and sampler is None:
                    traces[n] = R @ p
                if n in snapset and raster is not None:
                    snaps[n] = raster[0] @ p
        return dict(traces=traces, snapshots=snaps)
