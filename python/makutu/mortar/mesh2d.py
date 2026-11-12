"""Maillage 2-D en blocs réguliers non conformes pour les solveurs makutu (AcousticSolver).

    mesh = BlocksMesh2D("Vp_xz.H", f0=8.0)
    solver = AcousticSolver(implem_type=..., mesh_type=..., dt=..., ..., **mesh.solver_kwargs())

Le pavage (:func:`plan_blocks_global`), le maillage (:class:`Mortar2D`, propriétés aux nœuds par
interpolation bilinéaire du modèle SEP) et le modèle FUnTiDES (:func:`funtides_model`) sont construits
ici ; ``solver_kwargs`` donne au solveur le modèle, un localisateur de points (sources et récepteurs) et
le crochet ``solver_setup`` qui pose l'opérateur d'interface après chaque ``compute_fe_init``.
Coordonnées makutu : x, y (ignoré), z profondeur depuis le haut du modèle.
"""
from __future__ import annotations

import numpy as np

from .blocks2d import grid_sampler, plan_blocks, plan_blocks_global
from .funtides2d import funtides_model, set_interface_operator
from .solver2d import Mortar2D, lagrange


class BlocksMesh2D:
    """vp_file : en-tête SEP 2-D (axe 1 = x, axe 2 = profondeur), ou vel / origin / spacing (grille
    (nx, nz), coin, pas) ; rho_file ou rho (scalaire, défaut 1) ; f0 : pic du Ricker ; order, ppw : critère
    de taille ; free_surface : surface libre en haut ; plan : "global" (pas de temps global,
    :func:`plan_blocks_global`) ou "elements" (moins d'éléments, :func:`plan_blocks`) ; patches : pavage
    imposé (liste de :class:`Patch` couvrant la grille), pour les cas tests."""

    dim = 2

    def __init__(self, vp_file=None, f0=8.0, order=4, ppw=5.0, rho_file=None, rho=1.0, free_surface=True,
                 plan="global", vel=None, origin=(0.0, 0.0), spacing=None, patches=None, **plan_kwargs):
        if vp_file is not None:
            from makutu.mesh.seismesh import grid_from_sep
            g = grid_from_sep(str(vp_file))
            vel = g.vel[:, 0, :].astype(float)
            (x0, _), _, (z0, _) = g.extent
            origin, spacing = (x0, z0), (g.spacing[0], g.spacing[2])
        if rho_file is not None:
            from makutu.mesh.seismesh import grid_from_sep
            rho = grid_from_sep(str(rho_file)).vel[:, 0, :].astype(float)
        self.vel = np.asarray(vel, float)
        self.origin, self.spacing = tuple(origin), tuple(spacing)
        self.order, self.f0, self.ppw = order, f0, ppw
        if patches is not None:                      # imposed tiling (tests)
            from .blocks2d import BlockPlan
            self.plan = BlockPlan(list(patches), sum(q.n_elements for q in patches), float("nan"), {})
        else:
            make = plan_blocks_global if plan == "global" else plan_blocks
            self.plan = make(self.vel, self.origin, self.spacing, order=order, f0=f0, ppw=ppw, **plan_kwargs)
        self.nodal = grid_sampler(self.vel, self.origin, self.spacing, rho=rho)   # reads self.vel live
        self.mortar = Mortar2D(self.plan.patches, order=order, nodal=self.nodal, assemble=False,
                               nodal_inset=0.5 * min(self.spacing))
        self.model, self._arrays, self.on_nodes, self._loc = funtides_model(self.mortar, free_surface)
        self.top = self.mortar.bbox[2]
        self.n_segments = None
        # first element of each block in the FUnTiDES numbering
        self._elem0 = np.cumsum([0] + [q.n_elements for q in self.plan.patches])

    # -- solver --------------------------------------------------------------------------------
    def solver_kwargs(self):
        """Arguments des solveurs makutu : modèle, ordre, localisateur, opérateur d'interface,
        propriétés aux nœuds."""
        import pyfuntides.solver as FTSolver
        loc = FTSolver.ModelLocationType.ONNODES if self.on_nodes else FTSolver.ModelLocationType.ONELEMENTS
        return dict(model=self.model, order=self.order, locator=self, solver_setup=self.setup_solver,
                    model_location=loc)

    def setup_solver(self, fun_solver):
        """Pose l'opérateur d'interface SIPG (appelé par le solveur makutu après compute_fe_init)."""
        self.n_segments = set_interface_operator(fun_solver, self.mortar)

    def dt_max(self, iterations=80):
        """Pas stable du saute-mouton sur ce maillage (FUnTiDES, opérateur d'interface compris)."""
        from .funtides2d import FuntidesBlocks2D
        return FuntidesBlocks2D(self.mortar).dt_max(iterations)

    @property
    def n_elements(self):
        return self.model.get_number_of_elements()

    # -- model (inversion grid) ------------------------------------------------------------------
    def _interpolation(self):
        """Matrice creuse B (nœuds, cellules de la grille en ordre sismique : x rapide, puis z) de
        l'interpolation bilinéaire des centres de cellules (celle de grid_sampler)."""
        if getattr(self, "_B", None) is None:
            import scipy.sparse as sp
            m = self.mortar
            nx, nz = self.vel.shape
            (x0, z0), (dx, dz) = self.origin, self.spacing
            u = np.clip((m.x_model - x0) / dx - 0.5, 0, nx - 1)
            w = np.clip((m.z_model - z0) / dz - 0.5, 0, nz - 1)
            i = np.minimum(u.astype(int), max(nx - 2, 0))
            k = np.minimum(w.astype(int), max(nz - 2, 0))
            a, b = u - i, w - k
            i1, k1 = np.minimum(i + 1, nx - 1), np.minimum(k + 1, nz - 1)
            rows = np.tile(np.arange(m.ndof), 4)
            cols = np.concatenate([k * nx + i, k * nx + i1, k1 * nx + i, k1 * nx + i1])
            vals = np.concatenate([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b])
            self._B = sp.csr_matrix((vals, (rows, cols)), shape=(m.ndof, nx * nz))
        return self._B

    def getVelocityModel(self):
        """Vitesse de la grille, ordre sismique (x rapide, z depuis le haut ; forme (nz, nx))."""
        return self.vel.T.ravel().astype(np.float32)

    def setVelocityModel(self, vel):
        """Nouvelle vitesse de la grille (ordre sismique ; valeurs > 0 seulement) ; les nœuds sont
        réinterpolés (le solveur makutu recalcule ensuite ses matrices)."""
        nx, nz = self.vel.shape
        v = np.asarray(vel, float).reshape(nz, nx).T
        mask = v > 0
        self.vel[mask] = v[mask]
        vp_n = self._interpolation() @ self.vel.T.ravel()
        np.array(self._arrays["vp_node"], copy=False)[...] = vp_n.astype(np.float32)

    def _unit_mass(self):
        """Masse nodale pour un coefficient unité : somme sur les éléments de |det J| w."""
        if getattr(self, "_unit", None) is None:
            m = self.mortar
            ww = np.outer(m.w, m.w).ravel()
            u = np.zeros(m.ndof)
            for q, d in zip(m.patches, m.elem_dofs):
                u += np.bincount(d.reshape(-1, ww.size).ravel(),
                                 np.tile(ww * q.hx * q.hz / 4.0, q.n_elements), minlength=m.ndof)
            self._unit = u
        return self._unit

    def nodeGradient(self, lam_pdd, pdd2, parametrization="c"):
        """Gradient de J par rapport à la vitesse de la grille, à partir des sommes nodales
        sum_n dt lambda p'' (lam_pdd) et sum_n dt p''^2 (pdd2) : dJ/dm_i = masse unité_i x lam_pdd_i
        (m = 1 / (rho vp^2) aux nœuds), puis dérivée selon la paramétrisation aux nœuds, puis
        transposée de l'interpolation vers la grille. Retourne (gradient, illumination), ordre sismique."""
        u = self._unit_mass()
        vp, rho = (np.asarray(a, float) for a in self.nodal(self.mortar.x_model, self.mortar.z_model))
        vp = self._interpolation() @ self.vel.T.ravel()
        if parametrization == "c":
            dm = -2.0 / (vp ** 3 * rho)
        elif parametrization == "1/c":
            dm = 2.0 / (vp * rho)
        elif parametrization == "1/c2":
            dm = 1.0 / rho
        else:
            raise ValueError(f"Unknown parametrization: {parametrization}")
        B = self._interpolation()
        return B.T @ (u * lam_pdd * dm), B.T @ (u * pdd2 * dm ** 2)

    # -- point location ------------------------------------------------------------------------
    def locate(self, x, y, z, tol=1e-6):
        """Élément FUnTiDES contenant (x, z) et coordonnées de référence (xi_x, xi_z), z vers le haut."""
        try:
            ip, ex, ez, xi, eta = self.mortar.locate(x, z)
        except ValueError:
            return -1, None
        e = int(self._elem0[ip] + ex * self.plan.patches[ip].nz + ez)
        return e, np.array([xi, -eta])                 # depth axis reversed: FUnTiDES z is upwards

    def _weights(self, xi):
        return [lagrange(self.mortar.xi, xi[0])[0], lagrange(self.mortar.xi, xi[1])[0]]

    def sourceWeights(self, xi):
        """Poids d'une source ponctuelle, ordre du second membre FUnTiDES ``ix + iz n``."""
        w = self._weights(xi)
        return np.outer(w[1], w[0]).ravel().astype(np.float32)

    def receiverWeights(self, xi):
        """Poids d'un récepteur, ordre des récepteurs makutu ``ix + iz n`` (2-D)."""
        w = self._weights(xi)
        return np.outer(w[1], w[0]).ravel().astype(np.float32)
