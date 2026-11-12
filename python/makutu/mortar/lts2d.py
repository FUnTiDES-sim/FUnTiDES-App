"""Pas de temps local par bloc (LF-LTS multi-niveaux) pour :class:`makutu.mortar.Mortar2D` : prototype.

Chaque bloc reçoit un niveau l >= 0 : il avance avec le pas dt / 2^l, dt étant le grand pas. Le schéma
est le saute-mouton à pas local de Diaz et Grote, appliqué récursivement niveau par niveau :

    u_{n+1} = -u_{n-1} + 2 y(dt),   y'' = -(w + A P_{>=1} y), y(0) = u_n, y'(0) = 0, w = A P_0 u_n,

où A = M^-1 K, P_l la projection sur les nœuds de niveau l et P_{>=l} sur ceux de niveau >= l. Sur
[0, tau], y est obtenu par deux pas de saute-mouton de tau / 2 pour les nœuds de niveau l, les niveaux
plus fins étant traités de la même façon à l'intérieur de chaque pas (récursion). Les nœuds d'un bloc
appartiennent à ce seul bloc (blocs non conformes) : le niveau d'un nœud est celui de son bloc ; le
couplage entre niveaux passe par la raideur des éléments et par l'opérateur d'interface SIPG, tous deux
contenus dans K.

Références :
- J. Diaz, M. J. Grote, Energy conserving explicit local time stepping for second-order wave equations,
  SIAM J. Sci. Comput. 31 (2009) ;
- M. J. Grote, M. Mehlin, S. A. Sauter, Convergence analysis of energy conserving explicit local time-
  stepping methods for the wave equation, SIAM J. Numer. Anal. 56 (2018) ;
- M. Almquist, M. Mehlin, Multilevel local time-stepping methods of Runge-Kutta type for wave
  equations, J. Sci. Comput. 71 (2017) (version multi-niveaux).

Sans amortissement ni éponge, le schéma conserve une énergie discrète (vérifiée numériquement) ; les
bords absorbants (amortissement diagonal C) sont traités au grand pas, comme dans le solveur LTS 2-D de
FUnTiDES ; la source est appliquée au grand pas.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .blocks2d import gll_cfl


def block_levels(m, dt, safety=0.9, max_level=6):
    """Niveau de chaque bloc pour le grand pas dt : plus petit l tel que dt / 2^l <= safety x dt_bloc,
    dt_bloc = C min(hx, hz) / vp_max (vp aux nœuds si le modèle est nodal)."""
    C = gll_cfl(m.order)
    lv = []
    for q, cf in zip(m.patches, m.coef):
        dtb = C * min(q.hx, q.hz) / float(cf["vp"].max())
        lv.append(int(min(max_level, max(0, np.ceil(np.log2(dt / (safety * dtb)) - 1e-12)))))
    return np.array(lv)


def _scaled(m):
    s = 1.0 / np.sqrt(m.M)
    return (sp.diags(s) @ m.K @ sp.diags(s)).tocsr()


def _lam_max(S, idx):
    """Plus grande valeur propre de la sous-matrice principale S[idx, idx]."""
    if len(idx) == 0:
        return 0.0
    sub = S[idx][:, idx]
    if len(idx) < 300:
        return float(np.linalg.eigvalsh(sub.toarray())[-1])
    return float(spla.eigsh(sub, k=1, which="LA", tol=1e-4, return_eigenvectors=False)[0])


def promoted_node_levels(m, block_level):
    """Niveau des nœuds : celui de leur bloc, relevé au niveau le plus fin des deux blocs pour les
    éléments de chaque segment d'interface (la pénalité varie comme 1 / h du côté fin)."""
    node_level = np.zeros(m.ndof, int)
    owner = np.zeros(m.ndof, int)
    for ib, (lv, d) in enumerate(zip(block_level, m.elem_dofs)):
        node_level[d.ravel()] = lv
        owner[d.ravel()] = ib
    for f in m._faces:
        l = np.maximum(block_level[owner[f["dA"][:, 0]]], block_level[owner[f["dB"][:, 0]]])[:, None]
        for d in (f["dA"], f["dB"]):
            np.maximum.at(node_level, d, np.broadcast_to(l, d.shape))
    return node_level, owner


def grade_levels(m, node_level, max_iter=200):
    """Gradation : dans chaque élément, et dans chaque paire d'éléments d'un segment d'interface, les
    niveaux des nœuds diffèrent d'au plus 1 (les nœuds trop grossiers sont relevés ; propagé jusqu'à
    stabilité). Un saut de deux niveaux entre nœuds couplés rend le LF-LTS instable (mesuré)."""
    node_level = node_level.copy()
    groups = [d.reshape(-1, d.shape[-1]) for d in m.elem_dofs]
    groups += [np.concatenate([f["dA"], f["dB"]], axis=1) for f in m._faces]
    for _ in range(max_iter):
        changed = False
        for g in groups:
            lv = node_level[g]
            low = lv.max(1, keepdims=True) - 1
            bad = lv < low
            if bad.any():
                np.maximum.at(node_level, g[bad], np.broadcast_to(low, g.shape)[bad])
                changed = True
        if not changed:
            return node_level
    raise RuntimeError("grade_levels: pas de convergence")


def assign_levels(m, dt, safety=0.95, max_level=8, max_iter=50):
    """Niveaux des nœuds pour le grand pas dt, sur la stabilité exacte : pour chaque groupe de nœuds
    (bloc, niveau), la plus grande valeur propre lambda de M^-1/2 K M^-1/2 restreinte au groupe
    (interfaces comprises) doit vérifier (dt / 2^niveau) sqrt(lambda) / 2 <= safety ; sinon le groupe
    passe au niveau suivant. Départ : estimation par bloc (:func:`block_levels`), promotion des
    éléments d'interface, gradation (:func:`grade_levels`). Retourne (niveaux des blocs de départ,
    niveaux des nœuds)."""
    S = _scaled(m)
    bl = block_levels(m, dt, safety=safety, max_level=max_level)
    node_level, owner = promoted_node_levels(m, bl)
    node_level = grade_levels(m, node_level)
    for _ in range(max_iter):
        changed = False
        keys = owner * (max_level + 1) + node_level
        for key in np.unique(keys):
            idx = np.where(keys == key)[0]
            l = int(key % (max_level + 1))
            lam = _lam_max(S, idx)
            need = int(np.ceil(np.log2(max(dt * np.sqrt(lam) / (2.0 * safety), 1.0)) - 1e-12))
            if need > l:
                node_level[idx] = min(max_level, need)
                changed = True
        if not changed:
            break
        node_level = grade_levels(m, node_level)
    return bl, node_level


class BlockLTS:
    """Intégrateur LF-LTS multi-niveaux sur un :class:`Mortar2D` assemblé (K, M, C).

    levels : niveau de chaque bloc (voir :func:`block_levels`), ou node_levels : niveau de chaque nœud
    (voir :func:`assign_levels`) ; dt : grand pas ; nu : stabilisation (Grote, Michel, Sauter : le second
    sous-pas de chaque niveau utilise kappa = (2 delta^2 - 1) / delta^2, delta = 1 + nu / 4 ; nu = 0 :
    schéma non stabilisé, kappa = 1)."""

    def __init__(self, m, dt, levels=None, promote_interfaces=True, node_levels=None, nu=0.01):
        self.m, self.dt = m, dt
        delta = 1.0 + nu / 4.0
        self.kappa = (2 * delta ** 2 - 1) / delta ** 2
        if node_levels is not None:
            node_level = np.asarray(node_levels, int)
            self.promoted = 0
        else:
            self.levels = np.asarray(levels, int)
            if promote_interfaces:
                node_level, _ = promoted_node_levels(m, self.levels)
            else:
                node_level = np.zeros(m.ndof, int)
                for lv, d in zip(self.levels, m.elem_dofs):
                    node_level[d.ravel()] = lv
            base = np.zeros(m.ndof, int)
            for lv, d in zip(self.levels, m.elem_dofs):
                base[d.ravel()] = lv
            self.promoted = int((node_level != base).sum())
        self.node_level = node_level
        self.L = int(node_level.max())
        Minv = sp.diags(1.0 / m.M)
        K = m.K.tocsc()
        # A P_l: columns of level l only (sparse), and the rows they reach (support)
        self.AP = []
        self.rows = []
        for l in range(self.L + 1):
            cols = np.where(node_level == l)[0]
            Kl = sp.csc_matrix((K.shape[0], K.shape[1]))
            if len(cols):
                sel = sp.csc_matrix((np.ones(len(cols)), (cols, cols)), shape=K.shape)
                Kl = (K @ sel).tocsr()
            Al = (Minv @ Kl).tocsr()
            Al.eliminate_zeros()
            self.AP.append(Al)
            self.rows.append(np.unique(Al.nonzero()[0]))
        # work counter: nonzeros of K applied per grand pas, against global stepping at dt / 2^L
        self.nnz = [a.nnz for a in self.AP]
        self.work_per_step = sum(self.nnz[l] * 2 ** l for l in range(self.L + 1))
        self.work_global = m.K.nnz * 2 ** self.L

    def _inner(self, z0, w, l, tau):
        """y(tau) pour y'' = -(w + A P_{>=l} y), y(0) = z0, y'(0) = 0."""
        if l > self.L:
            return z0 - 0.5 * tau * tau * w
        s = 0.5 * tau
        a0 = self.AP[l] @ z0
        y1 = self._inner(z0, w + a0, l + 1, s)                    # first step of tau / 2 (zero velocity)
        wl = w + (1.0 - self.kappa) * a0 + self.kappa * (self.AP[l] @ y1)
        return 2.0 * self._inner(y1, wl, l + 1, s) - z0           # second step (stabilized forcing)

    def step(self, u, u_prev, f=None):
        """Un grand pas : renvoie u_{n+1} (f : force au temps t_n, déjà divisée par M, ou None)."""
        m, dt = self.m, self.dt
        w = self.AP[0] @ u
        if f is not None:
            w = w - f
        if self.L == 0:
            y = u - 0.5 * dt * dt * w
        else:
            y = self._inner(u, w, 1, dt)
        # u_{n+1} + u_{n-1} = 2 y without damping; with C: (M + dt C / 2) u+ = 2 M y - (M - dt C / 2) u-
        if np.any(m.C):
            a_p = m.M + 0.5 * dt * m.C
            a_m = m.M - 0.5 * dt * m.C
            return (2.0 * m.M * y - a_m * u_prev) / a_p
        return 2.0 * y - u_prev

    def run(self, nt, source=None, wavelet=None, receivers=None, p0=None, track_energy=False):
        """Comme :meth:`Mortar2D.run`, au grand pas dt (éponge appliquée au grand pas)."""
        m, dt = self.m, self.dt
        taper = np.exp(-m.sigma * dt)
        u = np.zeros(m.ndof) if p0 is None else p0[1].copy()
        u_prev = np.zeros(m.ndof) if p0 is None else p0[0].copy()
        fvec = None
        if source is not None:
            fvec = np.asarray(m.point_matrix([source]).todense()).ravel() / m.M
        R = m.point_matrix(receivers) if receivers else None
        traces = np.zeros((nt, len(receivers))) if receivers else None
        energy = []
        for n in range(nt):
            f = None if fvec is None else wavelet[n] * fvec
            u_new = self.step(u, u_prev, f)
            if m.sponge > 0:
                u_new *= taper
                u *= taper
            if track_energy:
                energy.append(m.energy(u_new, u, dt))
            u_prev, u = u, u_new
            if R is not None:
                traces[n] = R @ u
        return dict(traces=traces, energy=np.array(energy), p=(u_prev, u))
