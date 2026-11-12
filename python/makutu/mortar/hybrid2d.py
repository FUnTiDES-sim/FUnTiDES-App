"""Prototype hybride 2-D acoustique : SEM continu sur grilles régulières + Galerkin discontinu sur triangles.

Principe (ordre et points par longueur d'onde fixés dans chaque zone, taille des mailles donnée par la
vitesse et la fréquence) :
- zones SEM (:class:`QuadZone`) : grille régulière de rectangles, base de Lagrange GLL d'ordre propre à la
  zone (p-adaptivité : l'ordre change d'une zone à l'autre, à travers une interface DG), masse diagonale ;
- zones triangles (:class:`TriZone`) : Galerkin discontinu sur triangles droits, base modale orthonormée de
  Koornwinder-Dubiner (masse diagonale), propriétés constantes par triangle ; elles ne servent que le long
  des interfaces du modèle (bords courbes), que les arêtes des triangles suivent ;
- toutes les faces entre zones, et toutes les arêtes entre triangles, portent le flux de pénalité
  intérieure symétrique (SIPG) : c'est le seul couplage DG. Les bords de zones sont alignés sur les axes ;
  de part et d'autre, les arêtes se correspondent de façon quelconque (segments d'intersection).

Même forme faible et même schéma saute-mouton que :class:`makutu.mortar.Mortar2D` (voir
doc/mortar-maths-et-implementation.md). Prototype NumPy / SciPy (matrices creuses), sans GPU.

Références : J. S. Hesthaven, T. Warburton, Nodal Discontinuous Galerkin Methods, Springer (2008) (base
de Dubiner, inégalité de trace sur le triangle) ; M. J. Grote, A. Schneebeli, D. Schötzau, SIAM J. Numer.
Anal. 44 (2006) (SIPG pour l'équation des ondes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gamma as _gamma

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import legendre as _leg
from scipy.special import eval_jacobi

from .solver2d import derivative, gll, lagrange

# --------------------------------------------------------------------------------------------
# Base de Dubiner sur le triangle de référence {r, s >= -1, r + s <= 0}
# --------------------------------------------------------------------------------------------


def _jacobi_n(x, a, b, n):
    """Polynôme de Jacobi orthonormé P_n^(a,b)."""
    g = 2.0 ** (a + b + 1) / (2 * n + a + b + 1) * _gamma(n + a + 1) * _gamma(n + b + 1) / (
        _gamma(n + a + b + 1) * _gamma(n + 1))
    return eval_jacobi(n, a, b, x) / np.sqrt(g)


def _djacobi_n(x, a, b, n):
    if n == 0:
        return np.zeros_like(x)
    return np.sqrt(n * (n + a + b + 1.0)) * _jacobi_n(x, a + 1, b + 1, n - 1)


def dubiner(order, r, s):
    """Base orthonormée du triangle de référence et ses dérivées : (V, Vr, Vs), forme (..., n),
    n = (order + 1)(order + 2) / 2."""
    r, s = np.asarray(r, float), np.asarray(s, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(np.abs(1 - s) > 1e-14, 2 * (1 + r) / (1 - s) - 1, -1.0)
    b = s
    V, Vr, Vs = [], [], []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            fa, dfa = _jacobi_n(a, 0, 0, i), _djacobi_n(a, 0, 0, i)
            gb, dgb = _jacobi_n(b, 2 * i + 1, 0, j), _djacobi_n(b, 2 * i + 1, 0, j)
            V.append(np.sqrt(2.0) * fa * gb * (1 - b) ** i)
            dr = dfa * gb
            ds = dfa * (gb * (0.5 * (1 + a)))
            tmp = dgb * (0.5 * (1 - b)) ** i
            if i > 0:
                dr = dr * (0.5 * (1 - b)) ** (i - 1)
                ds = ds * (0.5 * (1 - b)) ** (i - 1)
                tmp = tmp - 0.5 * i * gb * (0.5 * (1 - b)) ** (i - 1)
            ds = ds + fa * tmp
            Vr.append(2.0 ** (i + 0.5) * dr)
            Vs.append(2.0 ** (i + 0.5) * ds)
    return np.stack(V, -1), np.stack(Vr, -1), np.stack(Vs, -1)


def triangle_quadrature(n):
    """Quadrature du triangle de référence (Gauss x Gauss-Jacobi, repliée) : exacte au degré 2 n - 1."""
    from scipy.special import roots_jacobi
    xa, wa = _leg.leggauss(n)
    xb, wb = roots_jacobi(n, 1, 0)                  # weight (1 - b)
    A, B = np.meshgrid(xa, xb, indexing="ij")
    r = 0.5 * (1 + A) * (1 - B) - 1
    w = 0.5 * np.outer(wa, wb)
    return r.ravel(), B.ravel(), w.ravel()


# --------------------------------------------------------------------------------------------
# Zones
# --------------------------------------------------------------------------------------------


@dataclass
class QuadZone:
    """Grille régulière nx x nz de rectangles sur [x0, x1] x [z0, z1], SEM continu d'ordre ``order``.
    model : fonction (x, z) -> (vp, rho) lue aux nœuds, ou (vp, rho) scalaires ; inset : distance à
    laquelle les nœuds de bord lisent le modèle à l'intérieur de la zone là où il saute de plus de 10 %."""

    x0: float
    x1: float
    z0: float
    z1: float
    nx: int
    nz: int
    order: int = 4
    model: object = (1500.0, 1000.0)
    inset: float = 0.0
    kind = "quad"

    @property
    def hx(self):
        return (self.x1 - self.x0) / self.nx

    @property
    def hz(self):
        return (self.z1 - self.z0) / self.nz

    @property
    def n_elements(self):
        return self.nx * self.nz

    def setup(self, off):
        N = self.order
        n1 = N + 1
        self.xi, self.w = gll(N)
        self.D = derivative(self.xi)
        nxn, nzn = self.nx * N + 1, self.nz * N + 1
        ex, ez, a, b = np.meshgrid(np.arange(self.nx), np.arange(self.nz), np.arange(n1), np.arange(n1),
                                   indexing="ij")
        self.elem_dofs = (off + (ex * N + a) * nzn + (ez * N + b)).reshape(self.nx, self.nz, n1 * n1)
        ux = np.concatenate([self.x0 + self.hx * (e + (self.xi[:-1] + 1) / 2) for e in range(self.nx)] + [[self.x1]])
        uz = np.concatenate([self.z0 + self.hz * (e + (self.xi[:-1] + 1) / 2) for e in range(self.nz)] + [[self.z1]])
        X, Z = np.meshgrid(ux, uz, indexing="ij")
        self.x, self.z = X.ravel(), Z.ravel()
        self.off, self.ndof = off, nxn * nzn
        if callable(self.model):
            xm, zm = self.x.copy(), self.z.copy()
            if self.inset > 0:
                # zone-edge nodes read the model inside the zone where it jumps across the edge (> 10 %)
                e, t = self.inset, 1e-9 * max(self.x1 - self.x0, self.z1 - self.z0)
                dxi = np.where(np.abs(xm - self.x0) < t, e, np.where(np.abs(xm - self.x1) < t, -e, 0.0))
                dzi = np.where(np.abs(zm - self.z0) < t, e, np.where(np.abs(zm - self.z1) < t, -e, 0.0))
                edge = (dxi != 0) | (dzi != 0)
                vin = np.asarray(self.model(xm[edge] + dxi[edge], zm[edge] + dzi[edge])[0], float)
                vout = np.asarray(self.model(xm[edge] - dxi[edge], zm[edge] - dzi[edge])[0], float)
                sharp = np.abs(vin - vout) / np.minimum(vin, vout) > 0.1
                idx = np.where(edge)[0][sharp]
                xm[idx] += dxi[idx]
                zm[idx] += dzi[idx]
            vp, rho = (np.asarray(q, float) for q in self.model(xm, zm))
        else:
            vp, rho = (np.full(self.ndof, float(q)) for q in self.model)
        self.vp_n, self.rho_n = vp, rho
        loc = self.elem_dofs - off
        self.b_en, self.c_en, self.imp_en = 1.0 / rho[loc], 1.0 / (rho * vp ** 2)[loc], (rho * vp)[loc]
        self.vmax = float(vp.max())

    def volume(self):
        N, n1, w, D = self.order, self.order + 1, self.w, self.D
        n = n1 * n1
        I = np.eye(n1)
        d = self.elem_dofs.reshape(-1, n)
        B = self.b_en.reshape(-1, n1, n1) * w[None, :, None] * w[None, None, :]
        Kx = (self.hz / self.hx) * np.einsum("ca,ecb,cd,bf->eabdf", D, B, D, I)
        Kz = (self.hx / self.hz) * np.einsum("cb,eac,cd,af->eabfd", D, B, D, I)
        Ke = (Kx + Kz).reshape(-1, n, n)
        me = (self.hx * self.hz / 4) * np.outer(w, w).ravel()[None, :] * self.c_en.reshape(-1, n)
        return d, Ke, me

    def evaluate(self, elems, pts):
        """Pour les éléments elems (S,) aux points pts (S, Q, 2) : dofs (S, n), V, Gx, Gz (S, Q, n), 1/rho (S, Q)."""
        n1 = self.order + 1
        ex, ez = elems // self.nz, elems % self.nz
        xi = 2 * (pts[..., 0] - self.x0 - ex[:, None] * self.hx) / self.hx - 1
        eta = 2 * (pts[..., 1] - self.z0 - ez[:, None] * self.hz) / self.hz - 1
        Lx, Lz = lagrange(self.xi, xi), lagrange(self.xi, eta)
        dLx, dLz = Lx @ self.D, Lz @ self.D                      # l_j'(x) = sum_i l_i(x) D[i, j]
        S, Q = xi.shape
        V = (Lx[..., :, None] * Lz[..., None, :]).reshape(S, Q, n1 * n1)
        Gx = (2 / self.hx) * (dLx[..., :, None] * Lz[..., None, :]).reshape(S, Q, n1 * n1)
        Gz = (2 / self.hz) * (Lx[..., :, None] * dLz[..., None, :]).reshape(S, Q, n1 * n1)
        dofs = self.elem_dofs[ex, ez]
        b = np.einsum("sqi,si->sq", V, self.b_en[ex, ez])
        return dofs, V, Gx, Gz, b

    def boundary_edges(self):
        """Arêtes de bord de zone : (elem, p0 (2,), p1 (2,), kappa de pénalité)."""
        N = self.order
        kx = 1.25 * N * (N + 1) / (2 * self.hx)
        kz = 1.25 * N * (N + 1) / (2 * self.hz)
        out = []
        zs = self.z0 + self.hz * np.arange(self.nz + 1)
        xs = self.x0 + self.hx * np.arange(self.nx + 1)
        for ex, X in ((0, self.x0), (self.nx - 1, self.x1)):
            for k in range(self.nz):
                out.append((ex * self.nz + k, (X, zs[k]), (X, zs[k + 1]), kx))
        for ez, Zc in ((0, self.z0), (self.nz - 1, self.z1)):
            for i in range(self.nx):
                out.append((i * self.nz + ez, (xs[i], Zc), (xs[i + 1], Zc), kz))
        return out

    def centroids(self):
        ex, ez = np.divmod(np.arange(self.n_elements), self.nz)
        return np.stack([self.x0 + (ex + 0.5) * self.hx, self.z0 + (ez + 0.5) * self.hz], 1)

    def locate(self, pts):
        """Élément contenant chaque point (-1 si hors de la zone)."""
        t = 1e-9 * max(self.x1 - self.x0, self.z1 - self.z0)
        inside = ((pts[:, 0] >= self.x0 - t) & (pts[:, 0] <= self.x1 + t) &
                  (pts[:, 1] >= self.z0 - t) & (pts[:, 1] <= self.z1 + t))
        ex = np.clip(((pts[:, 0] - self.x0) // self.hx).astype(int), 0, self.nx - 1)
        ez = np.clip(((pts[:, 1] - self.z0) // self.hz).astype(int), 0, self.nz - 1)
        return np.where(inside, ex * self.nz + ez, -1)

    def dof_points(self):
        return self.x, self.z

    def project(self, g):
        return g(self.x, self.z)


@dataclass
class GridZone:
    """Cellules actives (masque ``active`` (nx, nz)) d'une grille rectiligne : SEM continu et conforme d'ordre
    ``order`` sur une région quelconque (trous, bords en escalier). Lignes de la grille : ``xe`` (nx + 1,) et
    ``ze`` (nz + 1,) (pas variable par colonne et par ligne), ou à défaut pas constants hx, hz depuis (x0, z0).
    Sert de grande zone SEM sans interface non conforme. model : fonction (x, z) -> (vp, rho)."""

    x0: float
    z0: float
    hx: float
    hz: float
    active: np.ndarray
    order: int = 4
    model: object = (1500.0, 1000.0)
    xe: np.ndarray | None = None
    ze: np.ndarray | None = None
    kind = "grid"

    def __post_init__(self):
        act = np.asarray(self.active, bool)
        nx, nz = act.shape
        if self.xe is None:
            self.xe = self.x0 + self.hx * np.arange(nx + 1)
        if self.ze is None:
            self.ze = self.z0 + self.hz * np.arange(nz + 1)
        self.xe, self.ze = np.asarray(self.xe, float), np.asarray(self.ze, float)
        assert len(self.xe) == nx + 1 and len(self.ze) == nz + 1
        self.x0, self.z0 = float(self.xe[0]), float(self.ze[0])
        self.wx, self.wz = np.diff(self.xe), np.diff(self.ze)
        self.uniform = np.ptp(self.wx) < 1e-9 * self.wx.max() and np.ptp(self.wz) < 1e-9 * self.wz.max()
        self.hx, self.hz = float(self.wx.min()), float(self.wz.min())   # smallest pitches

    @property
    def x_end(self):
        return float(self.xe[-1])

    @property
    def z_end(self):
        return float(self.ze[-1])

    @property
    def n_elements(self):
        return int(self.cells.shape[0]) if hasattr(self, "cells") else int(np.asarray(self.active).sum())

    def cell_box(self, i, k):
        return self.xe[i], self.ze[k], self.xe[i + 1], self.ze[k + 1]

    def setup(self, off):
        N, n1 = self.order, self.order + 1
        act = np.asarray(self.active, bool)
        self.nx, self.nz = act.shape
        self.cells = np.argwhere(act)                                  # (ne, 2): i, k
        self.cell_index = -np.ones(act.shape, int)
        self.cell_index[act] = np.arange(len(self.cells))
        self.hx_e, self.hz_e = self.wx[self.cells[:, 0]], self.wz[self.cells[:, 1]]
        self.xi, self.w = gll(N)
        self.D = derivative(self.xi)
        nzn = self.nz * N + 1
        a, b = np.meshgrid(np.arange(n1), np.arange(n1), indexing="ij")
        I = self.cells[:, 0, None] * N + a.ravel()[None, :]
        K = self.cells[:, 1, None] * N + b.ravel()[None, :]
        g = I * nzn + K                                                # grid node ids (ne, n1^2)
        used, inv = np.unique(g, return_inverse=True)
        self.elem_dofs = off + inv.reshape(g.shape)
        self.off, self.ndof = off, len(used)
        gi, gk = used // nzn, used % nzn

        def coord(idx, e, w):
            c = idx // N
            return np.append(e, e[-1])[c] + (self.xi[idx % N] + 1) * 0.5 * np.append(w, 0.0)[c]
        self.x = coord(gi, self.xe, self.wx)
        self.z = coord(gk, self.ze, self.wz)
        if callable(self.model):
            vp, rho = (np.asarray(q, float) for q in self.model(self.x, self.z))
        else:
            vp, rho = (np.full(self.ndof, float(q)) for q in self.model)
        self.vp_n, self.rho_n = vp, rho
        loc = self.elem_dofs - off
        self.b_en, self.c_en, self.imp_en = 1.0 / rho[loc], 1.0 / (rho * vp ** 2)[loc], (rho * vp)[loc]
        self.vmax = float(vp.max())

    def volume(self):
        n1, w, D = self.order + 1, self.w, self.D
        n = n1 * n1
        I = np.eye(n1)
        hx, hz = self.hx_e[:, None, None, None, None], self.hz_e[:, None, None, None, None]
        B = self.b_en.reshape(-1, n1, n1) * w[None, :, None] * w[None, None, :]
        Kx = (hz / hx) * np.einsum("ca,ecb,cd,bf->eabdf", D, B, D, I)
        Kz = (hx / hz) * np.einsum("cb,eac,cd,af->eabfd", D, B, D, I)
        me = (self.hx_e * self.hz_e / 4)[:, None] * np.outer(w, w).ravel()[None, :] * self.c_en
        return self.elem_dofs, (Kx + Kz).reshape(-1, n, n), me

    def evaluate(self, elems, pts):
        n1 = self.order + 1
        ci, ck = self.cells[elems, 0], self.cells[elems, 1]
        hx, hz = self.wx[ci][:, None], self.wz[ck][:, None]
        xi = 2 * (pts[..., 0] - self.xe[ci][:, None]) / hx - 1
        eta = 2 * (pts[..., 1] - self.ze[ck][:, None]) / hz - 1
        Lx, Lz = lagrange(self.xi, xi), lagrange(self.xi, eta)
        dLx, dLz = Lx @ self.D, Lz @ self.D
        S, Q = xi.shape
        V = (Lx[..., :, None] * Lz[..., None, :]).reshape(S, Q, n1 * n1)
        Gx = (2 / hx)[..., None] * (dLx[..., :, None] * Lz[..., None, :]).reshape(S, Q, n1 * n1)
        Gz = (2 / hz)[..., None] * (Lx[..., :, None] * dLz[..., None, :]).reshape(S, Q, n1 * n1)
        b = np.einsum("sqi,si->sq", V, self.b_en[elems])
        return self.elem_dofs[elems], V, Gx, Gz, b

    def _free_sides(self):
        """(cellule, côté) des côtés de cellules actives sans voisine active : 0 x-, 1 x+, 2 z-, 3 z+."""
        act = np.asarray(self.active, bool)
        pad = np.pad(act, 1)
        out = []
        for side, (di, dk) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            nb = pad[1 + di:1 + di + self.nx, 1 + dk:1 + dk + self.nz]
            for e in np.where(~nb[act])[0]:
                out.append((int(e), side))
        return out

    def _side_geometry(self, e, side):
        xa, za, xb, zb = self.cell_box(*self.cells[e])
        return [((xa, za), (xa, zb)), ((xb, za), (xb, zb)), ((xa, za), (xb, za)), ((xa, zb), (xb, zb))][side]

    def boundary_edges(self):
        N = self.order
        c = 1.25 * N * (N + 1) / 2
        return [(e, *self._side_geometry(e, sd), c / (self.hx_e[e] if sd < 2 else self.hz_e[e]))
                for e, sd in self._free_sides()]

    def domain_faces(self, bbox, tol):
        """Côtés de cellules sur le bord du domaine : [(nom, dofs (N+1,), h, rho v (N+1,))]."""
        x0, x1, z0, z1 = bbox
        N, n1 = self.order, self.order + 1
        out = []
        for e, sd in self._free_sides():
            (xa, za), (xb, zb) = self._side_geometry(e, sd)
            d = self.elem_dofs[e].reshape(n1, n1)
            imp = self.imp_en[e].reshape(n1, n1)
            if sd == 0 and abs(xa - x0) < tol:
                out.append(("left", d[0, :], self.hz_e[e], imp[0, :]))
            elif sd == 1 and abs(xa - x1) < tol:
                out.append(("right", d[N, :], self.hz_e[e], imp[N, :]))
            elif sd == 2 and abs(za - z0) < tol:
                out.append(("top", d[:, 0], self.hx_e[e], imp[:, 0]))
            elif sd == 3 and abs(za - z1) < tol:
                out.append(("bottom", d[:, N], self.hx_e[e], imp[:, N]))
        return out

    def centroids(self):
        return np.stack([self.xe[self.cells[:, 0]] + 0.5 * self.hx_e, self.ze[self.cells[:, 1]] + 0.5 * self.hz_e], 1)

    def locate(self, pts):
        i = np.searchsorted(self.xe, pts[:, 0], "right") - 1
        k = np.searchsorted(self.ze, pts[:, 1], "right") - 1
        # points on a cell edge: try the neighbour too
        out = -np.ones(len(pts), int)
        t = 1e-9 * max(self.wx.max(), self.wz.max())
        for di in (0, -1):
            for dk in (0, -1):
                ii, kk = i + di, k + dk
                ok = (out < 0) & (ii >= 0) & (ii < self.nx) & (kk >= 0) & (kk < self.nz)
                cand = np.full(len(pts), -1)
                cand[ok] = self.cell_index[ii[ok], kk[ok]]
                inside = cand >= 0
                if inside.any():
                    iic, kkc = np.clip(ii, 0, self.nx - 1), np.clip(kk, 0, self.nz - 1)
                    inside &= ((pts[:, 0] >= self.xe[iic] - t) & (pts[:, 0] <= self.xe[iic + 1] + t)
                               & (pts[:, 1] >= self.ze[kkc] - t) & (pts[:, 1] <= self.ze[kkc + 1] + t))
                out[inside] = cand[inside]
        return out

    def dof_points(self):
        return self.x, self.z

    def project(self, g):
        return g(self.x, self.z)


@dataclass
class TriZone:
    """Triangles droits en Galerkin discontinu d'ordre ``order`` (base de Dubiner).
    nodes (n, 2), tris (m, 3) ; vp, rho : par triangle (m,) ou scalaires ; model : fonction
    (x, z, e) -> (vp, rho) aux points (x, z) des triangles e : propriétés lues aux points de quadrature
    (masse alors pleine par triangle, inversée une fois ; sinon constantes par triangle et masse diagonale)."""

    nodes: np.ndarray
    tris: np.ndarray
    order: int = 3
    vp: object = 1500.0
    rho: object = 1000.0
    tau: float = 0.5
    model: object = None
    kind = "tri"

    @property
    def n_elements(self):
        return len(self.tris)

    def setup(self, off):
        P = np.asarray(self.nodes, float)
        T = np.asarray(self.tris, int).copy()
        e2, e3 = P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]
        det = e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0]
        T[det < 0] = T[det < 0][:, [0, 2, 1]]                      # counter-clockwise
        self.tris = T
        self.v1 = P[T[:, 0]]
        e2, e3 = P[T[:, 1]] - self.v1, P[T[:, 2]] - self.v1
        det = e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0]
        self.area = 0.5 * det
        # inverse of A = [e2 e3] (columns): rows give grad(lambda2), grad(lambda3)
        self.Ainv = np.stack([np.stack([e3[:, 1], -e3[:, 0]], -1), np.stack([-e2[:, 1], e2[:, 0]], -1)], 1) / det[:, None, None]
        m = len(T)
        self.np_ = (self.order + 1) * (self.order + 2) // 2
        self.off, self.ndof = off, m * self.np_
        self.elem_dofs = off + np.arange(m * self.np_).reshape(m, self.np_)
        if self.model is not None:                          # centroid values: display, time step, penalty
            cen = (P[T[:, 0]] + P[T[:, 1]] + P[T[:, 2]]) / 3.0
            vc, rc = self.model(cen[:, 0], cen[:, 1], np.arange(m))
            self.vp, self.rho = np.asarray(vc, float), np.asarray(rc, float)
        self.vp_e = np.broadcast_to(np.asarray(self.vp, float), (m,)).copy()
        self.rho_e = np.broadcast_to(np.asarray(self.rho, float), (m,)).copy()
        self.vmax = float(self.vp_e.max())
        self.cen = (P[T[:, 0]] + P[T[:, 1]] + P[T[:, 2]]) / 3.0

    def _basis_at(self, elems, pts):
        d = pts - self.v1[elems][:, None, :]
        Ai = self.Ainv[elems]
        l2 = d[..., 0] * Ai[:, None, 0, 0] + d[..., 1] * Ai[:, None, 0, 1]
        l3 = d[..., 0] * Ai[:, None, 1, 0] + d[..., 1] * Ai[:, None, 1, 1]
        V, Vr, Vs = dubiner(self.order, 2 * l2 - 1, 2 * l3 - 1)
        Gx = 2 * (Ai[:, None, 0, 0, None] * Vr + Ai[:, None, 1, 0, None] * Vs)
        Gz = 2 * (Ai[:, None, 0, 1, None] * Vr + Ai[:, None, 1, 1, None] * Vs)
        return V, Gx, Gz

    def volume(self):
        r, s, w = triangle_quadrature(self.order + 1)
        V, Vr, Vs = dubiner(self.order, r, s)                      # (Q, n)
        Ai = self.Ainv
        Gx = 2 * (Ai[:, 0, 0, None, None] * Vr + Ai[:, 1, 0, None, None] * Vs)   # (m, Q, n)
        Gz = 2 * (Ai[:, 0, 1, None, None] * Vr + Ai[:, 1, 1, None, None] * Vs)
        detJ = self.area / 2.0
        if self.model is None:
            Ke = np.einsum("q,eqi,eqj->eij", w, Gx, Gx) + np.einsum("q,eqi,eqj->eij", w, Gz, Gz)
            Ke *= (detJ / self.rho_e)[:, None, None]
            me = np.repeat((detJ / (self.rho_e * self.vp_e ** 2))[:, None], self.np_, axis=1)
            return self.elem_dofs, Ke, me
        # properties at the quadrature points (exact for the polynomial part up to degree 2 p + 1)
        r2, s2, w2 = triangle_quadrature(self.order + 2)
        V2, Vr2, Vs2 = dubiner(self.order, r2, s2)
        P = np.asarray(self.nodes, float)
        e2, e3 = P[self.tris[:, 1]] - self.v1, P[self.tris[:, 2]] - self.v1
        X = (self.v1[:, None, :] + 0.5 * (r2 + 1)[None, :, None] * e2[:, None, :]
             + 0.5 * (s2 + 1)[None, :, None] * e3[:, None, :])
        m, Q = X.shape[:2]
        vq, rq = self.model(X[..., 0].ravel(), X[..., 1].ravel(), np.repeat(np.arange(m), Q))
        vq, rq = np.asarray(vq, float).reshape(m, Q), np.asarray(rq, float).reshape(m, Q)
        Gx2 = 2 * (Ai[:, 0, 0, None, None] * Vr2 + Ai[:, 1, 0, None, None] * Vs2)
        Gz2 = 2 * (Ai[:, 0, 1, None, None] * Vr2 + Ai[:, 1, 1, None, None] * Vs2)
        wb = w2[None, :] / rq * detJ[:, None]
        Ke = np.einsum("eq,eqi,eqj->eij", wb, Gx2, Gx2) + np.einsum("eq,eqi,eqj->eij", wb, Gz2, Gz2)
        Me = np.einsum("eq,qi,qj->eij", w2[None, :] / (rq * vq ** 2) * detJ[:, None], V2, V2)
        return self.elem_dofs, Ke, Me

    def evaluate(self, elems, pts):
        V, Gx, Gz = self._basis_at(elems, pts)
        if self.model is not None:
            S, Q = pts.shape[:2]
            _, rq = self.model(pts[..., 0].ravel(), pts[..., 1].ravel(), np.repeat(elems, Q))
            b = 1.0 / np.asarray(rq, float).reshape(S, Q)
        else:
            b = np.broadcast_to((1.0 / self.rho_e[elems])[:, None], pts.shape[:2])
        return self.elem_dofs[elems], V, Gx, Gz, b

    def edges(self):
        """(interior faces (eA, eB, p0, p1), boundary edges [(elem, p0, p1, kappa)])."""
        P, T = np.asarray(self.nodes, float), self.tris
        p = self.order
        ed = {}
        for e, t in enumerate(T):
            for k in range(3):
                a, b = int(t[k]), int(t[(k + 1) % 3])
                ed.setdefault((min(a, b), max(a, b)), []).append(e)
        inner, bnd = [], []
        for (a, b), es in ed.items():
            L = float(np.linalg.norm(P[b] - P[a]))
            if len(es) == 2:
                inner.append((es[0], es[1], P[a], P[b]))
            else:
                bnd.append((es[0], tuple(P[a]), tuple(P[b]), self.kappa(es[0], L)))
        return inner, bnd

    def kappa(self, e, L):
        p = self.order
        return self.tau * (p + 1) * (p + 2) / 2.0 * L / self.area[e]

    def boundary_edges(self):
        return self.edges()[1]

    def centroids(self):
        return self.cen

    def locate(self, pts):
        if getattr(self, "_finder", None) is None:
            from matplotlib.tri import Triangulation
            P = np.asarray(self.nodes, float)
            self._finder = Triangulation(P[:, 0], P[:, 1], self.tris).get_trifinder()
        return np.asarray(self._finder(pts[:, 0], pts[:, 1]), int)

    def dof_points(self):
        return np.repeat(self.cen[:, 0], self.np_), np.repeat(self.cen[:, 1], self.np_)

    def project(self, g):
        """Projection L2 d'une fonction g(x, z) sur la base modale."""
        r, s, w = triangle_quadrature(self.order + 2)
        V, _, _ = dubiner(self.order, r, s)
        P = np.asarray(self.nodes, float)
        e2, e3 = P[self.tris[:, 1]] - self.v1, P[self.tris[:, 2]] - self.v1
        X = self.v1[:, None, :] + 0.5 * (r + 1)[None, :, None] * e2[:, None, :] + 0.5 * (s + 1)[None, :, None] * e3[:, None, :]
        G = g(X[..., 0], X[..., 1])
        return np.einsum("q,eq,qi->ei", w, G, V).ravel()


# --------------------------------------------------------------------------------------------
# Solveur
# --------------------------------------------------------------------------------------------

_SIDES = ("left", "right", "top", "bottom")


@dataclass
class Hybrid2D:
    """Assemblage des zones et du couplage SIPG ; mêmes méthodes que :class:`Mortar2D`
    (dt_max, run, point_matrix, energy). Les bords absorbants et l'éponge ne portent que sur les zones SEM."""

    zones: list
    boundary: dict = field(default_factory=lambda: {s: "absorb" for s in _SIDES})
    sponge: float = 0.0
    sponge_R: float = 1e-3
    sponge_sides: tuple | None = None
    tol: float = 1e-6

    def __post_init__(self):
        off = 0
        for z in self.zones:
            z.setup(off)
            off += z.ndof
        self.ndof = off
        self.n_elements = sum(z.n_elements for z in self.zones)
        P = np.concatenate([np.array([[z.x0, z.z0], [z.x1, z.z1]]) if z.kind == "quad" else
                            np.array([[z.x0, z.z0], [z.x_end, z.z_end]]) if z.kind == "grid" else
                            np.asarray(z.nodes) for z in self.zones])
        self.bbox = (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max())
        self.vmax = max(z.vmax for z in self.zones)
        rows, cols, vals = [], [], []
        M = np.zeros(self.ndof)
        self._mass_blocks = []                    # (dofs (m, n), M (m, n, n)) of the zones with full element masses
        for z in self.zones:
            d, Ke, me = z.volume()
            n = d.shape[1]
            rows.append(np.repeat(d, n, axis=1).ravel())
            cols.append(np.tile(d, (1, n)).ravel())
            vals.append(Ke.ravel())
            if me.ndim == 3:
                self._mass_blocks.append((d, me))
                M[d.ravel()] = np.einsum("eii->ei", me).ravel()        # diagonal, for reporting only
            else:
                M += np.bincount(d.ravel(), me.ravel(), minlength=self.ndof)
        self.M = M
        self._build_mass_operators()
        self.n_faces = dict(tri_tri=0, zone=0)
        self._faces(rows, cols, vals)
        self.K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(self.ndof, self.ndof)).tocsr()
        self.K.sum_duplicates()
        self._boundary()

    def _build_mass_operators(self):
        """Masse complète Mfull, son inverse Minv et L^-1 (M = L L^T), tous diagonaux par blocs."""
        diag = np.ones(self.ndof, bool)
        r, c, vm, vi, vl = [], [], [], [], []
        for d, Mb in self._mass_blocks:
            diag[d.ravel()] = False
            n = d.shape[1]
            L = np.linalg.cholesky(Mb)
            Li = np.linalg.inv(L)
            r.append(np.repeat(d, n, axis=1).ravel()); c.append(np.tile(d, (1, n)).ravel())
            vm.append(Mb.ravel()); vi.append(np.linalg.inv(Mb).ravel()); vl.append(Li.ravel())
        idx = np.where(diag)[0]
        r.append(idx); c.append(idx)
        vm.append(self.M[idx]); vi.append(1.0 / self.M[idx]); vl.append(1.0 / np.sqrt(self.M[idx]))
        rr, cc = np.concatenate(r), np.concatenate(c)
        mk = lambda v: sp.csr_matrix((np.concatenate(v), (rr, cc)), shape=(self.ndof, self.ndof))
        self.Mfull, self.Minv, self.Linv = mk(vm), mk(vi), mk(vl)
        self._diag_dofs = diag

    # ---------------------------------------------------------------- SIPG faces
    def _add_faces(self, zA, eA, kA, zB, eB, kB, p0, p1, nrm, rows, cols, vals):
        """Faces entre les éléments eA de zA et eB de zB, segments [p0, p1], normale nrm de A vers B."""
        q = max(zA.order, zB.order) + 1
        xq, wq = _leg.leggauss(q)
        pts = 0.5 * (p0 + p1)[:, None, :] + 0.5 * xq[None, :, None] * (p1 - p0)[:, None, :]
        L = np.linalg.norm(p1 - p0, axis=1)
        om = 0.5 * L[:, None] * wq[None, :]
        dA, VA, GxA, GzA, bA = zA.evaluate(eA, pts)
        dB, VB, GxB, GzB, bB = zB.evaluate(eB, pts)
        dnA = GxA * nrm[:, None, 0, None] + GzA * nrm[:, None, 1, None]
        dnB = GxB * nrm[:, None, 0, None] + GzB * nrm[:, None, 1, None]
        J = np.concatenate([VA, -VB], axis=2)
        F = 0.5 * np.concatenate([dnA * bA[:, :, None], dnB * bB[:, :, None]], axis=2)
        gam = np.maximum(kA * bA.max(1), kB * bB.max(1))
        JwF = np.einsum("sq,sqi,sqj->sij", om, J, F)
        JwJ = np.einsum("sq,sqi,sqj->sij", om, J, J)
        Ks = gam[:, None, None] * JwJ - JwF - np.transpose(JwF, (0, 2, 1))
        dofs = np.concatenate([dA, dB], axis=1)
        m = dofs.shape[1]
        rows.append(np.repeat(dofs, m, axis=1).ravel())
        cols.append(np.tile(dofs, (1, m)).ravel())
        vals.append(Ks.ravel())

    def _faces(self, rows, cols, vals):
        tol = self.tol
        # 1. interior faces of the triangle zones
        for z in self.zones:
            if z.kind != "tri":
                continue
            inner, _ = z.edges()
            if not inner:
                continue
            eA = np.array([f[0] for f in inner]); eB = np.array([f[1] for f in inner])
            p0 = np.array([f[2] for f in inner]); p1 = np.array([f[3] for f in inner])
            t = p1 - p0
            L = np.linalg.norm(t, axis=1)
            nrm = np.stack([t[:, 1], -t[:, 0]], 1) / L[:, None]
            flip = np.einsum("sd,sd->s", z.cen[eB] - z.cen[eA], nrm) < 0
            nrm[flip] *= -1
            kA = np.array([z.kappa(e, l) for e, l in zip(eA, L)])
            kB = np.array([z.kappa(e, l) for e, l in zip(eB, L)])
            self._add_faces(z, eA, kA, z, eB, kB, p0, p1, nrm, rows, cols, vals)
            self.n_faces["tri_tri"] += len(eA)
        # 2. zone boundaries: axis-aligned lines, edges of the two sides matched by intersection
        recs = []
        for iz, z in enumerate(self.zones):
            cen = z.centroids()
            for e, a, b, kap in z.boundary_edges():
                a, b = np.asarray(a, float), np.asarray(b, float)
                if abs(a[0] - b[0]) < tol:
                    axis, coord, t0, t1 = 0, a[0], min(a[1], b[1]), max(a[1], b[1])
                elif abs(a[1] - b[1]) < tol:
                    axis, coord, t0, t1 = 1, a[1], min(a[0], b[0]), max(a[0], b[0])
                else:
                    raise ValueError("bord de zone non aligné sur les axes")
                side = 0 if cen[e, axis] < coord else 1
                recs.append((axis, coord, side, (t0, t1, iz, e, kap, coord)))
        # group edges by line with a tolerance (mesher vertices may be rounded: no fixed-grid rounding key)
        eps = max(10 * tol, 1e-9 * max(abs(v) for v in self.bbox) + 1e-5)
        lines = {}
        for axis in (0, 1):
            sub = sorted((r for r in recs if r[0] == axis), key=lambda r: r[1])
            key, last = None, None
            for r in sub:
                if last is None or r[1] - last > eps:
                    key = (axis, r[1])
                last = r[1]
                lines.setdefault(key, ([], []))[r[2]].append(r[3])
        self.outer_tri_edges = 0
        for (axis, _), (minus, plus) in lines.items():
            if not minus or not plus:
                self.outer_tri_edges += sum(1 for r in minus + plus if self.zones[r[2]].kind == "tri")
                continue
            A = sorted(minus); B = sorted(plus)
            a0 = np.array([r[0] for r in A]); a1 = np.array([r[1] for r in A])
            b0 = np.array([r[0] for r in B]); b1 = np.array([r[1] for r in B])
            br = np.unique(np.concatenate([a0, a1, b0, b1]))
            br = br[np.concatenate([[True], np.diff(br) > tol])]
            mid = 0.5 * (br[:-1] + br[1:])
            ia = np.searchsorted(a0, mid, "right") - 1
            ib = np.searchsorted(b0, mid, "right") - 1
            ok = (ia >= 0) & (ib >= 0)
            ok &= (mid < a1[np.clip(ia, 0, None)]) & (mid < b1[np.clip(ib, 0, None)])
            coord = A[0][5]
            sel = np.where(ok)[0]
            if not len(sel):
                continue
            pairs = {}
            for k in sel:
                ra, rb = A[ia[k]], B[ib[k]]
                pairs.setdefault((ra[2], rb[2]), []).append((ra[3], ra[4], rb[3], rb[4], br[k], br[k + 1]))
            for (za, zb), lst in pairs.items():
                arr = np.array(lst)
                s0, s1 = arr[:, 4], arr[:, 5]
                if axis == 0:
                    p0 = np.stack([np.full(len(s0), coord), s0], 1); p1 = np.stack([np.full(len(s0), coord), s1], 1)
                    nrm = np.tile([1.0, 0.0], (len(s0), 1))
                else:
                    p0 = np.stack([s0, np.full(len(s0), coord)], 1); p1 = np.stack([s1, np.full(len(s0), coord)], 1)
                    nrm = np.tile([0.0, 1.0], (len(s0), 1))
                self._add_faces(self.zones[za], arr[:, 0].astype(int), arr[:, 1], self.zones[zb],
                                arr[:, 2].astype(int), arr[:, 3], p0, p1, nrm, rows, cols, vals)
                self.n_faces["zone"] += len(s0)

    # ---------------------------------------------------------------- outer boundary (SEM zones)
    def _boundary(self):
        x0, x1, z0, z1 = self.bbox
        tol = self.tol
        C = np.zeros(self.ndof)
        sig = np.zeros(self.ndof)
        for z in self.zones:
            px, pz = z.dof_points()
            if self.sponge > 0:
                smax = -3.0 * self.vmax * np.log(self.sponge_R) / (2 * self.sponge)
                dist = {"left": px - x0, "right": x1 - px, "top": pz - z0, "bottom": z1 - pz}
                for s, d in dist.items():
                    on = s in self.sponge_sides if self.sponge_sides is not None else True
                    if on and self.boundary.get(s, "absorb") == "absorb":
                        sig[z.off:z.off + z.ndof] += smax * np.clip(1 - d / self.sponge, 0, None) ** 2
            if z.kind == "grid":
                for name, d, h, imp in z.domain_faces(self.bbox, tol):
                    if self.boundary.get(name, "absorb") == "absorb":
                        C += np.bincount(d, (h / 2) * z.w / imp, minlength=self.ndof)
                continue
            if z.kind != "quad":
                continue
            N, n1 = z.order, z.order + 1
            d4 = z.elem_dofs.reshape(z.nx, z.nz, n1, n1)
            i4 = z.imp_en.reshape(z.nx, z.nz, n1, n1)
            sides = []
            if abs(z.x0 - x0) < tol:
                sides.append(("left", d4[0, :, 0, :], z.hz, i4[0, :, 0, :]))
            if abs(z.x1 - x1) < tol:
                sides.append(("right", d4[-1, :, N, :], z.hz, i4[-1, :, N, :]))
            if abs(z.z0 - z0) < tol:
                sides.append(("top", d4[:, 0, :, 0], z.hx, i4[:, 0, :, 0]))
            if abs(z.z1 - z1) < tol:
                sides.append(("bottom", d4[:, -1, :, N], z.hx, i4[:, -1, :, N]))
            for name, d, h, imp in sides:
                if self.boundary.get(name, "absorb") == "absorb":
                    C += np.bincount(d.ravel(), ((h / 2) * z.w[None, :] / imp).ravel(), minlength=self.ndof)
        self.C, self.sigma = C, sig

    # ---------------------------------------------------------------- utilities
    def point_matrix(self, pts):
        """Matrice creuse (npts, ndof) d'évaluation du champ aux points (x, z)."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        todo = np.ones(len(pts), bool)
        rows, cols, vals = [], [], []
        for z in self.zones:
            idx = np.where(todo)[0]
            if not len(idx):
                break
            e = z.locate(pts[idx])
            hit = e >= 0
            if not hit.any():
                continue
            k = idx[hit]
            dofs, V, _, _, _ = z.evaluate(e[hit], pts[k][:, None, :])
            rows.append(np.repeat(k, dofs.shape[1])); cols.append(dofs.ravel()); vals.append(V[:, 0, :].ravel())
            todo[k] = False
        if todo.any():
            raise ValueError(f"{todo.sum()} point(s) hors du domaine, par exemple {pts[todo][0]}")
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(pts), self.ndof))

    def raster(self, nx, nz):
        x0, x1, z0, z1 = self.bbox
        gx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        gz = z0 + (np.arange(nz) + 0.5) * (z1 - z0) / nz
        X, Z = np.meshgrid(gx, gz, indexing="ij")
        return self.point_matrix(np.stack([X.ravel(), Z.ravel()], 1)), (x0, x1, z1, z0)

    def project(self, g):
        """Vecteur des inconnues représentant la fonction g(x, z)."""
        return np.concatenate([z.project(g) for z in self.zones])

    def _scaled(self):
        return (self.Linv @ self.K @ self.Linv.T).tocsr()

    def max_eigenvalue(self, tol=1e-4):
        return float(spla.eigsh(self._scaled(), k=1, which="LA", tol=tol, return_eigenvectors=False)[0])

    def dt_max(self):
        return 2.0 / np.sqrt(self.max_eigenvalue())

    def zone_dt_max(self, iz):
        """Pas stable de la zone iz seule (sous-matrice de ses inconnues, interfaces comprises)."""
        z = self.zones[iz]
        idx = np.arange(z.off, z.off + z.ndof)
        A = self._scaled().tocsr()[idx][:, idx]
        return 2.0 / np.sqrt(float(spla.eigsh(A, k=1, which="LA", tol=1e-4, return_eigenvectors=False)[0]))

    def min_eigenvalue(self):
        A = self._scaled().toarray()
        return float(np.linalg.eigvalsh(0.5 * (A + A.T))[0])

    def energy(self, p_new, p_old, dt):
        v = (p_new - p_old) / dt
        return 0.5 * v @ (self.Mfull @ v) + 0.5 * p_new @ (self.K @ p_old)

    def run(self, dt, nt, source=None, wavelet=None, receivers=None, snapshots=(), raster=None, p0=None,
            track_energy=False):
        """Saute-mouton (comme :meth:`Mortar2D.run`)."""
        M, C, K = self.M, self.C, self.K
        a_plus, a_minus = M + 0.5 * dt * C, M - 0.5 * dt * C
        ic = np.where(C > 0)[0]
        has_c = len(ic) > 0
        taper = np.exp(-self.sigma * dt)
        damp = bool(np.any(self.sigma))
        p = np.zeros(self.ndof) if p0 is None else p0[1].copy()
        p_old = np.zeros(self.ndof) if p0 is None else p0[0].copy()
        fvec = None if source is None else np.asarray(self.point_matrix([source]).todense()).ravel()
        R = self.point_matrix(receivers) if receivers else None
        traces = np.zeros((nt, len(receivers))) if receivers else None
        snaps, energy, snapset = {}, [], set(snapshots)
        for n in range(nt):
            r = K @ p
            if fvec is not None:
                r -= wavelet[n] * fvec
            p_new = 2 * p - p_old - dt * dt * (self.Minv @ r)
            if has_c:                                  # absorbing boundary (diagonal SEM dofs only)
                p_new[ic] = (2 * M[ic] * p[ic] - a_minus[ic] * p_old[ic] - dt * dt * r[ic]) / a_plus[ic]
            if damp:
                p_new *= taper
                p *= taper
            if track_energy:
                energy.append(self.energy(p_new, p, dt))
            p_old, p = p, p_new
            if R is not None:
                traces[n] = R @ p
            if n in snapset and raster is not None:
                snaps[n] = raster[0] @ p
        return dict(traces=traces, snapshots=snaps, energy=np.array(energy), p=(p_old, p))


# --------------------------------------------------------------------------------------------
# Maillage d'une zone de triangles (gmsh) et figure
# --------------------------------------------------------------------------------------------


def triangulate_ring(outer, inner, curve, size, order=3):
    """Triangles dans le rectangle ``outer`` (x0, x1, z0, z1) privé du rectangle ``inner`` (ou None),
    avec la courbe fermée ``curve`` (k, 2) (polygone) imposée comme arêtes, un triangle par côté du polygone
    (à donner à la taille des triangles : le pas de temps dépend du plus petit triangle) ; size(x, z) :
    taille visée ; les bords des rectangles sont divisés uniformément.
    Retourne (nodes (n, 2), tris (m, 3))."""
    import gmsh
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ring")
    g = gmsh.model.geo

    division = []                                   # (line, number of nodes): uniform, sized at the middle

    def loop(pts, one_element=False):
        pts = np.asarray(pts, float)
        ids = [g.addPoint(float(x), float(z), 0.0) for x, z in pts]
        ls = [g.addLine(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
        for i, l in enumerate(ls):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            mid = 0.5 * (a + b)
            n = 1 if one_element else max(1, int(round(np.linalg.norm(b - a) / float(size(mid[0], mid[1])))))
            division.append((l, n + 1))
        return g.addCurveLoop(ls), ls

    x0, x1, z0, z1 = outer
    lo, _ = loop([(x0, z0), (x1, z0), (x1, z1), (x0, z1)])
    loops = [lo]
    if inner is not None:
        i0, i1, k0, k1 = inner
        li, _ = loop([(i0, k0), (i1, k0), (i1, k1), (i0, k1)])
        loops.append(li)
    surf = g.addPlaneSurface(loops)
    _, cl = loop(curve, one_element=True)           # the curve is given at the triangle size
    g.synchronize()
    for l, n in division:
        gmsh.model.mesh.setTransfiniteCurve(l, n)
    gmsh.model.mesh.embed(1, cl, 2, surf)
    gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: float(size(x, y)))
    for name in ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature", "Mesh.MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber(name, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    et, _, conn = gmsh.model.mesh.getElements(2)
    tri = np.asarray(conn[list(et).index(2)], int).reshape(-1, 3)
    remap = -np.ones(int(tags.max()) + 1, int)
    used = np.unique(tri)
    remap[used] = np.arange(len(used))
    xyz = np.asarray(xyz).reshape(-1, 3)
    pos = np.zeros((int(tags.max()) + 1, 2))
    pos[np.asarray(tags, int)] = xyz[:, :2]
    gmsh.model.remove()
    if own:
        gmsh.finalize()
    return pos[used], remap[tri]


def mesh_cmap():
    """viridis limitée à [0, 0,72] : les vitesses fortes (sel) restent en vert moyen, arêtes blanches lisibles."""
    import matplotlib
    from matplotlib.colors import ListedColormap
    return ListedColormap(matplotlib.colormaps["viridis"](np.linspace(0.0, 0.72, 256)), name="viridis_mesh")


def draw_hybrid(ax, zones, color_by="vp", cmap=None, vmin=None, vmax=None, lw=0.3):
    """Maillage hybride : zones SEM (grille) et triangles, colorés par la vitesse. Palette par défaut :
    viridis tronquée (sans le jaune clair) pour que les arêtes blanches restent visibles dans les corps rapides."""
    from matplotlib.collections import LineCollection, PolyCollection
    if cmap is None:
        cmap = mesh_cmap()
    polys, vals, segs, thick = [], [], [], []
    for z in zones:
        if z.kind == "quad":
            n1 = z.order + 1
            v = z.vp_n[z.elem_dofs - z.off].reshape(z.nx, z.nz, -1).mean(-1)
            xs = z.x0 + z.hx * np.arange(z.nx + 1); zs = z.z0 + z.hz * np.arange(z.nz + 1)
            for i in range(z.nx):
                for k in range(z.nz):
                    polys.append([(xs[i], zs[k]), (xs[i + 1], zs[k]), (xs[i + 1], zs[k + 1]), (xs[i], zs[k + 1])])
                    vals.append(v[i, k])
            thick += [[(z.x0, z.z0), (z.x1, z.z0)], [(z.x1, z.z0), (z.x1, z.z1)], [(z.x1, z.z1), (z.x0, z.z1)],
                      [(z.x0, z.z1), (z.x0, z.z0)]]
        elif z.kind == "grid":
            v = z.vp_n[z.elem_dofs - z.off].mean(-1)
            for (i, k), vv in zip(z.cells, v):
                xa, za, xb, zb = z.cell_box(i, k)
                polys.append([(xa, za), (xb, za), (xb, zb), (xa, zb)])
                vals.append(vv)
            thick += [list(z._side_geometry(e, sd)) for e, sd in z._free_sides()]
        else:
            P = np.asarray(z.nodes)
            for t, v in zip(z.tris, z.vp_e):
                polys.append(P[t]); vals.append(v)
    pc = PolyCollection(polys, array=np.array(vals), cmap=cmap, edgecolors="w", linewidths=lw)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.add_collection(LineCollection(thick, colors="k", linewidths=1.0))
    P = np.concatenate([np.array(p) for p in polys])
    ax.set_xlim(P[:, 0].min(), P[:, 0].max()); ax.set_ylim(P[:, 1].max(), P[:, 1].min())
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("profondeur (m)")
    return pc
