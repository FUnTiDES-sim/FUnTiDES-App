"""Prototype NumPy/SciPy : éléments spectraux 2-D acoustiques sur blocs réguliers non conformes.

Le domaine est pavé de blocs rectangulaires alignés sur les axes. Chaque bloc porte une grille
régulière d'éléments (nx x nz, tailles hx x hz propres au bloc). À l'intérieur d'un bloc, la
discrétisation est le SEM continu classique (points GLL, masse diagonale). Les blocs ne partagent
aucun nœud : ils sont recollés par une pénalité intérieure symétrique (SIPG) intégrée sur les
segments d'interface, intersections des arêtes des deux côtés. Le rapport des tailles de part et
d'autre est quelconque (pas seulement 2:1).

Équation : (1 / (rho vp^2)) p_tt - div(1/rho grad p) = f. Schéma saute-mouton :
    M (p^{n+1} - 2 p^n + p^{n-1}) / dt^2 + C (p^{n+1} - p^{n-1}) / (2 dt) + K p^n = F^n,
M diagonale (GLL), C amortissement paraxial du premier ordre sur les bords absorbants, K symétrique
(SEM continu dans les blocs + termes SIPG sur les interfaces). Sans amortissement ni source,
l'énergie discrète E^{n+1/2} = 1/2 |dp|_M^2 / dt^2 + 1/2 p^{n+1}.K p^n est conservée.

Références :
- D. N. Arnold, An interior penalty finite element method with discontinuous elements,
  SIAM J. Numer. Anal. 19 (1982) ;
- P. F. Antonietti, I. Mazzieri, A. Quarteroni, F. Rapetti, Non-conforming high order approximations
  of the elastodynamics equation, CMAME 209-212 (2012) (DGSEM entre sous-domaines, SEM conforme
  à l'intérieur : code SPEED) ;
- M. J. Grote, A. Schneebeli, D. Schötzau, Discontinuous Galerkin finite element method for the
  wave equation, SIAM J. Numer. Anal. 44 (2006).

Prototype de validation (CPU, matrices creuses) avant un portage dans FUnTiDES (sem2d).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import legendre as _leg

# --------------------------------------------------------------------------------------------
# Bases 1-D
# --------------------------------------------------------------------------------------------


def gll(order):
    """Points et poids de Gauss-Lobatto-Legendre (order + 1 points sur [-1, 1])."""
    c = np.zeros(order + 1)
    c[order] = 1.0
    inner = np.sort(np.real(_leg.legroots(_leg.legder(c)))) if order > 1 else np.zeros(0)
    x = np.concatenate([[-1.0], inner, [1.0]])
    w = 2.0 / (order * (order + 1) * _leg.legval(x, c) ** 2)
    return x, w


def _bary_weights(xn):
    d = xn[:, None] - xn[None, :]
    np.fill_diagonal(d, 1.0)
    return 1.0 / d.prod(axis=1)


def lagrange(xn, xe):
    """Matrice (len(xe), len(xn)) des polynômes de Lagrange de nœuds xn évalués en xe."""
    xe = np.atleast_1d(np.asarray(xe, float))
    bw = _bary_weights(xn)
    diff = xe[..., None] - xn
    hit = np.abs(diff) < 1e-13
    with np.errstate(divide="ignore", invalid="ignore"):
        t = bw / diff
        L = t / t.sum(-1, keepdims=True)
    rows = hit.any(-1)
    L[rows] = hit[rows].astype(float)
    return L


def derivative(xn):
    """D[i, j] = l_j'(x_i)."""
    bw = _bary_weights(xn)
    d = xn[:, None] - xn[None, :]
    np.fill_diagonal(d, 1.0)
    D = (bw[None, :] / bw[:, None]) / d
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(1))
    return D


def ricker(t, f0, t0=None):
    t0 = 1.2 / f0 if t0 is None else t0
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


# --------------------------------------------------------------------------------------------
# Blocs
# --------------------------------------------------------------------------------------------


@dataclass
class Patch:
    """Bloc rectangulaire [x0, x1] x [z0, z1] maillé par nx x nz éléments identiques.

    vp, rho : valeurs par élément, tableaux (nx, nz) ou scalaires."""

    x0: float
    x1: float
    z0: float
    z1: float
    nx: int
    nz: int
    vp: np.ndarray | float = 1500.0
    rho: np.ndarray | float = 1000.0
    level: int = 0  # libre (classe de taille, niveau LTS...)

    def __post_init__(self):
        self.vp = np.broadcast_to(np.asarray(self.vp, float), (self.nx, self.nz)).copy()
        self.rho = np.broadcast_to(np.asarray(self.rho, float), (self.nx, self.nz)).copy()

    @property
    def hx(self):
        return (self.x1 - self.x0) / self.nx

    @property
    def hz(self):
        return (self.z1 - self.z0) / self.nz

    @property
    def n_elements(self):
        return self.nx * self.nz


def patch_grid(xb, zb, h, vp=1500.0, rho=1000.0):
    """Blocs du produit des coupures xb x zb ; h[i][j] taille cible du bloc (i, j) (nx = ceil(W/h))."""
    out = []
    h = np.broadcast_to(np.asarray(h, float), (len(xb) - 1, len(zb) - 1))
    for i in range(len(xb) - 1):
        for j in range(len(zb) - 1):
            W, H = xb[i + 1] - xb[i], zb[j + 1] - zb[j]
            nx, nz = max(1, int(np.ceil(W / h[i, j] - 1e-9))), max(1, int(np.ceil(H / h[i, j] - 1e-9)))
            out.append(Patch(xb[i], xb[i + 1], zb[j], zb[j + 1], nx, nz, vp, rho))
    return out


# --------------------------------------------------------------------------------------------
# Solveur
# --------------------------------------------------------------------------------------------

_SIDES = ("left", "right", "top", "bottom")  # x min, x max, z min, z max


@dataclass
class Mortar2D:
    """SEM 2-D acoustique sur blocs non conformes recollés par SIPG.

    boundary : dict côté du domaine -> "absorb" (paraxial) ou "rigid" (condition naturelle) ;
    alpha : facteur de pénalité, gamma = alpha (N + 1)^2 / h_n max(1/rho) (h_n : plus petite taille
    normale à l'interface des deux éléments). Défaut : 1,25 N / (2 (N + 1)), soit 1,25 fois la
    constante de trace GLL N (N + 1) / 2 ; la coercivité mesurée (plus petite valeur propre >= 0)
    demande alpha >= 0,22 / 0,32 / 0,37 aux ordres 2 / 4 / 6 (interface conforme, cas le pire),
    et toute pénalité au-delà réduit le pas de temps stable ;
    sponge : largeur (m) d'une couche éponge sur les côtés absorbants (0 : aucune), limitée aux côtés
    sponge_sides s'il est donné (par exemple sans "top" pour tirer et enregistrer près de la surface) ;
    assemble : False pour ne construire que la numérotation et les matrices d'interface (calcul fait par
    FUnTiDES, :mod:`makutu.mortar.funtides2d`) ;
    nodal : fonction (x, z) -> (vp, rho) ; si elle est donnée, les propriétés sont prises aux nœuds GLL
    (masse, raideur, flux et bords), comme le mode IS_MODEL_ON_NODES de FUnTiDES ; sinon les valeurs
    par élément des blocs ; nodal_inset : les nœuds des bords de blocs où le modèle saute (écart relatif
    > inset_jump entre les points à cette distance de part et d'autre du bord) lisent le modèle à cette
    distance à l'intérieur de leur bloc (interface nette sur un bord de bloc vue d'un seul côté) ; ailleurs
    ils le lisent à leur position (une lecture décalée y déplacerait localement le modèle) ;"""

    patches: list
    order: int = 4
    alpha: float | None = None
    boundary: dict = field(default_factory=lambda: {s: "absorb" for s in _SIDES})
    sponge: float = 0.0
    sponge_R: float = 1e-3
    sponge_sides: tuple | None = None
    tol: float = 1e-6
    nodal: object = None
    nodal_inset: float = 0.0
    inset_jump: float = 0.1
    assemble: bool = True

    def __post_init__(self):
        N = self.order
        if self.alpha is None:
            self.alpha = 1.25 * N / (2.0 * (N + 1))
        self.xi, self.w = gll(N)
        self.D = derivative(self.xi)
        self.xq, self.wq = _leg.leggauss(N + 1)
        p = self.patches
        self.bbox = (min(q.x0 for q in p), max(q.x1 for q in p), min(q.z0 for q in p), max(q.z1 for q in p))
        self._number()
        self._assemble_interfaces()
        self._assemble_boundary()
        if self.assemble:                         # False: numbering and interface matrices only (FUnTiDES)
            self._assemble_volume()
            self.K = (self._Kvol + self._Kint).tocsr()
            self.K.sum_duplicates()

    # ---------------------------------------------------------------- numbering
    def _number(self):
        N = self.order
        self.offsets, off = [], 0
        self.elem_dofs = []
        xs, zs = [], []
        for q in self.patches:
            nxn, nzn = q.nx * N + 1, q.nz * N + 1
            self.offsets.append(off)
            ex, ez, a, b = np.meshgrid(np.arange(q.nx), np.arange(q.nz), np.arange(N + 1), np.arange(N + 1),
                                       indexing="ij")
            I, J = ex * N + a, ez * N + b
            self.elem_dofs.append((off + I * nzn + J).reshape(q.nx, q.nz, (N + 1) ** 2))
            ux = np.concatenate([q.x0 + q.hx * (e + (self.xi[:-1] + 1) / 2) for e in range(q.nx)] + [[q.x1]])
            uz = np.concatenate([q.z0 + q.hz * (e + (self.xi[:-1] + 1) / 2) for e in range(q.nz)] + [[q.z1]])
            X, Z = np.meshgrid(ux, uz, indexing="ij")
            xs.append(X.ravel())
            zs.append(Z.ravel())
            off += nxn * nzn
        self.ndof = off
        self.x, self.z = np.concatenate(xs), np.concatenate(zs)
        self.n_elements = sum(q.n_elements for q in self.patches)
        # coefficients at the element nodes, (nx, nz, n) per patch: b = 1/rho, c = 1/(rho vp^2), imp = rho vp
        self.coef = []
        # points where the model is read: nodes on a block edge are moved nodal_inset into their own
        # block, so that a sharp interface lying on a block boundary is seen one-sided on each side
        self.x_model, self.z_model = self.x.copy(), self.z.copy()
        if self.nodal_inset > 0 and self.nodal is not None:
            # only where the model jumps across the block edge (relative jump > inset_jump between the
            # points inset inside and outside): elsewhere a shifted read would move the model locally
            e = self.nodal_inset
            for q, off in zip(self.patches, self.offsets):
                nn = (q.nx * self.order + 1) * (q.nz * self.order + 1)
                xs, zs = self.x_model[off:off + nn], self.z_model[off:off + nn]
                x, z = self.x[off:off + nn], self.z[off:off + nn]
                t = self.tol
                dx_in = np.where(np.abs(x - q.x0) < t, e, np.where(np.abs(x - q.x1) < t, -e, 0.0))
                dz_in = np.where(np.abs(z - q.z0) < t, e, np.where(np.abs(z - q.z1) < t, -e, 0.0))
                edge = (dx_in != 0) | (dz_in != 0)
                if not edge.any():
                    continue
                xi, zi = x[edge] + dx_in[edge], z[edge] + dz_in[edge]
                xo, zo = x[edge] - dx_in[edge], z[edge] - dz_in[edge]
                bbox = self.bbox
                xo = np.clip(xo, bbox[0], bbox[1]); zo = np.clip(zo, bbox[2], bbox[3])
                vin = np.asarray(self.nodal(xi, zi)[0], float)
                vout = np.asarray(self.nodal(xo, zo)[0], float)
                sharp = np.abs(vin - vout) / np.minimum(vin, vout) > self.inset_jump
                idx = np.where(edge)[0][sharp]
                xs[idx], zs[idx] = xi[sharp], zi[sharp]
        if self.nodal is not None:
            vp_n, rho_n = (np.asarray(a, float) for a in self.nodal(self.x_model, self.z_model))
            self.vp_nodes = vp_n
        for q, d in zip(self.patches, self.elem_dofs):
            if self.nodal is not None:
                vp, rho = vp_n[d], rho_n[d]
            else:
                vp = np.broadcast_to(q.vp[:, :, None], d.shape)
                rho = np.broadcast_to(q.rho[:, :, None], d.shape)
            self.coef.append(dict(b=1.0 / rho, c=1.0 / (rho * vp ** 2), imp=rho * vp, vp=vp))
        self.vmax = max(float(c["vp"].max()) for c in self.coef)

    # ---------------------------------------------------------------- volume (CG per patch)
    def _assemble_volume(self):
        N, w, D = self.order, self.w, self.D
        n1 = N + 1
        ww = np.outer(w, w).ravel()
        I = np.eye(n1)
        rows, cols, vals, mass = [], [], [], np.zeros(self.ndof)
        n = n1 ** 2
        for q, dofs, cf in zip(self.patches, self.elem_dofs, self.coef):
            d = dofs.reshape(-1, n)
            B = cf["b"].reshape(-1, n1, n1) * w[None, :, None] * w[None, None, :]     # w_a w_b / rho_ab
            # int 1/rho du/dx dv/dx (GLL quadrature) : delta_bb' (hz/hx) sum_c D[c,a] B[c,b] D[c,a']
            Kx = (q.hz / q.hx) * np.einsum("ca,ecb,cd,bf->eabdf", D, B, D, I)
            Kz = (q.hx / q.hz) * np.einsum("cb,eac,cd,af->eabfd", D, B, D, I)
            Ke = (Kx + Kz).reshape(-1, n, n)
            rows.append(np.repeat(d, n, axis=1).ravel())
            cols.append(np.tile(d, (1, n)).ravel())
            vals.append(Ke.ravel())
            me = (q.hx * q.hz / 4) * ww[None, :] * cf["c"].reshape(-1, n)
            mass += np.bincount(d.ravel(), me.ravel(), minlength=self.ndof)
        self._Kvol = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(self.ndof, self.ndof))
        self.M = mass

    # ---------------------------------------------------------------- interfaces (SIPG)
    def _face_rows(self, q, normal_axis, far, eta):
        """Lignes trace T et dérivée normale G (selon +axe) sur une face d'élément d'un bloc,
        face far (True : côté max de l'axe normal), points de référence tangentiels eta (S, Q)."""
        N = self.order
        Lt = lagrange(self.xi, eta)                      # (S, Q, N+1)
        S, Q = eta.shape
        f = N if far else 0
        T = np.zeros((S, Q, N + 1, N + 1))
        if normal_axis == 0:
            T[:, :, f, :] = Lt
            G = (2.0 / q.hx) * self.D[f][None, None, :, None] * Lt[:, :, None, :]
        else:
            T[:, :, :, f] = Lt
            G = (2.0 / q.hz) * Lt[:, :, :, None] * self.D[f][None, None, None, :]
        return T.reshape(S, Q, -1), G.reshape(S, Q, -1)

    def _assemble_interfaces(self):
        N, tol = self.order, self.tol
        rows, cols, vals = [], [], []
        self.interfaces = []
        self._segments = []                       # (dofs, J, F, om, gamma) per interface, see interface_factors
        self._faces = []                          # structured form per interface, see interface_faces
        P = self.patches
        for ia, A in enumerate(P):
            for ib, B in enumerate(P):
                if ia == ib:
                    continue
                if abs(A.x1 - B.x0) < tol:                       # A left of B, normal +x
                    axis, lo, hi = 0, max(A.z0, B.z0), min(A.z1, B.z1)
                elif abs(A.z1 - B.z0) < tol:                     # A above B (z grows), normal +z
                    axis, lo, hi = 1, max(A.x0, B.x0), min(A.x1, B.x1)
                else:
                    continue
                if hi - lo <= tol:
                    continue
                # tangential element breakpoints of both sides
                if axis == 0:
                    tA = A.z0 + A.hz * np.arange(A.nz + 1)
                    tB = B.z0 + B.hz * np.arange(B.nz + 1)
                    hA_t, hB_t, a0, b0 = A.hz, B.hz, A.z0, B.z0
                else:
                    tA = A.x0 + A.hx * np.arange(A.nx + 1)
                    tB = B.x0 + B.hx * np.arange(B.nx + 1)
                    hA_t, hB_t, a0, b0 = A.hx, B.hx, A.x0, B.x0
                br = np.concatenate([[lo, hi], tA, tB])
                br = np.sort(br[(br >= lo - tol) & (br <= hi + tol)])
                br = br[np.concatenate([[True], np.diff(br) > tol])]
                s0, s1 = br[:-1], br[1:]
                mid, L = 0.5 * (s0 + s1), s1 - s0
                kA = np.clip(((mid - a0) // hA_t).astype(int), 0, (A.nz if axis == 0 else A.nx) - 1)
                kB = np.clip(((mid - b0) // hB_t).astype(int), 0, (B.nz if axis == 0 else B.nx) - 1)
                sq = mid[:, None] + 0.5 * L[:, None] * self.xq[None, :]          # (S, Q) physical
                etaA = 2 * (sq - (a0 + kA[:, None] * hA_t)) / hA_t - 1
                etaB = 2 * (sq - (b0 + kB[:, None] * hB_t)) / hB_t - 1
                TA, GA = self._face_rows(A, axis, True, etaA)
                TB, GB = self._face_rows(B, axis, False, etaB)
                if axis == 0:
                    eA = (np.full_like(kA, A.nx - 1), kA)
                    eB = (np.zeros_like(kB), kB)
                    hnA, hnB = A.hx, B.hx
                else:
                    eA = (kA, np.full_like(kA, A.nz - 1))
                    eB = (kB, np.zeros_like(kB))
                    hnA, hnB = A.hz, B.hz
                # 1/rho at the quadrature points, interpolated from each side's face nodes
                irA = np.einsum("sqi,si->sq", TA, self.coef[ia]["b"][eA])
                irB = np.einsum("sqi,si->sq", TB, self.coef[ib]["b"][eB])
                J = np.concatenate([TA, -TB], axis=2)                           # (S, Q, 2n)
                F = 0.5 * np.concatenate([GA * irA[:, :, None], GB * irB[:, :, None]], axis=2)
                om = 0.5 * L[:, None] * self.wq[None, :]                        # (S, Q)
                gamma = self.alpha * (N + 1) ** 2 / min(hnA, hnB) * np.maximum(irA, irB).max(1)
                JwF = np.einsum("sq,sqi,sqj->sij", om, J, F)
                JwJ = np.einsum("sq,sqi,sqj->sij", om, J, J)
                Ks = -JwF - np.transpose(JwF, (0, 2, 1)) + gamma[:, None, None] * JwJ
                dofs = np.concatenate([self.elem_dofs[ia][eA], self.elem_dofs[ib][eB]], axis=1)  # (S, 2n)
                m = dofs.shape[1]
                rows.append(np.repeat(dofs, m, axis=1).ravel())
                cols.append(np.tile(dofs, (1, m)).ravel())
                vals.append(Ks.ravel())
                self._segments.append((dofs, J, F, om, gamma))
                self._faces.append(dict(axis=axis, dA=self.elem_dofs[ia][eA], dB=self.elem_dofs[ib][eB],
                                        etaA=etaA, etaB=etaB, irA=irA, irB=irB, om=om, gamma=gamma,
                                        hnA=hnA, hnB=hnB))
                self.interfaces.append(dict(a=ia, b=ib, axis=axis, lo=lo, hi=hi, n_segments=len(L),
                                            ratio=max(hnA, hnB) / min(hnA, hnB),
                                            ratio_t=max(hA_t, hB_t) / min(hA_t, hB_t)))
        if rows:
            self._Kint = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                       shape=(self.ndof, self.ndof))
        else:
            self._Kint = sp.coo_matrix((self.ndof, self.ndof))

    # ---------------------------------------------------------------- outer boundary
    def _assemble_boundary(self):
        N, w, tol = self.order, self.w, self.tol
        x0, x1, z0, z1 = self.bbox
        C = np.zeros(self.ndof)
        for q, dofs, cf in zip(self.patches, self.elem_dofs, self.coef):
            d4 = dofs.reshape(q.nx, q.nz, N + 1, N + 1)
            z4 = cf["imp"].reshape(q.nx, q.nz, N + 1, N + 1)
            sides = []
            if abs(q.x0 - x0) < tol:
                sides.append(("left", d4[0, :, 0, :], q.hz, z4[0, :, 0, :]))
            if abs(q.x1 - x1) < tol:
                sides.append(("right", d4[-1, :, N, :], q.hz, z4[-1, :, N, :]))
            if abs(q.z0 - z0) < tol:
                sides.append(("top", d4[:, 0, :, 0], q.hx, z4[:, 0, :, 0]))
            if abs(q.z1 - z1) < tol:
                sides.append(("bottom", d4[:, -1, :, N], q.hx, z4[:, -1, :, N]))
            for name, d, h, z in sides:
                if self.boundary.get(name, "absorb") != "absorb":
                    continue
                C += np.bincount(d.ravel(), ((h / 2) * w[None, :] / z).ravel(), minlength=self.ndof)
        self.C = C
        self.sigma = np.zeros(self.ndof)
        if self.sponge > 0:
            L = self.sponge
            vmax = self.vmax
            smax = -3.0 * vmax * np.log(self.sponge_R) / (2 * L)
            dist = {"left": self.x - x0, "right": x1 - self.x, "top": self.z - z0, "bottom": z1 - self.z}
            for s, d in dist.items():
                on = s in self.sponge_sides if self.sponge_sides is not None else True
                if on and self.boundary.get(s, "absorb") == "absorb":
                    self.sigma += smax * np.clip(1 - d / L, 0, None) ** 2

    # ---------------------------------------------------------------- utilities
    def locate(self, x, z):
        """(bloc, ex, ez, xi, eta) du point (x, z) ; ValueError s'il est hors domaine."""
        for ip, q in enumerate(self.patches):
            if q.x0 - self.tol <= x <= q.x1 + self.tol and q.z0 - self.tol <= z <= q.z1 + self.tol:
                ex = int(np.clip((x - q.x0) // q.hx, 0, q.nx - 1))
                ez = int(np.clip((z - q.z0) // q.hz, 0, q.nz - 1))
                xi = 2 * (x - q.x0 - ex * q.hx) / q.hx - 1
                eta = 2 * (z - q.z0 - ez * q.hz) / q.hz - 1
                return ip, ex, ez, xi, eta
        raise ValueError(f"point ({x}, {z}) hors du domaine")

    def point_matrix(self, pts):
        """Matrice creuse (npts, ndof) d'interpolation aux points (x, z) (premier bloc contenant le point)."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        npt, t = len(pts), self.tol
        todo = np.ones(npt, bool)
        rows, cols, vals = [], [], []
        for ip, q in enumerate(self.patches):
            x, z = pts[:, 0], pts[:, 1]
            k = np.where(todo & (x >= q.x0 - t) & (x <= q.x1 + t) & (z >= q.z0 - t) & (z <= q.z1 + t))[0]
            if not len(k):
                continue
            todo[k] = False
            ex = np.clip(((x[k] - q.x0) // q.hx).astype(int), 0, q.nx - 1)
            ez = np.clip(((z[k] - q.z0) // q.hz).astype(int), 0, q.nz - 1)
            xi = 2 * (x[k] - q.x0 - ex * q.hx) / q.hx - 1
            eta = 2 * (z[k] - q.z0 - ez * q.hz) / q.hz - 1
            wgt = (lagrange(self.xi, xi)[:, :, None] * lagrange(self.xi, eta)[:, None, :]).reshape(len(k), -1)
            rows.append(np.repeat(k, wgt.shape[1]))
            cols.append(self.elem_dofs[ip][ex, ez].ravel())
            vals.append(wgt.ravel())
        if todo.any():
            raise ValueError(f"{todo.sum()} point(s) hors du domaine, par exemple {pts[todo][0]}")
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(npt, self.ndof))

    def raster(self, nx, nz):
        """Matrice d'échantillonnage sur une grille régulière nx x nz (centres de cellules) et étendue."""
        x0, x1, z0, z1 = self.bbox
        gx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        gz = z0 + (np.arange(nz) + 0.5) * (z1 - z0) / nz
        X, Z = np.meshgrid(gx, gz, indexing="ij")
        return self.point_matrix(np.stack([X.ravel(), Z.ravel()], 1)), (x0, x1, z1, z0)

    def max_eigenvalue(self, tol=1e-4):
        """Plus grande valeur propre de M^-1 K (borne de stabilité dt < 2 / sqrt(lambda))."""
        s = 1.0 / np.sqrt(self.M)
        A = sp.diags(s) @ self.K @ sp.diags(s)
        return float(spla.eigsh(A, k=1, which="LA", tol=tol, return_eigenvectors=False)[0])

    def dt_max(self):
        return 2.0 / np.sqrt(self.max_eigenvalue())

    def min_eigenvalue(self):
        """Plus petite valeur propre de M^-1 K (dense : petits maillages seulement)."""
        s = 1.0 / np.sqrt(self.M)
        A = (sp.diags(s) @ self.K @ sp.diags(s)).toarray()
        return float(np.linalg.eigvalsh(0.5 * (A + A.T))[0])

    def interface_factors(self):
        """Facteurs SIPG des segments d'interface, dans la numérotation des nœuds de ce solveur :
        dofs (n_seg, m), P et G (n_seg, n_q, m), gamma (n_seg,), m = 2 (N + 1)^2, n_q = N + 1, avec
        K_s u = sum_q gamma P_q (P_q . u) - P_q (G_q . u) - G_q (P_q . u) ; P_q : saut de trace, G_q :
        moyenne du flux 1/rho dp/dn, prémultipliés par sqrt(poids de Gauss x longueur / 2)
        (pour FUnTiDES : Solver.set_interface_operator)."""
        m, nq = 2 * (self.order + 1) ** 2, self.order + 1
        if not self._segments:
            return (np.zeros((0, m), np.int64), np.zeros((0, nq, m)), np.zeros((0, nq, m)), np.zeros(0))
        dofs = np.concatenate([d for d, *_ in self._segments])
        P = np.concatenate([np.sqrt(om)[:, :, None] * J for _, J, _, om, _ in self._segments])
        G = np.concatenate([np.sqrt(om)[:, :, None] * F for _, _, F, om, _ in self._segments])
        gamma = np.concatenate([g for *_, g in self._segments])
        return dofs, P, G, gamma

    def interface_faces(self):
        """Opérateur d'interface sous forme structurée (blocs d'éléments rectangulaires), pour
        FUnTiDES ``Solver.set_interface_operator`` : dict de tableaux face (S, 2, n1), line (S, 2, n1, n1),
        dn (S, 2, n1), lag (S, 2, Q, n1), half_inv_rho (S, 2, Q), weight (S, Q), gamma (S,), n1 = N + 1.
        Côté 0 : bloc que quitte la normale unifiée (face max de l'axe normal), côté 1 : l'autre (face
        min). line[s, c, b, :] : nœuds de la ligne d'élément normale à la face passant par le nœud de face b ;
        dn : coefficients de la dérivée selon la normale unifiée à la face sur cette ligne."""
        N, n1 = self.order, self.order + 1
        out = {k: [] for k in ("face", "line", "dn", "lag", "half_inv_rho", "weight", "gamma")}
        for f in self._faces:
            S = len(f["gamma"])
            faces, lines, dns, lags, hirs = [], [], [], [], []
            for side, (d, eta, ir, hn) in enumerate(((f["dA"], f["etaA"], f["irA"], f["hnA"]),
                                                     (f["dB"], f["etaB"], f["irB"], f["hnB"]))):
                fi = N if side == 0 else 0
                d4 = d.reshape(S, n1, n1)                         # [a (x), b (z)]
                if f["axis"] == 0:                                # normal along x: face nodes (fi, b), lines (:, b)
                    faces.append(d4[:, fi, :])
                    lines.append(np.transpose(d4, (0, 2, 1)))     # [b, a]
                else:                                             # normal along z: face nodes (a, fi), lines (a, :)
                    faces.append(d4[:, :, fi])
                    lines.append(d4)                              # [a, c]
                dns.append(np.broadcast_to((2.0 / hn) * self.D[fi], (S, n1)))
                lags.append(lagrange(self.xi, eta))               # (S, Q, n1)
                hirs.append(0.5 * ir)
            out["face"].append(np.stack(faces, 1))
            out["line"].append(np.stack(lines, 1))
            out["dn"].append(np.stack(dns, 1))
            out["lag"].append(np.stack(lags, 1))
            out["half_inv_rho"].append(np.stack(hirs, 1))
            out["weight"].append(f["om"])
            out["gamma"].append(f["gamma"])
        if not self._faces:
            return None
        return {k: np.concatenate(v) for k, v in out.items()}

    def apply_interface_faces(self, u):
        """K_interface u calculé à partir de :meth:`interface_faces` (même formule que FUnTiDES)."""
        F = self.interface_faces()
        y = np.zeros_like(u)
        if F is None:
            return y
        t = u[F["face"]]                                          # (S, 2, n1)
        d = np.einsum("sca,scba->scb", F["dn"], u[F["line"]])     # (S, 2, n1)
        tr = np.einsum("scqb,scb->scq", F["lag"], t)
        dr = np.einsum("scqb,scb->scq", F["lag"], d)
        jump = tr[:, 0] - tr[:, 1]                                # (S, Q)
        flux = (F["half_inv_rho"] * dr).sum(1)
        c1 = F["weight"] * (F["gamma"][:, None] * jump - flux)
        c2 = -F["weight"] * jump
        sf = np.einsum("scqb,sq->scb", F["lag"], c1) * np.array([1.0, -1.0])[None, :, None]
        el = np.einsum("scqb,scq,sq->scb", F["lag"], F["half_inv_rho"], c2)
        np.add.at(y, F["face"].ravel(), sf.ravel())
        np.add.at(y, F["line"].ravel(), (el[:, :, :, None] * F["dn"][:, :, None, :]).ravel())
        return y

    def interface_operator(self):
        """Matrices SIPG denses des segments : (dofs (n_seg, m), K (n_seg, m, m))."""
        dofs, P, G, gamma = self.interface_factors()
        PP = np.einsum("sqi,sqj->sij", P, P)
        PG = np.einsum("sqi,sqj->sij", P, G)
        return dofs, gamma[:, None, None] * PP - PG - np.transpose(PG, (0, 2, 1))

    def energy(self, p_new, p_old, dt):
        v = (p_new - p_old) / dt
        return 0.5 * v @ (self.M * v) + 0.5 * p_new @ (self.K @ p_old)

    # ---------------------------------------------------------------- time loop
    def run(self, dt, nt, source=None, wavelet=None, receivers=None, snapshots=(), raster=None,
            p0=None, track_energy=False):
        """Saute-mouton. source (x, z), wavelet (nt,) ; receivers : liste de (x, z) ;
        snapshots : indices de pas ; raster : (S, extent) de :meth:`raster`. Retourne un dict."""
        M, C, K = self.M, self.C, self.K
        a_plus = M + 0.5 * dt * C
        a_minus = M - 0.5 * dt * C
        taper = np.exp(-self.sigma * dt)
        p = np.zeros(self.ndof) if p0 is None else p0[1].copy()
        p_old = np.zeros(self.ndof) if p0 is None else p0[0].copy()
        fvec = None
        if source is not None:
            fvec = np.asarray(self.point_matrix([source]).todense()).ravel()
        R = self.point_matrix(receivers) if receivers else None
        traces = np.zeros((nt, len(receivers))) if receivers else None
        snaps, energy = {}, []
        snapset = set(snapshots)
        for n in range(nt):
            r = K @ p
            if fvec is not None:
                r -= wavelet[n] * fvec
            p_new = (2 * M * p - a_minus * p_old - dt * dt * r) / a_plus
            if self.sponge > 0:
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
