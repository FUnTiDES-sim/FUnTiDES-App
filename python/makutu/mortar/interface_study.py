"""Erreur due aux interfaces non conformes, à ordre et points par longueur d'onde fixés.

Règle de maillage : h = N v / (fmax ppw) dans chaque bloc ; de part et d'autre d'une interface entre deux
milieux de vitesses v1 et v2 = c v1, le rapport des tailles de mailles vaut donc c. Onde plane en
incidence normale (bande à bords latéraux rigides, ce qui garde l'onde plane exacte), interface plane
horizontale à z = zi :
- coefficients de réflexion et de transmission en pression mesurés au pic des impulsions, comparés aux
  valeurs exactes R = (Z2 - Z1) / (Z2 + Z1), T = 1 + R (Z = rho v) ;
- trois maillages : blocs (h1 au-dessus, h2 = c h1 au-dessous, interface non conforme), conforme
  (min(h1, h2) des deux côtés, bord d'élément sur l'interface), référence fine (min / 3 des deux côtés).
L'écart blocs - conforme isole ce qu'apporte la non-conformité de l'interface.
"""
from __future__ import annotations

import numpy as np

from .solver2d import Mortar2D, Patch, ricker


def plane_wave_case(c, order=4, ppw=5.0, f0=8.0, v1=2000.0, rho=1000.0, mesh="blocs", width=None,
                    zi=3000.0, depth=6000.0, z_src=1000.0, z_rec1=1800.0, dz_rec2=200.0, sponge=800.0, dt=None):
    """Une simulation ; retourne dict(t, inc_ref, p1, p2, R, T, ...) pour le contraste c = v2 / v1."""
    fmax = 2.5 * f0
    v2 = c * v1
    h1 = order * v1 / (fmax * ppw)
    h2 = order * v2 / (fmax * ppw)
    if mesh == "conforme":                                         # same as blocs, without non-conformity
        h1 = h2 = min(h1, h2)
    if mesh == "fine":
        h1 = h2 = order * min(v1, v2) / (fmax * ppw) / 3.0
    W = width if width is not None else 3.37 * max(h1, h2)          # non-commensurate widths
    nx1, nx2 = max(1, int(np.ceil(W / h1))), max(1, int(np.ceil(W / h2)))
    nz1, nz2 = int(np.ceil(zi / h1)), int(np.ceil((depth - zi) / h2))
    patches = [Patch(0.0, W, 0.0, zi, nx1, nz1, v1, rho), Patch(0.0, W, zi, depth, nx2, nz2, v2, rho)]
    m = Mortar2D(patches, order=order, boundary={"left": "rigid", "right": "rigid", "top": "absorb",
                                                 "bottom": "absorb"},
                 sponge=sponge, sponge_sides=("top", "bottom"))
    if dt is None:
        dt = 0.5 * m.dt_max()
    T_end = (zi - z_src + zi - z_rec1) / v1 + 3.0 / f0
    nt = int(T_end / dt)
    # downgoing plane Ricker pulse centred at z_src at t = 0: p(z, t) = g(z - v1 t), g(z) = Ricker((z - z_src) / v1)
    def g(z):
        return ricker(-(z - z_src) / v1, f0, t0=0.0)

    in1 = m.z < zi
    p0 = np.where(in1, g(m.z), 0.0)
    pm = np.where(in1, g(m.z + v1 * dt), 0.0)                       # p(z, -dt)
    out = m.run(dt, nt, receivers=[(0.5 * W, z_rec1), (0.5 * W, zi + dz_rec2)], p0=(pm, p0))
    tr = out["traces"]
    t = np.arange(1, nt + 1) * dt                                   # traces[n] = p at (n + 1) dt
    p1, p2 = tr[:, 0], tr[:, 1]
    # windows: incident at z_rec1, reflected at z_rec1 later, transmitted at zi + dz_rec2
    t_inc = (z_rec1 - z_src) / v1
    t_ref = (zi - z_src + zi - z_rec1) / v1
    t_tr = (zi - z_src) / v1 + dz_rec2 / v2
    half = 1.0 / f0

    def peak(p, tc):
        w = (t > tc - half) & (t < tc + half)
        k = np.argmax(np.abs(p[w]))
        return p[w][k], t[w][k]

    a_inc, ti = peak(p1, t_inc)
    a_ref, tr_ = peak(p1, t_ref)
    a_tr, tt = peak(p2, t_tr)
    Z1, Z2 = rho * v1, rho * v2
    R_ex = (Z2 - Z1) / (Z2 + Z1)
    return dict(c=c, mesh=mesh, h1=h1, h2=h2, ratio=h2 / h1, n_elements=m.n_elements, dt=dt, t=t, p1=p1, p2=p2,
                R=a_ref / a_inc, T=a_tr / a_inc, R_exact=R_ex, T_exact=1 + R_ex,
                dt_ref=tr_ - t_ref, dt_tr=tt - t_tr, dt_inc=ti - t_inc,
                windows=(t_inc, t_ref, t_tr, half))


def window_errors(res, ref):
    """Écarts L2 relatifs des impulsions réfléchie et transmise d'un calcul à la référence (même dt)."""
    t_inc, t_ref, t_tr, half = res["windows"]
    t = res["t"]
    n = min(len(t), len(ref["t"]))
    wr = (t[:n] > t_ref - half) & (t[:n] < t_ref + half)
    wt = (t[:n] > t_tr - half) & (t[:n] < t_tr + half)
    wi = (t[:n] > t_inc - half) & (t[:n] < t_inc + half)
    inc = np.linalg.norm(ref["p1"][:n][wi])
    return (np.linalg.norm((res["p1"][:n] - ref["p1"][:n])[wr]) / inc,
            np.linalg.norm((res["p2"][:n] - ref["p2"][:n])[wt]) / inc)
