"""Tests du prototype hybride makutu.mortar.hybrid2d : SEM continu sur grilles régulières (ordre propre à
chaque zone) + triangles en Galerkin discontinu, recollés par SIPG. NumPy / SciPy / gmsh, sans GPU.
Chaque test enregistre une figure de contrôle."""
import numpy as np
import pytest

pytest.importorskip("gmsh")

from makutu.mortar import ricker  # noqa: E402
from makutu.mortar.hybrid2d import (Hybrid2D, QuadZone, TriZone, draw_hybrid, dubiner,  # noqa: E402
                                    triangle_quadrature, triangulate_ring)

RIGID = {s: "rigid" for s in ("left", "right", "top", "bottom")}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
    _plt().close(fig)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_dubiner_basis_is_orthonormal(order):
    r, s, w = triangle_quadrature(order + 2)
    V, Vr, Vs = dubiner(order, r, s)
    assert np.abs(np.einsum("q,qi,qj->ij", w, V, V) - np.eye(V.shape[1])).max() < 1e-12
    e = 1e-6
    assert np.abs((dubiner(order, r + e, s)[0] - dubiner(order, r - e, s)[0]) / (2 * e) - Vr).max() < 1e-6
    assert np.abs((dubiner(order, r, s + e)[0] - dubiner(order, r, s - e)[0]) / (2 * e) - Vs).max() < 1e-6


def _small(v_in=3000.0, rho_in=1500.0, boundary=RIGID, h=100.0):
    """Domaine 1200 x 1000 : 4 zones SEM d'ordres 4, 4, 4, 3 autour d'un anneau de triangles (ordre 3) qui
    suit un cercle de rayon 220 m, zone SEM d'ordre 5 au centre."""
    nc = int(round(2 * np.pi * 220 / h))
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    circ = np.stack([600 + 220 * np.cos(th), 500 + 220 * np.sin(th)], 1)
    nodes, tris = triangulate_ring((300, 900, 200, 800), (500, 700, 400, 600), circ, lambda x, z: h)
    cen = nodes[tris].mean(1)
    inside = np.hypot(cen[:, 0] - 600, cen[:, 1] - 500) < 220
    bg, inn = (2000.0, 1000.0), (v_in, rho_in)
    zones = [QuadZone(0, 1200, 0, 200, 10, 2, 4, bg), QuadZone(0, 1200, 800, 1000, 13, 2, 4, bg),
             QuadZone(0, 300, 200, 800, 3, 5, 4, bg), QuadZone(900, 1200, 200, 800, 4, 7, 3, bg),
             QuadZone(500, 700, 400, 600, 2, 2, 5, inn),
             TriZone(nodes, tris, vp=np.where(inside, v_in, 2000.0), rho=np.where(inside, rho_in, 1000.0))]
    return Hybrid2D(zones, boundary=boundary)


def test_hybrid_operator_is_symmetric_coercive_and_conserves_energy(figure_path):
    """SEM d'ordres 3, 4 et 5 et triangles d'ordre 3, contraste de vitesse et de densité sur le cercle :
    K symétrique semi-définie positive, énergie discrète conservée."""
    m = _small()
    assert m.n_faces["tri_tri"] > 0 and m.n_faces["zone"] > 0 and m.outer_tri_edges == 0
    assert abs(m.K - m.K.T).max() < 1e-12 * abs(m.K).max()
    assert m.min_eigenvalue() > -1e-9 * m.max_eigenvalue()
    p0 = m.project(lambda x, z: np.exp(-((x - 250) ** 2 + (z - 500) ** 2) / (2 * 60.0 ** 2)))
    dt = 0.9 * m.dt_max()
    E = m.run(dt, 3000, p0=(p0, p0), track_energy=True)["energy"]
    drift = np.abs(E / E[0] - 1)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    pc = draw_hybrid(ax[0], m.zones, lw=0.4)
    fig.colorbar(pc, ax=ax[0], label="vp (m/s)", shrink=0.8)
    ax[0].set_title(f"{m.n_elements} éléments dont {m.zones[-1].n_elements} triangles ; SEM d'ordres 4, 4, 4, 3 "
                    "et 5 (centre)", fontsize=9)
    ax[1].semilogy(np.arange(len(E)) * dt, drift + 1e-17)
    ax[1].set_xlabel("temps (s)"); ax[1].set_ylabel("|E(t) / E(0) - 1|")
    ax[1].set_title(f"énergie discrète, bords rigides (max {drift.max():.1e})", fontsize=9)
    fig.tight_layout()
    _save(fig, figure_path())
    assert drift.max() < 1e-10


def test_triangle_zone_time_step_stays_close_to_sem(figure_path):
    """Pas stable de chaque zone seule : les triangles (ordre 3, arêtes de la taille des mailles SEM) restent
    au-dessus de 40 % du pas de la zone SEM la plus contraignante de même vitesse."""
    m = _small(v_in=2000.0, rho_in=1000.0)
    dts = [m.zone_dt_max(i) for i in range(len(m.zones))]
    labels = [f"SEM ordre {z.order}" if z.kind == "quad" else f"triangles ordre {z.order}" for z in m.zones]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(len(dts)), [1e3 * d for d in dts], color=["C0"] * 5 + ["C1"])
    ax.set_xticks(range(len(dts)), labels, fontsize=7, rotation=15)
    ax.set_ylabel("pas stable de la zone (ms)")
    ax.set_title("vp = 2000 m/s partout, mailles de 75 à 120 m", fontsize=9)
    fig.tight_layout()
    _save(fig, figure_path())
    assert dts[-1] > 0.4 * min(dts[:4])


def exact_2d(t, r, f, dt, c, rho=1.0, n=2000):
    t0 = t - r / c
    if t0 <= 0:
        return 0.0
    w = np.linspace(0.0, np.sqrt(t0), n + 1)
    g = 2.0 * np.interp(t0 - w * w, np.arange(len(f)) * dt, f, left=0.0, right=0.0) / np.sqrt(c * c * w * w + 2.0 * c * r)
    simpson = np.ones(n + 1)
    simpson[1:-1:2], simpson[2:-1:2] = 4.0, 2.0
    return rho * c / (2.0 * np.pi) * (simpson * g).sum() * (w[1] - w[0]) / 3.0


def test_exact_solution_through_triangles_and_mixed_orders(figure_path):
    """Milieu homogène : l'onde traverse des zones SEM d'ordres 3, 4, 5 et un anneau de triangles DG ;
    les traces égalent la solution exacte 2-D (5 points par longueur d'onde)."""
    VP, f0, hq, R = 1500.0, 5.0, 96.0, 260.0
    X0, X1 = -600.0, 2600.0
    nc = int(round(2 * np.pi * R / hq))
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    circ = np.stack([1000 + R * np.cos(th), 1000 + R * np.sin(th)], 1)
    nodes, tris = triangulate_ring((600, 1400, 600, 1400), (880, 1120, 880, 1120), circ, lambda x, z: hq)
    n = lambda L, h: int(np.ceil(L / h))
    mdl = (VP, 1.0)
    zones = [QuadZone(X0, X1, X0, 600, n(X1 - X0, hq), n(600 - X0, hq), 4, mdl),
             QuadZone(X0, X1, 1400, X1, n(X1 - X0, 1.1 * hq), n(X1 - 1400, 1.1 * hq), 4, mdl),
             QuadZone(X0, 600, 600, 1400, n(600 - X0, hq), n(800, hq), 4, mdl),
             QuadZone(1400, X1, 600, 1400, n(X1 - 1400, 0.75 * hq), n(800, 0.75 * hq), 3, mdl),
             QuadZone(880, 1120, 880, 1120, 2, 2, 5, mdl), TriZone(nodes, tris, vp=VP, rho=1.0)]
    m = Hybrid2D(zones, sponge=500.0)
    dt = min(1e-3, 0.8 * m.dt_max())
    nt = int(1.3 / dt)
    w = ricker(np.arange(nt) * dt, f0, t0=1.0 / f0)
    src = (350.0, 950.0)
    recs = [(1000.0, 1000.0), (1650.0, 1100.0), (1250.0, 1250.0), (1000.0, 1700.0)]
    where = ["SEM ordre 5 (centre)", "SEM ordre 3", "triangles", "SEM ordre 4"]
    tr = m.run(dt, nt, src, w, recs)["traces"]
    t = (np.arange(nt) + 1) * dt
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})
    draw_hybrid(ax[0], m.zones, lw=0.3)
    ax[0].plot(*src, "r*", ms=12)
    errs = []
    for k, r in enumerate(recs):
        d = np.hypot(r[0] - src[0], r[1] - src[1])
        ref = np.array([exact_2d(tt, d, w, dt, VP) for tt in t])
        errs.append(np.linalg.norm(tr[:, k] - ref) / np.linalg.norm(ref))
        ax[0].plot(r[0], r[1], "v", color=f"C{k + 1}", ms=8)
        ax[1].plot(t, ref, "k", lw=2.5, alpha=0.35)
        ax[1].plot(t, tr[:, k], "--", color=f"C{k + 1}", label=f"{where[k]} : écart {errs[k]:.1e}")
    ax[0].set_xlim(0, 2000); ax[0].set_ylim(2000, 0)
    ax[0].set_title("source (étoile), récepteurs (triangles)", fontsize=9)
    ax[1].legend(fontsize=8); ax[1].set_xlabel("temps (s)")
    ax[1].set_title("traces (tirets) contre la solution exacte 2-D (gris)", fontsize=9)
    fig.tight_layout()
    _save(fig, figure_path())
    assert max(errs) < 2e-2, errs


def test_automatic_zones_from_a_grid_model(figure_path):
    """plan_hybrid sur un modèle grille (corps rapide elliptique dans un fond en gradient) : les zones
    pavent exactement le domaine, aucune arête de triangle ne reste libre à l'intérieur, les arêtes des
    triangles suivent le contour, les blocs SEM respectent la taille requise, l'énergie est conservée."""
    from makutu.mortar.hybrid_plan import plan_hybrid
    nx, nz, d = 200, 120, 10.0
    X, Z = np.meshgrid((np.arange(nx) + 0.5) * d, (np.arange(nz) + 0.5) * d, indexing="ij")
    v = 1800.0 + 0.5 * Z
    v[((X - 1000) / 450) ** 2 + ((Z - 600) / 250) ** 2 < 1] = 4000.0
    zones, info = plan_hybrid(v, (0.0, 0.0), (d, d), f0=8.0, order_body=6, interface_weight=0.25)
    m = Hybrid2D(zones, boundary=RIGID)                       # setup computes the triangle areas
    area = sum((z.x1 - z.x0) * (z.z1 - z.z0) if z.kind == "quad" else z.area.sum() for z in zones)
    assert abs(area - nx * nz * d * d) < 1e-6 * nx * nz * d * d
    for z in zones:                                            # SEM: h <= N v_min / (fmax ppw)
        if z.kind == "quad":
            i0, i1 = int(round(z.x0 / d)), int(round(z.x1 / d))
            k0, k1 = int(round(z.z0 / d)), int(round(z.z1 / d))
            inside = ((X[i0:i1, k0:k1] - 1000) / 450) ** 2 + ((Z[i0:i1, k0:k1] - 600) / 250) ** 2 < 1
            assert max(z.hx, z.hz) <= z.order * v[i0:i1, k0:k1].min() / (20.0 * 5.0) * 1.001 or inside.all()
    assert m.outer_tri_edges == 0 and info["n_triangles"] > 0
    p0 = m.project(lambda x, z: np.exp(-((x - 300) ** 2 + (z - 600) ** 2) / (2 * 60.0 ** 2)))
    E = m.run(0.9 * m.dt_max(), 1500, p0=(p0, p0), track_energy=True)["energy"]
    assert np.abs(E / E[0] - 1).max() < 1e-10
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    pc = draw_hybrid(ax, m.zones, lw=0.3)
    for c in info["contours"]:
        ax.plot(c[:, 0], c[:, 1], "r-", lw=0.8)
    fig.colorbar(pc, ax=ax, label="vp (m/s)", shrink=0.8)
    ax.set_title(f"zones automatiques : {info['n_quad_elements']} éléments SEM (ordre 4 dehors, 6 dedans), "
                 f"{info['n_triangles']} triangles ; contour lissé en rouge", fontsize=9)
    fig.tight_layout()
    _save(fig, figure_path())


def test_cg_grids_and_dg_band_without_mortar(figure_path):
    """plan_cg_hybrid : une grille SEM conforme pour le fond, une pour l'intérieur du corps (ordre propre),
    une bande de triangles DG entre les deux, seul couplage. Les zones pavent le domaine, tous les bords
    intérieurs sont couplés (y compris des sommets gmsh arrondis face aux côtés exacts des grilles),
    l'énergie est conservée."""
    from makutu.mortar.hybrid_plan import plan_cg_hybrid
    nx, nz, d = 200, 120, 10.0
    X, Z = np.meshgrid((np.arange(nx) + 0.5) * d, (np.arange(nz) + 0.5) * d, indexing="ij")
    v = 1800.0 + 0.5 * Z
    v[((X - 1000) / 450) ** 2 + ((Z - 600) / 250) ** 2 < 1] = 4000.0
    zones, info = plan_cg_hybrid(v, (0.0, 0.0), (d, d), f0=8.0, order_body=6)
    assert [z.kind for z in zones] == ["grid", "grid", "tri"] and zones[1].order == 6
    m = Hybrid2D(zones, boundary=RIGID)
    cell_area = lambda z: sum(z.wx[i] * z.wz[k] for i, k in np.argwhere(z.active))
    area = sum(cell_area(z) if z.kind == "grid" else z.area.sum() for z in zones)
    assert abs(area - nx * nz * d * d) < 1e-6 * nx * nz * d * d
    assert m.outer_tri_edges == 0
    # rectilinear background: each row respects h <= N v_min(row) / (fmax ppw), pitch grows with depth
    bg = zones[0]
    for k in range(len(bg.wz)):
        rows = slice(int(bg.ze[k] // d), int(np.ceil(bg.ze[k + 1] / d)))
        assert bg.wz[k] <= 4 * (1800.0 + 0.5 * Z[0, rows]).min() / 100.0 * 1.001
    assert bg.wz[-1] > 1.2 * bg.wz[0]
    # every free side of the grids not on the domain boundary is glued to triangles
    glued = 0.0
    for z in zones[:2]:
        for e, a, b, _ in z.boundary_edges():
            a, b = np.asarray(a), np.asarray(b)
            on_dom = (a[0] == b[0] and a[0] in (0.0, nx * d)) or (a[1] == b[1] and a[1] in (0.0, nz * d))
            glued += 0.0 if on_dom else np.linalg.norm(b - a)
    band = sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for e, a, b, _ in zones[2].boundary_edges())
    assert abs(glued - band) < 1e-6 * band
    p0 = m.project(lambda x, z: np.exp(-((x - 300) ** 2 + (z - 600) ** 2) / (2 * 60.0 ** 2)))
    E = m.run(0.9 * m.dt_max(), 1500, p0=(p0, p0), track_energy=True)["energy"]
    assert np.abs(E / E[0] - 1).max() < 1e-10
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    pc = draw_hybrid(ax, m.zones, lw=0.3)
    for c in info["contours"]:
        ax.plot(c[:, 0], c[:, 1], "r-", lw=0.8)
    fig.colorbar(pc, ax=ax, label="vp (m/s)", shrink=0.8)
    ax.set_title(f"CG + DG sans mortar : {info['n_quad_elements']} éléments SEM (grille du fond ordre 4, "
                 f"grille du corps ordre 6), {info['n_triangles']} triangles DG", fontsize=9)
    fig.tight_layout()
    _save(fig, figure_path())
