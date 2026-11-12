"""Tests du prototype makutu.mortar : blocs réguliers non conformes recollés par SIPG (2-D acoustique).

NumPy / SciPy seulement (pas de GPU). Chaque test enregistre une figure de contrôle (fixture figure_path).
"""
import numpy as np
import pytest

from makutu.mortar import Mortar2D, Patch, grid_sampler, patch_grid, plan_blocks, ricker, uniform_plan
from makutu.mortar.solver2d import derivative, gll, lagrange

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


def test_gll_basis():
    x, w = gll(4)
    assert np.isclose(w.sum(), 2.0)
    assert np.allclose(lagrange(x, x), np.eye(5))
    D = derivative(x)
    assert np.allclose(D @ x ** 3, 3 * x ** 2)            # exact up to degree N


@pytest.mark.parametrize("order", [2, 4])
def test_default_penalty_is_coercive_and_cheap(order, figure_path):
    """Pénalité par défaut : K symétrique semi-définie positive (seul le mode constant est nul) sur des
    interfaces conformes et non conformes, et pas de temps stable proche de celui sans interface."""
    cases = {"1 bloc": [Patch(0, 600, 0, 600, 6, 6)],
             "h | h": patch_grid([0, 300, 600], [0, 600], [[100], [100]]),
             "h | h/2": patch_grid([0, 300, 600], [0, 600], [[100], [50]]),
             "h | h/1,5": patch_grid([0, 300, 600], [0, 600], [[100], [66.7]]),
             "4 blocs, T": patch_grid([0, 250, 600], [0, 350, 600], [[100, 73], [73, 137]])}
    res = {}
    for k, p in cases.items():
        m = Mortar2D(p, order=order, boundary=RIGID)
        assert abs(m.K - m.K.T).max() < 1e-12 * abs(m.K).max()
        lmax = m.max_eigenvalue()
        lmin = m.min_eigenvalue()
        assert lmin > -1e-9 * lmax, k
        res[k] = 2 / np.sqrt(lmax)
    # a conforming split costs at most 15 % of the stable step
    assert res["h | h"] > 0.85 * res["1 bloc"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(list(res), [1e3 * v for v in res.values()], color="tab:blue")
    ax.set_ylabel("pas de temps stable (ms)")
    ax.set_title(f"ordre {order} : pas stable 2/sqrt(lambda_max), vp = 1500 m/s, éléments de 100 m (et 50-137 m)")
    _save(fig, figure_path())


def test_energy_is_conserved_across_nonconforming_interfaces(figure_path):
    """Boîte rigide, 4 blocs de rapports 1,37 à 3,5, contrastes de vitesse et de densité, propriétés aux
    nœuds : l'énergie discrète du schéma saute-mouton reste constante."""
    blk = patch_grid([0, 900, 2000], [0, 1100, 2000], [[250, 365], [185, 650]], vp=2000.0)

    def nodal(x, z):
        v = np.where(x > 900, np.where(z > 1100, 4500.0, 2000.0), np.where(z > 1100, 3000.0, 2000.0))
        return v + 300 * np.sin(x / 97.0) * np.cos(z / 131.0), 1000 + 1200 * (z > 1100)

    m = Mortar2D(blk, order=4, boundary=RIGID, nodal=nodal)
    p0 = np.exp(-((m.x - 700) ** 2 + (m.z - 900) ** 2) / (2 * 150.0 ** 2))
    dt = 0.9 * m.dt_max()
    E = m.run(dt, 3000, p0=(p0, p0), track_energy=True)["energy"]
    drift = np.abs(E / E[0] - 1)
    assert drift.max() < 1e-10
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(np.arange(len(E)) * dt, drift + 1e-17)
    ax.set_xlabel("temps (s)"); ax.set_ylabel("|E(t) / E(0) - 1|")
    ax.set_title("énergie discrète, 4 blocs non conformes, bords rigides (max %.1e)" % drift.max())
    _save(fig, figure_path())


def test_interface_reflection_is_small(figure_path):
    """Milieu homogène, interface verticale entre h et r h : la réflexion parasite mesurée côté source
    reste très inférieure à l'onde directe (référence : un seul bloc à h, même pas de temps)."""
    v, f0, X, Z, xi, h = 2000.0, 8.0, 3000.0, 2000.0, 1500.0, 40.0
    src, rec = (1000.0, 1000.0), [(600.0, 1000.0)]

    def shot(p, dt, nt):
        m = Mortar2D(p, order=4, sponge=500.0)
        return m.run(dt, nt, src, ricker(np.arange(nt) * dt, f0), rec)["traces"][:, 0]

    ref_p = [Patch(0, X, 0, Z, int(X / h), int(Z / h), v)]
    dt = 0.8 * Mortar2D(ref_p, order=4).dt_max()
    nt = int(1.2 / dt)
    t = np.arange(nt) * dt
    ref = shot(ref_p, dt, nt)
    tr = (2 * xi - src[0] - rec[0][0]) / v                     # reflected arrival time
    win = (t > tr - 0.12) & (t < tr + 0.2)
    plt = _plt()
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t, ref, "k", label="référence (un bloc, h = 40 m)")
    for r, tol in ((1.37, 2e-4), (2.0, 2e-3)):
        p = [Patch(0, xi, 0, Z, int(xi / h), int(Z / h), v),
             Patch(xi, X, 0, Z, int(np.ceil((X - xi) / (r * h))), int(np.ceil(Z / (r * h))), v)]
        d = shot(p, dt, nt)
        refl = np.abs(d - ref)[win].max() / np.abs(ref).max()
        assert refl < tol, (r, refl)
        ax[1].semilogy(t, np.abs(d - ref) / np.abs(ref).max() + 1e-12, label=f"rapport {r} : réflexion {refl:.1e}")
    ax[0].legend(); ax[0].set_ylabel("pression")
    ax[1].axvspan(t[win][0], t[win][-1], color="0.9", label="fenêtre de l'arrivée réfléchie")
    ax[1].legend(fontsize=8); ax[1].set_xlabel("temps (s)"); ax[1].set_ylabel("|écart| / max référence")
    ax[0].set_title("interface non conforme en milieu homogène : récepteur côté source")
    fig.tight_layout()
    _save(fig, figure_path())


def test_block_plan_follows_velocity(figure_path):
    """Modèle synthétique : fond lent en gradient, corps rapide, poche lente. Chaque élément respecte la
    taille requise, le corps rapide reçoit de grands éléments et la poche lente de petits, et le nombre
    d'éléments reste proche de la borne idéale."""
    nx, nz, d = 300, 150, 10.0
    x = (np.arange(nx) + 0.5) * d
    z = (np.arange(nz) + 0.5) * d
    X, Z = np.meshgrid(x, z, indexing="ij")
    v = 1800.0 + 0.6 * Z
    fast = ((X - 1200) / 600) ** 2 + ((Z - 700) / 300) ** 2 < 1
    slow = ((X - 2300) / 250) ** 2 + ((Z - 1000) / 150) ** 2 < 1
    v[fast] = 4500.0
    v[slow] = 1200.0
    order, f0, ppw = 4, 8.0, 5.0
    plan = plan_blocks(v, (0.0, 0.0), (d, d), order=order, f0=f0, ppw=ppw, interface_weight=0.25)
    hreq = order * v / (2.5 * f0 * ppw)
    for p in plan.patches:                                 # validity: h <= min h_req over each element
        i0, i1 = int(round(p.x0 / d)), int(round(p.x1 / d))
        k0, k1 = int(round(p.z0 / d)), int(round(p.z1 / d))
        assert max(p.hx, p.hz) <= hreq[i0:i1, k0:k1].min() * (1 + 1e-9)
    assert plan.stats["overhead"] < 1.6      # uniform blocks in a continuous gradient: ~1.5 x ideal

    def size_at(xp, zp):
        for p in plan.patches:
            if p.x0 <= xp < p.x1 and p.z0 <= zp < p.z1:
                return max(p.hx, p.hz)
    assert size_at(1200, 700) > 1.8 * size_at(600, 700)     # fast body: large elements
    assert size_at(2300, 1000) < 0.8 * size_at(2300, 300)   # slow pocket: small elements
    # work-aware ranking: per-block time steps (LTS) and a global time step target
    lts = plan_blocks(v, (0.0, 0.0), (d, d), order=order, f0=f0, ppw=ppw, rank="lts", interface_weight=0.25)
    assert lts.stats["work_lts"] <= plan.stats["work_lts"] * (1 + 1e-9)
    uni = uniform_plan(v, (0.0, 0.0), (d, d), order=order, f0=f0, ppw=ppw)
    glob = plan_blocks(v, (0.0, 0.0), (d, d), order=order, f0=f0, ppw=ppw, dt_min=uni.stats["dt_global"],
                       interface_weight=0.25)
    assert glob.stats["dt_global"] >= uni.stats["dt_global"] * (1 - 1e-9)
    assert glob.stats["work_global"] < uni.stats["work_global"]
    if figure_path() is not None:
        from makutu.mortar.plots2d import plot_plan
        path = figure_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_plan(path, plan, v, (0, nx * d, nz * d, 0), "synthétique", hreq=hreq)


def test_block_plan_shot_matches_fine_reference(figure_path):
    """Tir sur le pavage du modèle synthétique (propriétés aux nœuds) contre une grille régulière fine
    (2 cellules du modèle par élément) : les blocs ne font pas pire qu'une grille uniforme à la plus petite
    taille requise, avec moins d'éléments. L'écart des deux vient surtout du bord net du corps rapide, qui
    traverse les éléments (il disparaît sur un modèle lissé)."""
    nx, nz, d = 160, 80, 12.5
    x = (np.arange(nx) + 0.5) * d
    z = (np.arange(nz) + 0.5) * d
    X, Z = np.meshgrid(x, z, indexing="ij")
    v = 1800.0 + 0.5 * Z
    v[((X - 900) / 350) ** 2 + ((Z - 550) / 180) ** 2 < 1] = 4000.0
    order, f0, ppw = 4, 8.0, 5.0
    plan = plan_blocks(v, (0.0, 0.0), (d, d), order=order, f0=f0, ppw=ppw, interface_weight=0.25)

    nodal = grid_sampler(v, (0.0, 0.0), (d, d))           # bilinear: continuous model for every mesh

    ref = Mortar2D([Patch(0, nx * d, 0, nz * d, nx // 2, nz // 2)], order=order, sponge=300.0, nodal=nodal,
                   sponge_sides=("left", "right", "bottom"))
    blk = Mortar2D(plan.patches, order=order, sponge=300.0, nodal=nodal, sponge_sides=("left", "right", "bottom"))
    uplan = uniform_plan(v, (0.0, 0.0), (d, d), order=order, f0=f0, ppw=ppw)
    uni = Mortar2D(uplan.patches, order=order, sponge=300.0, nodal=nodal, sponge_sides=("left", "right", "bottom"))
    dt = 0.8 * min(ref.dt_max(), blk.dt_max(), uni.dt_max())
    nt = int(1.6 / dt)
    w = ricker(np.arange(nt) * dt, f0)
    recs = [(xr, 60.0) for xr in np.arange(350.0, 1700.0, 50.0)]
    a = ref.run(dt, nt, (1000.0, 40.0), w, recs)["traces"]
    b = blk.run(dt, nt, (1000.0, 40.0), w, recs)["traces"]
    u = uni.run(dt, nt, (1000.0, 40.0), w, recs)["traces"]
    err = np.linalg.norm(b - a) / np.linalg.norm(a)
    err_u = np.linalg.norm(u - a) / np.linalg.norm(a)
    assert blk.n_elements < uni.n_elements
    assert err < 1.2 * err_u and err < 0.15, (err, err_u)
    plt = _plt()
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.5))
    t = np.arange(nt) * dt
    c = 0.05 * np.abs(a).max()
    ext = (recs[0][0], recs[-1][0], t[-1], 0)
    for axi, D, lab in zip(ax, (a, b, b - a, u - a),
                           (f"référence (grille 25 m, {ref.n_elements} éléments)",
                            f"blocs ({plan.stats['n_blocks']} blocs, {blk.n_elements} éléments)",
                            f"blocs - référence (écart L2 {100 * err:.1f} %)",
                            f"uniforme ({uni.n_elements} él.) - référence ({100 * err_u:.1f} %)")):
        axi.imshow(D, extent=ext, aspect="auto", cmap="gray_r", vmin=-c, vmax=c)
        axi.set_title(lab, fontsize=9); axi.set_xlabel("x récepteur (m)")
    ax[0].set_ylabel("temps (s)")
    fig.tight_layout()
    _save(fig, figure_path())


def test_blocks_converge_on_a_smooth_model(figure_path):
    """Modèle analytique lisse prélevé exactement aux nœuds : l'écart des blocs à une référence fine
    décroît quand on augmente les points par longueur d'onde, et à 5 points il égale celui de la grille
    uniforme avec moins d'éléments. (Avec un modèle SEP lu cellule par cellule, les petits sauts aux bords
    des cellules empêchent cette convergence : d'où grid_sampler bilinéaire.)"""
    X, Z, d = 3000.0, 1500.0, 25.0
    fv = lambda x, z: 1500 + 0.9 * z + 400 * np.sin(x / 500.0) * np.sin(z / 300.0) ** 2
    xc, zc = (np.arange(int(X / d)) + 0.5) * d, (np.arange(int(Z / d)) + 0.5) * d
    V = fv(*np.meshgrid(xc, zc, indexing="ij"))
    kw = dict(order=4, sponge=400.0, nodal=lambda x, z: (fv(x, z), np.full(x.shape, 1000.0)),
              sponge_sides=("left", "right", "bottom"))
    src = (1507.0, 50.0)
    recs = [(x, 70.0) for x in np.arange(450.0, 2600.0, 25.0) if abs(x - src[0]) > 200]
    ref = Mortar2D([Patch(0, X, 0, Z, int(X / 12.5), int(Z / 12.5))], **kw)
    dt = 0.8 * ref.dt_max()
    nt = int(1.4 / dt)
    w = ricker(np.arange(nt) * dt, 8.0)
    R = ref.run(dt, nt, src, w, recs)["traces"]
    err, ne = {}, {}
    for ppw in (5.0, 6.0, 7.0):
        plan = plan_blocks(V, (0.0, 0.0), (d, d), order=4, f0=8.0, ppw=ppw, interface_weight=0.25)
        B = Mortar2D(plan.patches, **kw).run(dt, nt, src, w, recs)["traces"]
        err[f"blocs {ppw:g}"] = np.linalg.norm(B - R) / np.linalg.norm(R)
        ne[f"blocs {ppw:g}"] = plan.n_elements
    up = uniform_plan(V, (0.0, 0.0), (d, d), order=4, f0=8.0, ppw=5.0)
    U = Mortar2D(up.patches, **kw).run(dt, nt, src, w, recs)["traces"]
    err["uniforme 5"], ne["uniforme 5"] = np.linalg.norm(U - R) / np.linalg.norm(R), up.n_elements
    assert err["blocs 7"] < 0.5 * err["blocs 5"]
    assert err["blocs 5"] < 1.5 * err["uniforme 5"] and ne["blocs 5"] < ne["uniforme 5"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in err:
        ax.loglog(ne[k], err[k], "o" if k.startswith("blocs") else "s", label=f"{k} ppw : {100 * err[k]:.2f} %")
    ax.set_xlabel("nombre d'éléments"); ax.set_ylabel("écart L2 relatif des traces")
    ax.set_title("modèle analytique lisse : convergence des blocs non conformes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, figure_path())


def test_block_edges_follow_sharp_rectangular_body(figure_path):
    """Corps rectangulaire lent à bord net (600 m/s dans 1500 m/s, comme le bicouche de validation) : les
    bords de blocs passent sur ses bords (un saut fort au milieu d'un élément coûtait 8 % sur les traces) ;
    une interface horizontale nette sur toute la largeur est aussi un bord de bloc."""
    nx, nz, d = 300, 150, 10.0
    v = np.full((nx, nz), 1500.0)
    v[:, 80:] = 3000.0                                    # sharp interface at z = 800 m
    v[140:180, 30:60] = 600.0                             # body [1400, 1800] x [300, 600]
    plan = plan_blocks(v, (0.0, 0.0), (d, d), order=2, f0=2.0, ppw=6.0)
    xs = {p.x0 for p in plan.patches} | {p.x1 for p in plan.patches}
    zs = {p.z0 for p in plan.patches} | {p.z1 for p in plan.patches}
    assert {1400.0, 1800.0} <= xs and {300.0, 600.0, 800.0} <= zs
    # no block straddles a body edge or the interface
    for p in plan.patches:
        assert not (p.x0 < 1400.0 < p.x1 and p.z0 < 600.0 and p.z1 > 300.0)
        assert not (p.x0 < 1800.0 < p.x1 and p.z0 < 600.0 and p.z1 > 300.0)
        assert not (p.z0 < 800.0 < p.z1)
    if figure_path() is not None:
        from makutu.mortar.plots2d import plot_plan
        path = figure_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_plan(path, plan, v, (0, nx * d, nz * d, 0), "corps rectangulaire net et interface horizontale",
                  hreq=2 * v / (2.5 * 2.0 * 6.0))
