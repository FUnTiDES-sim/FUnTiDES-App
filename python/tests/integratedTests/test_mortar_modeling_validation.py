"""Validation de la modélisation acoustique 2-D sur blocs non conformes (makutu.mortar.BlocksMesh2D +
AcousticSolver de makutu + FUnTiDES SEM 2-D avec opérateur d'interface).

- solution exacte 2-D en milieu homogène, à travers des interfaces non conformes de rapports non entiers ;
- source sur un coin de blocs, récepteur sur une interface ;
- surface libre sur un modèle en couches, contre une grille fine uniforme ;
- stabilité en temps long (8 s) avec bords absorbants.
Chaque test enregistre une figure de contrôle.
"""
import numpy as np
import pytest

import pyfuntides.solver as FTSolver

pytestmark = pytest.mark.skipif(not hasattr(FTSolver.Solver, "set_interface_operator"),
                                reason="pyfuntides sans set_interface_operator")

from makutu.mortar import BlocksMesh2D, Patch  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402

VP = 1500.0


def exact_2d(t, r, f, dt, c=VP, rho=1.0, n=2000):
    """(rho c / 2 pi) int_0^(t - r/c) f(tau) / sqrt(c^2 (t - tau)^2 - r^2) dtau (as test_acoustic_2d)."""
    t0 = t - r / c
    if t0 <= 0:
        return 0.0
    w = np.linspace(0.0, np.sqrt(t0), n + 1)
    g = 2.0 * np.interp(t0 - w * w, np.arange(len(f)) * dt, f, left=0.0, right=0.0) \
        / np.sqrt(c * c * w * w + 2.0 * c * r)
    simpson = np.ones(n + 1)
    simpson[1:-1:2], simpson[2:-1:2] = 4.0, 2.0
    return rho * c / (2.0 * np.pi) * (simpson * g).sum() * (w[1] - w[0]) / 3.0


def _patches(xb, zb, H):
    out = []
    for i in range(len(xb) - 1):
        for j in range(len(zb) - 1):
            W, D = xb[i + 1] - xb[i], zb[j + 1] - zb[j]
            out.append(Patch(xb[i], xb[i + 1], zb[j], zb[j + 1], int(np.ceil(W / H[i][j])),
                             int(np.ceil(D / H[i][j])), VP, 1.0))
    return out


def _shoot(mesh, dt, tmax, f0, src, rcv, **kw):
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=dt, minTime=0.0, maxTime=tmax, dtSeismo=dt, sourceType="ricker2",
                            sourceFreq=f0, **mesh.solver_kwargs(), **kw)
    solver.initialize()
    solver.updateSourceAndReceiversFromCoords([list(src)], [list(r) for r in rcv])
    solver.evaluateSource()
    for t in range(solver.n_time_steps):
        solver.execute(t)
    return solver, solver.getPressureAtReceivers().astype(np.float64)


def _homogeneous_mesh(order, patches):
    nx = nz = 200
    return BlocksMesh2D(vel=np.full((nx, nz), VP), origin=(0.0, 0.0), spacing=(10.0, 10.0), f0=5.0,
                        order=order, rho=1.0, free_surface=False, patches=patches)


# irregular tiling, non-integer size ratios (1.15 to 1.9), T-junctions, corner at (1170, 850)
XB, ZB = [0.0, 700.0, 1170.0, 2000.0], [0.0, 850.0, 1400.0, 2000.0]
H = [[20.0, 15.0, 25.0], [17.0, 22.0, 13.0], [24.0, 18.0, 21.0]]


@pytest.mark.parametrize("order, case, tol", [(2, "interior", 3e-3), (3, "interior", 3e-3), (4, "interior", 3e-3),
                                              (3, "corner", 3e-3)])
def test_exact_solution_through_nonconforming_blocks(order, case, tol, figure_path):
    """Milieu homogène découpé en 9 blocs non conformes : la trace égale la solution exacte 2-D."""
    if case == "interior":
        src, rcv = (1010.0, 0.0, 990.0), [(1310.0, 0.0, 690.0), (520.0, 0.0, 1300.0)]
    else:                                     # source on the corner of four blocks, receivers on interfaces
        src, rcv = (1170.0, 0.0, 850.0), [(700.0, 0.0, 600.0), (1500.0, 0.0, 1400.0)]
    mesh = _homogeneous_mesh(order, _patches(XB, ZB, H))
    dt, tmax, f0 = 1e-3 if order < 4 else 5e-4, 0.7, 5.0
    solver, tr = _shoot(mesh, dt, tmax, f0, src, rcv)
    assert mesh.n_segments > 0
    f = solver.sourceValue[:, 0]
    errs, refs = [], []
    for k, r in enumerate(rcv):
        d = np.hypot(r[0] - src[0], r[2] - src[2])
        ref = np.array([exact_2d(n * dt, d, f, dt) for n in range(tr.shape[0])])
        refs.append(ref)
        errs.append(np.linalg.norm(tr[:, k] - ref) / np.linalg.norm(ref))
    path = figure_path()
    if path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from makutu.mortar.plots2d import draw_blocks
        t = np.arange(tr.shape[0]) * dt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
        draw_blocks(ax[0], mesh.plan.patches, None, None, edges=True, fine_color="0.6", fine_width=0.2)
        ax[0].plot(src[0], src[2], "r*", ms=12, label="source")
        for k, r in enumerate(rcv):
            ax[0].plot(r[0], r[2], "v", ms=9, label=f"récepteur {k}")
        ax[0].legend(fontsize=8); ax[0].set_title(f"9 blocs non conformes, ordre {order}", fontsize=9)
        for k in range(len(rcv)):
            ax[1].plot(t, refs[k], "k", lw=2.5, alpha=0.4, label="solution exacte" if k == 0 else None)
            ax[1].plot(t, tr[:, k], "--", label=f"récepteur {k} : écart L2 {errs[k]:.1e}")
        ax[1].legend(fontsize=8); ax[1].set_xlabel("temps (s)"); ax[1].set_ylabel("pression")
        ax[1].set_title(f"FUnTiDES + opérateur d'interface contre la solution exacte 2-D ({case})", fontsize=9)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
        plt.close(fig)
    assert max(errs) < tol, errs


def _layered():
    nx, nz, d = 200, 100, 10.0
    z = (np.arange(nz) + 0.5) * d
    x = (np.arange(nx) + 0.5) * d
    X, Z = np.meshgrid(x, z, indexing="ij")
    v = np.where(Z < 300.0, 1500.0, 2500.0 + 0.5 * (Z - 300.0))
    v[((X - 1300) / 300) ** 2 + ((Z - 650) / 120) ** 2 < 1] = 4200.0
    return v, d


def test_free_surface_matches_fine_uniform_grid(figure_path):
    """Surface libre, couches et corps rapide : blocs (pavage automatique, coupé sur l'interface nette)
    contre une grille fine uniforme (éléments deux fois plus petits que le plus petit requis), coupée elle
    aussi sur l'interface pour représenter le saut de la même façon."""
    v, d = _layered()
    nx, nz = v.shape
    f0, order = 8.0, 4
    mesh = BlocksMesh2D(vel=v, origin=(0.0, 0.0), spacing=(d, d), f0=f0, order=order, rho=1.0,
                        free_surface=True, interface_weight=0.25)
    # reference: uniform grid twice finer than the smallest required size, split at the sharp interface
    # (z = 300 m) so that it represents the jump exactly, as the blocks do
    hmin = order * v.min() / (2.5 * f0 * 5.0) / 2
    nxr = int(np.ceil(nx * d / hmin))
    ref_mesh = BlocksMesh2D(vel=v, origin=(0.0, 0.0), spacing=(d, d), f0=f0, order=order, rho=1.0,
                            free_surface=True,
                            patches=[Patch(0, nx * d, 0, 300.0, nxr, int(np.ceil(300.0 / hmin))),
                                     Patch(0, nx * d, 300.0, nz * d, nxr, int(np.ceil((nz * d - 300.0) / hmin)))])
    dt = 0.9 * min(mesh.dt_max(), ref_mesh.dt_max())
    src = (700.0, 0.0, 40.0)
    rcv = [(x, 0.0, 20.0) for x in np.arange(100.0, 1950.0, 50.0)]
    kw = dict(sponge_size=[300.0, 0.0, 300.0], surface_sponge=True)
    sol, a = _shoot(mesh, dt, 1.4, f0, src, rcv, **kw)
    p = np.array(sol.data.get_current_field(0), copy=False)
    surf = [i for i in range(mesh.model.get_number_of_nodes()) if mesh.model.is_free_surface(i)]
    assert len(surf) > 0 and np.all(p[surf] == 0.0)
    _, b = _shoot(ref_mesh, dt, 1.4, f0, src, rcv, **kw)
    far = np.abs(np.array([r[0] for r in rcv]) - src[0]) > 200.0
    err = np.linalg.norm((a - b)[:, far]) / np.linalg.norm(b[:, far])
    path = figure_path()
    if path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from makutu.mortar.plots2d import draw_blocks
        t = np.arange(a.shape[0]) * dt
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.5))
        draw_blocks(ax[0], mesh.plan.patches, v, (0, nx * d, nz * d, 0), edges=True, fine_width=0.3)
        ax[0].set_title(f"blocs : {len(mesh.plan.patches)} blocs, {mesh.n_elements} éléments "
                        f"(référence {ref_mesh.n_elements})", fontsize=9)
        c = 0.05 * np.abs(b).max()
        ext = (rcv[0][0], rcv[-1][0], t[-1], 0)
        for axi, D, lab in zip(ax[1:], (b, a, a - b), ("grille fine (2 blocs conformes)", "blocs",
                                                        f"blocs - référence (L2 {100 * err:.2f} %)")):
            axi.imshow(D, extent=ext, aspect="auto", cmap="gray_r", vmin=-c, vmax=c)
            axi.set_title(f"surface libre : {lab}", fontsize=9); axi.set_xlabel("x récepteur (m)")
        ax[1].set_ylabel("temps (s)")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
        plt.close(fig)
    assert err < 0.05, err


def test_long_run_is_stable(figure_path):
    """8 s simulées (plus de 7000 pas) sur blocs, bords absorbants et éponge : le champ décroît après le
    passage de l'onde, sans croissance ni valeur non finie."""
    v, d = _layered()
    nx, nz = v.shape
    mesh = BlocksMesh2D(vel=v, origin=(0.0, 0.0), spacing=(d, d), f0=8.0, order=4, rho=1.0,
                        free_surface=False, interface_weight=0.25)
    dt = 0.9 * mesh.dt_max()
    rcv = [(x, 0.0, z) for x in (300.0, 1000.0, 1700.0) for z in (100.0, 500.0, 900.0)]
    _, tr = _shoot(mesh, dt, 8.0, 8.0, (1000.0, 0.0, 400.0), rcv,
                   sponge_size=[300.0, 0.0, 300.0], surface_sponge=False)
    assert np.isfinite(tr).all()
    t = np.arange(tr.shape[0]) * dt
    early = np.abs(tr[t < 2.0]).max()
    late = np.abs(tr[t > 6.0]).max()
    path = figure_path()
    if path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.semilogy(t, np.abs(tr).max(1) + 1e-30)
        ax.set_xlabel("temps (s)"); ax.set_ylabel("max |p| aux 9 récepteurs")
        ax.set_title(f"stabilité, {len(t)} pas de {dt * 1e3:.2f} ms, {mesh.n_segments} segments d'interface : "
                     f"max(t > 6 s) / max(t < 2 s) = {late / early:.1e}", fontsize=9)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
        plt.close(fig)
    assert late < 1e-2 * early
