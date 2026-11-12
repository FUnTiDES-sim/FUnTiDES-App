"""Maillage en blocs non conformes (makutu.mortar.BlocksMesh2D) dans le solveur acoustique de makutu.

Le tir AcousticSolver (localisation des sources et récepteurs, opérateur d'interface posé par le crochet
solver_setup) doit reproduire le tir direct makutu.mortar.funtides2d sur le même maillage.
"""
import numpy as np
import pytest

import pyfuntides.solver as FTSolver

pytestmark = pytest.mark.skipif(not hasattr(FTSolver.Solver, "set_interface_operator"),
                                reason="pyfuntides sans set_interface_operator")

from makutu.mortar import BlocksMesh2D  # noqa: E402
from makutu.mortar.funtides2d import FuntidesBlocks2D  # noqa: E402
from makutu.solvers import AcousticSolver  # noqa: E402


def _model():
    nx, nz, d = 150, 75, 20.0
    x = (np.arange(nx) + 0.5) * d
    z = (np.arange(nz) + 0.5) * d
    X, Z = np.meshgrid(x, z, indexing="ij")
    v = 1600.0 + 0.6 * Z
    v[((X - 1500) / 500) ** 2 + ((Z - 800) / 250) ** 2 < 1] = 4000.0          # fast body: large elements
    return v, d


def test_acoustic_solver_on_blocks_matches_direct_run(figure_path):
    v, d = _model()
    mesh = BlocksMesh2D(vel=v, origin=(0.0, 0.0), spacing=(d, d), f0=8.0, order=4, free_surface=False,
                        interface_weight=0.25)
    assert len(mesh.plan.patches) > 1
    dt, tmax = 1.0e-3, 1.2
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=dt, minTime=0.0, maxTime=tmax, dtSeismo=dt, sourceType="ricker2",
                            sourceFreq=8.0, **mesh.solver_kwargs())
    solver.initialize()
    assert mesh.n_segments > 0
    src = [1000.0, 0.0, 300.0]
    rcv = [[x, 0.0, 100.0] for x in np.arange(200.0, 2900.0, 100.0)] + [[1500.0, 0.0, 1300.0]]
    solver.updateSourceAndReceiversFromCoords([src], rcv)
    solver.evaluateSource()
    for t in range(solver.n_time_steps):
        solver.execute(t)
    a = solver.getPressureAtReceivers().astype(float)
    w = np.asarray(solver.sourceValue[:, 0], float)

    direct = FuntidesBlocks2D(mesh.mortar)
    b = direct.run(dt, len(w), (src[0], src[2]), w, [(r[0], r[2]) for r in rcv])["traces"]
    n = min(len(a), len(b))
    # makutu records before the buffer swap: one step behind the direct run
    errs = {s: np.linalg.norm(a[s:n] - b[:n - s]) / np.linalg.norm(b[:n - s]) for s in (0, 1)}
    shift = min(errs, key=errs.get)
    assert np.abs(a).max() > 0
    assert errs[shift] < 1e-4, errs

    path = figure_path()
    if path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(n) * dt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        c = 0.05 * np.abs(b).max()
        ax[0].imshow(a[:, :-1], aspect="auto", cmap="gray_r", vmin=-c, vmax=c,
                     extent=(rcv[0][0], rcv[-2][0], t[-1], 0))
        ax[0].set_title(f"AcousticSolver makutu sur BlocksMesh2D ({len(mesh.plan.patches)} blocs, "
                        f"{mesh.n_elements} éléments, {mesh.n_segments} segments)", fontsize=9)
        ax[0].set_xlabel("x récepteur (m)"); ax[0].set_ylabel("temps (s)")
        ax[1].plot(t[shift:], a[shift:n, -1], label="AcousticSolver (récepteur profond)")
        ax[1].plot(t[:n - shift], b[:n - shift, -1], "--", label="tir direct funtides2d")
        ax[1].set_title(f"écart relatif L2 {errs[shift]:.1e} (décalage d'enregistrement {shift} pas)", fontsize=9)
        ax[1].legend(fontsize=8); ax[1].set_xlabel("temps (s)")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
        plt.close(fig)


@pytest.mark.parametrize("param", ["c", "1/c2"])
def test_gradient_on_blocks_matches_finite_differences(param, tmp_path, monkeypatch, figure_path):
    """FWI sur maillage en blocs : le gradient (adjoint, aux nœuds puis transposé de l'interpolation
    bilinéaire vers la grille du modèle) égale la dérivée par différences finies du coût."""
    monkeypatch.chdir(tmp_path)
    nx, nz, d = 100, 50, 10.0
    X, Z = np.meshgrid((np.arange(nx) + 0.5) * d, (np.arange(nz) + 0.5) * d)        # (nz, nx), z down
    v0 = 1800.0 + 0.8 * Z
    v0[((X - 650) / 180) ** 2 + ((Z - 280) / 90) ** 2 < 1] = 3500.0
    vtrue = v0 + 200.0 * np.exp(-((X - 500) ** 2 + (Z - 330) ** 2) / 60.0 ** 2)
    bump = np.exp(-((X - 450) ** 2 + (Z - 300) ** 2) / 80.0 ** 2)
    mesh = BlocksMesh2D(vel=v0.T.copy(), origin=(0.0, 0.0), spacing=(d, d), f0=10.0, order=3,
                        free_surface=False, interface_weight=0.25)
    assert len(mesh.plan.patches) > 1
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            dt=4e-4, minTime=0.0, maxTime=0.5, dtSeismo=2e-3, sourceType="ricker2",
                            sourceFreq=10.0, **mesh.solver_kwargs())
    solver.initialize()
    src, rcv = [[250.0, 0.0, 80.0]], [[x, 0.0, 50.0] for x in np.linspace(100, 900, 17)]

    def shoot(v, step=None):
        solver.updateVelocityModel(np.ravel(v))
        solver.updateSourceAndReceiversFromCoords(src, rcv)
        solver.evaluateSource()
        solver.resetWaveField()
        solver.runForward(step)
        return solver.getPressureAtReceivers().astype(np.float64)

    to_m, to_v = {"c": (lambda v: v, lambda m: m), "1/c2": (lambda v: v ** -2, lambda m: m ** -0.5)}[param]
    obs = shoot(vtrue)
    cost = lambda v: solver.dtSeismoRecorded / 2 * np.sum((obs - shoot(v)) ** 2)
    sim = shoot(v0, solver.snapshotStepFor())
    grad = solver.computeGradient(obs - sim, param).reshape(v0.shape)
    m0 = to_m(v0)
    delta = bump * 0.01 * m0
    eps = 0.5
    fd = (cost(to_v(m0 + eps * delta)) - cost(to_v(m0 - eps * delta))) / (2 * eps)
    adjoint = np.sum(grad * delta)
    assert adjoint == pytest.approx(fd, rel=5e-2)

    path = figure_path()
    if path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from makutu.mortar.plots2d import draw_blocks
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
        draw_blocks(ax[0], mesh.plan.patches, v0.T, (0, nx * d, nz * d, 0), edges=True, fine_width=0.3)
        ax[0].set_title(f"modèle initial et blocs ({len(mesh.plan.patches)} blocs, {mesh.n_elements} éléments)",
                        fontsize=9)
        c = np.abs(grad).max()
        im = ax[1].imshow(grad, extent=(0, nx * d, nz * d, 0), cmap="seismic", vmin=-c, vmax=c, aspect="equal")
        fig.colorbar(im, ax=ax[1], shrink=0.8)
        ax[1].set_title(f"gradient ({param}) sur la grille du modèle : adjoint {adjoint:.4e}, "
                        f"différences finies {fd:.4e}", fontsize=9)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
        plt.close(fig)
