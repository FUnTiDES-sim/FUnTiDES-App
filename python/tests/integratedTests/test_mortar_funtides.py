"""Blocs non conformes dans le solveur SEM 2-D de FUnTiDES (makutu.mortar.funtides2d).

FUnTiDES (float32, GPU ou CPU) doit reproduire le prototype makutu.mortar.Mortar2D (float64) sur le même
maillage en blocs : mêmes masses, mêmes forces K u, mêmes traces. Nécessite un pyfuntides construit avec
COMPILE_SEM_2D et Solver.set_interface_operator (FUnTiDES dev-2d).
"""
import numpy as np
import pytest

import pyfuntides.solver as FTSolver

pytestmark = pytest.mark.skipif(not hasattr(FTSolver.Solver, "set_interface_operator"),
                                reason="pyfuntides sans set_interface_operator")

from makutu.mortar import Mortar2D, patch_grid, ricker  # noqa: E402
from makutu.mortar.funtides2d import FuntidesBlocks2D  # noqa: E402


def _nodal(x, z):
    v = np.where(x > 900, np.where(z > 1100, 4500.0, 2000.0), np.where(z > 1100, 3000.0, 2000.0))
    return v + 300 * np.sin(x / 97.0) * np.cos(z / 131.0), 1000 + 1200 * (z > 1100)


@pytest.mark.parametrize("order", [2, 4])
def test_funtides_blocks_match_the_prototype(order, figure_path):
    blk = patch_grid([0, 900, 2000], [0, 1100, 2000], [[50, 73], [37, 130]], vp=2000.0)
    m = Mortar2D(blk, order=order, nodal=_nodal)
    f = FuntidesBlocks2D(m)
    assert f.n_segments > 0
    assert np.abs(f.mass() - m.M).max() < 1e-5 * m.M.max()
    # outer absorbing boundary only: interface edges are not damped, even at block corners
    damping = np.array(f.solver.get_damping_matrix(0), copy=False)
    assert np.abs(damping - m.C).max() < 1e-5 * m.C.max()
    u = np.random.default_rng(1).standard_normal(m.ndof)
    Ku = m.K @ u
    assert np.linalg.norm(f.stiffness_product(u) - Ku) < 1e-5 * np.linalg.norm(Ku)
    assert 0 <= f.dt_max() / m.dt_max() - 1 < 1e-2       # power iteration: lambda from below, dt from above

    dt = 0.8 * m.dt_max()
    nt = int(1.2 / dt)
    w = ricker(np.arange(nt) * dt, 8.0)
    src = (700.0, 600.0)
    recs = [(x, 300.0) for x in np.arange(100.0, 1900.0, 100.0)] + [(1500.0, 1600.0), (400.0, 1500.0)]
    a = m.run(dt, nt, src, w, recs)["traces"]
    b = f.run(dt, nt, src, w, recs)["traces"]
    err = np.linalg.norm(b - a) / np.linalg.norm(a)
    assert err < 1e-4
    c = FuntidesBlocks2D(m, with_interfaces=False).run(dt, nt, src, w, recs)["traces"]
    assert np.linalg.norm(c - a) / np.linalg.norm(a) > 0.5          # uncoupled blocks: the operator matters

    path = figure_path()
    if path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(nt) * dt
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for k, col in ((4, "C0"), (18, "C1")):
            ax[0].plot(t, a[:, k], col, label=f"prototype float64, récepteur x = {recs[k][0]:.0f} m, z = {recs[k][1]:.0f} m")
            ax[0].plot(t, b[:, k], col + "--", label="FUnTiDES float32")
            ax[0].plot(t, c[:, k], col + ":", lw=0.8, label="FUnTiDES sans opérateur d'interface")
        ax[0].legend(fontsize=7); ax[0].set_ylabel("pression")
        ax[1].semilogy(t, np.abs(b - a).max(1) / np.abs(a).max() + 1e-12)
        ax[1].set_ylabel("max |FUnTiDES - prototype| / max"); ax[1].set_xlabel("temps (s)")
        ax[0].set_title(f"ordre {order} : 4 blocs non conformes, écart relatif L2 {err:.1e}")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
        plt.close(fig)
