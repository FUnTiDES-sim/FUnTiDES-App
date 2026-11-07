"""
Adjoint and gradient of makutu on FUnTiDES, phase by phase (2-D, small mesh):

- reciprocity: the adjoint run injects the residual with the receiver weights, which must
  be the transpose of the recording (source <-> receiver swap, off the nodes);
- Taylor test: J(m + h dm) - J(m) - h <g, dm> = O(h^2) for a random perturbation;
- absorbing edges: their terms are not differentiated (the gradient is exact inside the
  domain, not on the edge elements: see makutu.imaging.GradientConditioner, edge);
- FUnTiDES sponge: it lowers the late-time energy left by the absorbing condition and
  leaves the direct arrivals unchanged.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import pyfuntides.model as FTModel
import pyfuntides.solver as FTSolver

from makutu.imaging import globals as gb
from makutu.mesh import CartesianMesh2D
from makutu.solvers import AcousticSolver

from test_gradient import PARAM, grid_2d, gradient_check, solver_2d

pytestmark = pytest.mark.skipif(not hasattr(FTModel, "CartesianUnstruct2DBuilder_f32_i32"),
                                reason="FUnTiDES built without COMPILE_SEM_2D")

SRC = [[300.0, 0.0, 100.0]]
RCV = [[x, 0.0, 60.0] for x in np.linspace(100, 900, 17)]


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield
    gb.solver = None


def background():
    X, Z = grid_2d()
    return X, Z, 2000.0 + 200.0 * Z / 500.0                  # vertical gradient, z down


def shoot(solver, v, src, rcv, step=None):
    solver.updateVelocityModel(np.ravel(v))
    solver.updateSourceAndReceiversFromCoords(src, rcv)
    solver.evaluateSource()
    solver.resetWaveField()
    solver.runForward(step)
    return solver.getPressureAtReceivers().astype(np.float64)


def test_reciprocity_off_the_nodes():
    _, _, v0 = background()
    solver = solver_2d(2, v0)
    for a, b in [([313.7, 0.0, 107.3], [688.1, 0.0, 271.9]), ([150.2, 0.0, 41.1], [833.3, 0.0, 433.3])]:
        ab = shoot(solver, v0, [a], [b])[:, 0]
        ba = shoot(solver, v0, [b], [a])[:, 0]
        assert np.linalg.norm(ab - ba) < 1e-3 * np.linalg.norm(ab)          # 2e-5 measured


@pytest.mark.parametrize("param", ["c", "1/c", "1/c2"])
def test_taylor_second_order(param):
    """The remainder of the first-order expansion is divided by 4 when h is halved."""
    X, Z, v0 = background()
    vtrue = v0 + 150.0 * np.exp(-((X - 600) ** 2 + (Z - 300) ** 2) / 80.0 ** 2)
    solver = solver_2d(2, v0)
    to_m, to_v = PARAM[param]
    obs = shoot(solver, vtrue, SRC, RCV)
    dtr = solver.dtSeismoRecorded
    J = lambda m: dtr / 2 * np.sum((obs - shoot(solver, to_v(m), SRC, RCV)) ** 2)
    sim = shoot(solver, v0, SRC, RCV, solver.snapshotStepFor())
    g = solver.computeGradient(obs - sim, param).reshape(v0.shape)

    m0 = to_m(v0)
    dm = gaussian_filter(np.random.default_rng(1).normal(size=v0.shape), 2.0)
    dm *= 0.01 * np.abs(m0).max() / np.abs(dm).max()                      # up to 1 % of the model
    J0, slope = J(m0), np.sum(g * dm)
    R = [abs(J(m0 + h * dm) - J0 - h * slope) for h in (2.0, 1.0, 0.5)]
    ratios = [R[0] / R[1], R[1] / R[2]]                                    # 4.05, 4.00 measured (c)
    assert all(3.5 < r < 4.5 for r in ratios), ratios


def test_gradient_inside_and_on_the_absorbing_edges():
    """Inside: the finite differences within 2 %; on a side edge the non-differentiated
    absorbing terms dominate (about 100 % measured), hence the edge mask of the
    gradient conditioner."""
    X, Z, v0 = background()
    vtrue = v0 + 150.0 * np.exp(-((X - 600) ** 2 + (Z - 300) ** 2) / 80.0 ** 2)
    solver = solver_2d(2, v0)
    rel = {}
    for name, (bx, bz) in [("inside", (500, 250)), ("side", (980, 250))]:
        bump = np.exp(-((X - bx) ** 2 + (Z - bz) ** 2) / 50.0 ** 2)
        adjoint, fd = gradient_check(solver, v0, vtrue, bump, "c", SRC, RCV)
        rel[name] = abs(adjoint - fd) / abs(fd)
    assert rel["inside"] < 2e-2                                            # 0.3 % measured
    assert rel["side"] > 0.2                                               # documents the edge error


def test_sponge_absorbs_without_growth():
    """Homogeneous model, interior 1000 x 500 m, free surface on top: with a 400 m sponge (two
    wavelengths) on the sides and the bottom, the energy after the direct arrivals (0.6 to
    2 s) falls below half of that left by the absorbing condition alone (0.15 measured), and
    the direct arrivals do not change."""
    def run(pad):
        lx, lz = 1000.0 + 2 * pad, 500.0 + pad
        ex, ez = int(lx / 25), int(lz / 25)
        mesh = CartesianMesh2D(lx, lz, ex, ez, order=2, vp=np.full((ez, ex), 2000.0))
        s = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                           model_location=FTSolver.ModelLocationType.ONELEMENTS, sponge_size=[pad] * 3,
                           surface_sponge=True, dt=1.67e-3, minTime=0.0, maxTime=2.0, dtSeismo=3.34e-3,
                           sourceType="ricker2", sourceFreq=10.0, **mesh.solver_kwargs())
        s.initialize()
        p = shoot(s, s.getVelocityModel(), [[pad + 500.0, 0.0, 25.0]],
                  [[pad + x, 0.0, z] for x, z in ((100.0, 25.0), (300.0, 25.0), (800.0, 25.0), (500.0, 400.0))])
        del s, mesh
        return p

    p0, p400 = run(0.0), run(400.0)
    late = slice(int(0.6 / 3.34e-3), None)
    early = slice(0, int(0.25 / 3.34e-3))                                  # before any edge reflection
    assert np.linalg.norm(p400[early] - p0[early]) < 1e-3 * np.linalg.norm(p0[early])
    assert np.linalg.norm(p400[late]) < 0.5 * np.linalg.norm(p0[late])
