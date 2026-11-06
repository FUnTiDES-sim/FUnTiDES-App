"""The X-axis MPI domain decomposition gives the single-process result."""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "mpi_shot.py")

pytestmark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not found")


def run_shot(nranks, output):
    cmd = ["mpirun", "-np", str(nranks), "--oversubscribe", sys.executable, SCRIPT, str(output)]
    # env: the Python copy of the environment. If MPI was initialised in this process
    # (makutu imports mpi4py.MPI), OpenMPI has changed the C environment, and an
    # mpirun child inheriting it fails.
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    return np.load(output).astype(np.float64)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    ref = run_shot(1, tmp_path_factory.mktemp("mpi") / "ref.npy")
    assert np.isfinite(ref).all() and np.abs(ref).max() > 0
    return ref


@pytest.mark.parametrize("nranks", [2, 3])
def test_decomposition_matches_single_process(tmp_path, reference, nranks):
    ref = reference
    split = run_shot(nranks, tmp_path / f"split{nranks}.npy")
    assert split.shape == ref.shape
    # Same discretization, only the summation order of the shared nodes differs
    assert np.linalg.norm(split - ref) / np.linalg.norm(ref) < 1e-4


def test_forward_chain_with_mpi(tmp_path):
    """Two ranks: seismograms exported by rank 0, cost function ~ 0 on these data."""
    script = os.path.join(os.path.dirname(__file__), "mpi_forward_cost.py")
    cmd = ["mpirun", "-np", "2", "--oversubscribe", sys.executable, script, str(tmp_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    r = np.load(tmp_path / "result.npz")
    assert r["data"].shape == (301, 14) and np.abs(r["data"]).max() > 0
    assert r["J"] < 1e-8 * r["energy"]
