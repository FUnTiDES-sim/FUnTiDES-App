"""Kokkos is initialized once for the whole test session (FUnTiDES needs it
before any pyfuntides object is built) and finalized at the end."""

import gc
import os
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(__file__))   # shots.py

kokkos = pytest.importorskip("kokkos")
pytest.importorskip("pyfuntides.solver")


@pytest.fixture(scope="session", autouse=True)
def kokkos_runtime():
    kokkos.initialize()
    yield
    gc.collect()   # every Kokkos view must be released before finalize
    kokkos.finalize()
