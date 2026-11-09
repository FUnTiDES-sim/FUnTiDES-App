"""Shared fixtures of the makutu tests.

``figure_path``: every test can save control images; they go to ``tests/figures/<module>/<test>[-<name>].png``
(git-ignored). Set ``MAKUTU_TEST_FIGURES=0`` to skip them, or ``MAKUTU_TEST_FIGURES=<dir>`` to change the
directory.
"""

import os
from pathlib import Path

import pytest

_FIGURES = Path(__file__).resolve().parent / "figures"


@pytest.fixture
def figure_path(request):
    """figure_path(name="") -> Path of a control image of the test, or None when figures are disabled."""
    setting = os.environ.get("MAKUTU_TEST_FIGURES", "1")

    def make(name=""):
        if setting == "0":
            return None
        root = _FIGURES if setting == "1" else Path(setting)
        module = Path(request.node.fspath).stem
        test = request.node.name.replace("[", "-").replace("]", "").replace("/", "_")
        return root / module / (f"{test}-{name}.png" if name else f"{test}.png")
    return make
