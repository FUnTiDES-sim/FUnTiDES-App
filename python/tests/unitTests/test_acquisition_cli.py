"""Command lines of apps/acquisition and the SEG-Y geometry convention:
regular acquisition -> SEG-Y headers -> check -> decision files -> readers."""

import os
import subprocess
import sys

import numpy as np
import pytest
import segyio

from makutu.acquisition import SEGYAcquisition, read_decision_file
from makutu.acquisition.decision_create import read_segy_geometry

APPS = os.path.join(os.path.dirname(__file__), "..", "..", "apps", "acquisition")
SRC_X = np.linspace(100, 900, 4)                 # 4 shots, sources at 10 m depth
RCV_X = np.linspace(100, 900, 11)                # 11 receivers, at 5 m depth


def run(script, *args, cwd):
    res = subprocess.run([sys.executable, os.path.join(APPS, script), *map(str, args)],
                         cwd=cwd, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    return res.stdout


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    d = tmp_path_factory.mktemp("acq")
    run("create_equispaced_segy.py", "--xmin", 0, "--xmax", 1000, "--ymin", 0, "--ymax", 0,
        "--margin", 100, "--n-src-x", 4, "--n-rcv-x", 11, "--n-samples", 51, "--dt", 4,
        "--output-dir", "acq", "--out", "survey", cwd=d)
    return d


def segy_files(survey):
    return sorted(str(p) for p in (survey / "acq").glob("*.segy"))


def test_segy_headers_are_standard(survey):
    """sx, sy, sdepth, gx, gy, gelev with scalco = scalel = -1000, read by segyio, by
    read_segy_geometry and by SEGYAcquisition."""
    files = segy_files(survey)
    assert len(files) == 4
    with segyio.open(files[1], ignore_geometry=True) as f:
        h = f.header[3]
        assert h[segyio.su.scalco] == -1000 and h[segyio.su.scalel] == -1000
        assert h[segyio.su.sx] == round(SRC_X[1] * 1000) and h[segyio.su.sdepth] == 10000
        assert h[segyio.su.gx] == round(RCV_X[3] * 1000) and h[segyio.su.gelev] == 5000
        assert len(f.samples) == 51 and segyio.tools.dt(f) == 4000
    src, rcv, ns, dt_us = read_segy_geometry(files[1])
    np.testing.assert_allclose(src, [[SRC_X[1], 0.0, 10.0]] * 11, atol=1e-3)
    np.testing.assert_allclose(rcv[:, 0], RCV_X, atol=1e-3)
    assert ns == 51 and dt_us == 4000
    acq = SEGYAcquisition(segdir=str(survey / "acq"))
    assert len(acq.shots) == 4
    np.testing.assert_allclose(acq.shots[2].getSourceCoords(), [[SRC_X[2], 0.0, 10.0]], atol=1e-3)
    np.testing.assert_allclose(np.array(acq.shots[2].getReceiverCoords())[:, 2], 5.0, atol=1e-3)


def test_decision_files_of_the_survey(survey):
    src, rcv, n = read_decision_file(str(survey / "acq" / "survey"))
    assert n == 4
    np.testing.assert_allclose([src[f"src{i + 1}"] for i in range(4)],
                               [[round(x), 0, 10] for x in SRC_X])     # integer metres
    np.testing.assert_allclose(np.sort(np.array(rcv["rcv_1"])[:, 0]), RCV_X)   # order not kept


def test_check_acquisition_report_and_plot(survey):
    out = run("check_acquisition.py", "--dir", "acq", "--out", "map", cwd=survey)
    assert "Shots             : 4" in out and "Receivers/shot    : 11" in out
    assert "Samples/trace     : 51" in out
    assert (survey / "map.png").stat().st_size > 0
    out_dec = run("check_acquisition.py", "--from-decision", "acq/survey", "--no-plot", cwd=survey)
    tail = lambda s: s[s.index("Receiver count"):].splitlines()[:4]          # noqa: E731
    assert "Shots             : 4" in out_dec and tail(out_dec)[0] == tail(out)[0]


@pytest.mark.parametrize("mode", ["one file per shot", "one file"])
def test_decision_files_from_segy(survey, mode, tmp_path):
    files = segy_files(survey)
    if mode == "one file":                       # all the traces in one SEG-Y file
        merged = tmp_path / "all.segy"
        spec = segyio.tools.metadata(segyio.open(files[0], ignore_geometry=True))
        spec.tracecount = 44
        with segyio.create(str(merged), spec) as out:
            k = 0
            for name in files:
                with segyio.open(name, ignore_geometry=True) as f:
                    for i in range(f.tracecount):
                        out.header[k], out.trace[k] = f.header[i], f.trace[i]
                        k += 1
            out.bin = segyio.open(files[0], ignore_geometry=True).bin
        files = [str(merged)]
    run("decision_create.py", "--from-segy", *files, "--out", tmp_path / "redo", cwd=survey)
    a = (survey / "acq" / "survey.mapped.data").read_bytes()
    b = (tmp_path / "redo.mapped.data").read_bytes()
    assert a == b                                # same records as written with the survey
