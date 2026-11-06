"""Unit tests of the acquisition classes: list, DAS, equispaced, SEG-Y, decision file."""

from pathlib import Path

import numpy as np
import pytest

from makutu.acquisition import (
    Acquisition,
    DecisionAcquisition,
    EQUISPACEDAcquisition,
    SEGYAcquisition,
    create_decision_files,
    read_decision_file,
)
from makutu.acquisition.SegyAcquisition import _int16

# Two shots of 300 receivers (shot_1.sgy, shot_2.sgy, decision.*). The SEG-Y headers hold
# integers with scalco = scalel = -10 (divide by 10); the decision file holds the raw integers
# without scalar words (see tests/data/acquisition/README.md). z is positive downwards: sdepth for
# the sources, gelev for the receivers.
ACQ_DIR = Path(__file__).resolve().parents[1] / "data/acquisition"

SOURCES = [[100.0, 200.0, 10.0], [300.0, 200.0, 10.0], [500.0, 200.0, 10.0]]
RECEIVERS = [[0.0, 200.0, 20.0], [50.0, 200.0, 20.0]]
DAS = [[0.0, 90.0, 10.0], [0.0, 90.0, 10.0]]       # dip, azimuth, gauge length


def src(shot):
    return shot.getSourceCoords()[0]


# --------------------------------------------------------------------------- list
class TestAcquisition:
    def test_one_shot_per_source(self):
        acq = Acquisition(sources=SOURCES, receivers=RECEIVERS, dt=1e-3)
        assert [s.id for s in acq.shots] == ["00001", "00002", "00003"]
        assert [src(s) for s in acq.shots] == SOURCES
        for shot in acq.shots:
            assert shot.getReceiverCoords() == RECEIVERS
            assert shot.dt == 1e-3
            assert shot.getDASGeometry() is None
        assert acq.id == "00001" and acq.limited_aperture is False

    def test_from_text_files(self, tmp_path):
        np.savetxt(tmp_path / "src.txt", SOURCES)
        np.savetxt(tmp_path / "rcv.txt", RECEIVERS)
        acq = Acquisition(sources=str(tmp_path / "src.txt"), receivers=str(tmp_path / "rcv.txt"))
        assert [src(s) for s in acq.shots] == SOURCES
        assert acq.shots[0].getReceiverCoords() == RECEIVERS

    def test_missing_receivers(self):
        with pytest.raises(ValueError):
            Acquisition(sources=SOURCES)

    def test_source_center(self):
        acq = Acquisition(sources=SOURCES, receivers=RECEIVERS)
        assert acq.getSourceCenter() == pytest.approx((300.0, 200.0, 10.0))

    def test_split(self):
        acq = Acquisition(sources=SOURCES, receivers=RECEIVERS, dt=2e-3)
        parts = acq.splitAcquisition()
        assert len(parts) == 3
        for part, shot in zip(parts, acq.shots):
            assert len(part.shots) == 1
            assert part.shots[0].id == shot.id and part.shots[0].dt == 2e-3
            assert src(part.shots[0]) == src(shot)
            assert part.shots[0].getReceiverCoords() == RECEIVERS

    def test_limited_aperture_not_implemented(self):
        acq = Acquisition(sources=SOURCES, receivers=RECEIVERS)
        with pytest.raises(NotImplementedError):
            acq.limitedAperture(100.0)
        assert acq.limited_aperture is False


# --------------------------------------------------------------------------- DAS
class TestDAS:
    def test_das_geometry(self):
        acq = Acquisition(sources=SOURCES, receivers=RECEIVERS, dasgeometry=DAS, dwords=None)
        for shot in acq.shots:
            assert shot.getDASGeometry() == DAS
            assert shot.getReceiverCoords() == RECEIVERS

    def test_das_requires_geometry(self):
        with pytest.raises(ValueError):
            Acquisition(sources=SOURCES, receivers=RECEIVERS, dwords=None)

    def test_split_keeps_das(self):
        acq = Acquisition(sources=SOURCES, receivers=RECEIVERS, dasgeometry=DAS, dwords=None)
        for part in acq.splitAcquisition():
            assert part.shots[0].getDASGeometry() == DAS


# --------------------------------------------------------------------------- equispaced
class TestEquispaced:
    def test_single_lines(self):
        acq = EQUISPACEDAcquisition(
            startFirstSourceLine=[0.0, 0.0, 10.0], endFirstSourceLine=[400.0, 0.0, 10.0], sourcesPerLine=5,
            startFirstReceiversLine=[0.0, 0.0, 20.0], endFirstReceiversLine=[1000.0, 0.0, 20.0],
            receiversPerLine=11)
        assert acq.type == "equispacedAcquisition"
        assert [src(s) for s in acq.shots] == [[x, 0.0, 10.0] for x in (0.0, 100.0, 200.0, 300.0, 400.0)]
        rcv = np.array(acq.shots[0].getReceiverCoords())
        np.testing.assert_allclose(rcv[:, 0], np.linspace(0.0, 1000.0, 11))
        np.testing.assert_allclose(rcv[:, 1:], [[0.0, 20.0]] * 11)

    def test_several_lines(self):
        acq = EQUISPACEDAcquisition(
            startFirstSourceLine=[0.0, 0.0, 10.0], endFirstSourceLine=[100.0, 0.0, 10.0],
            startLastSourceLine=[0.0, 200.0, 10.0], endLastSourceLine=[100.0, 200.0, 10.0],
            numberOfSourceLines=3, sourcesPerLine=[2, 3, 2],
            startFirstReceiversLine=[0.0, 0.0, 20.0], endFirstReceiversLine=[100.0, 0.0, 20.0],
            startLastReceiversLine=[0.0, 100.0, 20.0], endLastReceiversLine=[100.0, 100.0, 20.0],
            numberOfReceiverLines=2, receiversPerLine=[3, 5])
        sources = [src(s) for s in acq.shots]
        assert len(sources) == 7
        assert sources[2:5] == [[0.0, 100.0, 10.0], [50.0, 100.0, 10.0], [100.0, 100.0, 10.0]]
        rcv = acq.shots[0].getReceiverCoords()
        assert len(rcv) == 8
        assert rcv[:3] == [[0.0, 0.0, 20.0], [50.0, 0.0, 20.0], [100.0, 0.0, 20.0]]
        assert rcv[3] == [0.0, 100.0, 20.0] and rcv[-1] == [100.0, 100.0, 20.0]

    @pytest.mark.parametrize("key, value, error", [
        ("receiversPerLine", [3], ValueError),
        ("receiversPerLine", 3.0, TypeError),
        ("sourcesPerLine", [2], ValueError),
        ("sourcesPerLine", "2", TypeError),
    ])
    def test_bad_counts(self, key, value, error):
        kwargs = dict(
            startFirstSourceLine=[0.0, 0.0, 10.0], endFirstSourceLine=[100.0, 0.0, 10.0],
            startLastSourceLine=[0.0, 100.0, 10.0], endLastSourceLine=[100.0, 100.0, 10.0],
            numberOfSourceLines=2, sourcesPerLine=2,
            startFirstReceiversLine=[0.0, 0.0, 20.0], endFirstReceiversLine=[100.0, 0.0, 20.0],
            startLastReceiversLine=[0.0, 100.0, 20.0], endLastReceiversLine=[100.0, 100.0, 20.0],
            numberOfReceiverLines=2, receiversPerLine=3)
        kwargs[key] = value
        with pytest.raises(error):
            EQUISPACEDAcquisition(**kwargs)


# --------------------------------------------------------------------------- SEG-Y
class TestSegy:
    @pytest.mark.parametrize("raw, value", [(0, 0), (10, 10), (0xFFF6, -10), (0x8000, -32768),
                                            (0x7FFF, 32767), (0xFFF60001, 1)])
    def test_int16(self, raw, value):
        assert _int16(raw) == value

    def test_example_files(self):
        acq = SEGYAcquisition(segdir=str(ACQ_DIR))
        assert acq.type == "segyAcquisition"
        assert len(acq.shots) == 2
        # scalco = scalel = -10: header integers divided by 10 (sx, sy, sdepth)
        assert src(acq.shots[0]) == pytest.approx([399.1, 499.1, 20.1])
        assert src(acq.shots[1]) == pytest.approx([599.1, 499.1, 20.1])
        assert len(acq.shots[0].getReceiverCoords()) == 300

    def test_receivers_match_headers(self):
        segyio = pytest.importorskip("segyio")
        acq = SEGYAcquisition(segdir=str(ACQ_DIR))
        for shot, name in zip(acq.shots, ("shot_1.sgy", "shot_2.sgy")):
            with segyio.open(ACQ_DIR / name, ignore_geometry=True) as f:
                expected = [[h[segyio.su.gx] / 10, h[segyio.su.gy] / 10, h[segyio.su.gelev] / 10]
                            for h in f.header]
            np.testing.assert_allclose(shot.getReceiverCoords(), expected)

    def test_ignore(self):
        acq = SEGYAcquisition(segdir=str(ACQ_DIR), ignore=("shot_2.sgy",))
        assert len(acq.shots) == 1
        assert src(acq.shots[0]) == pytest.approx([399.1, 499.1, 20.1])

    def test_no_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SEGYAcquisition(segdir=str(tmp_path))


# --------------------------------------------------------------------------- decision file
class TestDecision:
    root = str(ACQ_DIR / "decision")

    def test_example_raw(self):
        # no scalar words in the file: raw integers
        acq = DecisionAcquisition(decision_filename=self.root)
        assert acq.type == "decisionAcquisition" and acq.numberOfSources == 2
        assert [src(s) for s in acq.shots] == [[3991.0, 4991.0, 201.0], [5991.0, 4991.0, 201.0]]
        assert all(len(s.getReceiverCoords()) == 300 for s in acq.shots)

    def test_example_matches_segy(self):
        dec = DecisionAcquisition(decision_filename=self.root, scalco=-10, scalel=-10)
        seg = SEGYAcquisition(segdir=str(ACQ_DIR))
        for d, s in zip(dec.shots, seg.shots):
            np.testing.assert_allclose(src(d), src(s))
            np.testing.assert_allclose(d.getReceiverCoords(), s.getReceiverCoords())

    def test_one_scalar_only(self):
        acq = DecisionAcquisition(decision_filename=self.root, scalco=-10)
        assert src(acq.shots[0]) == pytest.approx([399.1, 499.1, 201.0])

    def test_wrong_words(self):
        with pytest.raises(ValueError):
            read_decision_file(self.root, swords=[22, 23, 15])

    def test_round_trip(self, tmp_path):
        shots = [{"src": [10, 20, 5], "rcv": [[0, 20, 1], [40, 20, 2]]},
                 {"src": [30, 20, 5], "rcv": [[0, 20, 1], [40, 20, 2], [80, 20, 3]]}]
        create_decision_files(shots, str(tmp_path / "survey"))   # default words read back by default
        acq = DecisionAcquisition(decision_filename=str(tmp_path / "survey"), sorted=True)
        assert [src(s) for s in acq.shots] == [[10.0, 20.0, 5.0], [30.0, 20.0, 5.0]]
        assert acq.shots[1].getReceiverCoords() == [[0.0, 20.0, 1.0], [40.0, 20.0, 2.0], [80.0, 20.0, 3.0]]

    def test_das(self, tmp_path):
        das = [[0, 90, 10], [5, 45, 10]]
        shots = [{"src": [10, 20, 5], "rcv": [[0, 20, 1], [40, 20, 2]],
                  "words": {60: [d[0] for d in das], 61: [d[1] for d in das], 62: 10}}]
        create_decision_files(shots, str(tmp_path / "das"), extra_mapped_words=[60, 61, 62])
        acq = DecisionAcquisition(decision_filename=str(tmp_path / "das"), sorted=True, dwords=None)
        assert acq.shots[0].getReceiverCoords() == [[0.0, 20.0, 1.0], [40.0, 20.0, 2.0]]
        assert acq.shots[0].getDASGeometry() == [[0.0, 90.0, 10.0], [5.0, 45.0, 10.0]]
        assert acq.splitAcquisition()[0].shots[0].getDASGeometry() == acq.shots[0].getDASGeometry()
