import pytest
from contextlib import nullcontext as does_not_raise

from utilities.acquisition import ShotPoint, Source, Receiver


@pytest.mark.parametrize('sp', [ShotPoint, Source, Receiver])
class TestShotPoint:
    # Init ShotPoint, Source and Receiver
    @pytest.mark.parametrize(
        "x, y, z, expectation",
        [
            (1, "iul", 3, pytest.raises(AssertionError)),
            (1, "451", 35.2, does_not_raise()),
        ],
    )
    def test_input_coords(self,  sp, x, y, z, expectation):
        with expectation:
            assert sp(x, y, z) is not None

    # Update coordinate
    @pytest.mark.parametrize(
            "axis, expectation",
            [
                (0, [10, 2, 3]),
                (1, [1, 10, 3]),
                (2, [1, 2, 10]),
            ]
    )
    def test_update_correct_coord(self, sp, axis, expectation):
        shotp = sp(1, 2, 3)
        shotp.updateCoordinate(axis, 10)
        
        assert shotp.coords == expectation


    @pytest.mark.parametrize(
            "axis, value, expectation",
            [
                (0, "125", pytest.raises(AssertionError)),
                (3, 125, pytest.raises(AssertionError)),
            ],
    )
    def test_assertion_update_coord(self, sp, axis, value, expectation):
        shotp = sp(1,2,3)

        with expectation:
            assert shotp.updateCoordinate(axis, value) is not None

    # Get position
    def test_get_x(self,  sp):
        assert sp(1, 2, 3).x() == 1

    def test_get_y(self,  sp):
        assert sp(1, 2, 3).y() == 2

    def test_get_z(self,  sp):
        assert sp(1, 2, 3).z() == 3

    @pytest.mark.parametrize("coords, isinbounds",
                        [
                            ((30, 30, 30), True),
                            ((9, 30, 30), False),
                            ((10, 20, 30), True),
                        ])
    def test_in_bounds(self, sp, coords, isinbounds):
        bounds = (10, 50, 20, 40, 30, 70)
        assert sp(*coords).isinBounds((bounds)) == isinbounds
