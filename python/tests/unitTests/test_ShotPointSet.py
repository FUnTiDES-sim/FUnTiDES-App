import pytest

from utilities.acquisition import ShotPointSet, Source, ShotPoint, Receiver, ReceiverSet, SourceSet


# Initialization tests
# Test empty, test correct list
@pytest.mark.parametrize("slist, expected_slist, expected_number",
                         [
                            (None, [], 0),
                            ([ShotPoint(1,2,3)], [ShotPoint(1,2,3)], 1),
                            ([ShotPoint(1,2,3), ShotPoint(4,5,6)], [ShotPoint(1,2,3), ShotPoint(4,5,6)], 2),
                            ((ShotPoint(1,2,3), ShotPoint(4,5,6)), [ShotPoint(1,2,3), ShotPoint(4,5,6)], 2),

                         ])
def test_init_ShotPointSet(slist, expected_slist, expected_number):
    assert ShotPointSet(slist).number == expected_number
    assert ShotPointSet(slist).list == expected_slist

# Try different shotpoint sets init
@pytest.mark.parametrize("splist",
                         [
                             [[1,2,3], [4,5,6]],
                             [Receiver(1,2,3), Receiver(4,5,6), [7,8,9]],
                             Source(1,2,3)
                         ])
def test_assertion_ShotPointSet(splist):
    with pytest.raises(AssertionError):
        ShotPointSet(splist)


# empty new lsit, wrnog list
@pytest.mark.parametrize("newlist, expected_new",
                         [
                            ([], ShotPointSet([])),
                            ([Source(7,8,9)], ShotPointSet([Source(7,8,9)])),
                            ((Source(7,8,9), Receiver(10,11,12)), ShotPointSet([Source(7,8,9), Receiver(10,11,12)]))
                         ])
def test_updateListOfShotPoints(newlist, expected_new):
    sps = ShotPointSet([ShotPoint(1,2,3), ShotPoint(4,5,6)])
    sps.updateList(newlist)

    assert sps == expected_new

@pytest.mark.parametrize("newsp, expected_new",
                         [
                             (None, ShotPointSet([ShotPoint(1,2,3), ShotPoint(4,5,6)])),
                             (Source(1,2,3), ShotPointSet([ShotPoint(1,2,3), ShotPoint(4,5,6), Source(1,2,3)])),
                         ])
def test_appendSingle_to_ShotPointSet(newsp, expected_new):
    sps = ShotPointSet([ShotPoint(1,2,3), ShotPoint(4,5,6)])
    sps.append(newsp)
    assert sps == expected_new


list = [ShotPoint(7,8,9), Source(10,11,12), Receiver(13,14,15)]
expected = ShotPointSet([ShotPoint(1,2,3), ShotPoint(4,5,6), ShotPoint(7,8,9), Source(10,11,12), Receiver(13,14,15)])
@pytest.mark.parametrize("newset, expected_new",
                        [
                        (list, expected),
                        (ShotPointSet(list), expected),
                        ])
def test_appendSet_to_ShotPointSet(newset, expected_new):
    sps = ShotPointSet([ShotPoint(1,2,3), ShotPoint(4,5,6)])
    sps.appendSet(newset)

    assert sps == expected_new

## ReceiverSet
@pytest.mark.parametrize("rs, expected",
                         [
                            (ReceiverSet([Receiver(10,10,10)]), ReceiverSet()),
                            (ReceiverSet([Receiver(10,30,30)]), ReceiverSet([Receiver(10,30,30)])),
                            (ReceiverSet([Receiver(20,30,40), Receiver(10,30,71)]), ReceiverSet([Receiver(20,30,40)]))
                         ])
def test_receiver_within_bounds(rs, expected):
    rs.keepReceiversWithinBounds(bounds=(10, 50, 20, 40, 30, 70))

    assert rs == expected

## SourceSet
@pytest.mark.parametrize("cs, center",
                         [
                             (SourceSet([Source(1,2,3)]), (1,2,3)),
                             (SourceSet([Source(100,100,100), Source(0,0,0)]), (50,50,50)),
                             (SourceSet(), None)
                         ])
def test_source_get_center(cs, center):
    assert cs.getCenter() == center