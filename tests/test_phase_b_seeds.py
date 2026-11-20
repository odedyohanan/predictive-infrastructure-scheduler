import pytest

from pis.phase_b import seeds


def test_ranges():
    assert seeds.HELD_OUT == range(3000, 3100)
    assert seeds.HISTORY == range(3100, 3400)
    assert seeds.CALIBRATION == range(3400, 3450)
    assert seeds.SELECTION == range(3450, 3500)


def test_parse():
    assert seeds.parse("3100-3103,3120") == [3100, 3101, 3102, 3103, 3120]
    assert seeds.parse(" 3400 , 3402-3402 ") == [3400, 3402]


@pytest.mark.parametrize("bad", ["", "3105-3100", "abc", "3100,,3101", "3100-"])
def test_parse_rejects_garbage(bad):
    with pytest.raises(ValueError):
        seeds.parse(bad)


@pytest.mark.parametrize("name", ["HELD_OUT", "HISTORY", "CALIBRATION", "SELECTION"])
@pytest.mark.parametrize("refused", [1050, 1150, 2050])
def test_old_seeds_refused_everywhere(name, refused):
    with pytest.raises(ValueError):
        seeds.check([refused], name)


def test_held_out_seed_allowed_only_for_held_out():
    seeds.check([3050], "HELD_OUT")
    for name in ("HISTORY", "CALIBRATION", "SELECTION"):
        with pytest.raises(ValueError):
            seeds.check([3050], name)


def test_check_accepts_its_own_range():
    seeds.check(list(seeds.HISTORY), "HISTORY")
    seeds.check([3400, 3449], "CALIBRATION")
    seeds.check([3450, 3499], "SELECTION")


def test_check_rejects_neighbours_and_unknown_names():
    with pytest.raises(ValueError):
        seeds.check([3100, 3400], "HISTORY")
    with pytest.raises(ValueError):
        seeds.check([3100], "TRAINING")
    with pytest.raises(ValueError):
        seeds.check([], "HISTORY")
