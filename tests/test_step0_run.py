import pytest

from tests._load import load_script

_m = load_script("step0_run")
DEFAULTS, check_run = _m.DEFAULTS, _m.check_run


def _args(**kw):
    return {**DEFAULTS, "attempt": 1, "label": "", **kw}


def test_default_run_is_clean():
    errors, dev = check_run(_args(), existing="", dirty=False)
    assert errors == [] and dev == []


@pytest.mark.parametrize("kw", [{"states": 100}, {"k": 8}, {"n_seeds": 4}, {"h2": 1}])
def test_size_changes_are_protocol_deviations(kw):
    errors, dev = check_run(_args(**kw), existing="", dirty=False)
    assert errors == [] and dev == [next(iter(kw))]


def test_seeds_outside_probe_range_rejected():
    errors, _ = check_run(_args(seeds="1100-1160"), existing="", dirty=False)
    assert errors
    errors, _ = check_run(_args(seeds="1150-1199", label="smoke", attempt=0), existing="", dirty=False)
    assert errors == []


def test_at_most_two_attempts_and_no_repeat():
    assert check_run(_args(attempt=3), existing="", dirty=False)[0]
    assert check_run(_args(attempt=0), existing="", dirty=False)[0]          # 0 is for smoke only
    assert check_run(_args(attempt=1), existing="## Attempt 1: **FAIL**\n", dirty=False)[0]
    assert check_run(_args(attempt=2), existing="## Attempt 1: **FAIL**\n", dirty=False)[0] == []


def test_dirty_code_refused_unless_smoke():
    assert check_run(_args(), existing="", dirty=True)[0]
    assert check_run(_args(label="smoke", attempt=0), existing="", dirty=True)[0] == []
