import math
import os
from pathlib import Path

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.forecaster import load_world_model  # noqa: E402
from pis.harness import (BaseGreedyController, forecast_vs_fact, render_report,  # noqa: E402
                         run_scenario, write_report)

SEED = 1100


def _controller():
    return BaseGreedyController(adapters.make_spec(SEED))


def _strip(records):
    return [{"t": r["t"], "tel": r["tel"], "policy": r["policy"]} for r in records]


@pytest.fixture(scope="module")
def records():
    return run_scenario(SEED, _controller(), duration_s=20.0)


def test_run_is_deterministic(records):
    again = run_scenario(SEED, _controller(), duration_s=20.0)
    assert _strip(records) == _strip(again)


def test_record_count_matches_duration(records):
    window = adapters.make_spec(SEED).policy_period_s
    assert len(records) == round(20.0 / window)
    assert all(set(r) == {"t", "tel", "policy", "info"} for r in records)
    assert all(r["info"]["decide_ms"] >= 0 for r in records)


def test_forecast_errors_finite_and_horizon_ordered_keys(records):
    from worldmodel.features import KEY_TARGETS
    res = forecast_vs_fact(records, load_world_model(), adapters.make_spec(SEED), horizons=(1, 5, 10))
    assert list(res["model"]) == [1, 5, 10] and list(res["persistence"]) == [1, 5, 10]
    for h in (1, 5, 10):
        assert set(res["model"][h]) == set(KEY_TARGETS) | {"reward"}
        assert all(math.isfinite(v) for v in res["model"][h].values())
        assert res["n_starts"][h] > 0


def test_report_contains_no_absolute_paths(records, tmp_path):
    res = forecast_vs_fact(records, load_world_model(), adapters.make_spec(SEED), horizons=(1, 5))
    leaky = f"checkpoint at {Path.home()}/x/model.pt and {os.getcwd()}/y"
    text = render_report(res, {"note": leaky})
    out = tmp_path / "report.md"
    write_report(text, out)
    body = out.read_text()
    assert str(Path.home()) not in body and os.getcwd() not in body
    assert "/" + "Users" + "/" not in body
