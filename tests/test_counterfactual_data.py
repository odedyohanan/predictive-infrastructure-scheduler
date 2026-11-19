import json

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis.counterfactual import (CF_SEEDS, history_arrays_without_context,  # noqa: E402
                                make_seed_records, read_records, write_records)
from worldmodel.data import split_of  # noqa: E402
from worldmodel.features import INPUT_NAMES, STATE_NAMES, build_arrays  # noqa: E402

SEED = 1150


@pytest.fixture(scope="module")
def records():
    return make_seed_records(SEED, drivers=("adaptive",), every=50, n_policies=2, branch_len=2,
                             history=8)


def test_counterfactual_records_have_sibling_format(records, tmp_path):
    assert records
    for r in records:
        assert {"run", "t", "input", "applied", "outcome"} <= set(r)
        assert set(r["input"]) == {"telemetry", "current_policy", "known"}
    X, S, Y, M = build_arrays(records)
    n = len(records)
    assert X.shape == (n, len(INPUT_NAMES)) and S.shape == Y.shape == M.shape == (n, len(STATE_NAMES))
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    # JSON round trip is lossless (what the sibling's load_records reads back)
    write_records(records, tmp_path, SEED)
    back = read_records([str(tmp_path)])
    assert [json.dumps(r, sort_keys=True) for r in back] == [
        json.dumps({**r, "source": str(tmp_path)}, sort_keys=True) for r in records]


def test_counterfactual_data_uses_dev_scenarios_only(records):
    assert set(CF_SEEDS) <= set(range(1100, 1200))
    assert not set(CF_SEEDS) & set(range(1100, 1150))      # probe seeds stay unseen
    assert {r["scenario_seed"] for r in records} == {SEED}
    assert {split_of(r["run"]) for r in records} == {split_of(SEED)}   # split follows the scenario
    for bad in (2000, 1100, 5):
        with pytest.raises(ValueError):
            make_seed_records(bad, drivers=("adaptive",), every=50, n_policies=1, branch_len=1)


def test_branches_carry_real_context_and_distinct_policies(records):
    runs = {}
    for r in records:
        runs.setdefault(r["run"], []).append(r)
    assert len(runs) >= 2
    for recs in runs.values():
        ctx = [r for r in recs if r["context"]]
        br = [r for r in recs if not r["context"]]
        assert len(ctx) == 7 and len(br) == 2
        assert all(r["applied"] == br[0]["applied"] for r in br)      # candidate held fixed
        assert ctx[-1]["outcome"] == br[0]["input"]["telemetry"]      # branch starts at the real state


def test_context_records_only_provide_history(records):
    from worldmodel.features import encode_input, encode_state
    H, S, Y, M, kept = history_arrays_without_context(records, 8)
    assert H.shape == (len(kept), 8, len(INPUT_NAMES))
    assert kept and not any(r["context"] for r in kept)
    assert len(kept) == sum(not r["context"] for r in records)
    first = next(i for i, r in enumerate(records) if not r["context"])   # first branch row of run 1
    ctx = records[first - 1]
    want = encode_input(encode_state(ctx["input"]["telemetry"]), ctx["applied"], ctx["input"]["known"])
    i = next(j for j, r in enumerate(kept) if r is records[first] or r == records[first])
    np.testing.assert_array_equal(H[i, -2], want)


def test_run_configs_contain_no_absolute_paths():
    from tests._load import load_script
    manifest = load_script("make_counterfactual_data").manifest
    recipe = load_script("train_world_model").recipe
    home = "/" + "Users" + "/someone/x"
    m = manifest({"out": home + "/cf", "seeds": [1150]})
    assert home not in json.dumps(m) and m["out"] == "<abs>/cf"
    r = recipe({"out": home + "/m.pt", "data": "x"}, [home + "/cf", "worldmodel:data/adaptive_v1"])
    assert home not in json.dumps(r)
    assert r["data"] == ["<abs>/cf", "worldmodel:data/adaptive_v1"] and r["out"] == "<abs>/m.pt"
