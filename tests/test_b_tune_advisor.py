import json
import pickle

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis.phase_b import predictors  # noqa: E402
from pis.phase_b.incidents import IncidentRule  # noqa: E402
from tests._load import load_script  # noqa: E402


@pytest.fixture(scope="module")
def tune():
    return load_script("b_tune_advisor")


def _ep(seed, arm, t, v, chronic=False, inc=1):
    return {"seed": seed, "arm": arm, "time_in_incident_s": t, "value_per_s": v, "chronic": chronic,
            "incidents": inc, "mode_share": 0.1, "fallback_share": 0.0, "decide_ms_p95": 5.0,
            "decide_ms_max": 9.0, "seconds": 1.0}


def test_grid_has_six_configs(tune):
    g = tune.grid()
    assert len(g) == 6 and {c["eps"] for c in g} == {0.05, 0.1, 0.2} and {c["n"] for c in g} == {8, 16}
    assert len({tune.arm_name(c) for c in g}) == 6


def test_summarise_excludes_chronic_from_time_only(tune):
    cfg = [{"eps": 0.1, "n": 8}]
    a = tune.arm_name(cfg[0])
    eps = []
    for s, (tr, ta, chronic) in enumerate([(10, 6, False), (20, 15, False), (200, 0, True), (8, 8, False)]):
        eps += [_ep(s, "r6", tr, 100.0 + s, chronic), _ep(s, a, ta, 100.0 + s - 0.1 * s)]
    out = tune.summarise(eps, cfg)[a]
    assert out["non_chronic"] == 3 and out["seeds"] == 4
    assert out["time_advisor"] == pytest.approx((6 + 15 + 8) / 3)
    assert out["time_r6"] == pytest.approx((10 + 20 + 8) / 3)
    assert out["value_diff"]["n"] == 4                                   # value over all episodes
    assert out["value_diff"]["mean"] == pytest.approx(-0.15)
    assert out["r6_value"] == pytest.approx(101.5)
    with pytest.raises(ValueError):
        tune.summarise(eps[:-1], cfg)                                   # a missing pair


def test_choose_config_value_constraint_and_ties(tune):
    def s(t, n, eps, ok):
        return {"time_advisor": t, "n": n, "eps": eps, "value_ok": ok}
    summ = {"a": s(5.0, 16, 0.1, True), "b": s(5.0, 8, 0.2, True), "c": s(1.0, 8, 0.05, False),
            "d": s(6.0, 8, 0.05, True)}
    assert tune.choose_config(summ) == "b"                               # c violates the value bound
    assert tune.choose_config({"c": summ["c"]}) is None


def test_value_ok_uses_one_percent_of_r6_value(tune):
    cfg = [{"eps": 0.1, "n": 8}]
    a = tune.arm_name(cfg[0])
    rng = np.random.default_rng(0)
    eps = []
    for s in range(10):
        r6 = 100.0 + rng.normal()
        eps += [_ep(s, "r6", 5, r6), _ep(s, a, 4, r6 - 0.5 + 0.01 * rng.normal())]
    assert tune.summarise(eps, cfg)[a]["value_ok"]                       # -0.5 % > -1 %
    eps = [dict(e, value_per_s=e["value_per_s"] - (2.0 if e["arm"] == a else 0.0)) for e in eps]
    assert not tune.summarise(eps, cfg)[a]["value_ok"]                   # -2.5 %


@pytest.mark.parametrize("args", [
    ["--seeds", "3450-3460"],                                            # part of the range
    ["--seeds", "3100-3104"],                                            # history in a real run
    ["--seeds", "3000-3004", "--label", "smoke", "--smoke-dir", "data/phase_b/smoke_test_refuse"],
    ["--seeds", "3400-3404", "--label", "smoke", "--smoke-dir", "data/phase_b/smoke_test_refuse"],
    ["--seeds", "3450", "--label", "smoke"],                              # no smoke dir
    ["--seeds", "3450", "--label", "smoke", "--smoke-dir", "/tmp/elsewhere"],
    ["--duration", "30"],                                                # short real run
    ["--label", "other"],
])
def test_main_refuses(tune, args):
    with pytest.raises(SystemExit) as e:
        tune.main(args)
    assert "refusing" in str(e.value)


def test_tune_end_to_end_and_resume(tune, tmp_path):
    from tests.test_phase_b_wm_prob import _tiny_model
    wm = _tiny_model(context=4, room=64, hi=np.full(43, 1e6, np.float32))
    wm.save(tmp_path / "wm.pt")
    q = predictors.QueueingModel(IncidentRule(3, 12, 0.5))
    (tmp_path / "pred.pkl").write_bytes(pickle.dumps({"queueing": q}))
    inc = {"k": 3, "interval": 12, "majority": 0.5, "warmup": 20, "H": 6, "min_lead_windows": 1}
    alarms = {"predictor": "queueing", "threshold": 0.0, "low": 0.0, "H": 6}           # always on after warm-up
    step_a = {"slo_lam": 40.0, "mu": 8.0, "change_cost": 0.0}
    args = dict(seeds=[3450], duration_s=11.0, inc=inc, alarms=alarms, world_model=tmp_path / "wm.pt",
                fitted=tmp_path / "pred.pkl", wm_n=4, step_a=step_a, out=tmp_path / "out", note=tmp_path / "n.md",
                config_path=tmp_path / "advisor.json", procs=1, label="smoke")
    res = tune.tune(**args)
    arms = {e["arm"] for e in res["episodes"]}
    assert arms == {"r6"} | {tune.arm_name(c) for c in tune.grid()}
    adv = [e for e in res["episodes"] if e["arm"] != "r6"]
    assert all(e["windows"] == 22 and e["mode_share"] == pytest.approx(2 / 22) for e in adv)
    cfg = json.loads((tmp_path / "advisor.json").read_text())
    assert cfg["chosen"] == res["chosen"]
    if cfg["chosen"]:
        assert cfg["params"]["h"] == 6 and cfg["params"]["m"] == 4 and cfg["params"]["slo_lam"] == 40.0
    note = (tmp_path / "n.md").read_text()
    assert "## Decision" in note and "projected" in note
    again = tune.tune(**args)
    assert again["shards"]["skipped"] and not again["shards"]["done"]
    with pytest.raises(ValueError):
        tune.tune(**{**args, "duration_s": 12.0})                         # changed configuration refused


def test_shadow_wm_n_real_runs_need_the_shadow_config(tune, tmp_path):
    fitted = tmp_path / "shadow" / "predictors.pkl"
    fitted.parent.mkdir()
    fitted.write_bytes(b"x")
    with pytest.raises(ValueError, match="config.json"):
        tune.shadow_wm_n(fitted, "abc", None, smoke=False)              # no shadow config
    (fitted.parent / "config.json").write_text(json.dumps({"wm_n": 16, "world_model_sha256": "abc"}))
    assert tune.shadow_wm_n(fitted, "abc", None, smoke=False) == 16
    with pytest.raises(ValueError, match="world model"):
        tune.shadow_wm_n(fitted, "def", None, smoke=False)              # another world model
    with pytest.raises(ValueError, match="wm"):
        tune.shadow_wm_n(fitted, "abc", 32, smoke=False)                # another number of futures
    (fitted.parent / "config.json").write_text(json.dumps({"world_model_sha256": "abc"}))
    with pytest.raises(ValueError, match="wm_n"):
        tune.shadow_wm_n(fitted, "abc", None, smoke=False)              # no silent default
    assert tune.shadow_wm_n(fitted, "def", None, smoke=True) == 32      # smoke: stand-ins allowed
    assert tune.shadow_wm_n(fitted, "def", 8, smoke=True) == 8
