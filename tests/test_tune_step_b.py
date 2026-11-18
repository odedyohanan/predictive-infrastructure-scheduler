import json
import math
from pathlib import Path

import pytest

from tests._load import load_script

tune = load_script("tune_step_b")
GATE = "worldmodel:models/gru_mix_bounded.pt"


def _check(a, dirty=False, note_exists=False, config_exists=False):
    return tune.check_run(a, dirty=dirty, note_exists=note_exists, config_exists=config_exists)


def _args(**kw):
    a = {**tune.DEFAULTS, "label": "", "out": "x/report.md", "config_out": "x/step_b.json"}
    a.update(kw)
    return a


def test_final_and_used_seeds_are_refused_even_in_smoke():
    for seeds in ("2000-2009", "2050", "1000-1010", "1099", "1200", "1150-1205"):
        for label in ("", "smoke"):
            errs = _check(_args(seeds=seeds, label=label))
            assert any("development range" in e for e in errs), (seeds, label)


def test_guards():
    assert _check(_args()) == []
    assert _check(_args(max_configs=7))                       # default grid has 8 configs
    assert _check(_args(), dirty=True)
    assert _check(_args(), note_exists=True)
    assert any("configs/step_b.json" in e for e in _check(_args(), config_exists=True))
    assert _check(_args(duration=6.0))
    assert _check(_args(grid_rev_lam="0,1"))                  # the penalty may never be off
    assert _check(_args(grid_mech="sampling"))


def test_smoke_cannot_write_official_outputs():
    for key, path in (("out", tune.OFFICIAL_NOTE), ("config_out", tune.OFFICIAL_CONFIG)):
        errs = _check(_args(label="smoke", **{key: path}), dirty=True)
        assert any("smoke" in e for e in errs)
    assert _check(_args(label="smoke", duration=6.0), dirty=True, note_exists=True, config_exists=True) == []


def test_default_grid():
    g = tune.grid(tune.DEFAULTS)
    assert len(g) == 8 <= tune.DEFAULTS["max_configs"]
    assert {c["mechanism"] for c in g} == {"gradient", "neighbourhood"}
    assert all(c["rev_lam"] > 0 for c in g)
    assert tune.OFFICIAL_NOTE == "results/step_b/report.md"


def _row(value, slo_hi, mech="gradient", steps=2, n=10, lo=None, p95=5.0, mx=20.0, va=1.0, va_lo=0.5):
    return {"cfg": {"mechanism": mech, "steps": steps, "rev_lam": 1.0},
            "vs_base": {"n": n, "value": {"mean": value, "lo": value - 1.0 if lo is None else lo},
                        "slo_v2": {"hi": slo_hi}, "decide_p95_worst": p95, "decide_max": mx},
            "vs_step_a": {"n": n, "value": {"mean": va, "lo": va_lo}}}


def test_select_best():
    rows = [_row(100.0, 1.5), _row(50.0, 0.9, steps=4), _row(50.0, 0.0, steps=2), _row(90.0, 0.0, p95=51.0),
            _row(80.0, 0.0, mx=501.0), _row(70.0, 0.0, n=9)]
    best, feas, beats, beats_a = tune.select_best(rows, 1.0, n_seeds=10)
    assert feas == [False, True, True, False, False, False]
    assert best == 2 and beats and beats_a                    # tie on value: fewer steps
    _, _, beats, beats_a = tune.select_best([_row(5.0, 0.0, lo=-1.0, va=-1.0)], 1.0, n_seeds=10)
    assert not beats and not beats_a
    assert tune.select_best([_row(5.0, 2.0)], 1.0, n_seeds=10)[0] is None


def test_revision_stats_and_exploit_rate():
    cons = {"a_p99_ms": -1.0, "a_queue_work_ms": 0.0, "b_rate": 2.0, "c_work_core_ms": 0.0, "text": ""}
    def rec(**i):
        return {"info": i}
    recs = [rec(rev_trigger=None, revised=False),
            rec(rev_trigger="slo", revised=False, rev_ms=3.0),
            rec(rev_trigger="untrusted", revised=True, rev_ms=4.0, rev_pred_reward_gain=0.5, rev_real_gain=-0.1,
                consequences=cons),
            rec(rev_trigger="slo", revised=True, rev_ms=5.0, rev_pred_reward_gain=0.5, rev_real_gain=0.3,
                consequences=cons)]
    s = tune.revision_stats(recs)
    assert (s["triggers"], s["triggers_slo"], s["triggers_untrusted"], s["revised"], s["exploited"]) == (3, 2, 1, 2, 1)
    assert s["rev_ms_max"] == 5.0 and s["consequences_mean"]["b_rate"] == 2.0
    agg = tune.aggregate_revisions([s, tune.revision_stats(recs[:2])])
    assert agg["exploit_rate"] == pytest.approx(0.5) and agg["real_gain_mean"] == pytest.approx(0.1)
    assert math.isnan(tune.aggregate_revisions([tune.revision_stats(recs[:2])])["exploit_rate"])


def test_tiny_end_to_end_run(tmp_path):
    step_a = tmp_path / "step_a.json"
    step_a.write_text(json.dumps({"members": [GATE, GATE], "beats_base": False,
                                  "params": {"n_candidates_m": 2, "horizon_h": 1, "lam": 1.0, "mu": 2.0,
                                             "change_cost": 0.0, "tau": "inf", "slo_lam": 10.0}}))
    out, cfg, raw = tmp_path / "r.md", tmp_path / "c.json", tmp_path / "r.json"
    res = tune.main(["--step-a-config", str(step_a), "--seeds", "1150,1151", "--grid-steps-gradient", "1",
                     "--grid-steps-neighbourhood", "1", "--grid-rev-lam", "1", "--slo-margin", "100",
                     "--duration", "4", "--procs", "1", "--label", "smoke", "--out", str(out),
                     "--config-out", str(cfg), "--raw", str(raw)])
    assert len(res["rows"]) == 2 and res["best"] is not None and res["mechanisms"] is not None
    text = out.read_text()
    for s in ("## Grid", "**chosen**", "## Model exploitation", "## Gradient vs neighbourhood search",
              "Step A loop vs base"):
        assert s in text
    assert "_rev_lr" not in text
    assert str(Path.home()) not in text and str(tmp_path) not in text
    def _no_nan(x):
        raise ValueError(f"non-strict JSON constant {x}")
    c = json.loads(cfg.read_text(), parse_constant=_no_nan)
    assert c["members"] == [GATE, GATE] and c["reviser"]["rev_lam"] > 0
    assert isinstance(c["beats_base"], bool) and isinstance(c["beats_step_a"], bool)
    assert c["step_a_config"].startswith("<abs>/")
    assert json.loads(raw.read_text())["best"] == res["best"]


def test_missing_step_a_config_refuses(tmp_path):
    with pytest.raises(SystemExit):
        tune.main(["--step-a-config", str(tmp_path / "nope.json"), "--label", "smoke", "--duration", "4",
                   "--out", str(tmp_path / "r.md"), "--config-out", str(tmp_path / "c.json")])
