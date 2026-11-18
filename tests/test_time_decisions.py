import json
from pathlib import Path

import pytest

from tests._load import load_script

td = load_script("time_decisions")
GATE = "worldmodel:models/gru_mix_bounded.pt"
PARAMS = {"n_candidates_m": 2, "horizon_h": 1, "lam": 1.0, "mu": 2.0, "change_cost": 0.0, "tau": "inf",
          "slo_lam": 10.0}


def _args(**kw):
    a = {"seeds": "1100-1102", "duration": 60.0, "label": "", "out": "x/timing.md"}
    a.update(kw)
    return a


def test_guards():
    assert td.check_run(_args(), dirty=False, note_exists=False) == []
    for seeds in ("2000", "1099", "1200", "1100-1205"):
        for label in ("", "smoke"):
            assert any("development range" in e for e in td.check_run(_args(seeds=seeds, label=label), False, False))
    assert td.check_run(_args(), dirty=True, note_exists=False)
    assert td.check_run(_args(), dirty=False, note_exists=True)
    assert td.check_run(_args(duration=5.0), dirty=False, note_exists=False)
    assert td.check_run(_args(label="smoke", duration=5.0), dirty=True, note_exists=True) == []
    for official in td.NOTES.values():
        assert td.check_run(_args(label="smoke", out=official), False, False)


def test_load_config_kinds(tmp_path):
    a, b, bad = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "bad.json"
    a.write_text(json.dumps({"members": [GATE], "params": PARAMS}))
    b.write_text(json.dumps({"members": [GATE], "step_a_params": PARAMS,
                             "reviser": {"mechanism": "gradient", "rev_steps": 1, "rev_lam": 1.0, "rev_lr": 0.6}}))
    bad.write_text('{"members": ["x"], "params": {"tau": NaN}}')
    assert td.load_config(str(a))["kind"] == "a" and td.load_config(str(b))["kind"] == "b"
    with pytest.raises(ValueError):
        td.load_config(str(bad))


def test_verdict():
    v = td.verdict({1: {"p95": 40.0, "max": 100.0}, 2: {"p95": 55.0, "max": 300.0}})
    assert v["worst_p95"] == 55.0 and not v["p95_ok"] and v["max_ok"]


@pytest.mark.parametrize("kind", ["a", "b"])
def test_tiny_run(tmp_path, kind):
    cfg = tmp_path / "c.json"
    body = {"members": [GATE, GATE]}
    if kind == "a":
        body["params"] = PARAMS
    else:
        body.update(step_a_params=PARAMS, reviser={"mechanism": "gradient", "rev_steps": 1, "rev_lam": 1.0,
                                                   "rev_lr": 0.6})
    cfg.write_text(json.dumps(body))
    out = tmp_path / "t.md"
    res = td.main(["--step", kind, "--config", str(cfg), "--seeds", "1100,1101", "--duration", "5",
                   "--label", "smoke", "--out", str(out)])
    assert set(res["per_seed"]) == {1100, 1101}
    text = out.read_text()
    assert "Official limits" in text and "| all |" in text and "sha256" in text
    assert str(Path.home()) not in text and str(tmp_path) not in text
