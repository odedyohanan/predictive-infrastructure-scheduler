import json
import math
from pathlib import Path

import pytest

from tests._load import load_script

tune = load_script("tune_step_a")
GATE = "worldmodel:models/gru_mix_bounded.pt"


def _args(**kw):
    a = {**tune.DEFAULTS, "label": "", "out": "x/tuning.md", "config_out": "x/step_a.json"}
    a.update(kw)
    return a


def test_final_and_used_seeds_are_refused_even_in_smoke():
    for seeds in ("2000-2009", "2050", "1000-1010", "1099", "1200"):
        for label in ("", "smoke"):
            errs = tune.check_run(_args(seeds=seeds, label=label), dirty=False, note_exists=False, n_members=5)
            assert any("development range" in e for e in errs), (seeds, label)
    errs = tune.check_run(_args(calib_seeds="2000"), dirty=False, note_exists=False, n_members=5)
    assert any("development range" in e for e in errs)


def test_seed_overlap_grid_cap_and_members_checked():
    assert tune.check_run(_args(), dirty=False, note_exists=False, n_members=5) == []
    assert tune.check_run(_args(calib_seeds="1110-1112"), dirty=False, note_exists=False, n_members=5)
    assert tune.check_run(_args(max_configs=10), dirty=False, note_exists=False, n_members=5)
    assert tune.check_run(_args(), dirty=False, note_exists=False, n_members=3)          # N=5 > 3 members
    assert tune.check_run(_args(), dirty=True, note_exists=False, n_members=5)
    assert tune.check_run(_args(), dirty=False, note_exists=True, n_members=5)
    assert tune.check_run(_args(duration=6.0), dirty=False, note_exists=False, n_members=5)


def test_smoke_cannot_write_official_outputs():
    for key, path in (("out", tune.OFFICIAL_NOTE), ("config_out", tune.OFFICIAL_CONFIG)):
        errs = tune.check_run(_args(label="smoke", **{key: path}), dirty=True, note_exists=False, n_members=5)
        assert any("smoke" in e for e in errs)
    assert tune.check_run(_args(label="smoke", duration=6.0), dirty=True, note_exists=True, n_members=5) == []


def test_default_grid_size():
    assert len(tune.grid(tune.DEFAULTS)) == 48 <= tune.DEFAULTS["max_configs"]
    g = tune.grid({**tune.DEFAULTS, "grid_tau_q": "95,inf"})
    assert {c["tau_q"] for c in g} == {95.0, math.inf}


def _row(n, m, h, value, slo_hi, n_pairs=10):
    return {"cfg": {"n": n, "m": m, "h": h}, "cmp": {"n": n_pairs, "value": {"mean": value},
                                                     "slo_v2": {"hi": slo_hi}}}


def test_select_best_respects_slo_constraint_and_ties():
    rows = [_row(5, 8, 5, 100.0, 1.5), _row(3, 4, 3, 50.0, 0.9), _row(5, 8, 5, 50.0, 0.0), _row(3, 4, 1, 10.0, 0.0)]
    best, feas = tune.select_best(rows, 1.0)
    assert feas == [False, True, True, True]
    assert best == 1                                  # tie on value: smaller N*M*H wins
    assert tune.select_best([_row(5, 8, 5, 100.0, 2.0)], 1.0)[0] is None


def test_calibrate_percentiles():
    pools = [{"3,1": [0.0, 1.0, 2.0]}, {"3,1": [3.0, 4.0, float("inf")]}]
    t = tune.calibrate(pools, [50.0, math.inf])
    assert t["3,1"][50.0] == pytest.approx(2.0) and math.isinf(t["3,1"][math.inf])


def test_tiny_end_to_end_run(tmp_path):
    out, cfg, raw = tmp_path / "t.md", tmp_path / "c.json", tmp_path / "r.json"
    res = tune.main(["--members", f"{GATE},{GATE}", "--calib-seeds", "1100", "--seeds", "1110,1111",
                     "--grid-n", "2", "--grid-m", "2", "--grid-h", "1", "--grid-lam", "0,1", "--grid-tau-q", "90",
                     "--slo-margin", "100", "--duration", "5", "--procs", "1", "--label", "smoke",
                     "--out", str(out), "--config-out", str(cfg), "--raw", str(raw)])
    assert len(res["rows"]) == 2 and res["best"] is not None
    text = out.read_text()
    assert "## Grid" in text and "**chosen**" in text and "## Ablations" in text and "oracle" in text
    assert "_qs" not in text
    assert str(Path.home()) not in text and str(tmp_path) not in text
    c = json.loads(cfg.read_text())
    assert c["members"] == [GATE, GATE] and set(c["params"]) >= {"n_candidates_m", "horizon_h", "lam", "tau"}
    assert json.loads(raw.read_text())["best"] == res["best"]
