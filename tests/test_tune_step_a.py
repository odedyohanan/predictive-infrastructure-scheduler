import json
import math
from pathlib import Path

import pytest

from tests._load import load_script

tune = load_script("tune_step_a")
GATE = "worldmodel:models/gru_mix_bounded.pt"


def _check(a, dirty=False, note_exists=False, config_exists=False, n_members=5):
    return tune.check_run(a, dirty=dirty, note_exists=note_exists, config_exists=config_exists,
                          n_members=n_members)


def _args(**kw):
    a = {**tune.DEFAULTS, "label": "", "out": "x/tuning.md", "config_out": "x/step_a.json"}
    a.update(kw)
    return a


def test_final_and_used_seeds_are_refused_even_in_smoke():
    for seeds in ("2000-2009", "2050", "1000-1010", "1099", "1200"):
        for label in ("", "smoke"):
            errs = _check(_args(seeds=seeds, label=label))
            assert any("development range" in e for e in errs), (seeds, label)
    errs = _check(_args(calib_seeds="2000"))
    assert any("development range" in e for e in errs)


def test_seed_overlap_grid_cap_and_members_checked():
    assert _check(_args()) == []
    assert _check(_args(calib_seeds="1110-1112"))
    assert _check(_args(max_configs=10))
    assert _check(_args(), n_members=3)                       # N=5 > 3 members
    assert _check(_args(), dirty=True)
    assert _check(_args(), note_exists=True)
    assert _check(_args(duration=6.0))


def test_non_smoke_never_overwrites_the_config():
    errs = _check(_args(out="elsewhere/note.md"), config_exists=True)
    assert any("configs/step_a.json" in e for e in errs)


def test_smoke_cannot_write_official_outputs():
    for key, path in (("out", tune.OFFICIAL_NOTE), ("config_out", tune.OFFICIAL_CONFIG)):
        errs = _check(_args(label="smoke", **{key: path}), dirty=True)
        assert any("smoke" in e for e in errs)
    assert _check(_args(label="smoke", duration=6.0), dirty=True, note_exists=True, config_exists=True) == []


def test_default_grid_size():
    assert len(tune.grid(tune.DEFAULTS)) == 48 <= tune.DEFAULTS["max_configs"]
    g = tune.grid({**tune.DEFAULTS, "grid_tau_q": "95,inf"})
    assert {c["tau_q"] for c in g} == {95.0, math.inf}


def _row(n, m, h, value, slo_hi, n_pairs=10, lo=None, tau=0.5, p95=5.0, mx=20.0):
    return {"cfg": {"n": n, "m": m, "h": h}, "params": {"tau": tau, "lam": 1.0},
            "cmp": {"n": n_pairs, "value": {"mean": value, "lo": value - 1.0 if lo is None else lo},
                    "slo_v2": {"hi": slo_hi}, "slo_v1": {"hi": 99.0}, "decide_p95_worst": p95, "decide_max": mx}}


def test_select_best_respects_slo_v2_constraint_and_ties():
    rows = [_row(5, 8, 5, 100.0, 1.5), _row(3, 4, 3, 50.0, 0.9), _row(5, 8, 5, 50.0, 0.0), _row(3, 4, 1, 10.0, 0.0)]
    best, feas, beats = tune.select_best(rows, 1.0, n_seeds=10)
    assert feas == [False, True, True, True]
    assert best == 1 and beats                        # tie on value: smaller N*M*H wins
    assert tune.select_best([_row(5, 8, 5, 100.0, 2.0)], 1.0, n_seeds=10)[0] is None
    # SLO v1 is reported only: a huge v1 upper bound does not make a config infeasible (set in _row)


def test_select_best_needs_all_seeds():
    rows = [_row(3, 4, 3, 90.0, 0.0, n_pairs=9), _row(3, 4, 3, 10.0, 0.0)]
    best, feas, _ = tune.select_best(rows, 1.0, n_seeds=10)
    assert feas == [False, True] and best == 1


def test_select_best_excludes_non_finite_tau_and_slow_configs():
    rows = [_row(3, 4, 3, 90.0, 0.0, tau=float("nan")), _row(3, 4, 3, 80.0, 0.0, p95=101.0),
            _row(3, 4, 3, 70.0, 0.0, mx=1001.0), _row(3, 4, 3, 60.0, 0.0, tau="inf"), _row(3, 4, 3, 5.0, 0.0)]
    best, feas, _ = tune.select_best(rows, 1.0, n_seeds=10)
    assert feas == [False, False, False, True, True] and best == 3      # "inf" = deliberate no-gate config
    # background-QoS gate: 2x the official 50 / 500 ms limits
    assert (tune.GATE_P95_MS, tune.GATE_MAX_MS) == (100.0, 1000.0)
    assert (tune.DECIDE_P95_MS, tune.DECIDE_MAX_MS) == (50.0, 500.0)
    assert tune.select_best([_row(3, 4, 3, 5.0, 0.0, p95=99.0, mx=999.0)], 1.0, n_seeds=10)[1] == [True]


def test_select_best_flags_no_gain():
    best, feas, beats = tune.select_best([_row(3, 4, 3, -5.0, 0.0)], 1.0, n_seeds=10)
    assert best == 0 and not beats
    best, _, beats = tune.select_best([_row(3, 4, 3, 5.0, 0.0, lo=-1.0)], 1.0, n_seeds=10)
    assert best == 0 and not beats                    # CI lower bound <= 0


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
    assert "background-QoS limits" in text and "time_decisions.py" in text
    assert str(Path.home()) not in text and str(tmp_path) not in text
    def _no_nan(x):
        raise ValueError(f"non-strict JSON constant {x}")
    c = json.loads(cfg.read_text(), parse_constant=_no_nan)
    assert c["members"] == [GATE, GATE] and set(c["params"]) >= {"n_candidates_m", "horizon_h", "lam", "tau"}
    assert isinstance(c["beats_base"], bool)
    if not c["beats_base"]:
        assert "No configuration beats the base scheduler" in text
    assert json.loads(raw.read_text())["best"] == res["best"]
