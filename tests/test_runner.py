import getpass
import json
import math
import socket
from pathlib import Path

import pytest

import pis.runner as runner
from tests._load import load_script

fe = load_script("final_eval")
GATE = "worldmodel:models/gru_mix_bounded.pt"
PARAMS = {"n_candidates_m": 2, "horizon_h": 1, "lam": 1.0, "mu": 2.0, "change_cost": 0.0, "tau": 0.5,
          "slo_lam": 10.0}
REVISER = {"mechanism": "gradient", "rev_steps": 1, "rev_lam": 1.0, "rev_lr": 0.6}


def _m(value, slo2=0.0, **kw):
    m = {"value_per_s": value, "slo_v2_pct": slo2, "slo_v1_pct": 0.0, "a_late_pct": 0.0, "a_queue_overdue_end": 0,
         "policy_changes": 1, "decide_ms_mean": 1.0, "decide_ms_p95": 2.0, "decide_ms_max": 3.0,
         "decisions_over_500ms": 0, "fallback_rate": math.nan, "gru_error": math.nan,
         "rank_agreement": math.nan, "top1_agreement": math.nan}
    m.update(kw)
    return m


def _ok(arm, seed, value, **kw):
    return {"seed": seed, "arm": arm, "failed": False, "metrics": _m(value, **kw), "seconds": 1.0}


# ------------------------------------------------------------------ runner
def test_failed_run_recorded_with_scrubbed_message(monkeypatch):
    secret = str(Path.home() / "private" / "member.pt")

    def boom(*a, **k):
        raise RuntimeError(f"cannot load {secret}")
    monkeypatch.setattr(runner, "make_controller", boom)
    r = runner.run_arm("base_r6", 1150, {"base_r6": {"kind": "base", "checkpoint": "x"}}, 1.0)
    assert r["failed"] is True and r["seed"] == 1150 and r["arm"] == "base_r6"
    assert "RuntimeError" in r["error"] and "<path>" in r["error"]
    assert str(Path.home()) not in r["error"]


def test_failed_seed_not_dropped_from_report():
    raw = [_ok("base_r6", s, 100.0) for s in (1, 2, 3)] + [_ok("loop_a", 1, 110.0), _ok("loop_a", 3, 112.0),
                                                           {"seed": 2, "arm": "loop_a", "failed": True,
                                                            "error": "ValueError: boom", "seconds": 0.1}]
    text = runner.report(raw, seeds=[1, 2, 3])
    assert "FAILED loop_a seed 2: ValueError: boom" in text
    assert "5 succeeded, 1 failed of 6" in text
    assert "Measured under background QoS" in text and "experiments/time_decisions.py" in text
    row = next(line for line in text.splitlines() if line.startswith("| 2 |") and "/" in line)
    assert "FAILED" in row
    assert runner.paired(runner.per_arm(raw), "loop_a", "base_r6", "value_per_s")["n"] == 2


def test_report_pairs_by_seed():
    # arm = reference + 10 in every scenario; the records arrive in a different order per arm
    base = {s: 100.0 + 37.0 * s for s in range(1, 6)}
    raw = [_ok("base_r6", s, v) for s, v in base.items()] + [_ok("loop_a", s, base[s] + 10.0) for s in (5, 3, 1, 4, 2)]
    d = runner.paired(runner.per_arm(raw), "loop_a", "base_r6", "value_per_s")
    assert d["n"] == 5 and d["mean"] == pytest.approx(10.0) and d["sd"] == pytest.approx(0.0)
    assert d["seeds"] == [1, 2, 3, 4, 5]
    # unpaired seeds are left out, not matched by position
    raw.append(_ok("loop_a", 9, 1e6))
    assert runner.paired(runner.per_arm(raw), "loop_a", "base_r6", "value_per_s")["n"] == 5


def test_success_rule():
    rule = {"reference": "base_r6", "confidence": 0.95, "slo_v2_margin_pp": 1.0}
    good = {"worst_p95": 10.0, "worst_max": 100.0, "p95_ok": True, "max_ok": True, "source": "x"}
    raw = [_ok("base_r6", s, 100.0 + s, slo2=1.0) for s in range(10)]
    raw += [_ok("loop_a", s, 105.0 + s + (s % 3) * 0.1, slo2=1.0 + (s % 2) * 0.5) for s in range(10)]
    per = runner.per_arm(raw)
    s = runner.success(per, "loop_a", rule, good)
    assert s["value_ok"] and s["slo_ok"] and s["time_ok"] and s["met"]
    assert not runner.success(per, "loop_a", rule, None)["met"]          # no timing note: not met
    assert not runner.success(per, "loop_a", {**rule, "slo_v2_margin_pp": 0.2}, good)["met"]
    assert not runner.success(per, "loop_a", rule, {**good, "p95_ok": False})["met"]
    worse = [_ok("base_r6", s, 100.0) for s in range(10)] + [_ok("loop_a", s, 100.0 - s % 2) for s in range(10)]
    assert not runner.success(runner.per_arm(worse), "loop_a", rule, good)["value_ok"]


def test_spearman():
    assert runner.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert runner.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(runner.spearman([1, 2, 3], [5, 5, 5]))
    assert math.isnan(runner.spearman([1, 2], [1, 2]))
    assert runner.spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, abs=1e-6)


def test_probe_metrics():
    recs = [{"info": {}}, {"info": {"probe": {"gru_error": 0.2, "rank_agreement": 0.5, "top1": True}}},
            {"info": {"probe": {"gru_error": 0.4, "rank_agreement": math.nan, "top1": False}}}]
    m = runner.probe_metrics(recs)
    assert m["probes"] == 2 and m["gru_error"] == pytest.approx(0.3) and m["rank_agreement"] == 0.5
    assert m["rank_probes"] == 1 and m["top1_agreement"] == 0.5
    assert math.isnan(runner.probe_metrics([{"info": {}}])["gru_error"])


def test_arm_definitions():
    defs = runner.arm_definitions(fe.make_protocol(), {"members": [GATE], "params": PARAMS},
                                  {"members": [GATE], "step_a_params": PARAMS, "reviser": REVISER})
    assert tuple(defs) == runner.ARMS
    assert defs["loop_no_uncertainty"]["options"] == {"use_uncertainty": False}
    assert defs["loop_no_fallback"]["options"] == {"use_fallback": False}
    assert defs["oracle_selection"]["options"] == {"selection": "oracle"}
    assert defs["loop_a"]["params"] == PARAMS and defs["loop_b"]["reviser"] == REVISER
    assert defs["adaptive_c9"] == {"kind": "adaptive", "h0": 2.0, "r_hi": 0.6, "hold": 6}


# ------------------------------------------------------------------ protocol and guards
def test_protocol():
    p = fe.make_protocol()
    assert fe.validate_protocol(p) == []
    assert fe.protocol_seeds(p) == list(range(2000, 2100)) and p["duration_s"] == 60.0
    assert p["success_rule"]["slo_v2_margin_pp"] == 1.0 and p["success_rule"]["confidence"] == 0.95
    assert p["decision_time_ms"] == {"p95_target": 50.0, "hard_limit": 500.0}
    assert "mpc_v2" not in p["arms"]
    from pis.config import setup_imports
    setup_imports()
    from sched.controllers import ADAPTIVE_GRID
    assert {k: p["arms"]["adaptive_c9"][k] for k in ("h0", "r_hi", "hold")} == ADAPTIVE_GRID[9]
    assert fe.validate_protocol({**p, "arms": {k: v for k, v in p["arms"].items() if k != "base_r3"}})


def test_committed_protocol_is_the_generated_one():
    path = Path(fe.__file__).resolve().parent.parent / fe.PROTOCOL
    if not path.exists():
        pytest.skip("protocol not frozen yet")
    assert json.loads(path.read_text()) == fe.make_protocol()


def _state(**kw):
    s = {"dirty": False, "committed": {fe.PROTOCOL: True, "configs/step_a.json": True, "configs/step_b.json": True},
         "report_text": "", "out_exists": False, "raw_exists": False, "prev_raw_exists": False}
    s.update(kw)
    return s


def _args(**kw):
    a = {"seeds_list": list(range(2000, 2100)), "label": "", "duration": 60.0, "out": fe.OFFICIAL_REPORT,
         "raw_dir": fe.OFFICIAL_RAW, "attempt": 1}
    a.update(kw)
    return a


def _check(args, state=None):
    return fe.check_run(args, fe.make_protocol(), state or _state())


def test_official_guards():
    assert _check(_args()) == []
    assert _check(_args(seeds_list=list(range(2000, 2050))))
    assert _check(_args(label="rehearsal"))
    assert _check(_args(duration=10.0))
    assert _check(_args(priority="normal"))
    assert _check(_args(), _state(dirty=True))
    for path in (fe.PROTOCOL, "configs/step_a.json", "configs/step_b.json"):
        c = {**_state()["committed"], path: False}
        assert any(path in e for e in _check(_args(), _state(committed=c)))
    assert _check(_args(out="elsewhere/report.md"))
    assert _check(_args(raw_dir="elsewhere/raw"))
    assert _check(_args(), _state(raw_exists=True))
    assert _check(_args(), _state(report_text="# x\n\n## Attempt 1\n"))
    assert _check(_args(attempt=2))                                     # no attempt 1 yet
    assert _check(_args(attempt=2), _state(prev_raw_exists=True, report_text="## Attempt 1\n")) == []
    assert _check(_args(attempt=4), _state(prev_raw_exists=True))


def test_used_seeds_never_accepted():
    for seeds in ([1000], [1099], list(range(1050, 1060))):
        for label in ("", "rehearsal", "smoke"):
            errs = _check(_args(seeds_list=seeds, label=label, out="o/r.md", raw_dir="o/raw"))
            assert any("never used" in e for e in errs), (seeds, label)


def test_rehearsal_guards():
    ok = _args(seeds_list=[1150, 1151], label="rehearsal", duration=10.0, out="scratch/r.md", raw_dir="scratch/raw")
    assert _check(ok, _state(dirty=True, committed={})) == []
    assert any("label" in e for e in _check({**ok, "label": ""}))
    assert _check({**ok, "seeds_list": [1200]})
    assert _check({**ok, "seeds_list": [1150, 2000]})                  # mixing in held-out seeds
    assert _check({**ok, "out": fe.OFFICIAL_REPORT})
    assert _check({**ok, "raw_dir": fe.OFFICIAL_RAW})
    assert _check(ok, _state(out_exists=True))
    assert _check(ok, _state(raw_exists=True))


def test_read_timing(tmp_path):
    note = tmp_path / "timing.md"
    note.write_text("# Decision time, Step A at normal priority\n\n- worst per-scenario p95 12.5 ms: within the "
                    "50 ms limit\n- worst max 612.0 ms: **over** the 500 ms limit\n")
    t = fe.read_timing(str(note))
    assert t["worst_p95"] == 12.5 and t["p95_ok"] and not t["max_ok"] and t["worst_max"] == 612.0
    assert fe.read_timing(str(tmp_path / "missing.md")) is None


# ------------------------------------------------------------------ end to end on development seeds
@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("final")
    proto = fe.make_protocol()
    proto["metrics"]["probe_every"] = 2
    (d / "protocol.json").write_text(json.dumps(proto))
    (d / "step_a.json").write_text(json.dumps({"members": [GATE, GATE], "params": PARAMS, "label": "smoke"}))
    (d / "step_b.json").write_text(json.dumps({"members": [GATE, GATE], "step_a_params": PARAMS,
                                               "reviser": REVISER, "label": "smoke"}))
    res = fe.main(["--protocol", str(d / "protocol.json"), "--step-a-config", str(d / "step_a.json"),
                   "--step-b-config", str(d / "step_b.json"), "--seeds", "1150", "--duration", "7",
                   "--procs", "1", "--label", "rehearsal", "--out", str(d / "report.md"),
                   "--raw-dir", str(d / "raw"), "--priority", "normal"])
    return d, res


def test_dry_run_on_dev_seeds_produces_all_arms(dry_run):
    d, res = dry_run
    raw = res["raw"]
    assert sorted(r["arm"] for r in raw) == sorted(runner.ARMS)
    assert not any(r["failed"] for r in raw), [r.get("error") for r in raw]
    lines = (d / "raw" / "attempt1" / "runs.jsonl").read_text().splitlines()
    assert len(lines) == len(runner.ARMS)
    text = (d / "report.md").read_text()
    for s in ("## Attempt 1 (rehearsal)", "### Success rule", "Measured at normal priority",
              "### Paired differences vs `base_r6`", "### Secondary comparisons"):
        assert s in text
    loop = next(r for r in raw if r["arm"] == "loop_a")["metrics"]
    assert loop["probes"] >= 1 and math.isfinite(loop["gru_error"])
    with pytest.raises(SystemExit):                                   # never overwrites
        fe.main(["--protocol", str(d / "protocol.json"), "--step-a-config", str(d / "step_a.json"),
                 "--step-b-config", str(d / "step_b.json"), "--seeds", "1150", "--duration", "7", "--procs", "1",
                 "--label", "rehearsal", "--out", str(d / "report.md"), "--raw-dir", str(d / "raw")])


def test_run_config_has_no_absolute_paths_or_hostnames(dry_run):
    d, _ = dry_run
    texts = [(d / "raw" / "attempt1" / "run_config.json").read_text(), (d / "report.md").read_text(),
             (d / "raw" / "attempt1" / "runs.jsonl").read_text()]
    host, user = socket.gethostname().split(".")[0], getpass.getuser()
    for t in texts:
        assert str(Path.home()) not in t and str(d) not in t and "/" + "Users/" not in t
        assert host not in t and user not in t
    cfg = json.loads(texts[0])
    assert cfg["protocol"]["path"].startswith("<abs>/") and len(cfg["protocol"]["sha256"]) == 64
    assert all(len(v) == 64 for v in cfg["models_sha256"].values()) and "base_r6 sched:models/rl_r6/final.pt" in \
        cfg["models_sha256"]
