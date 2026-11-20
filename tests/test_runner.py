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


def test_success_needs_complete_runs():
    rule = {"reference": "base_r6", "confidence": 0.95, "slo_v2_margin_pp": 1.0}
    good = {"worst_p95": 10.0, "worst_max": 100.0, "p95_ok": True, "max_ok": True, "source": "x"}
    raw = [_ok("base_r6", s, 100.0 + s, slo2=1.0) for s in range(10)]
    raw += [_ok("loop_a", s, 105.0 + s + (s % 3) * 0.1, slo2=1.0) for s in range(7)]
    per = runner.per_arm(raw)
    s = runner.success(per, "loop_a", rule, good, seeds=list(range(10)))
    assert not s["met"] and s["value"]["n"] == 7 and "not decidable" in s["undecidable"]
    assert s["value_ok"] and s["slo_ok"]                                 # paired stats still reported
    ref_short = runner.per_arm([r for r in raw if not (r["arm"] == "base_r6" and r["seed"] == 9)])
    assert not runner.success(ref_short, "loop_a", rule, good, seeds=list(range(7)) + [9])["met"]
    assert runner.success(per, "loop_a", rule, good, seeds=list(range(7)))["met"]
    text = runner.report(raw, fe.make_protocol(), timing={"loop_a": good}, seeds=list(range(10)))
    assert "not decidable" in text and "met the rule" not in text


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
         "report_text": "", "out_exists": False, "raw_exists": False, "prev_raw_exists": False,
         "freeze": {"commits": ["c" * 40], "files": [fe.PROTOCOL]}}
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
    assert _check(_args(raw_dir=fe.OFFICIAL_RAW + "/alt"))              # exact paths, not subdirectories
    assert _check(_args(out="results/final/other/report.md"))
    assert _check(_args(), _state(raw_exists=True))
    assert _check(_args(), _state(report_text="# x\n\n## Attempt 1\n"))
    assert _check(_args(attempt=2))                                     # no attempt 1 yet
    assert _check(_args(attempt=2), _state(prev_raw_exists=True, report_text="## Attempt 1\n")) == []
    assert _check(_args(attempt=4), _state(prev_raw_exists=True))


def test_official_reads_the_frozen_protocol_and_its_configs():
    assert _check(_args(protocol=fe.PROTOCOL, step_a_config="configs/step_a.json",
                        step_b_config="configs/step_b.json")) == []
    assert any("--protocol" in e for e in _check(_args(protocol="other/final_eval.json")))
    assert any("--step-a-config" in e for e in _check(_args(step_a_config="other/step_a.json")))
    assert any("--step-b-config" in e for e in _check(_args(step_b_config="configs/step_a.json")))
    for freeze in ({"commits": [], "files": []}, {"commits": ["a" * 40, "b" * 40], "files": [fe.PROTOCOL]},
                   {"commits": ["a" * 40], "files": [fe.PROTOCOL, "pis/runner.py"]}):
        assert any("exactly one commit" in e for e in _check(_args(), _state(freeze=freeze)))
    # rehearsals may use scratch protocols and configs
    ok = _args(seeds_list=[1150], label="rehearsal", duration=10.0, out="s/r.md", raw_dir="s/raw",
               protocol="s/p.json", step_a_config="s/a.json")
    assert _check(ok, _state(freeze={"commits": [], "files": []})) == []


def test_freeze_state_of_this_repo():
    st = fe.freeze_state()
    if not st["commits"]:
        pytest.skip("protocol not frozen yet")
    assert len(st["commits"]) == 1 and st["files"] == [fe.PROTOCOL]


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


SHA = {GATE: "a" * 64, "models/ens_1.pt": "b" * 64}


def _timing_note(kind="A", label="", params=PARAMS, reviser=None, members=((GATE, "a" * 16),), p95="within"):
    rev = f"; reviser {json.dumps(reviser)}" if reviser else ""
    return (f"# Decision time, Step {kind} at normal priority{' (' + label + ')' if label else ''}\n\n"
            "- code: git HEAD abc\n- config: configs/step_x.json\n- members: "
            + ", ".join(f"{m} (sha256 {h})" for m, h in members) + "\n"
            f"- controller params: {json.dumps(params)}{rev}\n\n"
            f"- worst per-scenario p95 12.5 ms: {p95} the 50 ms limit\n- worst max 99.0 ms: within the 500 ms limit\n")


def test_timing_counts_only_for_the_evaluated_config(tmp_path):
    a_cfg = {"members": [GATE], "params": PARAMS}
    b_cfg = {"members": [GATE], "step_a_params": PARAMS, "reviser": REVISER}
    note = tmp_path / "t.md"

    def tf(text, kind="a", cfg=a_cfg, official=True, committed=True):
        note.write_text(text)
        return fe.timing_for(str(note), kind, cfg, SHA, official, committed)
    t = tf(_timing_note())
    assert "unavailable" not in t and t["p95_ok"] and t["max_ok"]
    assert "unavailable" not in tf(_timing_note("B", reviser=REVISER), "b", b_cfg)
    assert "not committed" in tf(_timing_note(), committed=False)["unavailable"]
    assert "unavailable" not in tf(_timing_note(), official=False, committed=False)
    assert "labelled 'smoke'" in tf(_timing_note(label="smoke"))["unavailable"]
    assert "params differ" in tf(_timing_note(params={**PARAMS, "horizon_h": 3}))["unavailable"]
    assert "Step A timing note" in tf(_timing_note("B", reviser=REVISER))["unavailable"]
    assert "reviser differs" in tf(_timing_note("B", reviser={**REVISER, "rev_steps": 2}), "b", b_cfg)["unavailable"]
    assert "differs from the recorded" in tf(_timing_note(members=((GATE, "f" * 16),)))["unavailable"]
    assert "missing" in fe.timing_for(str(tmp_path / "none.md"), "a", a_cfg, SHA, True, True)["unavailable"]
    # the report shows the reason and does not count part (c)
    raw = [_ok("base_r6", s, 100.0) for s in range(3)] + [_ok("loop_a", s, 110.0 + s) for s in range(3)]
    text = runner.report(raw, fe.make_protocol(), timing={"loop_a": {"unavailable": "x.md: labelled 'smoke'"}},
                         seeds=[0, 1, 2])
    assert "not available: x.md: labelled 'smoke'" in text


def test_member_prefixes_against_tuning_notes():
    tuning = "# Step A tuning\n\n- code: git HEAD x\n- members: " + ", ".join(
        f"{m} (sha256 {h[:16]})" for m, h in SHA.items()) + "\n- calibration seeds: 1100-1109\n"
    assert fe.recorded_members(tuning) == [(GATE, "a" * 16), ("models/ens_1.pt", "b" * 16)]
    assert fe.member_mismatch(tuning, [GATE], SHA, exact=False) is None          # Step A config: first N
    assert fe.member_mismatch(tuning, [GATE], SHA, exact=True)
    assert "differs" in fe.member_mismatch(tuning, [GATE], {GATE: "c" * 64}, exact=False)
    assert "not recorded" in fe.member_mismatch(tuning, ["models/ens_9.pt"], {"models/ens_9.pt": "d" * 64}, False)
    assert fe.member_mismatch("# no members\n", [GATE], SHA, exact=False) == "no member sha256 recorded"


def test_replace_section_and_attempt_summary(tmp_path):
    text = "# T\n\n## Attempt 1\n\nold one\n\n## Attempt 2 (started)\n\nstub\n\n## Attempt 3\n\nthree\n"
    out = fe.replace_section(text, 2, "## Attempt 2\n\nfull\n\n")
    assert "stub" not in out and "old one" in out and "three" in out and "full" in out
    assert fe.replace_section("# T\n\n", 1, "## Attempt 1\n") == "# T\n\n## Attempt 1\n"
    d = tmp_path / "attempt1"
    d.mkdir()
    (d / "run_config.json").write_text(json.dumps({"n_seeds": 2, "arms": {"x": 1, "y": 1}}))
    (d / "runs.jsonl").write_text(json.dumps({"failed": False}) + "\n" + json.dumps({"failed": True}) + "\n")
    assert fe.summarise_attempt(d) == "2 of 4 runs recorded (1 ok, 1 failed), did not complete"


def test_short_n_warning():
    raw = [_ok("base_r6", s, 100.0) for s in range(4)] + [_ok("loop_a", s, 101.0) for s in range(3)]
    raw.append({"seed": 3, "arm": "loop_a", "failed": True, "error": "E: x", "seconds": 0.0})
    text = runner.report(raw, fe.make_protocol(), seeds=[0, 1, 2, 3])
    assert "fewer scenarios than the protocol's" in text and "loop_a: n = 3 of 4" in text
    assert "loop_a 1 failed or missing" in text and "base_r6 0 failed or missing" in text
    full = runner.report([r for r in raw if r["seed"] < 3], fe.make_protocol(), seeds=[0, 1, 2])
    assert "loop_a: n =" not in full and "loop_b: n = 0 of 3" in full      # loop_b has no runs at all


def test_gate_fallback_rate():
    recs = [{"info": {"reason": r}} for r in ("short_history", "short_history", "model", "untrusted", "no_finite",
                                              "revised", "model", "model")]
    assert runner.gate_fallback_rate(recs) == pytest.approx(2 / 8)
    assert math.isnan(runner.gate_fallback_rate([{"info": {"decide_ms": 1.0}}]))


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
    assert "(started)" not in text and text.count("## Attempt 1") == 1
    assert "gate fallback rate" in text and "not available: " in text      # scratch configs have no timing note
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


def test_crashed_attempt_leaves_its_section_and_is_summarised_by_the_next(tmp_path, monkeypatch):
    proto = tmp_path / "protocol.json"
    proto.write_text(json.dumps(fe.make_protocol()))
    (tmp_path / "a.json").write_text(json.dumps({"members": [GATE], "params": PARAMS}))
    (tmp_path / "b.json").write_text(json.dumps({"members": [GATE], "step_a_params": PARAMS, "reviser": REVISER}))
    monkeypatch.setattr(fe, "_init_worker", lambda *a: fe._W.update(arms=a[0], duration=a[1], probe_every=a[2]))
    calls = []

    def fake(arm, seed, arms, duration, probe_every):
        calls.append(arm)
        if len(calls) == 3 and "crashed" not in calls:
            calls.append("crashed")
            raise RuntimeError("worker died")
        return _ok(arm, seed, 100.0)
    monkeypatch.setattr(fe, "run_arm", fake)
    common = ["--protocol", str(proto), "--step-a-config", str(tmp_path / "a.json"), "--step-b-config",
              str(tmp_path / "b.json"), "--seeds", "1150", "--label", "rehearsal", "--procs", "1",
              "--raw-dir", str(tmp_path / "raw")]
    with pytest.raises(RuntimeError):
        fe.main(common + ["--out", str(tmp_path / "r1.md")])
    first = (tmp_path / "r1.md").read_text()
    assert "## Attempt 1 (rehearsal) (started)" in first and "has not finished" in first
    fe.main(common + ["--out", str(tmp_path / "r2.md"), "--attempt", "2"])
    second = (tmp_path / "r2.md").read_text()
    assert "## Attempt 2 (rehearsal)" in second and "(started)" not in second
    assert "previous attempt: attempt 1: 2 of 9 runs recorded (2 ok, 0 failed), did not complete" in second


def test_attempts_are_pinned_to_the_previous_configuration():
    prev = {"protocol": {"sha256": "p" * 64},
            "step_configs": {"step_a": {"sha256": "a" * 64}, "step_b": {"sha256": "b" * 64}},
            "models_sha256": {"member m1": "1" * 64, "base_r6 x": "2" * 64}}
    assert fe.pinned_mismatch(prev, prev) == []
    cur = json.loads(json.dumps(prev))
    cur["step_configs"]["step_b"]["sha256"] = "c" * 64
    cur["models_sha256"]["member m1"] = "3" * 64
    cur["code"] = "git HEAD other"                                         # a code fix is allowed
    bad = fe.pinned_mismatch(prev, cur)
    assert len(bad) == 2 and any("step_b" in b for b in bad) and any("member m1" in b for b in bad)
    assert fe.pinned_mismatch({}, prev)                                    # nothing recorded: refuse
    assert fe.pinned_mismatch(None, prev)


def test_official_timing_checked_before_start(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "TIMING_NOTES", {"loop_a": str(tmp_path / "a" / "timing.md"),
                                             "loop_b": str(tmp_path / "b" / "timing.md")})
    steps = {"step_a": {"members": ["m"], "params": {}}, "step_b": {"members": ["m"], "reviser": {}}}
    bad = fe.timing_refusals(steps, {"m": "a" * 64}, lambda path: False)
    assert len(bad) == 2 and all("unavailable" in b or "missing" in b for b in bad)
