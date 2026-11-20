import json
import pickle

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis.phase_b import incidents, predictors  # noqa: E402
from pis.phase_b.incidents import IncidentRule  # noqa: E402
from tests._load import load_script  # noqa: E402

RULE = {"k": 3, "interval": 12, "majority": 0.5, "warmup": 20, "H": 20, "min_lead_windows": 1, "chronic": 0.5}


@pytest.fixture(scope="module")
def fe():
    return load_script("b_final_eval")


@pytest.fixture(autouse=True)
def _primary_from_shadow(fe, monkeypatch):
    """The stand-in shadow runs have no store: question 1's primary predictor is given."""
    monkeypatch.setattr(fe, "primary_predictor", lambda shadow_cfg, names, root=None: "world_model")


# ------------------------------------------------------------------ protocol files
def _files(tmp, gate="GATE: pass", advisor=True):
    """Stand-in configs, notes and shadow outputs; returns the paths make_protocol takes."""
    from tests.test_phase_b_wm_prob import _tiny_model
    tmp.mkdir(parents=True, exist_ok=True)
    wm = _tiny_model(context=4, room=64, hi=np.full(43, 1e6, np.float32))
    sha = wm.save(tmp / "wm.pt")
    shadow = tmp / "shadow"
    (shadow / "s0000").mkdir(parents=True)
    (shadow / "s0000" / "risks.parquet").write_bytes(b"risks")
    (shadow / "s0000" / "done").write_text("")
    (shadow / "predictors.pkl").write_bytes(pickle.dumps({"queueing": predictors.QueueingModel(IncidentRule(3, 12))}))
    (shadow / "config.json").write_text(json.dumps({
        "label": "", "names": ["world_model", "queueing"], "rule": {k: RULE[k] for k in
                                                                     ("k", "interval", "majority", "warmup", "H",
                                                                      "min_lead_windows")},
        "calibration": list(range(3400, 3410)), "selection": [3450], "store": "unused", "wm_n": 4,
        "world_model_sha256": sha}))
    alarms = {"predictor": "queueing", "threshold": 0.0, "low": 0.0, "H": 20, "min_lead_windows": 1, "alpha": 0.1}
    adv = {"params": {"m": 2, "n": 4, "h": 6, "eps": 0.1, "slo_lam": 40.0, "mu": 8.0, "change_cost": 0.0},
           "alarms": {k: alarms[k] for k in ("predictor", "threshold", "low", "H")}, "predictor": "queueing",
           "rule": {k: RULE[k] for k in ("k", "interval", "majority", "warmup")}, "wm_n": 4,
           "world_model_sha256": sha, "fitted_sha256": None, "label": ""}
    from pis.phase_b.wm_prob import sha256
    adv["fitted_sha256"] = sha256(shadow / "predictors.pkl")
    for name, obj in (("incident.json", RULE), ("alarms.json", alarms), ("advisor.json", adv)):
        (tmp / name).write_text(json.dumps(obj))
    (tmp / "shadow_dev.md").write_text(f"# Shadow\n\n{gate}\n\nReason: test.\n")
    (tmp / "advisor_timing.md").write_text("# Advisor decision time at normal priority\n")
    paths = {"incident": tmp / "incident.json", "alarms": tmp / "alarms.json", "world_model": tmp / "wm.pt",
             "shadow_dir": shadow, "shadow_note": tmp / "shadow_dev.md"}
    if advisor:
        paths.update(advisor=tmp / "advisor.json", timing_note=tmp / "advisor_timing.md")
    return {k: str(v) for k, v in paths.items()}


def test_make_protocol_full(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path))
    assert fe.validate_protocol(p) == []
    assert p["variant"] == "full" and tuple(p["arms"]) == fe.ARMS
    assert fe.ARMS == ("r6", "advisor", "loop_a", "advisor_random", "advisor_oracle", "always_on")
    assert fe.protocol_seeds(p) == list(range(3000, 3100)) and p["duration_s"] == 300.0
    assert p["reference_arm"] == "r6"
    for key in ("incident", "alarms", "advisor", "world_model", "timing_note", "shadow_note", "fitted",
                "shadow_config", "step_a", "phase_a_protocol"):
        assert len(p["files"][key]["sha256"]) == 64, key
    assert len(p["files"]["shadow_risks"]["sha256"]) == 64
    q2, q1 = p["success_rule"]["question_2"], p["success_rule"]["question_1"]
    assert q2["value_tol"] == 0.01 and q2["decide_p95_ms"] == 50.0 and q2["decide_max_ms"] == 500.0
    assert q2["candidate"] == "advisor" and q2["reference"] == "r6" and q2["confidence"] == 0.95
    assert q1["bootstrap"] == {"reps": 2000, "tag": 14_000, "conf": 0.95} and q1["cause"] == "building_up"
    assert q1["names"] == ["world_model", "queueing"] and q1["min_lead_windows"] == 1
    json.dumps(p, allow_nan=False)                                   # strict JSON


def test_make_protocol_shadow_only_after_the_gate(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path, gate="GATE: stop before prevention", advisor=False))
    assert fe.validate_protocol(p) == []
    assert p["variant"] == "shadow_only" and list(p["arms"]) == ["r6"]
    assert p["success_rule"]["question_2"] == "not attempted"
    assert p["files"]["advisor"] is None and p["files"]["timing_note"] is None


def test_make_protocol_refuses_missing_or_unusable_inputs(fe, tmp_path):
    paths = _files(tmp_path)
    with pytest.raises(ValueError):
        fe.make_protocol({**paths, "advisor": str(tmp_path / "nope.json")})
    (tmp_path / "shadow_dev.md").write_text("# Shadow\n\nno gate line\n")
    with pytest.raises(ValueError, match="GATE"):
        fe.make_protocol(paths)
    (tmp_path / "shadow_dev.md").write_text("GATE: pass\n")
    (tmp_path / "incident.json").write_text(json.dumps({**RULE, "chronic": 0.6}))
    with pytest.raises(ValueError, match="chronic"):
        fe.make_protocol(paths)


def test_validate_protocol(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path))
    assert fe.validate_protocol({**p, "reference_arm": "advisor"})
    assert fe.validate_protocol({**p, "arms": {k: v for k, v in p["arms"].items() if k != "loop_a"}})
    assert fe.validate_protocol({k: v for k, v in p.items() if k != "success_rule"})


# ------------------------------------------------------------------ guards
def _official_args(fe, **kw):
    base = {"seeds_list": list(range(3000, 3100)), "label": "", "duration": 300.0, "out": fe.OFFICIAL_REPORT,
            "raw_dir": fe.OFFICIAL_RAW, "attempt": 1, "resume": False,
            "protocol": fe.PROTOCOL}
    return {**base, **kw}


def _state(**kw):
    base = {"dirty": False, "committed": {"configs/phase_b/final_eval.json": True}, "sha_mismatch": [],
            "report_text": "", "out_exists": False, "raw_exists": False, "prev_raw_exists": False,
            "freeze": {"commits": ["abc"], "files": ["configs/phase_b/final_eval.json"]},
            "run_config": None, "code_changed": False, "background": True, "priority": "nice 10"}
    return {**base, **kw}


@pytest.fixture()
def official_protocol(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path))
    p["files"] = {k: (None if v is None else {**v, "path": fe.OFFICIAL_FILES.get(k, v["path"])})
                  for k, v in p["files"].items()}
    return p


def test_official_run_accepted_when_everything_is_in_place(fe, official_protocol):
    assert fe.check_run(_official_args(fe), official_protocol, _state()) == []


@pytest.mark.parametrize("args,state,why", [
    ({}, {"dirty": True}, "clean"),
    ({"seeds_list": list(range(3000, 3050))}, {}, "exactly"),
    ({"seeds_list": list(range(3000, 3100)) + [3450]}, {}, "exactly"),
    ({"seeds_list": list(range(3000, 3100)) + [3000]}, {}, "exactly"),
    ({"label": "rehearsal"}, {}, "label"),
    ({"duration": 30.0}, {}, "duration"),
    ({}, {"background": False}, "gentle"),
    ({}, {"committed": {"configs/phase_b/final_eval.json": False}}, "committed"),
    ({}, {"sha_mismatch": ["alarms"]}, "sha256"),
    ({}, {"freeze": {"commits": ["a", "b"], "files": ["configs/phase_b/final_eval.json"]}}, "one commit"),
    ({}, {"freeze": {"commits": ["a"], "files": ["configs/phase_b/final_eval.json", "x.py"]}}, "one commit"),
    ({"protocol": "data/phase_b/smoke_x/final_eval.json"}, {}, "reads"),
    ({"out": "data/phase_b/smoke_x/final.md"}, {}, "writes"),
    ({"attempt": 2}, {}, "previous"),
    ({"attempt": 4}, {"prev_raw_exists": True}, "attempt"),
    ({}, {"report_text": "# Final\n\n## Attempt 1 (started)\n"}, "already has a section"),
    ({}, {"raw_exists": True}, "never overwritten"),
])
def test_official_run_refused(fe, official_protocol, args, state, why):
    errors = fe.check_run(_official_args(fe, **args), official_protocol, _state(**state))
    assert errors and any(why in e for e in errors), errors


def test_official_run_refuses_a_protocol_pointing_elsewhere(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path))                          # stand-in paths, not the frozen files
    errors = fe.check_run(_official_args(fe), p, _state())
    assert any("frozen" in e for e in errors)


def test_attempt_two_needs_the_first(fe, official_protocol):
    assert fe.check_run(_official_args(fe, attempt=2), official_protocol, _state(prev_raw_exists=True)) == []


def test_resume(fe, official_protocol):
    stub = "# Final\n\n## Attempt 1 (started)\n\n- code: x\n"
    ok = _state(report_text=stub, raw_exists=True, run_config={"attempt": 1})
    assert fe.check_run(_official_args(fe, resume=True), official_protocol, ok) == []
    assert fe.check_run(_official_args(fe, resume=True), official_protocol, _state(report_text=stub))  # no raw
    assert fe.check_run(_official_args(fe, resume=True), official_protocol,
                        {**ok, "report_text": "## Attempt 1\n\nfinished\n"})                       # finished
    assert fe.check_run(_official_args(fe, resume=True), official_protocol, {**ok, "code_changed": True})
    assert fe.check_run(_official_args(fe, resume=True), official_protocol, {**ok, "run_config": None})


def _rehearsal_args(fe, tmp_path, **kw):
    base = {"seeds_list": [3450, 3451, 3452], "label": "rehearsal", "duration": 30.0,
            "out": str(tmp_path / "final.md"), "raw_dir": str(tmp_path / "raw"), "attempt": 1, "resume": False,
            "protocol": str(tmp_path / "p.json")}
    return {**base, **kw}


def test_rehearsal_accepted_on_selection_seeds(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path))
    assert fe.check_run(_rehearsal_args(fe, tmp_path), p, _state(dirty=True)) == []
    assert fe.check_run(_rehearsal_args(fe, tmp_path, label="smoke"), p, _state()) == []


@pytest.mark.parametrize("args,state,why", [
    ({"seeds_list": [3400]}, {}, "SELECTION"),                      # calibration
    ({"seeds_list": [3100]}, {}, "SELECTION"),                      # history
    ({"seeds_list": [2000]}, {}, "earlier"),                        # Phase A held out
    ({"seeds_list": [1150]}, {}, "earlier"),
    ({"label": ""}, {}, "label"),
    ({"out": "results/phase_b/final.md"}, {}, "must not write"),
    ({"raw_dir": "results/raw/phase_b/final/x"}, {}, "must not write"),
    ({}, {"out_exists": True}, "never overwritten"),
    ({}, {"raw_exists": True}, "never overwritten"),
    ({}, {"sha_mismatch": ["incident"]}, "sha256"),
])
def test_rehearsal_refused(fe, tmp_path, args, state, why):
    p = fe.make_protocol(_files(tmp_path))
    errors = fe.check_run(_rehearsal_args(fe, tmp_path, **args), p, _state(**state))
    assert errors and any(why in e for e in errors), errors


def test_main_never_runs_on_held_out_seeds_without_the_frozen_protocol(fe, tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps(fe.make_protocol(_files(tmp_path / "f"))))
    for argv in (["--protocol", str(p)], ["--protocol", str(p), "--seeds", "3000-3002", "--label", "rehearsal",
                                            "--out", str(tmp_path / "o.md"), "--raw-dir", str(tmp_path / "r")]):
        with pytest.raises(SystemExit) as e:
            fe.main(argv)
        assert "refusing" in str(e.value)
    assert not (tmp_path / "r").exists() and not (tmp_path / "o.md").exists()


# ------------------------------------------------------------------ timing note
def _timing_note(tmp_path, cfg, alarms, sha, label="", p95=20.0, mx=40.0, dirty=False, selection="model"):
    tim = load_script("b_time_advisor")
    per = {3450: {"n": 600, "mode_windows": 10, "fallbacks": 0, "mean": 10.0, "p50": 9.0, "p95": p95, "max": mx}}
    res = {"per_seed": per, "all": per[3450], "in_mode": {"n": 10, "p50": 12.0, "p95": 15.0, "max": 16.0},
           "breakdown": {"risk_ms_mean": 1.0, "risk_ms_p95": 2.0, "select_ms_mean_in_mode": 3.0,
                         "select_ms_p95_in_mode": 4.0},
           "wall_s": {3450: 20.0}, "worst_p95": p95, "worst_max": mx, "p95_ok": p95 <= 50, "max_ok": mx <= 500}
    text = tim.note_text(res, {"label": label, "seeds": "3450-3452", "duration": 300.0,
                               "config": "configs/phase_b/advisor.json", "selection": selection}, cfg, sha, alarms)
    text = text.replace(" (uncommitted changes)", "")              # the test tree itself may be dirty
    if dirty:
        text = text.replace("\n- config:", " (uncommitted changes)\n- config:", 1)
    path = tmp_path / "timing.md"
    path.write_text(text)
    return str(path)


def test_timing_note_verdict(fe, tmp_path):
    cfg = {"params": {"m": 4, "n": 8, "h": 20, "eps": 0.05, "slo_lam": 40.0, "mu": 8.0, "change_cost": 0.0}}
    alarms = {"predictor": "hazard", "threshold": 0.4, "low": 0.28, "H": 20}
    sha = "a" * 64
    ok = fe.timing_for(_timing_note(tmp_path, cfg, alarms, sha), cfg, alarms, sha, official=True, committed=True)
    assert "unavailable" not in ok and ok["p95_ok"] and ok["max_ok"] and ok["worst_p95"] == 20.0
    over = fe.timing_for(_timing_note(tmp_path, cfg, alarms, sha, mx=600.0), cfg, alarms, sha, True, True)
    assert "unavailable" not in over and not over["max_ok"]

    def bad(**kw):
        note = _timing_note(tmp_path, cfg, alarms, sha, **{k: v for k, v in kw.items()
                                                            if k in ("label", "dirty", "selection")})
        t = fe.timing_for(note, kw.get("cfg", cfg), kw.get("alarms", alarms), kw.get("sha", sha), True,
                          kw.get("committed", True))
        return t.get("unavailable")
    assert "labelled" in bad(label="smoke")
    assert "uncommitted" in bad(dirty=True)
    assert "selection" in bad(selection="always")
    assert "params" in bad(cfg={"params": {**cfg["params"], "eps": 0.1}})
    assert "alarm" in bad(alarms={**alarms, "threshold": 0.5})
    assert "world model" in bad(sha="b" * 64)
    assert "committed" in bad(committed=False)
    assert "missing" in fe.timing_for(str(tmp_path / "none.md"), cfg, alarms, sha, True, True)["unavailable"]


# ------------------------------------------------------------------ question 2 on synthetic raw results
def _rec(seed, arm, t, v, chronic=False, slo=5.0, bu=(1, 1.0), su=(0, 0.0), failed=None):
    if failed:
        return {"seed": seed, "arm": arm, "failed": failed, "seconds": 1.0}
    return {"seed": seed, "arm": arm, "failed": None, "time_in_incident_s": t, "incidents": bu[0] + su[0],
            "value_per_s": v, "slo_v2_pct": slo, "chronic": chronic, "mode_share": 0.2, "fallback_share": 0.1,
            "risk_nan": 0, "decide_ms_p95": 5.0, "decide_ms_max": 9.0, "seconds": 1.0, "windows": 600,
            "by_cause": {"building_up": {"incidents": bu[0], "time_s": bu[1]},
                         "sudden": {"incidents": su[0], "time_s": su[1]}}}


def _episodes(adv_times=(5, 14, 26, 10, 20, 0), value_delta=-0.1):
    r6_times = (10, 20, 30, 15, 25, 250)
    out = []
    for s, (tr, ta) in enumerate(zip(r6_times, adv_times)):
        seed = 3000 + s
        out.append(_rec(seed, "r6", tr, 100.0 + s, chronic=tr > 150, bu=(1 if s != 3 else 0, tr)))
        out.append(_rec(seed, "advisor", ta, 100.0 + s + value_delta, bu=(1 if s != 3 else 0, ta)))
    return out


TIMING_OK = {"worst_p95": 20.0, "worst_max": 40.0, "p95_ok": True, "max_ok": True}


def test_question_two_success_and_chronic_exclusion(fe, tmp_path):
    rule = fe.make_protocol(_files(tmp_path))["success_rule"]["question_2"]
    q = fe.question_two(_episodes(), ["r6", "advisor"], rule, TIMING_OK)
    a = q["arms"]["advisor"]
    assert a["non_chronic"] == 5 and a["chronic"] == 1 and a["pairs"] == 6
    assert a["time_diff"]["n"] == 5 and a["time_diff"]["mean"] == pytest.approx(-5.0)
    assert a["time_diff_chronic"]["n"] == 1 and a["time_diff_chronic"]["mean"] == pytest.approx(-250.0)
    assert a["value_diff"]["n"] == 6 and a["value_diff"]["mean"] == pytest.approx(-0.1)
    assert a["by_cause"]["building_up"]["time_diff"]["n"] == 5
    assert a["quiet_value_diff"]["n"] == 1                         # seed 3003: no counted incident for R6
    s = q["success"]
    assert s["time_ok"] and s["value_ok"] and s["timing_ok"] and s["met"]


def test_question_two_failure_cases(fe, tmp_path):
    rule = fe.make_protocol(_files(tmp_path))["success_rule"]["question_2"]
    spread = fe.question_two(_episodes(adv_times=(5, 25, 26, 20, 20, 0)), ["r6", "advisor"], rule, TIMING_OK)
    assert not spread["success"]["time_ok"] and not spread["success"]["met"]          # CI crosses zero
    costly = fe.question_two(_episodes(value_delta=-2.0), ["r6", "advisor"], rule, TIMING_OK)
    assert costly["success"]["time_ok"] and not costly["success"]["value_ok"] and not costly["success"]["met"]
    slow = fe.question_two(_episodes(), ["r6", "advisor"], rule, {**TIMING_OK, "max_ok": False})
    assert not slow["success"]["timing_ok"] and not slow["success"]["met"]
    gone = fe.question_two(_episodes(), ["r6", "advisor"], rule, {"unavailable": "missing"})
    assert not gone["success"]["timing_ok"] and "missing" in gone["success"]["timing"]


def test_question_two_failed_runs_are_listed_not_paired(fe, tmp_path):
    rule = fe.make_protocol(_files(tmp_path))["success_rule"]["question_2"]
    eps = [e for e in _episodes() if not (e["seed"] == 3001 and e["arm"] == "advisor")]
    eps.append(_rec(3001, "advisor", 0, 0, failed="RuntimeError: boom"))
    q = fe.question_two(eps, ["r6", "advisor"], rule, TIMING_OK)
    assert q["arms"]["advisor"]["pairs"] == 5 and q["failed"] == [(3001, "advisor", "RuntimeError: boom")]


# ------------------------------------------------------------------ question 1 on synthetic risks
def _risk_case(n_cal=10, n_out=10, T=100):
    rule = IncidentRule(3, 12, 0.5)
    risks, labels, active = {}, {}, {}
    for e in list(range(3400, 3400 + n_cal)) + list(range(3000, 3000 + n_out)):
        viol = np.zeros(T, bool)
        viol[60:80] = True
        act = np.arange(T) >= 40
        labels[e] = incidents.label_series(viol, act, rule, 20)
        active[e] = act
        good = np.zeros(T)
        good[50:60] = 1.0
        good[:20] = np.nan
        flat = np.zeros(T)
        flat[:20] = np.nan
        risks[e] = {"world_model": good, "queueing": flat}
    return risks, labels, active


def test_question_one_scores_held_out_with_calibration_thresholds(fe):
    risks, labels, active = _risk_case()
    cal, out = list(range(3400, 3410)), list(range(3000, 3010))
    q = fe.question_one(risks, labels, active, ["world_model", "queueing"], cal, out, RULE)
    res = q["predictors"]
    assert res["world_model"]["threshold"] == 1.0 and res["queueing"]["threshold"] is None
    assert res["world_model"]["selection"]["by_cause"]["building_up"]["recall"] == 1.0
    d = q["q1"]["diffs"]["world_model"]["building_up"]
    assert q["q1"]["non_ml"] == "queueing" and d["diff"] == 1.0 and d["lo"] > 0
    assert q["verdict"]["world_model"] is True
    b = q["outcome_brier"]["world_model"]
    assert b["all"]["n"] > 0 and 0.0 <= b["all"]["brier"] <= 1.0


def test_question_one_thresholds_never_see_held_out(fe):
    risks, labels, active = _risk_case()
    cal, out = list(range(3400, 3410)), list(range(3000, 3010))
    base = fe.question_one(risks, labels, active, ["world_model", "queueing"], cal, out, RULE)
    for e in out:                                                  # held-out risks changed: thresholds stay
        risks[e]["world_model"] = np.where(np.isnan(risks[e]["world_model"]), np.nan, 0.5)
    again = fe.question_one(risks, labels, active, ["world_model", "queueing"], cal, out, RULE)
    assert again["predictors"]["world_model"]["threshold"] == base["predictors"]["world_model"]["threshold"]
    assert again["verdict"]["world_model"] is False


# ------------------------------------------------------------------ report
def test_report_text_full_and_shadow_only(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path / "a"))
    q2 = fe.question_two(_episodes(), ["r6", "advisor"], p["success_rule"]["question_2"], TIMING_OK)
    risks, labels, active = _risk_case()
    q1 = fe.question_one(risks, labels, active, ["world_model", "queueing"], list(range(3400, 3410)),
                         list(range(3000, 3010)), RULE)
    meta = {"title": "Attempt 1 (rehearsal)", "code": "git HEAD x"}
    text = fe.report_text(p, meta, q2, q1, TIMING_OK, warnings=["w1"])
    assert text.startswith("## Attempt 1 (rehearsal)")
    assert "Question 2" in text and "Question 1" in text and "SUCCESS" in text and "w1" in text
    assert "chronic" in text
    p2 = fe.make_protocol(_files(tmp_path / "b", gate="GATE: stop before prevention", advisor=False))
    q2s = fe.question_two([e for e in _episodes() if e["arm"] == "r6"], ["r6"], "not attempted", None)
    text2 = fe.report_text(p2, meta, q2s, q1, None, warnings=[])
    assert "not attempted" in text2 and "SUCCESS" not in text2


# ------------------------------------------------------------------ end to end on a development seed
def test_rehearsal_end_to_end(fe, tmp_path, monkeypatch):
    paths = _files(tmp_path / "files")
    proto = tmp_path / "protocol.json"
    assert fe.main(["--write-protocol", "--protocol", str(proto)] +
                   [x for k, v in paths.items() for x in (f"--{k.replace('_', '-')}", v)]) is None
    with pytest.raises(SystemExit):                                # never rewritten
        fe.main(["--write-protocol", "--protocol", str(proto)] +
                [x for k, v in paths.items() for x in (f"--{k.replace('_', '-')}", v)])
    risks, labels, active = _risk_case(n_out=0, T=40)
    monkeypatch.setattr(fe, "calibration_inputs", lambda shadow_cfg, names: (
        {e: {n: r["world_model"] if n == "world_model" else r["queueing"] for n in names}
         for e, r in risks.items()}, labels, active))
    out, raw = tmp_path / "final.md", tmp_path / "raw"
    argv = ["--protocol", str(proto), "--seeds", "3450", "--duration", "11", "--label", "smoke",
            "--out", str(out), "--raw-dir", str(raw), "--procs", "1"]
    res = fe.main(argv)
    eps = res["episodes"]
    assert {e["arm"] for e in eps} == set(fe.ARMS) and all(e["failed"] is None for e in eps)
    assert all(e["windows"] == 22 for e in eps)
    always = next(e for e in eps if e["arm"] == "always_on")
    assert always["mode_share"] == pytest.approx(2 / 22)
    r6 = next(e for e in eps if e["arm"] == "r6")
    assert len(r6["viol"]) == 22 and set(r6["viol"]) <= {"0", "1"}
    text = out.read_text()
    assert "## Attempt 1 (smoke)" in text and "(started)" not in text and "Projected" in text
    assert (raw / "attempt_1" / "run_config.json").exists()
    att = raw / "attempt_1"
    assert (att / fe.PRIMARY_FILE).exists() and (att / fe.ORACLE_SHARDS / "s0000" / "done").exists()
    primary = json.loads((att / fe.SHARDS / "s0000" / fe.EPISODES_FILE).read_text())
    assert {e["arm"] for e in primary} == set(fe.PRIMARY_ARMS)                 # the oracle runs on its own
    assert (att / fe.ORACLE_SHARDS / "s0000" / "done").stat().st_mtime >= \
        (att / fe.SHARDS / "s0000" / "done").stat().st_mtime
    assert "Diagnostic oracle arm" in text and "| advisor_oracle |" in text
    with pytest.raises(SystemExit) as e:                           # never overwritten
        fe.main(argv)
    assert "refusing" in str(e.value)


# ------------------------------------------------------------------ review fixes
class RaisingPredictor:
    """A predictor whose risk always raises (scoring failure)."""
    name = "queueing"

    def risk(self, rows, H, episode=None):
        raise RuntimeError("predictor broke")


def test_scoring_failure_keeps_the_r6_run(fe, tmp_path):
    from pis.shards import Shard
    (tmp_path / "pred.pkl").write_bytes(pickle.dumps({"queueing": RaisingPredictor()}))
    params = {"incident": RULE, "alarms": {"predictor": None, "threshold": None, "low": None, "H": 20},
              "duration_s": 12.0, "rule": RULE, "arms": ["r6"], "names": ["queueing"], "world_model": None,
              "fitted": str(tmp_path / "pred.pkl"), "wm_n": 4, "advisor_params": None}
    d = tmp_path / "s0000"
    d.mkdir()
    fe.run_shard(Shard("s0000", [3450]), d, params)
    (rec,) = json.loads((d / fe.EPISODES_FILE).read_text())
    assert rec["failed"] is None and rec["windows"] == 24 and "value_per_s" in rec
    assert "predictor broke" in rec["scoring_failed"]
    assert not (d / fe.RISKS_FILE).exists()
    q = fe.question_two([rec], ["r6"], "not attempted", None)
    assert q["failed"] == [] and q["scoring_failed"] == [(3450, rec["scoring_failed"])]


def test_question_two_not_decidable_when_r6_or_advisor_failed(fe, tmp_path):
    rule = fe.make_protocol(_files(tmp_path))["success_rule"]["question_2"]
    for arm in ("r6", "advisor"):
        eps = [e for e in _episodes() if not (e["seed"] == 3001 and e["arm"] == arm)]
        eps.append(_rec(3001, arm, 0, 0, failed="RuntimeError: boom"))
        s = fe.question_two(eps, ["r6", "advisor"], rule, TIMING_OK)["success"]
        assert s["time_ok"] and s["value_ok"] and s["timing_ok"]               # would pass on the rest
        assert not s["decidable"] and not s["met"]
    eps = _episodes() + [_rec(3001, "loop_a", 0, 0, failed="RuntimeError: other arm")]
    s = fe.question_two(eps, ["r6", "advisor", "loop_a"], rule, TIMING_OK)["success"]
    assert s["decidable"] and s["met"]                                          # another arm does not block


def test_report_says_not_decidable(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path))
    eps = [e for e in _episodes() if not (e["seed"] == 3001 and e["arm"] == "r6")]
    eps.append(_rec(3001, "r6", 0, 0, failed="RuntimeError: boom"))
    q2 = fe.question_two(eps, ["r6", "advisor"], p["success_rule"]["question_2"], TIMING_OK)
    risks, labels, active = _risk_case()
    q1 = fe.question_one(risks, labels, active, ["world_model", "queueing"], list(range(3400, 3410)),
                         list(range(3000, 3010)), RULE, blocking=[(3001, "r6", "RuntimeError: boom")])
    assert not q1["decidable"] and q1["verdict"]["world_model"] is True
    text = fe.report_text(p, {"title": "Attempt 1"}, q2, q1, TIMING_OK, warnings=[])
    assert "**Question 2: not decidable**" in text and "**Question 1: not decidable**" in text
    assert "SUCCESS" not in text
    assert "not_decidable" in p["success_rule"]


def test_question_one_refuses_overlapping_episodes(fe):
    risks, labels, active = _risk_case()
    with pytest.raises(ValueError, match="overlap"):
        fe.question_one(risks, labels, active, ["world_model", "queueing"], list(range(3400, 3410)),
                        [3400, 3000], RULE)


def test_validate_protocol_seeds_and_duration(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path))
    assert any("held-out" in e for e in fe.validate_protocol({**p, "seeds": {"first": 3000, "last": 3049}}))
    assert any("duration" in e for e in fe.validate_protocol({**p, "duration_s": 60.0}))


def test_value_bound_uses_the_absolute_r6_mean(fe, tmp_path):
    rule = fe.make_protocol(_files(tmp_path))["success_rule"]["question_2"]
    eps = [dict(e, value_per_s=-e["value_per_s"]) for e in _episodes()]
    s = fe.question_two(eps, ["r6", "advisor"], rule, TIMING_OK)["success"]
    assert s["value_bound"] < 0


# ------------------------------------------------------------------ live rows vs store rows
def test_live_rows_keep_the_exact_known_values():
    from sched.randomize import known_params
    from pis.phase_b import features
    from pis.phase_b.episode import run_episode
    from pis.phase_b.scenarios import scenario_for
    from pis.harness import BaseGreedyController
    sc = scenario_for(3450)
    import dataclasses
    sc = dataclasses.replace(sc, duration_s=2.0)
    known = known_params(sc.spec)
    rows = features.rows_from_records(run_episode(sc, BaseGreedyController(sc.spec), "r6"), known, episode=3450)
    for name, col in zip(features.KNOWN_NAMES, features.KNOWN_COLS):
        assert rows[col].dtype == np.float64 and (rows[col] == float(known[name])).all()


PROBE_STORE, PROBE_SHADOW = "data/phase_b/timing_probe", "data/phase_b/smoke_g5/shadow"
PROBE_WM, PROBE_SEED, PROBE_WINDOWS = "models/smoke_g4/pgru_smoke.pt", 3104, 160


def _probe_paths():
    from tests._load import ROOT
    return ROOT / PROBE_STORE, ROOT / PROBE_SHADOW, ROOT / PROBE_WM


@pytest.mark.skipif(not all(p.exists() for p in _probe_paths()), reason="probe store and smoke models not present")
def test_live_rows_give_the_store_risks():
    """An R6 probe episode rerun live: every predictor's risk series from `rows_from_records` equals the one
    from the store rows the shadow run reads (world model with its fixed per-episode seed)."""
    from sched.randomize import known_params
    from pis.phase_b import features, policies, runs
    from pis.phase_b.episode import run_episode
    from pis.phase_b.history import Store
    from pis.phase_b.scenarios import scenario_for
    store, shadow_dir, wm = _probe_paths()
    shadow = runs.load_script("b_shadow")
    st = Store(store)
    eps = st.episodes()
    assert eps.set_index("seed").loc[PROBE_SEED, "source"] == "r6"
    stored = features.join_known(st.window_rows(PROBE_SEED, 0, 600), eps)
    sc = scenario_for(PROBE_SEED)
    live = features.rows_from_records(run_episode(sc, policies.make("r6", sc.spec, PROBE_SEED), "r6"),
                                      known_params(sc.spec), episode=PROBE_SEED)
    cfg = json.loads((shadow_dir / "config.json").read_text())
    names, rule = cfg["names"], cfg["rule"]
    assert set(names) == {"world_model", "hazard", "analogue", "queueing", "change_point"}
    preds = [shadow.load_predictors(shadow_dir / "predictors.pkl", names, rule, wm, 8) for _ in range(2)]
    a = shadow.episode_risks(preds[0], names, stored.iloc[:PROBE_WINDOWS], rule["H"], rule["warmup"], PROBE_SEED)
    b = shadow.episode_risks(preds[1], names, live.iloc[:PROBE_WINDOWS], rule["H"], rule["warmup"], PROBE_SEED)
    for n in names:
        np.testing.assert_allclose(b[n], a[n], rtol=0, atol=1e-9, err_msg=n)


# ------------------------------------------------------------------ oracle sizing and the primary predictor
def test_protocol_oracle_fields_and_primary_predictor(fe, tmp_path):
    p = fe.make_protocol(_files(tmp_path))
    assert p["oracle"]["branches"] == 4 and p["oracle"]["horizon"] == 6            # h of the advisor config
    q1 = p["success_rule"]["question_1"]
    assert q1["primary"] == "world_model" and q1["secondary"] == []
    p2 = fe.make_protocol(_files(tmp_path / "b"), oracle_branches=2, oracle_horizon=3)
    assert p2["oracle"]["branches"] == 2 and p2["oracle"]["horizon"] == 3
    assert any("oracle" in e for e in fe.validate_protocol({**p, "oracle": None}))
    bad = json.loads(json.dumps(p))
    bad["success_rule"]["question_1"]["primary"] = "queueing"
    assert any("primary" in e for e in fe.validate_protocol(bad))
    with pytest.raises(ValueError):
        fe.make_protocol(_files(tmp_path / "c"), oracle_branches=0)
    shadow_only = fe.make_protocol(_files(tmp_path / "d", gate="GATE: stop before prevention", advisor=False))
    assert shadow_only["oracle"] is None and fe.validate_protocol(shadow_only) == []


def test_choose_primary_best_selection_recall_then_false_alarms(fe):
    def r(recall, fa, thr=0.5):
        return {"threshold": thr, "selection": {"by_cause": {"building_up": {"recall": recall}},
                                                "false_alarm_rate": fa}}
    res = {"world_model": r(0.4, 0.08), "hazard": r(0.6, 0.09), "analogue": r(0.6, 0.05), "queueing": r(0.9, 0.0)}
    assert fe.choose_primary(res, ["world_model", "hazard", "analogue"]) == "analogue"     # tie -> lower FA
    res["analogue"] = r(0.6, 0.05, thr=None)                                                # infeasible = 0
    assert fe.choose_primary(res, ["world_model", "hazard", "analogue"]) == "hazard"
    assert fe.choose_primary(res, []) is None


def test_question_one_success_is_judged_on_the_primary_only(fe):
    risks, labels, active = _risk_case()
    for e in risks:                                     # hazard: as good as the world model
        risks[e]["hazard"] = risks[e]["world_model"].copy()
    for e in range(3000, 3010):                         # world model fails on held-out
        risks[e]["world_model"] = np.where(np.isnan(risks[e]["world_model"]), np.nan, 0.5)
    names = ["world_model", "hazard", "queueing"]
    cal, out = list(range(3400, 3410)), list(range(3000, 3010))
    q = fe.question_one(risks, labels, active, names, cal, out, RULE, primary="world_model")
    assert q["verdict"]["hazard"] is True and q["primary_above"] is False
    p = {"success_rule": {"question_2": "not attempted"}, "files": {"shadow_note": {"path": "x"}}}
    q2 = fe.question_two([e for e in _episodes() if e["arm"] == "r6"], ["r6"], "not attempted", None)
    text = fe.report_text(p, {"title": "Attempt 1"}, q2, q, None, [])
    assert "**Question 1: not met (world_model" in text and "secondary (no success claim)" in text
    good = fe.question_one(risks, labels, active, names, cal, out, RULE, primary="hazard")
    assert good["primary_above"] is True
    assert "**Question 1: SUCCESS (hazard" in fe.report_text(p, {"title": "A"}, q2, good, None, [])


def test_oracle_projection_from_tuning_raw(fe, tmp_path):
    d = tmp_path / "tune" / "s0000"
    d.mkdir(parents=True)
    eps = [{"seed": 3450, "arm": "r6", "seconds": 6.0, "windows": 600, "mode_share": 0.0},
           {"seed": 3450, "arm": "eps0.05_n8", "seconds": 9.0, "windows": 600, "mode_share": 0.1},
           {"seed": 3451, "arm": "eps0.05_n8", "seconds": 9.0, "windows": 600, "mode_share": 0.3}]
    (d / "episodes.json").write_text(json.dumps(eps))
    (d / "done").write_text("")
    adv = {"chosen": "eps0.05_n8", "params": {"m": 4, "h": 20}}
    proj = fe.oracle_projection(tmp_path / "tune", adv, RULE, branches=4, horizon=20, n_seeds=100)
    from pis.adapters import initial_policy
    from pis.probe import named_interventions
    cands = 1 + 4 + len(named_interventions(initial_policy()))
    want_s = 0.2 * 600 * cands * 4 * (20 + 3 - 1) * 0.01
    assert proj["mode_share_mean"] == pytest.approx(0.2) and proj["mode_share_max"] == pytest.approx(0.3)
    assert proj["one_proc_h"] == pytest.approx(100 * want_s / 3600)
    assert proj["four_procs_h"] == pytest.approx(proj["one_proc_h"] / 4)
    assert fe.oracle_projection(tmp_path / "none", adv, RULE, 4, 20) is None


# ------------------------------------------------------------------ code identity with the shadow run
SMOKE_SHADOW = "data/phase_b/smoke_g5/shadow"


def _smoke_shadow():
    from tests._load import ROOT
    return ROOT / SMOKE_SHADOW, ROOT / PROBE_STORE, ROOT / PROBE_WM


@pytest.mark.skipif(not all(p.exists() for p in _smoke_shadow()), reason="smoke shadow run not present")
def test_rescore_check_reproduces_and_names_a_changed_predictor(tmp_path):
    import shutil
    import pandas as pd
    shadow = load_script("b_shadow")
    src, _, wm = _smoke_shadow()
    assert shadow.rescore_check(src, wm, episodes=2, windows=60) == []
    copy = tmp_path / "shadow"
    shutil.copytree(src, copy)
    first = json.loads((copy / "config.json").read_text())["calibration"][0]
    for d in sorted(copy.glob("s*")):
        f = d / "risks.parquet"
        df = pd.read_parquet(f)
        hit = (df.episode == first) & (df.window == 40)
        if hit.any():
            df.loc[hit, "hazard"] += 1e-6
            df.to_parquet(f, index=False)
    assert shadow.rescore_check(copy, wm, episodes=2, windows=60) == ["hazard"]
    refusals, _ = shadow.identity_check(copy, wm, episodes=2, windows=60)
    assert refusals and "hazard" in refusals[0]
    refusals, _ = shadow.identity_check(tmp_path / "missing", wm)
    assert refusals and "could not be re-scored" in refusals[0]


def test_code_drift(tmp_path):
    shadow = load_script("b_shadow")
    assert "no code commit" in shadow.code_drift({})
    from pis.phase_b import runs
    assert shadow.code_drift({"code_head": runs.git_head()}) is None
    assert shadow.code_drift({"code_head": "aa71319"})                       # pis/phase_b changed since then


def test_background_priority_is_read_from_the_process(monkeypatch):
    import os
    from pis.phase_b import runs

    def prio(nice, darwin):
        def get(which, who):
            if which == runs.PRIO_DARWIN_PROCESS:
                if darwin is None:
                    raise OSError("not macOS")
                return darwin
            return nice
        return get
    monkeypatch.setattr(os, "getpriority", prio(0, 0))
    assert runs.background_priority() == (False, "nice 0")
    monkeypatch.setattr(os, "getpriority", prio(0, 1))
    assert runs.background_priority() == (True, "macOS background QoS")
    monkeypatch.setattr(os, "getpriority", prio(5, None))
    assert runs.background_priority() == (True, "nice 5")


def test_protocol_takes_world_model_and_shadow_dir_from_the_alarms_config(fe, tmp_path):
    paths = _files(tmp_path)
    alarms = json.loads((tmp_path / "alarms.json").read_text())
    (tmp_path / "alarms.json").write_text(json.dumps({**alarms, "world_model": paths["world_model"],
                                                      "shadow_dir": paths["shadow_dir"]}))
    rest = {k: v for k, v in paths.items() if k not in ("world_model", "shadow_dir")}
    p = fe.make_protocol(rest)
    assert p["files"]["world_model"]["path"] == paths["world_model"]
    assert p["files"]["shadow_risks"]["path"] == paths["shadow_dir"]
    assert p["files"]["fitted"]["path"] == paths["shadow_dir"] + "/predictors.pkl"
    (tmp_path / "alarms.json").write_text(json.dumps(alarms))          # a config that records neither
    with pytest.raises(ValueError):                                     # -> the first run's files (absent here)
        fe.make_protocol(rest)


def test_official_files_follow_the_alarms_config(fe, tmp_path):
    assert fe.official_files({})["world_model"] == "models/pgru_b1.pt"
    assert fe.official_files({})["shadow_risks"] == "results/raw/phase_b/shadow"
    off = fe.official_files({"world_model": "models/rssm_b2.pt", "shadow_dir": "results/raw/phase_b/shadow_rssm_b2"})
    assert off["world_model"] == "models/rssm_b2.pt"
    assert off["fitted"] == "results/raw/phase_b/shadow_rssm_b2/predictors.pkl"
    assert off["shadow_config"] == "results/raw/phase_b/shadow_rssm_b2/config.json"
    assert {k: v for k, v in off.items() if k not in ("world_model", "fitted", "shadow_config", "shadow_risks")} \
        == {k: v for k, v in fe.OFFICIAL_FILES.items() if k not in ("world_model", "fitted", "shadow_config",
                                                                     "shadow_risks")}
    p = fe.make_protocol(_files(tmp_path))
    p["files"] = {k: (None if v is None else {**v, "path": off.get(k, v["path"])}) for k, v in p["files"].items()}
    assert fe.check_run(_official_args(fe), p, _state(official_files=off)) == []
    errors = fe.check_run(_official_args(fe), p, _state())              # the first run's files are expected
    assert any("world_model" in e for e in errors) and any("shadow_risks" in e for e in errors)
