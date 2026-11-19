import json

import numpy as np
import pandas as pd
import pytest

from pis.phase_b import alarms, incidents
from pis.phase_b.incidents import Incident, IncidentRule, Labels
from tests._load import load_script


def _labels(T, incs):
    inc = np.zeros(T, bool)
    for o, e, _ in incs:
        inc[o:e] = True
    return Labels(in_incident=inc, incidents=[Incident(o, e, c) for o, e, c in incs])


# ------------------------------------------------------------------ hysteresis
def test_alarm_events_hysteresis():
    r = np.array([0.0, 0.5, 0.9, 0.8, 0.6, 0.5, 0.9, 0.3, 0.95])
    assert alarms.alarm_events(r, 0.8, 0.56) == [2, 6, 8]


def test_alarm_events_nan_is_off_and_stays_off():
    r = np.array([np.nan, 0.9, 0.9, np.nan, 0.9, 0.1, 0.9])
    assert alarms.alarm_events(r, 0.8, 0.56) == [1, 4, 6]


def test_alarm_events_no_alarm_and_bad_low():
    assert alarms.alarm_events(np.zeros(10), 0.5, 0.35) == []
    assert alarms.alarm_events(np.ones(10), 0.5, 0.35) == [0]
    with pytest.raises(ValueError):
        alarms.alarm_events(np.zeros(3), 0.5, 0.6)


# ------------------------------------------------------------------ scoring
def test_lead_time_and_true_false_events():
    lab = _labels(200, [(50, 60, "building_up"), (120, 130, "sudden")])
    # 35: onset 50 within 20 -> true, lead 15; 45: true but not the first in [30, 49];
    # 80: no onset in (80, 100] -> false; 117: true, lead 3
    s = alarms.score_episode([35, 45, 80, 117], lab, H=20, min_lead=4)
    assert (s["events"], s["true"], s["false"], s["censored"]) == (4, 3, 1, 0)
    assert s["incidents"] == {"building_up": 1, "sudden": 1}
    assert s["caught"] == {"building_up": 1, "sudden": 0}          # lead 3 < min lead 4
    assert s["leads"] == {"building_up": [15], "sudden": []}
    assert s["early"] == {"building_up": [15], "sudden": [3]}
    assert alarms.episode_loss(s) == pytest.approx(0.25)


def test_first_event_in_window_sets_the_lead():
    lab = _labels(200, [(100, 110, "building_up")])
    s = alarms.score_episode([70, 85, 95], lab, H=20, min_lead=1)   # 70 is before onset - H
    assert s["leads"]["building_up"] == [15]
    assert s["false"] == 1 and s["true"] == 2


def test_event_at_onset_does_not_catch_it():
    lab = _labels(200, [(100, 110, "building_up")])
    s = alarms.score_episode([100], lab, H=20, min_lead=1)
    assert s["caught"]["building_up"] == 0 and s["false"] == 1


def test_event_inside_incident_counts_only_new_onsets():
    lab = _labels(200, [(50, 90, "building_up"), (100, 110, "sudden")])
    s = alarms.score_episode([60, 85], lab, H=20, min_lead=1)
    assert s["false"] == 1 and s["true"] == 1                       # 60: in force, next onset 100 > 80


def test_censored_events_neither_true_nor_false():
    lab = _labels(100, [])
    s = alarms.score_episode([70, 80], lab, H=20, min_lead=1, tail=2)   # 70+20+2 = 92 <= 99; 80 -> 102
    assert (s["true"], s["false"], s["censored"]) == (0, 1, 1)


def test_onset_in_warmup_not_counted():
    viol = np.zeros(100, bool)
    viol[5:15] = True
    viol[60:70] = True
    lab = incidents.label_series(viol, np.zeros(100, bool), IncidentRule(3, 12), warmup=20)
    s = alarms.score_episode([0, 50], lab, H=20, min_lead=1)
    assert sum(s["incidents"].values()) == 1
    assert s["caught"]["sudden"] == 1 and s["false"] == 1


def test_quiet_episode_alarm_rate():
    quiet = _labels(200, [])
    busy = _labels(200, [(100, 110, "building_up")])
    scores = [alarms.score_episode([10, 50], quiet, 20, 1), alarms.score_episode([], quiet, 20, 1),
              alarms.score_episode([90], busy, 20, 1)]
    sm = alarms.summarize(scores)
    assert sm["quiet_episodes"] == 2
    assert sm["alarms_per_quiet_episode"] == pytest.approx(1.0)
    assert sm["false_alarm_rate"] == pytest.approx(2 / 3)
    assert sm["by_cause"]["building_up"]["recall"] == 1.0
    assert sm["by_cause"]["sudden"]["recall"] is None
    assert alarms.episode_loss(scores[1]) == 0.0


def test_summary_without_events_or_quiet_episodes():
    busy = _labels(200, [(100, 110, "building_up")])
    sm = alarms.summarize([alarms.score_episode([], busy, 20, 1)])
    assert sm["false_alarm_rate"] is None
    assert sm["alarms_per_quiet_episode"] is None
    assert sm["by_cause"]["building_up"]["recall"] == 0.0
    assert sm["by_cause"]["building_up"]["median_lead"] is None
    assert alarms.summarize([])["events"] == 0


# ------------------------------------------------------------------ conformal threshold
def _synthetic(rng, n, T=200, H=20):
    """Episodes with a background risk, occasional false spikes and, in half of them, a rise before an onset."""
    out = []
    for _ in range(n):
        r = rng.uniform(0.0, 0.4, T)
        incs = []
        if rng.random() < 0.5:
            o = int(rng.integers(60, 180))
            lead = int(rng.integers(2, 18))
            r[o - lead:o] = rng.uniform(0.5, 1.0) + rng.uniform(0, 0.1, lead)
            incs.append((o, min(o + 10, T), "building_up"))
        for _ in range(rng.poisson(0.6)):
            s = int(rng.integers(0, T - 3))
            r[s:s + 3] = rng.uniform(0.3, 0.95)
        out.append((r, _labels(T, incs)))
    return out


def _mean_loss(eps, thr, H):
    return float(np.mean([alarms.episode_loss(alarms.score_episode(
        alarms.alarm_events(r, thr, alarms.LOW_RATIO * thr), lab, H, 1)) for r, lab in eps]))


def test_conformal_threshold_meets_alpha_on_fresh_episodes():
    """Conformal risk control bounds the expected false-alarm share of a fresh episode, over calibration
    draws: the mean over 200 repetitions of the fresh loss stays below alpha. (A single calibration draw
    may land above it: about a quarter of the draws do here, as the guarantee is not conditional.)"""
    rng = np.random.default_rng(0)
    H, losses = 20, []
    for _ in range(200):
        thr = alarms.conformal_threshold(_synthetic(rng, 50), 0.10, H=H)
        assert thr is not None
        losses.append(_mean_loss(_synthetic(rng, 200), thr, H))
    assert np.mean(losses) <= 0.10
    assert np.mean(np.array(losses) <= 0.10) >= 0.6


def test_conformal_threshold_infeasible():
    lab = _labels(100, [])
    eps = [(np.full(100, 0.9), lab) for _ in range(20)]        # every alarm is false at any threshold
    assert alarms.conformal_threshold(eps, 0.10, H=20) is None
    assert alarms.conformal_threshold(eps[:5], 0.10, H=20) is None   # 1/(n+1) > alpha


def test_conformal_scan_is_monotone():
    """A low threshold that happens to pass is not chosen when a higher one fails."""
    T, H = 200, 20
    good = _labels(T, [(100, 110, "building_up")])
    eps = []
    for _ in range(30):
        r = np.zeros(T)
        r[85:112] = 0.5                         # one true alarm event at 85 for thresholds up to 0.5
        r[105:107] = 0.8                        # inside it; a separate (false) event only above 0.5
        eps.append((r, good))
    # at 0.8: one event at 105, no onset in (105, 125] -> loss 1; at 0.5: one true event -> loss 0
    assert _mean_loss(eps, 0.5, H) == 0.0 and _mean_loss(eps, 0.8, H) == 1.0
    assert alarms.conformal_threshold(eps, 0.10, H=H) is None
    for r, _ in eps:
        r[105:107] = 0.5
    assert alarms.conformal_threshold(eps, 0.10, H=H) == pytest.approx(0.5)


def test_threshold_grid_capped():
    g = alarms.threshold_grid([np.linspace(0, 1, 10_001), np.array([np.nan, 0.5])], max_grid=100)
    assert len(g) <= 100 and g[0] == 0.0 and g[-1] == 1.0 and np.all(np.diff(g) > 0)
    assert len(alarms.threshold_grid([np.array([0.2, 0.2, 0.7])])) == 2


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_recall_diff():
    ca = np.array([1, 1, 0, 1, 1, 0, 1, 1])
    cb = np.array([0, 1, 0, 0, 1, 0, 0, 1])
    n = np.array([1, 1, 1, 1, 1, 0, 1, 1])
    b = alarms.bootstrap_recall_diff(ca, cb, n, reps=500)
    assert b["diff"] == pytest.approx(6 / 7 - 3 / 7)
    assert b["lo"] <= b["diff"] <= b["hi"]
    assert b == alarms.bootstrap_recall_diff(ca, cb, n, reps=500)         # fixed rng tag
    none = alarms.bootstrap_recall_diff(np.zeros(3), np.zeros(3), np.zeros(3), reps=50)
    assert none["diff"] is None and none["lo"] is None


# ------------------------------------------------------------------ the shadow script
@pytest.fixture(scope="module")
def shadow():
    return load_script("b_shadow")


def _result(recall_bu, fa, feasible=True):
    return {"threshold": 0.5 if feasible else None, "low": 0.35 if feasible else None,
            "selection": {"false_alarm_rate": fa,
                          "by_cause": {"building_up": {"recall": recall_bu}, "sudden": {"recall": 0.0}}}}


def test_choose_best_predictor(shadow):
    res = {"world_model": _result(0.5, 0.2), "hazard": _result(0.5, 0.1), "queueing": _result(0.9, 0.0, False),
           "change_point": _result(0.3, 0.0)}
    assert shadow.choose(res) == "hazard"
    assert shadow.choose({"queueing": _result(0.0, None, False)}) is None


def test_gate(shadow):
    assert shadow.gate(0.21)[0] == "GATE: pass"
    assert shadow.gate(0.2)[0] == "GATE: stop before prevention"
    assert shadow.gate(None)[0] == "GATE: stop before prevention"


def test_better_non_ml(shadow):
    res = {"queueing": _result(0.3, 0.2), "change_point": _result(0.4, 0.5), "hazard": _result(0.9, 0.0)}
    assert shadow.better_non_ml(res) == "change_point"


@pytest.mark.parametrize("args", [
    ["--calibration", "3100-3104", "--selection", "3450-3499"],                 # history as calibration
    ["--calibration", "3400-3449", "--selection", "3100-3104"],                 # history as selection
    ["--calibration", "3400-3420", "--selection", "3450-3499"],                 # part of the range
    ["--calibration", "3000-3049", "--selection", "3450-3499"],                 # held out
    ["--label", "smoke", "--calibration", "3000-3004", "--selection", "3114",
     "--smoke-dir", "data/phase_b/smoke_test_refuse"],                            # held out, even in smoke
    ["--label", "smoke", "--calibration", "3104,3108", "--selection", "3108",
     "--smoke-dir", "data/phase_b/smoke_test_refuse"],                            # overlap
    ["--label", "smoke", "--calibration", "3104", "--selection", "3114",
     "--fit-seeds", "3104,3105", "--smoke-dir", "data/phase_b/smoke_test_refuse"],  # fit overlaps
    ["--label", "smoke", "--calibration", "3104", "--selection", "3114"],       # no smoke dir
    ["--label", "other"],
])
def test_main_refuses(shadow, args):
    with pytest.raises(SystemExit) as e:
        shadow.main(args + ["--store", "/nonexistent"])
    assert "refusing" in str(e.value)


# ------------------------------------------------------------------ end to end on a synthetic store
def _store(root):
    from tests.test_phase_b_predictors import synth
    from pis import shards
    from pis.phase_b import features, history
    d = root / "store" / "s0"
    d.mkdir(parents=True)
    rng = np.random.default_rng(3)
    seeds = list(range(3100, 3112)) + list(range(3400, 3420)) + list(range(3450, 3456))
    for i, seed in enumerate(seeds):
        onset = int(rng.integers(90, 150)) if i % 2 else None
        df = synth(seed, onset)
        if onset is not None:
            df.loc[df.window >= onset - 30, "deg_active"] = True
        df[history.WINDOW_COLUMNS].to_parquet(d / f"windows-{seed}.parquet")
        ep = {"seed": seed, "kind": "none" if onset is None else "load_ramp", "source": "r6",
              "windows": len(df), **{c: float(df[c].iloc[0]) for c in features.KNOWN_COLS}}
        pd.DataFrame([ep]).to_parquet(d / f"episodes-{seed}.parquet")
    (d / shards.DONE).write_text("")
    return root / "store"


def _tiny_wm(path, train_seeds):
    from tests.test_phase_b_wm_prob import _tiny_model
    m = _tiny_model()
    m.meta["train_seeds"] = train_seeds
    m.save(path)
    return path


def test_shadow_end_to_end(shadow, tmp_path):
    store = _store(tmp_path)
    wm = _tiny_wm(tmp_path / "wm.pt", list(range(3100, 3108)))
    mc = pd.DataFrame({"episode": [3451, 3452, 3453], "window": [100, 60, 110], "risk": [0.6, 0.0, 0.3]})
    mc.to_parquet(tmp_path / "labels.parquet")
    rule = {"k": 3, "interval": 12, "majority": 0.5, "warmup": 20, "H": 20, "min_lead_windows": 2}
    args = dict(store=store, rule=rule, names=list(shadow.ALL),
                fit_seeds=list(range(3100, 3112)), calibration=list(range(3400, 3420)),
                selection=list(range(3450, 3456)), world_model=wm, wm_n=4, mc_path=tmp_path / "labels.parquet",
                out=tmp_path / "shadow", note=tmp_path / "note.md", alarms_path=tmp_path / "alarms.json",
                procs=1, label="smoke", overrides={"analogue": {"k": 5}, "hazard": {"iters": 50}})
    res = shadow.shadow(**args)
    raw = pd.concat([pd.read_parquet(p) for p in (tmp_path / "shadow").glob("s*/risks.parquet")])
    assert set(raw.episode) == set(range(3400, 3420)) | set(range(3450, 3456))
    assert raw[raw.window < 20][list(shadow.ALL)].isna().all().all()
    assert raw[raw.window >= 20][list(shadow.ALL)].notna().all().all()
    cfg = json.loads((tmp_path / "alarms.json").read_text())
    assert set(cfg) == {"predictor", "threshold", "low", "H", "min_lead_windows", "alpha"}
    assert cfg["predictor"] == res["best"] and cfg["predictor"] is not None      # the synthetic rise is easy
    assert cfg["low"] == pytest.approx(0.7 * cfg["threshold"])
    p = res["predictors"][cfg["predictor"]]
    assert p["calibration"]["adjusted_risk"] <= 0.10
    assert res["gate"][0] == ("GATE: pass" if p["selection"]["by_cause"]["building_up"]["recall"] > 0.2
                              else "GATE: stop before prevention")
    note = (tmp_path / "note.md").read_text()
    assert "GATE:" in note
    assert res["ms_per_window"] > 0
    assert "brier" in res["predictors"]["hazard"]["selection"]
    again = shadow.shadow(**args)                                  # finished shards are reused
    assert again["shards"]["skipped"] and not again["shards"]["done"]
    with pytest.raises(ValueError):
        shadow.shadow(**{**args, "rule": {**rule, "H": 10}})        # a changed configuration is refused


def test_shadow_refuses_world_model_trained_on_evaluated_episodes(shadow, tmp_path):
    store = _store(tmp_path)
    wm = _tiny_wm(tmp_path / "wm.pt", [3100, 3450])
    rule = {"k": 3, "interval": 12, "majority": 0.5, "warmup": 20, "H": 20, "min_lead_windows": 2}
    with pytest.raises(ValueError, match="trained on evaluated|history seeds only"):
        shadow.shadow(store, rule, ["world_model"], list(range(3100, 3112)), list(range(3400, 3406)),
                      list(range(3450, 3456)), wm, 4, None, tmp_path / "out", tmp_path / "n.md",
                      tmp_path / "a.json", 1)


def test_shadow_real_run_refuses_world_model_without_history_seeds(shadow, tmp_path):
    store = _store(tmp_path)
    rule = {"k": 3, "interval": 12, "majority": 0.5, "warmup": 20, "H": 20, "min_lead_windows": 2}
    for i, seen in enumerate(([], [3100, 3300 + 99, 3420])):
        wm = _tiny_wm(tmp_path / f"wm{i}.pt", seen)
        with pytest.raises(ValueError, match="history seeds only"):
            shadow.shadow(store, rule, ["world_model"], list(range(3100, 3112)), list(range(3400, 3406)),
                          list(range(3450, 3456)), wm, 4, None, tmp_path / "out", tmp_path / "n.md",
                          tmp_path / "a.json", 1)


def test_shadow_refuses_non_r6_episodes(shadow, tmp_path):
    store = _store(tmp_path)
    p = store / "s0" / "episodes-3450.parquet"
    pd.read_parquet(p).assign(source="random").to_parquet(p)
    rule = {"k": 3, "interval": 12, "majority": 0.5, "warmup": 20, "H": 20, "min_lead_windows": 2}
    with pytest.raises(ValueError, match="R6"):
        shadow.shadow(store, rule, ["queueing"], list(range(3100, 3112)), list(range(3400, 3406)),
                      list(range(3450, 3456)), None, 4, None, tmp_path / "out", tmp_path / "n.md",
                      tmp_path / "a.json", 1)
