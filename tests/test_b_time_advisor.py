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
def timing():
    return load_script("b_time_advisor")


def _args(**kw):
    base = {"seeds": "3450-3452", "duration": 300.0, "label": "", "smoke_dir": None, "selection": "model"}
    return {**base, **kw}


def test_summarise_percentiles(timing):
    s = timing.summarise(np.arange(1, 101, dtype=float))
    assert s["n"] == 100 and s["p50"] == pytest.approx(50.5) and s["max"] == 100.0
    assert s["p95"] == pytest.approx(np.percentile(np.arange(1, 101), 95))
    assert timing.summarise(np.zeros(0))["n"] == 0


def test_check_run(timing):
    assert timing.check_run(_args(), dirty=False, note_exists=False) == []
    assert timing.check_run(_args(), dirty=True, note_exists=False)
    assert timing.check_run(_args(), dirty=False, note_exists=True)
    assert timing.check_run(_args(duration=30.0), dirty=False, note_exists=False)
    assert timing.check_run(_args(seeds="3450-3451"), dirty=False, note_exists=False)       # three seeds
    assert timing.check_run(_args(selection="always"), dirty=False, note_exists=False)
    assert timing.check_run(_args(seeds="3100-3102"), dirty=False, note_exists=False)        # history
    assert timing.check_run(_args(smoke_dir="data/phase_b/smoke_x"), dirty=False, note_exists=False)
    smoke = _args(label="smoke", seeds="3100", duration=20.0, selection="always", smoke_dir="data/phase_b/smoke_x")
    assert timing.check_run(smoke, dirty=True, note_exists=True) == []
    assert timing.check_run({**smoke, "smoke_dir": None}, dirty=False, note_exists=False)
    assert timing.check_run({**smoke, "seeds": "3000"}, dirty=False, note_exists=False)       # held out


def test_main_refuses(timing):
    with pytest.raises(SystemExit) as e:
        timing.main(["--seeds", "3450-3451"])
    assert "refusing" in str(e.value)


def test_time_advisor_smoke_end_to_end(timing, tmp_path):
    from tests.test_phase_b_wm_prob import _tiny_model
    wm = _tiny_model(context=4, room=64, hi=np.full(43, 1e6, np.float32))
    sha = wm.save(tmp_path / "wm.pt")
    (tmp_path / "pred.pkl").write_bytes(pickle.dumps({"queueing": predictors.QueueingModel(IncidentRule(3, 12))}))
    inc = {"k": 3, "interval": 12, "majority": 0.5, "warmup": 20, "H": 6}
    alarms = {"predictor": "queueing", "threshold": 0.0, "low": 0.0, "H": 6}
    cfg = {"params": {"m": 2, "n": 4, "h": 6, "eps": 0.1, "slo_lam": 40.0, "mu": 8.0, "change_cost": 0.0},
           "alarms": alarms, "world_model_sha256": sha, "wm_n": 4}
    for name, obj in (("inc.json", inc), ("alarms.json", alarms), ("advisor.json", cfg)):
        (tmp_path / name).write_text(json.dumps(obj))
    res = timing.measure(seeds=[3450, 3451], duration_s=12.0, cfg=cfg, inc=inc, alarms=alarms,
                         world_model=tmp_path / "wm.pt", fitted=tmp_path / "pred.pkl", selection="model")
    assert set(res["per_seed"]) == {3450, 3451}
    assert res["all"]["n"] == 48 and res["in_mode"]["n"] == 8
    assert set(res) >= {"worst_p95", "worst_max", "p95_ok", "max_ok", "breakdown"}
    text = timing.note_text(res, {"label": "smoke", "seeds": "3450-3451", "duration": 12.0,
                                  "config": str(tmp_path / "advisor.json"), "selection": "model"}, cfg, sha,
                            alarms)
    assert sha in text and "p95" in text and "in intervention mode" in text


def test_frozen_inputs_of_a_real_run(timing):
    inc = {"k": 7, "interval": 28, "majority": 0.5, "warmup": 20, "H": 20, "chronic": 0.5}
    alarms = {"predictor": "hazard", "threshold": 0.4, "low": 0.28, "H": 20, "alpha": 0.1}
    cfg = {"alarms": {"predictor": "hazard", "threshold": 0.4, "low": 0.28, "H": 20},
           "rule": {"k": 7, "interval": 28, "majority": 0.5, "warmup": 20}}
    a = {"incident": "configs/phase_b/incident.json", "alarms": "configs/phase_b/alarms.json"}
    assert timing.frozen_problems(a, cfg, inc, alarms) == []
    assert timing.frozen_problems({**a, "alarms": "data/phase_b/smoke_x/alarms.json"}, cfg, inc, alarms)
    assert timing.frozen_problems({**a, "incident": "data/phase_b/smoke_x/incident.json"}, cfg, inc, alarms)
    assert timing.frozen_problems(a, cfg, inc, {**alarms, "threshold": 0.5})
    assert timing.frozen_problems(a, {**cfg, "alarms": None}, inc, alarms)
    assert timing.frozen_problems(a, cfg, {**inc, "k": 6}, alarms)


def _tune_raw(tmp_path):
    d = tmp_path / "tune" / "s0000"
    d.mkdir(parents=True)
    shares = {3450: 0.1, 3451: 0.4, 3452: 0.0, 3453: 0.4, 3454: 0.3, 3100: 0.9}
    eps = [{"seed": s, "arm": "eps0.1_n8", "mode_share": m} for s, m in shares.items()]
    eps += [{"seed": s, "arm": "eps0.2_n8", "mode_share": 1.0} for s in (3450, 3452)]
    (d / "episodes.json").write_text(json.dumps(eps))
    (d / "done").write_text("")
    return tmp_path / "tune"


def test_top_mode_seeds(timing, tmp_path):
    tune = _tune_raw(tmp_path)
    assert timing.top_mode_seeds(tune, "eps0.1_n8") == [3451, 3453, 3454]       # history seed 3100 ignored
    assert timing.top_mode_seeds(tune, "eps0.2_n8") == [3450, 3452]
    assert timing.top_mode_seeds(tune, None) == [] and timing.top_mode_seeds(tmp_path / "none", "x") == []


def test_real_run_times_the_seeds_with_the_most_intervention(timing, tmp_path):
    tune = _tune_raw(tmp_path)
    (tmp_path / "advisor.json").write_text(json.dumps({"chosen": "eps0.1_n8", "params": {"m": 4}}))
    with pytest.raises(SystemExit) as e:
        timing.main(["--config", str(tmp_path / "advisor.json"), "--tune-dir", str(tune), "--seeds", "3450-3452"])
    assert "refusing" in str(e.value) and "3451" in str(e.value)


def test_note_reports_in_mode_p95(timing):
    per = {3450: {"n": 10, "mode_windows": 2, "fallbacks": 0, "mean": 1.0, "p50": 1.0, "p95": 2.0, "max": 3.0,
                  "in_mode_p95": 7.5}}
    res = {"per_seed": per, "all": per[3450], "in_mode": {"n": 2, "p50": 7.0, "p95": 7.5, "max": 8.0},
           "breakdown": {"risk_ms_mean": 1.0, "risk_ms_p95": 1.0, "select_ms_mean_in_mode": 1.0,
                         "select_ms_p95_in_mode": 1.0},
           "wall_s": {3450: 1.0}, "worst_p95": 2.0, "worst_max": 3.0, "p95_ok": True, "max_ok": True,
           "worst_in_mode_p95": 7.5}
    text = timing.note_text(res, {"label": "smoke", "seeds": "3450", "duration": 5.0, "config": "c.json",
                                  "selection": "model"}, {"params": {}}, "x", {})
    assert "- in-mode p95 (worst episode) 7.5 ms" in text and "| 7.5 |" in text
    assert "- worst per-episode p95 2.0 ms: within" in text


def test_main_takes_world_model_and_fitted_from_the_alarms_config(timing, tmp_path, monkeypatch):
    from pis.phase_b import runs
    from tests.test_phase_b_wm_prob import _tiny_model
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    seen = {}

    def fake(seeds, duration_s, cfg, inc, alarms, world_model, fitted, selection="model"):
        seen.update(world_model=world_model, fitted=fitted)
        return {"p95_ok": True, "max_ok": True, "worst_p95": 1.0, "worst_max": 1.0}
    monkeypatch.setattr(timing, "measure", fake)
    monkeypatch.setattr(timing, "note_text", lambda *a, **kw: "note\n")
    _tiny_model().save(tmp_path / "w.pt")
    inc = {"k": 3, "interval": 12, "majority": 0.5, "warmup": 20, "H": 6}
    alarms = {"predictor": "queueing", "threshold": 0.0, "low": 0.0, "H": 6, "world_model": str(tmp_path / "w.pt"),
              "shadow_dir": str(tmp_path / "sh")}
    cfg = {"params": {"m": 2}, "alarms": {k: alarms[k] for k in ("predictor", "threshold", "low", "H")}}
    for name, obj in (("inc.json", inc), ("alarms.json", alarms), ("advisor.json", cfg)):
        (tmp_path / name).write_text(json.dumps(obj))
    args = ["--label", "smoke", "--seeds", "3450", "--duration", "10", "--config", str(tmp_path / "advisor.json"),
            "--incident", str(tmp_path / "inc.json"), "--alarms", str(tmp_path / "alarms.json"),
            "--tune-dir", str(tmp_path / "none"), "--smoke-dir", str(tmp_path / "data/phase_b/smoke_t")]
    timing.main(args)
    assert seen == {"world_model": str(tmp_path / "w.pt"), "fitted": str(tmp_path / "sh" / "predictors.pkl")}
    timing.main(args + ["--fitted", "x/p.pkl"])
    assert seen == {"world_model": str(tmp_path / "w.pt"), "fitted": "x/p.pkl"}
