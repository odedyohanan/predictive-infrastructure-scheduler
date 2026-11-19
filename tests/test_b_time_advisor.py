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
