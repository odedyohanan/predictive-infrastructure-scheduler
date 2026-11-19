from pathlib import Path

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.ensemble import DISAGREEMENT_KEYS, Ensemble, disagreement  # noqa: E402
from pis.forecaster import load_world_model, rollout_batch  # noqa: E402

SEED = 1100
GATE_MODEL = "worldmodel:models/gru_mix_bounded.pt"


def _policies(n: int) -> list[dict]:
    from sched.controllers import isolated_policy
    from sched.dataset import perturb_policy, policy_dict
    spec = adapters.make_spec(SEED)
    rng = np.random.default_rng(3)
    p = isolated_policy()
    out = [policy_dict(p)]
    for _ in range(n - 1):
        p = perturb_policy(p, rng, spec)
        out.append(policy_dict(p))
    return out


def _history(n: int) -> list[tuple[dict, dict]]:
    from sim.policy import policy_from_dict
    sim = adapters.make_sim(SEED)
    pols = _policies(n + 1)
    tels = [sim.run_window(policy_from_dict(pols[0]))]
    for pd in pols[1:n]:
        tels.append(sim.run_window(policy_from_dict(pd)))
    return list(zip(tels, pols[1:]))


@pytest.fixture(scope="module")
def ens2():
    return Ensemble.from_paths([GATE_MODEL, GATE_MODEL])


def test_disagreement_zero_for_identical_members():
    rng = np.random.default_rng(0)
    one = rng.normal(size=(1, 4, 3, 39)).astype(np.float32)
    roll = np.repeat(one, 5, axis=0)
    d = disagreement(roll, DISAGREEMENT_KEYS, np.ones(39, np.float32))
    assert d.shape == (4,)
    assert np.allclose(d, 0.0)


def test_disagreement_increases_with_member_spread():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(1, 3, 2, 39)).astype(np.float32)
    noise = rng.normal(size=(5, 1, 1, 39)).astype(np.float32)
    scales = np.array([0.0, 0.1, 1.0], np.float32)[None, :, None, None]
    roll = base + noise * scales                      # candidate i spreads by scales[i]
    d = disagreement(roll, DISAGREEMENT_KEYS, np.ones(39, np.float32))
    assert d[0] < d[1] < d[2]
    # normalisation: doubling a key's training std halves its contribution
    d2 = disagreement(roll, DISAGREEMENT_KEYS, np.full(39, 2.0, np.float32))
    assert np.allclose(d2, d / 2.0, rtol=1e-5)


def test_disagreement_non_finite_is_inf():
    roll = np.zeros((3, 2, 2, 39), np.float32)
    roll[1, 1, 0, 0] = np.nan
    d = disagreement(roll, DISAGREEMENT_KEYS, np.ones(39, np.float32))
    assert d[0] == 0.0 and np.isinf(d[1])


@pytest.mark.parametrize("n_hist", [12, 1])
def test_shapes(ens2, n_hist):
    hist, known = _history(n_hist), adapters.known_of(adapters.make_spec(SEED))
    out = ens2.rollout_batch(hist, _policies(3), known, 4)
    assert out.shape == (2, 3, 4, 39)
    assert np.isfinite(out).all()
    assert ens2.disagreement(out).shape == (3,)
    assert ens2.train_std.shape == (39,) and (ens2.train_std > 0).all()
    assert ens2.history == 8 and len(ens2) == 2


@pytest.mark.parametrize("n_hist", [12, 2, 1])
def test_ensemble_rollout_member_0_equals_forecaster(ens2, n_hist):
    hist, known = _history(n_hist), adapters.known_of(adapters.make_spec(SEED))
    cands = _policies(4)
    ref = rollout_batch(load_world_model(), hist, cands, known, 5)
    out = ens2.rollout_batch(hist, cands, known, 5)
    assert np.array_equal(out[0], ref)
    assert np.array_equal(out[1], ref)              # same checkpoint twice: identical members
    assert np.allclose(ens2.disagreement(out), 0.0)


def test_describe_has_no_absolute_paths(ens2):
    rows = ens2.describe()
    assert len(rows) == 2
    assert all(r["path"] == GATE_MODEL and len(r["sha256"]) == 64 for r in rows)


def test_train_ensemble_guards(tmp_path):
    from tests._load import load_script
    te = load_script("train_ensemble")
    assert te.parse_members("1-4") == [1, 2, 3, 4] and te.parse_members("3,1") == [1, 3]
    with pytest.raises(ValueError):
        te.parse_members("0-2")                      # member 0 is the gate-passing model
    ok = {"label": "", "out_dir": "models", "force": False}
    assert te.check_run(ok, dirty=False, existing=[]) == []
    assert te.check_run(ok, dirty=True, existing=[])                     # dirty tree refused
    assert te.check_run(ok, dirty=False, existing=["models/ens_1.pt"])   # no silent overwrite
    assert te.check_run({**ok, "label": "smoke"}, dirty=True, existing=[])   # smoke never into models/
    assert te.check_run({**ok, "label": "smoke", "out_dir": str(tmp_path)}, dirty=True, existing=[]) == []
    assert te.RECIPE == {"arch": "gru", "history": 8, "hidden": 128, "layers": 2, "lr": 1e-3,
                         "batch": 512, "epochs": 300, "patience": 20}


def test_train_ensemble_smoke_guard_uses_repo_root_and_data_string(tmp_path, monkeypatch):
    from tests._load import ROOT, load_script
    te = load_script("train_ensemble")
    monkeypatch.chdir(tmp_path)                       # cwd is not the repo
    smoke = {"label": "smoke", "force": False}
    assert te.check_run({**smoke, "out_dir": str(ROOT / "models")}, dirty=False, existing=[])
    assert te.check_run({**smoke, "out_dir": "models"}, dirty=False, existing=[]) == []   # tmp/models
    job = {"member": 2, "seed": 2, "data": ["worldmodel:data/adaptive_v1", "old_data:teacher_v2"],
           "out_dir": "models", "code": "git HEAD x", "label": "", "recipe": dict(te.RECIPE)}
    rec = te.member_record(job, Path("models/ens_2.pt"))
    assert rec["data"] == "worldmodel:data/adaptive_v1,old_data:teacher_v2" and rec["seed"] == 2
