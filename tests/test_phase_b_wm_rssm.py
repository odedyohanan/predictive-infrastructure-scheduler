import numpy as np
import pandas as pd
import pytest
import torch

from pis.config import setup_imports

setup_imports()

from pis.phase_b import features, history, incidents, predictors, wm_prob, wm_rssm, world_models  # noqa: E402
from tests.test_phase_b_wm_prob import _linear_episodes, _synthetic_store, _tiny_model, _window  # noqa: E402


def _tiny_rssm(seed=0, lo=None, hi=None, context=4, room=0, latent=4):
    torch.manual_seed(seed)
    net = wm_rssm.RSSM(hidden=8, latent=latent, layers=1)
    from worldmodel.model import Normalizer
    xn = Normalizer(np.zeros(79, np.float32), np.ones(79, np.float32))
    yn = Normalizer(np.zeros(43, np.float32), np.full(43, 5.0, np.float32))
    lo = np.zeros(43, np.float32) if lo is None else lo
    hi = np.full(43, 3.0, np.float32) if hi is None else hi
    return wm_rssm.RSSMWorldModel(net, xn, yn, {"context": context, "room": room, "state_min": lo.tolist(),
                                                 "state_max": hi.tolist()})


# ------------------------------------------------------------------ learning on a known system
@pytest.fixture(scope="module")
def linear_fit():
    sigma = np.array([0.1, 0.3])
    tr = _linear_episodes(24, 120, sigma, 0)
    va = _linear_episodes(6, 120, sigma, 1)
    model, hist = wm_rssm.train(tr, va, context=8, burn_in=2, room=0, hidden=16, latent=4, layers=1, lr=1e-2,
                                batch=64, epochs=40, patience=40, seed=0)
    return sigma, va, model, hist


def test_nll_decreases_and_variance_matches_noise(linear_fit):
    sigma, va, model, hist = linear_fit
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert min(h["val_loss"] for h in hist) < hist[0]["val_loss"]
    x = np.stack([e[0][40:48] for e in va])
    mean, var = model.predict(x)
    ratio = var.mean(axis=0) / sigma ** 2
    assert np.all(np.abs(ratio - 1) < 0.3), ratio
    s, u = x[:, -1, :2], x[:, -1, 2:3]                   # the mean follows the system: 0.8 s + 0.5 u - s
    truth = -0.2 * s + 0.5 * u
    assert np.abs(mean - truth).mean() < 0.5 * sigma.mean()


def test_training_records_the_recipe_and_is_deterministic(linear_fit):
    _, _, model, hist = linear_fit
    assert model.meta["kind"] == "rssm" and model.meta["free_bits"] == 1.0 and model.meta["latent"] == 4
    assert model.meta["epochs_run"] == len(hist) and model.meta["best_val_loss"] == min(h["val_loss"] for h in hist)
    sigma = np.array([0.1, 0.3])
    tr, va = _linear_episodes(4, 40, sigma, 0), _linear_episodes(2, 40, sigma, 1)
    a, ha = wm_rssm.train(tr, va, context=8, burn_in=2, room=4, hidden=8, latent=2, epochs=2, batch=16, seed=3)
    b, hb = wm_rssm.train(tr, va, context=8, burn_in=2, room=4, hidden=8, latent=2, epochs=2, batch=16, seed=3)
    assert [h["val_loss"] for h in ha] == [h["val_loss"] for h in hb]
    assert all(torch.equal(p, q) for p, q in zip(a.net.state_dict().values(), b.net.state_dict().values()))


def test_free_bits_floor_the_kl_term():
    torch.manual_seed(0)
    net = wm_rssm.RSSM(in_dim=3, out_dim=2, hidden=4, latent=2, layers=1)
    x, y = torch.zeros(2, 5, 3), torch.zeros(2, 5, 2)
    m, keep = torch.ones(2, 5, 2), torch.ones(2, 5, 1)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        nll, kl = wm_rssm.elbo_terms(net, x, y, m, keep, generator=g)
    low = wm_rssm.objective(nll, kl * 0.0, free_bits=1.0, out_dim=2)
    assert float(low) == pytest.approx(float(nll) + 0.5)               # KL below the free bits counts as 1 nat
    high = wm_rssm.objective(nll, kl * 0.0 + 3.0, free_bits=1.0, out_dim=2)
    assert float(high) == pytest.approx(float(nll) + 1.5)


# ------------------------------------------------------------------ rollouts
def test_sample_rollouts_shape_and_clipping():
    m = _tiny_rssm()
    w = _window()
    out = m.sample_rollouts(w, w.policies[-1], None, horizon=7, n=5, seed=1)
    assert out.shape == (5, 7, 43)
    assert out.min() >= 0.0 and out.max() <= 3.0
    assert np.ptp(out[:, -1], axis=0).max() > 0          # samples differ
    np.testing.assert_array_equal(out, m.sample_rollouts(w, w.policies[-1], None, horizon=7, n=5, seed=1))
    assert np.abs(out - m.sample_rollouts(w, w.policies[-1], None, horizon=7, n=5, seed=2)).max() > 0
    per_step = np.tile(w.policies[-1], (7, 1))
    np.testing.assert_array_equal(m.sample_rollouts(w, per_step, None, 7, 5, seed=1), out)


def test_sample_rollouts_batch_of_candidates():
    m = _tiny_rssm(context=4, room=8)
    w = _window()
    p, q = w.policies[-1], w.policies[-2]
    one = m.sample_rollouts_batch(w, [p], None, horizon=6, n=5, seed=3)
    assert one.shape == (1, 5, 6, 43)
    np.testing.assert_array_equal(one[0], m.sample_rollouts(w, p, None, 6, 5, seed=3))
    two = m.sample_rollouts_batch(w, [p, q], None, horizon=6, n=5, seed=3)
    assert two.shape == (2, 5, 6, 43) and np.abs(two[0] - two[1]).max() > 0
    long = m.sample_rollouts_batch(w, [p, q], None, horizon=20, n=3, seed=3)   # restarts past the room
    assert long.shape == (2, 3, 20, 43) and np.isfinite(long).all() and long.max() <= 3.0


def test_rollouts_condition_on_the_observed_history():
    """The history is filtered through the posterior: a different past changes the futures (same seed)."""
    m = _tiny_rssm(context=6, hi=np.full(43, 50.0, np.float32))
    w = _window(T=6, seed=4)
    other = features.Window(w.states.copy(), w.policies, w.known)
    other.states[1] += 1.0                                       # same newest state, another past
    a = m.sample_rollouts(w, w.policies[-1], None, horizon=3, n=4, seed=0)
    b = m.sample_rollouts(other, w.policies[-1], None, horizon=3, n=4, seed=0)
    assert np.abs(a - b).max() > 1e-4
    short = features.Window(w.states[-1:], w.policies[-1:], w.known)            # one window of history works
    assert m.sample_rollouts(short, w.policies[-1], None, horizon=3, n=2, seed=0).shape == (2, 3, 43)


def test_restart_beyond_the_room_refilters_the_last_context():
    hist = _window(T=10, seed=4)
    a = _tiny_rssm(seed=2, context=3, room=2, hi=np.full(43, 50.0, np.float32))
    b = _tiny_rssm(seed=2, context=3, room=100, hi=np.full(43, 50.0, np.float32))
    ra = a.sample_rollouts(hist, hist.policies[-1], None, horizon=12, n=3, seed=5)
    rb = b.sample_rollouts(hist, hist.policies[-1], None, horizon=12, n=3, seed=5)
    np.testing.assert_array_equal(ra[:, :3], rb[:, :3])          # identical until the first restart
    assert np.abs(ra - rb).max() > 1e-4


def test_save_load_round_trip(tmp_path):
    m = _tiny_rssm(seed=3)
    p = tmp_path / "m.pt"
    m.save(p)
    m2 = wm_rssm.RSSMWorldModel.load(p)
    w = _window()
    a = m.sample_rollouts(w, w.policies[-1], None, horizon=4, n=3, seed=7)
    np.testing.assert_array_equal(a, m2.sample_rollouts(w, w.policies[-1], None, horizon=4, n=3, seed=7))
    assert m2.meta["sha256"] and len(m2.meta["sha256"]) == 64 and m2.context == 4
    assert m2.net.params() == m.net.params()


# ------------------------------------------------------------------ store training
def test_training_refuses_seeds_outside_history(tmp_path):
    _synthetic_store(tmp_path, [3100, 3101])
    st = history.Store(tmp_path)
    for bad in ([3100, 3400], [3450], [3000], [1100]):
        with pytest.raises(ValueError):
            wm_rssm.fit(st, [3100], bad, epochs=1)
        with pytest.raises(ValueError):
            wm_rssm.fit(st, bad, [3101], epochs=1)
    with pytest.raises(ValueError, match="overlap"):
        wm_rssm.fit(st, [3100, 3101], [3101], epochs=1)


def test_fit_from_store_runs_and_feeds_the_shared_tools(tmp_path):
    _synthetic_store(tmp_path, [3100, 3101, 3108])
    st = history.Store(tmp_path)
    model, hist = wm_rssm.fit(st, [3100, 3101], [3108], context=8, burn_in=2, room=8, hidden=8, latent=2,
                              layers=1, epochs=2)
    assert len(hist) == 2 and model.bounds[0].shape == (43,)
    assert model.meta["train_seeds"] == [3100, 3101] and model.meta["kind"] == "rssm"
    r = wm_prob.predictive_scores(model, st, [3108], horizon=3, n=4, every=10, warmup=8)
    assert r["starts"] > 0
    eps = st.episodes()
    rows = features.join_known(st.window_rows(3108, 0, 20), eps)
    rule = incidents.IncidentRule(k=3, interval=12)
    pred = predictors.WorldModelPredictor(model, rule, n=4, warmup=4)
    risk = pred.risk(rows, 5)
    assert 0.0 <= risk <= 1.0


# ------------------------------------------------------------------ loading by kind
def test_loader_dispatches_on_the_kind(tmp_path):
    r, g = _tiny_rssm(seed=1), _tiny_model(seed=1)
    r.save(tmp_path / "r.pt")
    g.save(tmp_path / "g.pt")
    assert isinstance(world_models.load_world_model(tmp_path / "r.pt"), wm_rssm.RSSMWorldModel)
    lg = world_models.load_world_model(tmp_path / "g.pt")
    assert type(lg) is wm_prob.ProbWorldModel
    assert world_models.kind_of(tmp_path / "r.pt") == "rssm" and world_models.kind_of(tmp_path / "g.pt") == "pgru"
    with pytest.raises(ValueError, match="rssm"):
        wm_prob.ProbWorldModel.load(tmp_path / "r.pt")
    with pytest.raises(ValueError, match="pgru"):
        wm_rssm.RSSMWorldModel.load(tmp_path / "g.pt")


def test_loader_reads_a_legacy_checkpoint_without_kind(tmp_path):
    """Checkpoints written before kinds existed (the rung-1 model) load as the probabilistic GRU."""
    g = _tiny_model(seed=2)
    p = tmp_path / "legacy.pt"
    torch.save({"state_dict": g.net.state_dict(), "arch": g.net.params(),
                "x_norm": (g.x_norm.mean, g.x_norm.std), "y_norm": (g.y_norm.mean, g.y_norm.std),
                "meta": dict(g.meta)}, p)
    m = world_models.load_world_model(p)
    assert type(m) is wm_prob.ProbWorldModel and m.meta["sha256"] == wm_prob.sha256(p)
    w = _window()
    np.testing.assert_array_equal(m.sample_rollouts(w, w.policies[-1], None, 4, 3, seed=1),
                                  g.sample_rollouts(w, w.policies[-1], None, 4, 3, seed=1))
    with pytest.raises(ValueError, match="unknown"):
        torch.save({"kind": "transformer"}, tmp_path / "x.pt")
        world_models.load_world_model(tmp_path / "x.pt")


def test_fit_by_arch():
    assert world_models.trainer("pgru") is wm_prob.fit and world_models.trainer("rssm") is wm_rssm.fit
    with pytest.raises(ValueError):
        world_models.trainer("lstm")
    assert pd.api.types.is_list_like(world_models.ARCHS)


# ------------------------------------------------------------------ training script
def test_script_smoke_run_trains_an_rssm(tmp_path, monkeypatch):
    import json
    from pis.phase_b import runs
    from tests._load import load_script
    script = load_script("b_train_world_model")
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    _synthetic_store(tmp_path / "store", list(range(3100, 3110)))
    inc = tmp_path / "incident.json"
    inc.write_text(json.dumps({"k": 3, "interval": 12, "majority": 0.5, "warmup": 8, "H": 3}))
    sdir, mdir = tmp_path / "data/phase_b/smoke_t", tmp_path / "models/smoke_t"
    res = script.main(["--arch", "rssm", "--label", "smoke", "--seeds", "3100-3109", "--store",
                       str(tmp_path / "store"), "--calib-seeds", "3109", "--incident", str(inc), "--mc-labels",
                       str(tmp_path / "none.parquet"), "--smoke-dir", str(sdir), "--models-dir", str(mdir),
                       "--horizon", "3", "--epochs", "2", "--hidden", "8", "--latent", "2", "--context", "8",
                       "--burn-in", "2", "--room", "4", "--samples", "4", "--every", "10", "--rank-states", "0"])
    rec = res["recipe"]
    assert rec["arch"] == "rssm" and rec["latent"] == 2 and rec["free_bits"] == 1.0 and rec["layers"] == 1
    assert rec["trainer"] == "pis.phase_b.wm_rssm.fit" and rec["out"] == "rssm_b2.pt"
    from pis.phase_b.world_models import load_world_model
    m = load_world_model(mdir / "rssm_b2.pt")
    assert isinstance(m, wm_rssm.RSSMWorldModel) and m.meta["sha256"] == rec["sha256"]
    note = (sdir / "wm_rssm_b2.md").read_text()
    assert note.startswith("# Recurrent state-space (RSSM) world model rssm_b2")
    assert "free bits" in note and "latent 2" in note and "Ladder criterion" in note
    assert json.loads((mdir / "rssm_b2.recipe.json").read_text())["arch"] == "rssm"
