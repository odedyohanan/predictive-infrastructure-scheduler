import json

import numpy as np
import pytest
import torch

from pis.config import setup_imports

setup_imports()

from pis.phase_b import features, history, incidents, predictors, wm_prob, wm_rssm, wm_transformer  # noqa: E402
from pis.phase_b import world_models  # noqa: E402
from tests.test_phase_b_wm_prob import _linear_episodes, _synthetic_store, _tiny_model, _window  # noqa: E402
from tests.test_phase_b_wm_rssm import _tiny_rssm  # noqa: E402


def _tiny_tf(seed=0, lo=None, hi=None, context=4, layers=2):
    torch.manual_seed(seed)
    net = wm_transformer.CausalTransformer(d_model=8, heads=2, ff=16, layers=layers, context=context)
    from worldmodel.model import Normalizer
    xn = Normalizer(np.zeros(79, np.float32), np.ones(79, np.float32))
    yn = Normalizer(np.zeros(43, np.float32), np.full(43, 5.0, np.float32))
    lo = np.zeros(43, np.float32) if lo is None else lo
    hi = np.full(43, 3.0, np.float32) if hi is None else hi
    return wm_transformer.TransformerWorldModel(net, xn, yn, {"context": context, "room": 0,
                                                              "state_min": lo.tolist(), "state_max": hi.tolist()})


# ------------------------------------------------------------------ learning on a known system
@pytest.fixture(scope="module")
def linear_fit():
    sigma = np.array([0.1, 0.3])
    tr = _linear_episodes(24, 120, sigma, 0)
    va = _linear_episodes(6, 120, sigma, 1)
    model, hist = wm_transformer.train(tr, va, context=8, burn_in=2, d_model=16, heads=2, ff=32, layers=2,
                                       lr=3e-3, batch=64, epochs=40, patience=40, seed=0)
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
    _, va, model, hist = linear_fit
    assert model.meta["kind"] == "transformer" and model.meta["room"] == 0 and model.meta["layers"] == 2
    assert model.meta["d_model"] == 16 and model.meta["heads"] == 2 and model.meta["ff"] == 32
    assert model.meta["epochs_run"] == len(hist) and model.meta["best_val_loss"] == min(h["val_loss"] for h in hist)
    sigma = np.array([0.1, 0.3])
    tr, va2 = _linear_episodes(4, 40, sigma, 0), _linear_episodes(2, 40, sigma, 1)
    kw = dict(context=8, burn_in=2, d_model=8, heads=2, ff=16, layers=1, epochs=2, batch=16, seed=3)
    a, ha = wm_transformer.train(tr, va2, **kw)
    b, hb = wm_transformer.train(tr, va2, **kw)
    assert [h["val_loss"] for h in ha] == [h["val_loss"] for h in hb]
    assert all(torch.equal(p, q) for p, q in zip(a.net.state_dict().values(), b.net.state_dict().values()))
    with pytest.raises(ValueError, match="room"):
        wm_transformer.train(tr, va2, **{**kw, "epochs": 1}, room=4)


def test_attention_is_causal():
    """Changing the inputs after position j leaves the outputs up to j unchanged."""
    torch.manual_seed(0)
    net = wm_transformer.CausalTransformer(in_dim=5, out_dim=3, d_model=8, heads=2, ff=16, layers=3, context=10)
    net.eval()
    x = torch.randn(2, 10, 5)
    y = x.clone()
    y[:, 6:] += torch.randn(2, 4, 5)
    with torch.no_grad():
        (ma, la), (mb, lb) = net(x), net(y)
        short, _ = net(x[:, :7])
    torch.testing.assert_close(ma[:, :6], mb[:, :6], rtol=0, atol=1e-6)
    torch.testing.assert_close(la[:, :6], lb[:, :6], rtol=0, atol=1e-6)
    assert (ma[:, 6:] - mb[:, 6:]).abs().max() > 1e-4
    torch.testing.assert_close(short, ma[:, :7], rtol=0, atol=1e-6)     # a prefix gives the same outputs
    with pytest.raises(ValueError, match="context"):
        net(torch.zeros(1, 11, 5))


def test_shared_prefix_pass_equals_the_full_pass():
    """Running the shared history prefix once gives the outputs of the full pass on every row."""
    torch.manual_seed(1)
    net = wm_transformer.CausalTransformer(in_dim=5, out_dim=3, d_model=8, heads=2, ff=16, layers=3, context=10)
    net.eval()
    for P, Q in ((0, 1), (0, 4), (6, 1), (6, 4), (9, 1)):
        prefix, suffix = torch.randn(P, 5), torch.randn(3, Q, 5)
        full = torch.cat([prefix.expand(3, -1, -1), suffix], dim=1)
        with torch.no_grad():
            mean, logvar = net(full)
            m2, l2 = net.last_shared(prefix, suffix)
        torch.testing.assert_close(m2, mean[:, -1], rtol=0, atol=1e-5)
        torch.testing.assert_close(l2, logvar[:, -1], rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="context"):
        net.last_shared(torch.zeros(8, 5), torch.zeros(1, 3, 5))


# ------------------------------------------------------------------ rollouts
def test_sample_rollouts_shape_and_clipping():
    m = _tiny_tf()
    w = _window()
    out = m.sample_rollouts(w, w.policies[-1], None, horizon=7, n=5, seed=1)
    assert out.shape == (5, 7, 43)
    assert out.min() >= 0.0 and out.max() <= 3.0
    assert np.ptp(out[:, -1], axis=0).max() > 0          # samples differ
    np.testing.assert_array_equal(out, m.sample_rollouts(w, w.policies[-1], None, horizon=7, n=5, seed=1))
    assert np.abs(out - m.sample_rollouts(w, w.policies[-1], None, horizon=7, n=5, seed=2)).max() > 0
    per_step = np.tile(w.policies[-1], (7, 1))
    np.testing.assert_array_equal(m.sample_rollouts(w, per_step, None, 7, 5, seed=1), out)
    short = features.Window(w.states[-1:], w.policies[-1:], w.known)            # one window of history works
    assert m.sample_rollouts(short, w.policies[-1], None, horizon=3, n=2, seed=0).shape == (2, 3, 43)


def test_sample_rollouts_batch_of_candidates():
    m = _tiny_tf(context=4)
    w = _window()
    p, q = w.policies[-1], w.policies[-2]
    one = m.sample_rollouts_batch(w, [p], None, horizon=6, n=5, seed=3)
    assert one.shape == (1, 5, 6, 43)
    np.testing.assert_array_equal(one[0], m.sample_rollouts(w, p, None, 6, 5, seed=3))
    two = m.sample_rollouts_batch(w, [p, q], None, horizon=6, n=5, seed=3)
    assert two.shape == (2, 5, 6, 43) and np.abs(two[0] - two[1]).max() > 0
    long = m.sample_rollouts_batch(w, [p, q], None, horizon=20, n=3, seed=3)    # the window slides
    assert long.shape == (2, 3, 20, 43) and np.isfinite(long).all() and long.max() <= 3.0


def test_rollout_step_matches_the_network_on_the_sliding_window():
    """Each sampled step is the Gaussian at the last position of the last `context` inputs, replayed by hand."""
    hi = np.full(43, 50.0, np.float32)
    m = _tiny_tf(seed=4, context=3, hi=hi)
    w = _window(T=6, seed=4)
    H, n, seed = 5, 2, 9
    out = m.sample_rollouts(w, w.policies[-1], None, horizon=H, n=n, seed=seed)
    k = features.known_vector(w.known)
    rng = np.random.default_rng(seed)
    inputs = [np.concatenate([w.states[j], w.policies[j + 1], k]) for j in range(len(w.states) - 1)]
    inputs = [np.broadcast_to(x, (n, len(x))) for x in inputs]
    s = np.broadcast_to(w.states[-1], (n, 43)).astype(np.float32)
    for h in range(H):
        inputs.append(np.concatenate([s, np.broadcast_to(w.policies[-1], (n, 33)), np.broadcast_to(k, (n, 3))], 1))
        win = np.stack(inputs[-3:], axis=1).astype(np.float32)
        mean, var = m.predict(win)
        noise = rng.standard_normal(mean.shape)
        s = np.clip(s + (mean + np.sqrt(var) * noise).astype(np.float32), 0.0, hi)
        np.testing.assert_allclose(out[:, h], s, rtol=1e-4, atol=1e-4)


def test_rollouts_condition_on_the_history_within_the_context():
    m = _tiny_tf(context=4, hi=np.full(43, 50.0, np.float32))
    w = _window(T=8, seed=4)
    near = features.Window(w.states.copy(), w.policies, w.known)
    near.states[-2] += 1.0                                       # inside the context: futures change
    far = features.Window(w.states.copy(), w.policies, w.known)
    far.states[1] += 1.0                                         # before the context: futures unchanged
    a = m.sample_rollouts(w, w.policies[-1], None, horizon=3, n=4, seed=0)
    assert np.abs(a - m.sample_rollouts(near, w.policies[-1], None, horizon=3, n=4, seed=0)).max() > 1e-4
    np.testing.assert_array_equal(a, m.sample_rollouts(far, w.policies[-1], None, horizon=3, n=4, seed=0))


def test_save_load_round_trip(tmp_path):
    m = _tiny_tf(seed=3)
    p = tmp_path / "m.pt"
    m.save(p)
    m2 = wm_transformer.TransformerWorldModel.load(p)
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
            wm_transformer.fit(st, [3100], bad, epochs=1)
        with pytest.raises(ValueError):
            wm_transformer.fit(st, bad, [3101], epochs=1)
    with pytest.raises(ValueError, match="overlap"):
        wm_transformer.fit(st, [3100, 3101], [3101], epochs=1)


def test_fit_from_store_runs_and_feeds_the_shared_tools(tmp_path):
    _synthetic_store(tmp_path, [3100, 3101, 3108])
    st = history.Store(tmp_path)
    model, hist = wm_transformer.fit(st, [3100, 3101], [3108], context=8, burn_in=2, d_model=8, heads=2, ff=16,
                                     layers=1, epochs=2)
    assert len(hist) == 2 and model.bounds[0].shape == (43,)
    assert model.meta["train_seeds"] == [3100, 3101] and model.meta["kind"] == "transformer"
    r = wm_prob.predictive_scores(model, st, [3108], horizon=3, n=4, every=10, warmup=8)
    assert r["starts"] > 0
    eps = st.episodes()
    rows = features.join_known(st.window_rows(3108, 0, 20), eps)
    rule = incidents.IncidentRule(k=3, interval=12)
    pred = predictors.WorldModelPredictor(model, rule, n=4, warmup=4)
    assert 0.0 <= pred.risk(rows, 5) <= 1.0


# ------------------------------------------------------------------ loading by kind
def test_loader_dispatches_on_the_kind(tmp_path):
    t, r, g = _tiny_tf(seed=1), _tiny_rssm(seed=1), _tiny_model(seed=1)
    for name, m in (("t", t), ("r", r), ("g", g)):
        m.save(tmp_path / f"{name}.pt")
    lt = world_models.load_world_model(tmp_path / "t.pt")
    assert type(lt) is wm_transformer.TransformerWorldModel
    assert type(world_models.load_world_model(tmp_path / "r.pt")) is wm_rssm.RSSMWorldModel
    assert type(world_models.load_world_model(tmp_path / "g.pt")) is wm_prob.ProbWorldModel
    assert world_models.kind_of(tmp_path / "t.pt") == "transformer"
    w = _window()
    np.testing.assert_array_equal(lt.sample_rollouts(w, w.policies[-1], None, 4, 3, seed=1),
                                  t.sample_rollouts(w, w.policies[-1], None, 4, 3, seed=1))
    with pytest.raises(ValueError, match="transformer"):
        wm_prob.ProbWorldModel.load(tmp_path / "t.pt")
    with pytest.raises(ValueError, match="rssm"):
        wm_transformer.TransformerWorldModel.load(tmp_path / "r.pt")
    assert world_models.trainer("transformer") is wm_transformer.fit and "transformer" in world_models.ARCHS


# ------------------------------------------------------------------ training script
def test_script_smoke_run_trains_a_transformer(tmp_path, monkeypatch):
    from pis.phase_b import runs
    from tests._load import load_script
    script = load_script("b_train_world_model")
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    _synthetic_store(tmp_path / "store", list(range(3100, 3110)))
    inc = tmp_path / "incident.json"
    inc.write_text(json.dumps({"k": 3, "interval": 12, "majority": 0.5, "warmup": 8, "H": 3}))
    sdir, mdir = tmp_path / "data/phase_b/smoke_t", tmp_path / "models/smoke_t"
    args = ["--arch", "transformer", "--label", "smoke", "--seeds", "3100-3109", "--store",
            str(tmp_path / "store"), "--calib-seeds", "3109", "--incident", str(inc), "--mc-labels",
            str(tmp_path / "none.parquet"), "--smoke-dir", str(sdir), "--models-dir", str(mdir),
            "--horizon", "3", "--epochs", "2", "--d-model", "8", "--heads", "2", "--ff", "16", "--layers", "1",
            "--context", "8", "--burn-in", "2", "--samples", "4", "--every", "10", "--rank-states", "0"]
    res = script.main(args)
    rec = res["recipe"]
    assert rec["arch"] == "transformer" and rec["d_model"] == 8 and rec["heads"] == 2 and rec["ff"] == 16
    assert rec["layers"] == 1 and rec["room"] == 0 and rec["dropout"] == 0.0 and "hidden" not in rec
    assert rec["trainer"] == "pis.phase_b.wm_transformer.fit" and rec["out"] == "tf_b3.pt"
    m = world_models.load_world_model(mdir / "tf_b3.pt")
    assert isinstance(m, wm_transformer.TransformerWorldModel) and m.meta["sha256"] == rec["sha256"]
    note = (sdir / "wm_tf_b3.md").read_text()
    assert note.startswith("# Causal transformer world model tf_b3")
    assert "d_model 8" in note and "heads 2" in note and "feed-forward 16" in note and "Ladder criterion" in note
    assert json.loads((mdir / "tf_b3.recipe.json").read_text())["arch"] == "transformer"
    with pytest.raises(SystemExit, match="room"):
        script.main(args + ["--room", "4"])


def test_script_defaults_by_arch():
    from tests._load import load_script
    script = load_script("b_train_world_model")
    assert script.DEFAULT_OUT["transformer"] == "models/tf_b3.pt"
    assert script.DEFAULT_LAYERS == {"pgru": 2, "rssm": 1, "transformer": 4}
    assert script.DEFAULT_ROOM == {"pgru": 32, "rssm": 32, "transformer": 0}
