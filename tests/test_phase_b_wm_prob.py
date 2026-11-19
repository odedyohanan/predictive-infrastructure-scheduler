import dataclasses

import numpy as np
import pandas as pd
import pytest

from pis.config import setup_imports

setup_imports()

from pis import shards  # noqa: E402
from pis.phase_b import features, history, scenarios, wm_prob  # noqa: E402
from pis.phase_b.episode import run_episode  # noqa: E402
from pis.phase_b.incidents import IncidentRule  # noqa: E402
from sched.randomize import known_params  # noqa: E402
from sim.slo import window_violation  # noqa: E402
from tests._load import load_script  # noqa: E402
from worldmodel.features import KNOWN_NAMES, POLICY_NAMES, STATE_NAMES  # noqa: E402

script = load_script("b_train_world_model")


# ------------------------------------------------------------------ features
def test_feature_dimensions():
    assert features.STATE_B[:39] == STATE_NAMES
    assert features.STATE_B[39:] == ["a.queue_overdue", "a.late_new", "a.completed_ontime", "a.arrivals"]
    assert len(features.STATE_B) == features.N_STATE == 43
    assert features.INPUT_B == features.STATE_B + POLICY_NAMES + KNOWN_NAMES
    assert len(features.INPUT_B) == features.N_INPUT == 79


class _Hold:
    def decide(self, tel, current, history):
        return current, {"decide_ms": 0.0}


@pytest.fixture(scope="module")
def overloaded():
    """A short episode whose A load is tripled half-way: quiet windows, then violations."""
    sc = scenarios.scenario_for(3100)
    sc = dataclasses.replace(sc, duration_s=12.0,
                             degradation=scenarios.Degradation("load_ramp", 5.0, 0.0, 2.0))
    recs = run_episode(sc, _Hold(), "hold")
    return sc, recs


def test_predicted_violation_agrees_with_slo_v2(overloaded):
    sc, recs = overloaded
    win = features.encode_records(recs, known_params(sc.spec))
    pred = features.predicted_violation(features.decode(win.states))
    truth = np.array([window_violation(r["tel"], sc.spec.a.slo_p99_ms, "v2") for r in recs])
    assert truth.any() and not truth.all()
    np.testing.assert_array_equal(pred, truth)
    for r, s in zip(recs, features.decode(win.states)):      # one decoded window at a time too
        assert features.predicted_violation(s) == window_violation(r["tel"], sc.spec.a.slo_p99_ms, "v2")


def test_store_rows_and_records_encode_alike(overloaded):
    sc, recs = overloaded
    frame = history.window_frame(sc, recs, "hold")
    known = known_params(sc.spec)
    a = features.encode_rows(frame, known)
    b = features.encode_records(recs, known)
    np.testing.assert_allclose(a.states, b.states, rtol=1e-6)
    np.testing.assert_allclose(a.policies, b.policies)
    np.testing.assert_allclose(a.known, b.known)
    rows = features.rows_from_records(recs, known)
    np.testing.assert_allclose(features.encode_rows(rows).states, a.states, rtol=1e-6)
    np.testing.assert_array_equal(rows.slo_v2_violation.to_numpy(), frame.slo_v2_violation.to_numpy())


def test_steps_pair_state_with_next_policy():
    T = 6
    s = np.arange(T, dtype=np.float32)[:, None] * np.ones((1, 43), np.float32)
    p = np.arange(T, dtype=np.float32)[:, None] * np.ones((1, 33), np.float32) / 10
    w = features.Window(s, p, np.array([1.0, 2.0, 3.0], np.float32))
    x, y, m = features.steps(w)
    assert x.shape == (T - 1, 79) and y.shape == (T - 1, 43) and m.shape == (T - 1, 43)
    np.testing.assert_allclose(x[:, 0], np.arange(T - 1))                 # state_t
    np.testing.assert_allclose(x[:, 43], np.arange(1, T) / 10)            # policy of window t+1
    np.testing.assert_allclose(x[:, -3:], np.tile([1, 2, 3], (T - 1, 1)))
    np.testing.assert_allclose(y, np.ones((T - 1, 43)))                   # next state - state


def test_mask_drops_latency_without_completions():
    s = np.ones((2, 43), np.float32)
    s[1, STATE_NAMES.index("a.p99_ms")] = 0.0
    m = features.target_mask(s)
    assert m[0].all()
    assert m[1, STATE_NAMES.index("a.p99_ms")] == 0 and m[1, STATE_NAMES.index("a.p50_ms")] == 0
    assert m[1].sum() == 41


# ------------------------------------------------------------------ learning on a known system
def _linear_episodes(n_eps, T, sigma, seed):
    """s' = 0.8 s + 0.5 u + noise; X = [s, u, known], Y = s' - s."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_eps):
        s = np.zeros((T, 2))
        u = rng.uniform(-1, 1, size=(T, 1))
        for t in range(T - 1):
            s[t + 1] = 0.8 * s[t] + 0.5 * u[t] + rng.normal(0, sigma, size=2)
        x = np.concatenate([s[:-1], u[:-1], np.ones((T - 1, 1))], axis=1).astype(np.float32)
        y = (s[1:] - s[:-1]).astype(np.float32)
        out.append((x, y, np.ones_like(y)))
    return out


def test_nll_decreases_and_variance_matches_noise():
    sigma = np.array([0.1, 0.3])
    tr = _linear_episodes(24, 120, sigma, 0)
    va = _linear_episodes(6, 120, sigma, 1)
    model, hist = wm_prob.train(tr, va, context=8, burn_in=2, hidden=16, layers=1, lr=1e-2, batch=64,
                                epochs=40, patience=40, seed=0)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert min(h["val_loss"] for h in hist) < hist[0]["val_loss"]
    x = np.stack([e[0][40:48] for e in va])
    _, var = model.predict(x)
    ratio = var.mean(axis=0) / sigma ** 2
    assert np.all(np.abs(ratio - 1) < 0.3), ratio


# ------------------------------------------------------------------ rollouts
def _tiny_model(seed=0, lo=None, hi=None, context=4, room=0):
    import torch
    torch.manual_seed(seed)
    net = wm_prob.ProbGRU(hidden=8, layers=1)
    from worldmodel.model import Normalizer
    xn = Normalizer(np.zeros(79, np.float32), np.ones(79, np.float32))
    yn = Normalizer(np.zeros(43, np.float32), np.full(43, 5.0, np.float32))
    lo = np.zeros(43, np.float32) if lo is None else lo
    hi = np.full(43, 3.0, np.float32) if hi is None else hi
    return wm_prob.ProbWorldModel(net, xn, yn, {"context": context, "room": room, "state_min": lo.tolist(),
                                                  "state_max": hi.tolist()})


def _window(T=6, seed=0):
    rng = np.random.default_rng(seed)
    return features.Window(rng.uniform(0, 2, (T, 43)).astype(np.float32),
                           rng.uniform(0, 1, (T, 33)).astype(np.float32), np.array([1.0, 0.2, 30.0], np.float32))


def test_sample_rollouts_shape_and_clipping():
    m = _tiny_model()
    out = m.sample_rollouts(_window(), _window().policies[-1], None, horizon=7, n=5, seed=1)
    assert out.shape == (5, 7, 43)
    assert out.min() >= 0.0 and out.max() <= 3.0
    assert np.ptp(out[:, -1], axis=0).max() > 0          # samples differ
    again = m.sample_rollouts(_window(), _window().policies[-1], None, horizon=7, n=5, seed=1)
    np.testing.assert_array_equal(out, again)
    per_step = np.tile(_window().policies[-1], (7, 1))    # a policy per step is accepted too
    np.testing.assert_array_equal(m.sample_rollouts(_window(), per_step, None, 7, 5, seed=1), out)


def _reference_rollout(m, hist, pol, horizon, n, seed):
    """Rollout by re-running the network on the whole input sequence of the current segment at every step.

    A segment starts at the last `context` inputs and runs while it has fewer than `context + room` inputs
    before the new one; then a new segment starts at the last `context` inputs (fresh recurrent state)."""
    import torch
    C, longest = m.context, m.context + m.room
    k = hist.known
    S, P = hist.states[-C:], hist.policies[-C:]
    inputs = [np.tile(np.concatenate([S[j], P[j + 1], k]), (n, 1)) for j in range(len(S) - 1)]
    s = np.tile(S[-1], (n, 1)).astype(np.float32)
    lo, hi = m.bounds
    rng = np.random.default_rng(seed)
    out, b = [], None
    for h in range(horizon):
        inputs.append(np.concatenate([s, np.tile(pol, (n, 1)), np.tile(k, (n, 1))], axis=1))
        cnt = len(inputs)
        if b is None or cnt - 1 - b >= longest:
            b = max(cnt - C, 0)
        x = np.stack(inputs[b:cnt], axis=1).astype(np.float32)
        with torch.no_grad():
            mean, logvar, _ = m.net(torch.from_numpy(m.x_norm.transform(x)))
        mean, logvar = mean[:, -1].numpy(), logvar[:, -1].numpy()
        delta = (mean + np.exp(0.5 * logvar) * rng.standard_normal(mean.shape)) * m.y_norm.std + m.y_norm.mean
        s = np.minimum(np.maximum(s + delta.astype(np.float32), np.maximum(lo, 0)), hi)
        out.append(s)
    return np.stack(out, axis=1)


@pytest.mark.parametrize("context,room,T,horizon", [(8, 8, 4, 6), (8, 8, 20, 6), (3, 2, 10, 12)])
def test_stepwise_rollout_equals_whole_sequence(context, room, T, horizon):
    """Each incremental step equals the network run on the whole concatenated sequence (same noise); the last
    case crosses context + room several times and pins the restart."""
    hist = _window(T=T, seed=4)
    m = _tiny_model(seed=2, context=context, room=room, hi=np.full(43, 50.0, np.float32))
    a = m.sample_rollouts(hist, hist.policies[-1], None, horizon=horizon, n=3, seed=5)
    b = _reference_rollout(m, hist, hist.policies[-1], horizon, 3, 5)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
    if room == 2:          # without the restart the trajectory differs: the test can tell
        no_restart = _tiny_model(seed=2, context=context, room=100, hi=np.full(43, 50.0, np.float32))
        c = no_restart.sample_rollouts(hist, hist.policies[-1], None, horizon=horizon, n=3, seed=5)
        assert np.abs(c - a).max() > 1e-3


def test_sample_rollouts_accepts_policy_dict():
    from pis.adapters import initial_policy
    m = _tiny_model()
    out = m.sample_rollouts(_window(), initial_policy(), {"a_service_ms": 1.0, "a_interference": 0.2,
                                                         "thermal_ambient": 30.0}, horizon=3, n=2, seed=0)
    assert out.shape == (2, 3, 43)


def test_save_load_round_trip(tmp_path):
    m = _tiny_model(seed=3)
    p = tmp_path / "m.pt"
    m.save(p)
    m2 = wm_prob.ProbWorldModel.load(p)
    a = m.sample_rollouts(_window(), _window().policies[-1], None, horizon=4, n=3, seed=7)
    b = m2.sample_rollouts(_window(), _window().policies[-1], None, horizon=4, n=3, seed=7)
    np.testing.assert_array_equal(a, b)
    assert m2.meta["sha256"] and len(m2.meta["sha256"]) == 64
    assert m2.context == 4


# ------------------------------------------------------------------ store training
def _synthetic_store(root, seeds, n=40):
    d = root / f"s{seeds[0]}"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for seed in seeds:
        rows = {"episode": [seed] * n, "window": list(range(n)), "t": [0.5 * (w + 1) for w in range(n)]}
        for c in history.STATE_COLUMNS.values():
            rows[c] = rng.uniform(0, 1, n).astype(float)
        for c in history.RAW_COLUMNS.values():
            rows[c] = rng.integers(0, 5, n)
        for c in history.POLICY_COLUMNS.values():
            rows[c] = rng.uniform(0, 1, n).astype(float)
        rows.update(value=[1.0] * n, slo_v2_violation=[False] * n, deg_kind=["none"] * n,
                    deg_active=[False] * n, deg_factor=[0.0] * n, alive_cpus=[8] * n, source=["r6"] * n)
        pd.DataFrame(rows, columns=history.WINDOW_COLUMNS).to_parquet(d / f"windows-{seed}.parquet")
        ep = {"seed": seed, "kind": "none", "source": "r6", "windows": n,
              **{c: float(i + 1) for i, c in enumerate(history.KNOWN_COLUMNS.values())}}
        pd.DataFrame([ep]).to_parquet(d / f"episodes-{seed}.parquet")
    (d / shards.DONE).write_text("")


def test_episode_data_from_store(tmp_path):
    _synthetic_store(tmp_path, [3100, 3101])
    st = history.Store(tmp_path)
    eps = features.episode_data(st, [3100, 3101])
    assert [e.states.shape for e in eps] == [(40, 43), (40, 43)]
    np.testing.assert_allclose(eps[0].known, [1.0, 2.0, 3.0])
    w = st.window_rows(3100, 0, 40)
    np.testing.assert_allclose(eps[0].states[:, 39], np.log1p(w.a_queue_overdue.to_numpy()), rtol=1e-6)


def test_training_refuses_seeds_outside_history(tmp_path):
    _synthetic_store(tmp_path, [3100, 3101])
    st = history.Store(tmp_path)
    for bad in ([3100, 3400], [3450], [3000], [1100]):
        with pytest.raises(ValueError):
            wm_prob.fit(st, [3100], bad, epochs=1)
        with pytest.raises(ValueError):
            wm_prob.fit(st, bad, [3101], epochs=1)


def test_fit_from_store_runs(tmp_path):
    _synthetic_store(tmp_path, [3100, 3101, 3108])
    st = history.Store(tmp_path)
    model, hist = wm_prob.fit(st, [3100, 3101], [3108], context=8, burn_in=2, room=8, hidden=8, layers=1,
                              epochs=2)
    assert len(hist) == 2 and model.bounds[0].shape == (43,)
    assert model.meta["train_seeds"] == [3100, 3101]


# ------------------------------------------------------------------ calibration pieces
def test_interval_coverage_and_crps():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=(400, 2000))
    truth = rng.normal(0, 1, size=2000)
    cov = wm_prob.interval_coverage(samples, truth, 0.8)
    assert abs(cov - 0.8) < 0.03
    narrow = wm_prob.interval_coverage(samples * 0.3, truth, 0.8)
    assert narrow < 0.5
    good = wm_prob.crps(samples, truth)
    assert good < wm_prob.crps(samples + 2.0, truth)


def test_rollout_risk_counts_new_onsets_only():
    class Stub:
        context = 4

        def __init__(self, viol_from):
            self.viol_from = viol_from

        def sample_rollouts(self, hist, policy, known, horizon, n, seed=None):
            out = np.zeros((n, horizon, 43), np.float32)
            out[..., 41] = np.log1p(1000)                  # completed on time
            for i in range(n // 2):                        # half the samples: every window late
                out[i, self.viol_from:, 40] = np.log1p(500)
            return out

    rule = IncidentRule(k=3, interval=12)
    w = 40
    rows = pd.DataFrame({"window": np.arange(w), "slo_v2_violation": np.zeros(w, bool),
                         "deg_active": np.zeros(w, bool)})
    win = _window(w)
    risk = wm_prob.rollout_risk(Stub(0), win, rows, rule, horizon=5, n=8, warmup=20)
    assert risk == 0.5
    # a run of k starting at the horizon's last window needs the k-1 tail to be seen
    assert wm_prob.rollout_risk(Stub(4), win, rows, rule, horizon=5, n=8, warmup=20) == 0.5
    assert wm_prob.rollout_risk(Stub(5), win, rows, rule, horizon=5, n=8, warmup=20) == 0.0
    # in incident at t: continuing violations are no new onset
    rows2 = rows.copy()
    rows2.loc[30:, "slo_v2_violation"] = True
    assert wm_prob.rollout_risk(Stub(0), win, rows2, rule, horizon=5, n=8, warmup=20) == 0.0


def test_ladder_uses_nonzero_coverage():
    cov = {"a.queue_overdue": {"10": 0.95}, "a.late_new": {"10": 0.97}, "a.p99_ms": {"10": 0.8}}
    nz = {"a.queue_overdue": {"10": 0.8}, "a.late_new": {"10": 0.75}}
    cal = {"horizon": 10, "coverage": cov, "coverage_nonzero": nz, "brier_skill": 0.1}
    assert wm_prob.ladder_pass(cal) is True                   # all-window coverage inflated by zeros: ignored
    assert wm_prob.ladder_pass({**cal, "brier_skill": -0.1}) is False
    assert wm_prob.ladder_pass({**cal, "brier_skill": None}) is None
    bad = {"a.queue_overdue": {"10": 0.6}, "a.late_new": {"10": 0.75}}
    assert wm_prob.ladder_pass({**cal, "coverage_nonzero": bad}) is False
    assert wm_prob.ladder_pass({**cal, "coverage": {**cov, "a.p99_ms": {"10": 0.95}}}) is False


def test_predictive_scores_report_nonzero_windows(tmp_path):
    _synthetic_store(tmp_path, [3108])
    st = history.Store(tmp_path)
    m = _tiny_model(context=4, hi=np.full(43, 50.0, np.float32))
    r = wm_prob.predictive_scores(m, st, [3108], horizon=3, n=8, every=1, warmup=4)
    for o in ("a.queue_overdue", "a.late_new"):
        assert set(r["coverage_nonzero"][o]) == {"1", "3"}
        assert 0 < r["counts_nonzero"][o]["3"] < r["counts"][o]["3"]
    assert "a.p99_ms" not in r["coverage_nonzero"]


# ------------------------------------------------------------------ script guards
def test_script_refuses_non_history_seeds(tmp_path):
    with pytest.raises(SystemExit, match="refusing"):
        script.main(["--label", "smoke", "--seeds", "3100-3101,3450", "--smoke-dir",
                     "data/phase_b/smoke_g4/t", "--models-dir", "models/smoke_g4"])


def test_script_smoke_outputs_only_in_smoke_places():
    with pytest.raises(SystemExit, match="refusing"):
        script.main(["--label", "smoke", "--seeds", "3100-3101", "--smoke-dir", "data/phase_b/smoke_g4/t",
                     "--models-dir", "models"])
    with pytest.raises(SystemExit, match="refusing"):
        script.main(["--label", "smoke", "--seeds", "3100-3101", "--smoke-dir", "results/x",
                     "--models-dir", "models/smoke_g4"])


def test_script_real_run_refuses_dirty_tree(monkeypatch):
    from pis.phase_b import runs
    monkeypatch.setattr(runs, "tree_dirty", lambda ignore=(): True)
    with pytest.raises(SystemExit, match="uncommitted"):
        script.main(["--seeds", "3100-3101"])


def test_mc_labels_must_match_the_rule(tmp_path):
    inc = {"k": 7, "interval": 28, "majority": 0.5, "warmup": 20, "H": 6}
    d = tmp_path / "mc"
    d.mkdir()
    (d / "labels.parquet").write_text("")
    with pytest.raises(ValueError, match="config"):
        script.check_mc_config(d / "labels.parquet", inc)
    import json
    cfg = {"k": 7, "interval": 28, "majority": 0.5, "warmup": 20, "horizon": 20}
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="horizon"):
        script.check_mc_config(d / "labels.parquet", inc)
    (d / "config.json").write_text(json.dumps({**cfg, "horizon": 6, "k": 5}))
    with pytest.raises(ValueError, match="k"):
        script.check_mc_config(d / "labels.parquet", inc)
    (d / "config.json").write_text(json.dumps({**cfg, "horizon": 6}))
    script.check_mc_config(d / "labels.parquet", inc)


def test_script_needs_incident_config(tmp_path, monkeypatch):
    from pis.phase_b import runs
    monkeypatch.setattr(runs, "tree_dirty", lambda ignore=(): False)
    with pytest.raises(SystemExit, match="incident"):
        script.main(["--seeds", "3100-3101", "--incident", str(tmp_path / "missing.json")])
