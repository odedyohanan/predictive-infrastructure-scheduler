import numpy as np
import pandas as pd
import pytest

from pis.config import setup_imports

setup_imports()

from pis import shards  # noqa: E402
from pis.adapters import initial_policy  # noqa: E402
from pis.phase_b import features, history, incidents, predictors  # noqa: E402
from pis.phase_b.incidents import IncidentRule  # noqa: E402
from worldmodel.features import POLICY_NAMES, encode_policy  # noqa: E402

RULE = IncidentRule(k=3, interval=12)
T = 200


def synth(ep: int, onset: int | None = None, ramp: int = 25, seed: int = 0) -> pd.DataFrame:
    """One episode of store-shaped rows. Quiet: A at 1500 req/s on 4 exclusive cores. With an onset: the
    arrival rate, queue and p99 climb over `ramp` windows, then requests turn overdue and late (SLO v2
    violations) from `onset` on."""
    rng = np.random.default_rng(seed + ep)
    t = np.arange(T)
    f = np.zeros(T)
    if onset is not None:
        f = np.clip((t - (onset - ramp)) / ramp, 0, 1)
    after = (onset is not None) & (t >= (onset if onset is not None else T))
    rows = {"episode": np.full(T, ep), "window": t, "t": 0.5 * (t + 1)}
    for c in history.STATE_COLUMNS.values():
        rows[c] = np.full(T, 0.3)
    rows["a_rate"] = 1500 * (1 + 1.6 * f) + rng.normal(0, 30, T)
    rows["a_queue_len"] = np.log1p(2 + 300 * f ** 2 + rng.uniform(0, 1, T))
    rows["a_queue_work_ms"] = np.log1p(2 + 300 * f ** 2 + rng.uniform(0, 1, T))
    rows["a_p99_ms"] = np.log1p(4 + 8 * f + rng.uniform(0, 0.3, T))
    rows["a_p50_ms"] = np.log1p(np.full(T, 1.0))
    rows["a_cpu_cores"] = 1.5 + 2.4 * f
    for k in range(8):
        rows[f"usage_0_{k}"] = np.full(T, 0.4 if k < 4 else 0.0)
        rows[f"usage_1_{k}"] = np.full(T, 0.0 if k < 4 else 0.5)
        rows[f"usage_2_{k}"] = np.full(T, 0.0 if k < 4 else 0.1)
    arrivals = (rows["a_rate"] * 0.5).astype(int)
    rows["a_arrivals"] = arrivals
    rows["a_queue_overdue"] = np.where(after, 40 + 10 * np.maximum(t - (onset or 0), 0), 0)
    rows["a_late_new"] = np.where(after, 120, 0)
    rows["a_completed_ontime"] = np.where(after, arrivals - 120, arrivals)
    pol = encode_policy(initial_policy())
    for i, n in enumerate(POLICY_NAMES):
        rows[history.POLICY_COLUMNS[n]] = np.full(T, float(pol[i]))
    rows.update(value=np.ones(T), slo_v2_violation=after.astype(bool), deg_kind=["none"] * T,
                deg_active=np.zeros(T, bool), deg_factor=np.zeros(T), alive_cpus=np.full(T, 8),
                source=["r6"] * T)
    df = pd.DataFrame(rows)
    for c, v in zip(features.KNOWN_COLS, (1.0, 0.2, 30.0)):
        df[c] = v
    return df


def frame(n_eps=30, first=3100):
    """Half the episodes quiet, half building up to an onset between windows 90 and 150."""
    rng = np.random.default_rng(1)
    eps = []
    for i in range(n_eps):
        onset = int(rng.integers(90, 150)) if i % 2 else None
        eps.append(synth(first + i, onset))
    return pd.concat(eps, ignore_index=True)


def _rows(df, upto):
    return df[df.window <= upto].reset_index(drop=True)


def test_violations_follow_slo_v2():
    df = synth(3100, onset=100)
    from sim.slo import window_violation
    for r in df.itertuples():
        tel = {"a": {"late_new": r.a_late_new, "completed_ontime": r.a_completed_ontime,
                     "queue_overdue": r.a_queue_overdue}}
        assert window_violation(tel, 10.0, "v2") == r.slo_v2_violation


def test_onset_distance():
    v = np.zeros(60, bool)
    v[30:35] = True
    d = predictors.onset_distance(v, np.zeros(60, bool), RULE, warmup=20)
    assert d[25] == 5 and d[29] == 1
    assert d[30] == predictors.NO_ONSET                 # an incident already started is no onset ahead
    for t in range(60):
        for h in (1, 5, 10):
            lab = incidents.label_series(v, np.zeros(60, bool), RULE, 20)
            assert (d[t] <= h) == incidents.onset_within(lab, t, h)


def test_rolling_is_causal():
    x = np.random.default_rng(0).normal(size=(30, 2))
    full = predictors.rolling(x, 8)
    part = predictors.rolling(x[12:21], 8)
    for a, b in zip(full, part):
        np.testing.assert_allclose(a[20], b[-1])
    _, _, slope = predictors.rolling(np.arange(10.0)[:, None] * 3, 4)
    np.testing.assert_allclose(slope[5:], 3.0)


def _rising_vs_flat(p, H=10):
    rising = synth(3990, onset=150)
    flat = synth(3991)
    early = p.risk(_rows(rising, 110), H)                # before the build-up starts
    late = p.risk(_rows(rising, 145), H)                 # 5 windows before the onset
    quiet = p.risk(_rows(flat, 145), H)
    return early, late, quiet


@pytest.fixture(scope="module")
def train_frame():
    return frame()


def test_hazard_rising_vs_flat(train_frame):
    p = predictors.HazardPredictor(RULE, max_h=20, iters=300)
    p.fit_frames(train_frame)
    early, late, quiet = _rising_vs_flat(p)
    assert late > 0.5 and late > early + 0.3
    assert quiet < 0.1
    h = p.hazards(_rows(synth(3990, onset=150), 145), 10)
    assert h.shape == (10,) and np.all((h >= 0) & (h <= 1))
    assert p.risk(_rows(synth(3990, onset=150), 145), 10) == pytest.approx(1 - np.prod(1 - h))
    with pytest.raises(ValueError):
        p.hazards(_rows(synth(3990), 50), 21)


def test_analogue_rising_vs_flat(train_frame):
    p = predictors.AnalogueSearch(RULE, k=20, max_h=20)
    p.fit_frames(train_frame)
    early, late, quiet = _rising_vs_flat(p)
    assert late > 0.5 and late > early + 0.3
    assert quiet < 0.1


def test_analogue_excludes_same_episode():
    ref = pd.concat([synth(3100, onset=120), synth(3101), synth(3102)], ignore_index=True)
    p = predictors.AnalogueSearch(RULE, k=5, max_h=20)
    p.fit_frames(ref)
    query = _rows(synth(3100, onset=120), 112)
    assert p.risk(query, 10) == 0.0                     # the only build-up is its own: excluded
    assert not (p.ep[p.neighbours(query)] == 3100).any()
    other = query.assign(episode=3555)
    assert p.risk(other, 10) > 0.5


def test_queueing_rising_vs_flat():
    p = predictors.QueueingModel(RULE)
    early, late, quiet = _rising_vs_flat(p)
    assert late > 0.5 and late > early + 0.3
    assert quiet < 0.1


def test_queueing_capacity():
    df = synth(3100)
    cores, busy = predictors.a_capacity(df)
    np.testing.assert_allclose(cores, 4.0)              # A alone on CPUs 0-3
    np.testing.assert_allclose(busy, 0.0)
    mu = predictors.a_service_rate(df)
    np.testing.assert_allclose(mu, 4000.0)


def test_queueing_calibration_fits_scale():
    p = predictors.QueueingModel(RULE)
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 4000)
    y = rng.uniform(size=4000) < 1 / (1 + np.exp(-z / 1.5))
    assert p.calibrate_arrays(z, y) == pytest.approx(1.5, rel=0.25)


def test_change_point_rising_vs_flat(train_frame):
    p = predictors.ChangePoint()
    p.fit_frames(train_frame[train_frame.episode % 2 == 0])
    early, late, quiet = _rising_vs_flat(p)
    assert late > 0.5 and late > early + 0.3
    assert quiet < 0.1


def test_bocpd_detects_mean_shift_within_ten_windows():
    rng = np.random.default_rng(3)
    z = np.concatenate([rng.normal(0, 1, 150), rng.normal(3, 1, 30)])
    m = predictors.bocpd_short_mass(z)
    assert m[20:150].max() < 0.5                      # no false detection before the change
    first = int(np.argmax(m[150:] > 0.5))
    assert m[150 + first] > 0.5 and first < 10


def test_world_model_predictor_share_of_futures():
    class Stub:
        context = 4

        def sample_rollouts(self, hist, policy, known, horizon, n, seed=None):
            assert known is None and hist.known is not None
            out = np.zeros((n, horizon, 43), np.float32)
            out[..., features.IDX["a.completed_ontime"]] = np.log1p(1000)
            out[: n // 4, :, features.IDX["a.late_new"]] = np.log1p(500)
            return out

    p = predictors.WorldModelPredictor(Stub(), RULE, n=8)
    assert p.risk(_rows(synth(3100), 60), 10) == 0.25
    in_incident = _rows(synth(3100, onset=50, ramp=5), 60)
    assert p.risk(in_incident, 10) == 0.0              # the incident in force continues: no new onset


class _NoStore:
    def __getattr__(self, name):
        raise AssertionError("the store must not be read")


@pytest.mark.parametrize("make", [
    lambda: predictors.HazardPredictor(RULE), lambda: predictors.AnalogueSearch(RULE),
    lambda: predictors.ChangePoint(), lambda: predictors.WorldModelPredictor(object(), RULE)])
@pytest.mark.parametrize("seeds", [[3100, 3400], [3450], [3000], [1100], [2050]])
def test_fit_refuses_non_history_seeds(make, seeds):
    with pytest.raises(ValueError):
        make().fit(_NoStore(), seeds)


@pytest.mark.parametrize("seeds", [[3100], [3450], [3400, 3460], [3000]])
def test_queueing_calibrates_on_calibration_seeds_only(seeds):
    with pytest.raises(ValueError):
        predictors.QueueingModel(RULE).calibrate(_NoStore(), seeds, 10)
    assert not hasattr(predictors.QueueingModel(RULE), "fit")


def _store(root, dfs):
    d = root / "s0"
    d.mkdir(parents=True)
    for df in dfs:
        seed = int(df.episode.iloc[0])
        df[history.WINDOW_COLUMNS].to_parquet(d / f"windows-{seed}.parquet")
        ep = {"seed": seed, "kind": "none", "source": "r6", "windows": len(df),
              **{c: float(df[c].iloc[0]) for c in features.KNOWN_COLS}}
        pd.DataFrame([ep]).to_parquet(d / f"episodes-{seed}.parquet")
    (d / shards.DONE).write_text("")
    return history.Store(root)


def test_fit_from_store(tmp_path):
    dfs = [synth(3100 + i, 120 if i % 2 else None) for i in range(6)] + [synth(3400, 120), synth(3401)]
    st = _store(tmp_path, dfs)
    hz = predictors.HazardPredictor(RULE, max_h=20, iters=50)
    hz.fit(st, range(3100, 3106))
    an = predictors.AnalogueSearch(RULE, k=5, max_h=20)
    an.fit(st, range(3100, 3106))
    cp = predictors.ChangePoint()
    cp.fit(st, range(3100, 3106))
    q = predictors.QueueingModel(RULE)
    q.calibrate(st, [3400, 3401], 10)
    rows = features.join_known(st.window_rows(3400, 0, 118), st.episodes())
    for p in (hz, an, cp, q):
        r = p.risk(rows, 10)
        assert 0.0 <= r <= 1.0
