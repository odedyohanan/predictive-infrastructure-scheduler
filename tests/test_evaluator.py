import copy

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.evaluator import Params, choose, predicted_rewards, score_candidates  # noqa: E402
from pis.reward import queue_penalty, reward_from_state  # noqa: E402

from worldmodel.features import STATE_NAMES, decode_state, encode_state  # noqa: E402

SEED = 1100
I = {k: STATE_NAMES.index(k) for k in ("b.rate", "c.work_core_ms", "a.p99_ms", "a.queue_work_ms")}
ONES = np.ones(39, np.float32)


@pytest.fixture(scope="module")
def spec():
    return adapters.make_spec(SEED)


@pytest.fixture(scope="module")
def s0():
    from sim.policy import policy_from_dict
    sim = adapters.make_sim(SEED)
    tel = None
    for _ in range(6):
        tel = sim.run_window(policy_from_dict(adapters.initial_policy()))
    v = encode_state(tel)
    v[I["a.p99_ms"]] = 0.0
    v[I["a.queue_work_ms"]] = 0.0
    return v


def _policies(n):
    from pis.candidates import random_valid_policies
    cur = adapters.initial_policy()
    out, seen = [cur], set()
    from pis.candidates import policy_key
    seen.add(policy_key(cur))
    rng = np.random.default_rng(0)
    while len(out) < n:
        p = random_valid_policies(None, cur, 1, 0.5, rng)[0]
        if policy_key(p) not in seen:
            seen.add(policy_key(p))
            out.append(p)
    return out


def _rollouts(s0, b_rates, n_members=3, horizon=2):
    """[N, n_cand, H, 39]: every member predicts s0 with candidate i's B rate set to b_rates[i]."""
    r = np.broadcast_to(s0, (n_members, len(b_rates), horizon, 39)).copy()
    for i, b in enumerate(b_rates):
        r[:, i, :, I["b.rate"]] = b
    return r


def P(**kw):
    base = dict(n_candidates_m=4, horizon_h=2, lam=1.0, mu=2.0, change_cost=0.0, tau=1.0, slo_lam=10.0)
    return Params(**{**base, **kw})


def test_vectorised_reward_matches_reward_module(spec):
    rng = np.random.default_rng(5)
    enc = np.abs(rng.normal(size=(2, 3, 4, 39))).astype(np.float32) * 3
    got = predicted_rewards(spec, enc, slo_lam=10.0, mu=2.0)
    assert got.shape == (2, 3, 4)
    for idx in np.ndindex(2, 3, 4):
        st = decode_state(enc[idx])
        want = (reward_from_state(spec, st, spec.policy_period_s, 10.0) - queue_penalty(spec, st, 2.0))
        assert got[idx] == pytest.approx(want, rel=1e-5, abs=1e-6)


def test_picks_best_finite_candidate(spec, s0):
    pols = _policies(4)
    scores, dis = score_candidates(spec, _rollouts(s0, [1.0, 2.0, 5.0, 3.0]), pols, pols[0], P(),
                                   train_std=ONES)
    assert np.allclose(dis, 0.0)
    d = choose(scores, dis, P())
    assert d.chosen == 2 and not d.fallback and d.reason == "model"


def test_nan_candidate_never_chosen(spec, s0):
    pols = _policies(3)
    r = _rollouts(s0, [1.0, 9.0, 2.0])
    r[1, 1, 1, 0] = np.nan                       # one member, one step, one number
    scores, dis = score_candidates(spec, r, pols, pols[0], P(), train_std=ONES)
    assert scores[1] == -np.inf and np.isinf(dis[1])
    d = choose(scores, dis, P())
    assert d.chosen == 2


def test_all_nan_falls_back(spec, s0):
    pols = _policies(2)
    r = _rollouts(s0, [1.0, 2.0])
    r[0, :, 0, 3] = np.inf
    scores, dis = score_candidates(spec, r, pols, pols[0], P(), train_std=ONES)
    d = choose(scores, dis, P())
    assert d.fallback and d.chosen == -1 and d.reason == "no_finite"
    d = choose(scores, dis, P(), use_fallback=False)          # nothing to pick, even without the gate
    assert d.fallback and d.reason == "no_finite"


def test_high_disagreement_falls_back(spec, s0):
    pols = _policies(2)
    r = _rollouts(s0, [1.0, 5.0])
    r[0, 1, :, I["b.rate"]] = 50.0               # members disagree wildly on the best candidate
    params = P(lam=0.0, tau=0.5)
    scores, dis = score_candidates(spec, r, pols, pols[0], params, train_std=ONES)
    assert dis[1] > 0.5 and dis[0] == 0.0
    d = choose(scores, dis, params)
    assert d.fallback and d.reason == "untrusted"
    assert choose(scores, dis, params, use_fallback=False).chosen == 1


def test_disagreement_penalty_changes_ranking(spec, s0):
    pols = _policies(2)
    r = _rollouts(s0, [4.0, 5.0])
    r[0, 1, :, I["b.rate"]] = 9.0
    r[1, 1, :, I["b.rate"]] = 1.0                # candidate 1: mean 5.0 but uncertain
    s_off, _ = score_candidates(spec, r, pols, pols[0], P(lam=0.0), train_std=ONES)
    s_on, _ = score_candidates(spec, r, pols, pols[0], P(lam=5.0), train_std=ONES)
    assert np.argmax(s_off) == 1 and np.argmax(s_on) == 0


def test_single_identical_candidate_no_crash(spec, s0):
    pols = _policies(1)
    scores, dis = score_candidates(spec, _rollouts(s0, [1.0], n_members=1, horizon=1), pols, pols[0], P(),
                                   train_std=ONES)
    d = choose(scores, dis, P())
    assert d.chosen == 0 and not d.fallback


def test_queue_growth_candidate_loses_to_empty_queue_candidate_when_p99_is_zero(spec, s0):
    pols = _policies(2)
    r = _rollouts(s0, [2.0, 2.0])
    assert (r[..., I["a.p99_ms"]] == 0).all()    # no completions: p99 reads 0 for both
    r[:, 1, :, I["a.queue_work_ms"]] = np.log1p(np.array([50.0, 200.0]))   # growing overdue backlog
    scores, _ = score_candidates(spec, r, pols, pols[0], P(), train_std=ONES)
    assert scores[0] > scores[1]
    assert choose(scores, np.zeros(2), P()).chosen == 0


def test_change_cost_penalises_change(spec, s0):
    pols = _policies(2)
    r = _rollouts(s0, [2.0, 2.0])
    s, _ = score_candidates(spec, r, pols, copy.deepcopy(pols[0]), P(change_cost=0.3), train_std=ONES)
    assert s[0] - s[1] == pytest.approx(0.3)


def test_params_round_trip():
    p = P(tau=float("inf"))
    assert Params.from_dict(p.to_dict()) == p
