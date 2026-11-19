import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.reward import queue_penalty, reward_from_state  # noqa: E402


def _real_windows(n: int):
    from sim.policy import policy_from_dict
    out = []
    for seed in (1100, 1101):
        sim = adapters.make_sim(seed)
        cur = adapters.initial_policy()
        spec = adapters.make_spec(seed)
        for _ in range(n // 2):
            out.append((spec, sim.run_window(policy_from_dict(cur))))
    return out


def test_matches_rl_reward_on_real_telemetry():
    from sched.rl import reward
    from worldmodel.features import decode_state, encode_state
    lam = 0.7
    for spec, tel in _real_windows(20):
        want = reward(spec, tel, lam)[0]
        got = reward_from_state(spec, decode_state(encode_state(tel)), 0.5, lam)
        assert got == pytest.approx(want, abs=1e-3)


def test_queue_penalty_grows_with_queue():
    spec = adapters.make_spec(1100)
    vals = [queue_penalty(spec, {"a.queue_work_ms": q}, 1.0)
            for q in (0.0, spec.a.slo_p99_ms, 3 * spec.a.slo_p99_ms)]
    assert vals[0] == 0.0 and vals[0] < vals[1] < vals[2]


def test_queue_penalty_is_capped():
    spec = adapters.make_spec(1100)
    huge = queue_penalty(spec, {"a.queue_work_ms": 1e9}, 2.0)
    assert huge == pytest.approx(2.0 * 5)


def test_zero_p99_with_big_queue_scores_worse_than_empty_queue():
    spec = adapters.make_spec(1100)
    base = {"a.p99_ms": 0.0, "b.rate": 1.0, "c.work_core_ms": 1.0}
    empty = {**base, "a.queue_work_ms": 0.0}
    big = {**base, "a.queue_work_ms": 4 * spec.a.slo_p99_ms}
    mu = 0.5
    r_empty = reward_from_state(spec, empty, 0.5, 1.0) - queue_penalty(spec, empty, mu)
    r_big = reward_from_state(spec, big, 0.5, 1.0) - queue_penalty(spec, big, mu)
    assert r_big < r_empty
