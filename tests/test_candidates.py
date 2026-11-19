import copy

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.candidates import is_valid, random_valid_policies, scheduler_candidates  # noqa: E402

SEED = 1100
CKPT = "models/rl_r6/final.pt"


@pytest.fixture(scope="module")
def spec():
    return adapters.make_spec(SEED)


def _tel(current):
    from sim.policy import policy_from_dict
    return adapters.make_sim(SEED).run_window(policy_from_dict(current))


def _cands(spec, m=8, seed=0):
    base = adapters.BaseScheduler(CKPT, spec)
    cur = adapters.initial_policy()
    return cur, scheduler_candidates(base, _tel(cur), cur, m, np.random.default_rng(seed))


def test_all_candidates_valid(spec):
    _, c = _cands(spec)
    assert c and all(is_valid(p) for p in c)


def test_no_duplicates(spec):
    from pis.candidates import policy_key
    _, c = _cands(spec, m=16)
    assert len({policy_key(p) for p in c}) == len(c)


def test_greedy_is_first(spec):
    base = adapters.BaseScheduler(CKPT, spec)
    cur = adapters.initial_policy()
    tel = _tel(cur)
    greedy, _ = adapters.BaseScheduler(CKPT, spec).propose(tel, cur, 0, np.random.default_rng(0))
    c = scheduler_candidates(base, tel, cur, 8, np.random.default_rng(0))
    assert c[0] == greedy


def test_keep_current_present(spec):
    from pis.candidates import policy_key
    cur, c = _cands(spec)
    assert policy_key(cur) in {policy_key(p) for p in c}


@pytest.mark.parametrize("strength", [0.1, 0.5, 1.0])
def test_random_policies_valid_at_all_strengths(spec, strength):
    cur = adapters.initial_policy()
    pols = random_valid_policies(spec, cur, 200, strength, np.random.default_rng(1))
    assert len(pols) == 200 and all(is_valid(p) for p in pols)


def test_random_policies_differ_from_current_and_do_not_mutate_input(spec):
    from pis.candidates import policy_key
    cur = adapters.initial_policy()
    before = copy.deepcopy(cur)
    pols = random_valid_policies(spec, cur, 50, 0.5, np.random.default_rng(2))
    assert cur == before
    assert sum(policy_key(p) != policy_key(cur) for p in pols) > 25


def test_is_valid_rejects_unused_cpu_and_cpuless_workload():
    p = adapters.initial_policy()
    bad = copy.deepcopy(p)
    bad["workloads"]["A"]["preferred_cpus"] = [0, 1]
    bad["workloads"]["B"]["preferred_cpus"] = [4, 5, 6]
    bad["workloads"]["C"]["preferred_cpus"] = [4, 5, 6]
    assert not is_valid(bad)                      # CPUs 2, 3, 7 unused
    assert is_valid(p)


def test_all_identical_samples_collapse_to_one():
    cur = adapters.initial_policy()

    class Stub:
        def propose(self, tel, current, m, rng):
            return copy.deepcopy(current), [copy.deepcopy(current) for _ in range(m)]

    assert scheduler_candidates(Stub(), {}, cur, 6, np.random.default_rng(0)) == [cur]
