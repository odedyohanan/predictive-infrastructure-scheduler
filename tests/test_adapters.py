import dataclasses

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402

CKPT = "models/rl_r6/final.pt"


def _policy_sequence(seed: int, n: int = 8) -> list[dict]:
    """A fixed, varied policy sequence (perturbations of the isolated policy)."""
    from sched.controllers import isolated_policy
    from sched.dataset import perturb_policy, policy_dict
    spec = adapters.make_spec(seed)
    rng = np.random.default_rng(seed)
    p = isolated_policy()
    out = [policy_dict(p)]
    for _ in range(n - 1):
        p = perturb_policy(p, rng, spec)
        out.append(policy_dict(p))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_physics_equivalent(seed):
    from sched.randomize import random_spec as sched_random_spec
    from sched.simulator import Simulator as SchedSim
    from sim.policy import policy_from_dict as sim_pfd
    from sched.policy import policy_from_dict as sched_pfd

    sim_a = adapters.make_sim(seed)
    spec_b = sched_random_spec(np.random.default_rng(900000 + seed))
    assert dataclasses.asdict(spec_b) == dataclasses.asdict(adapters.make_spec(seed))
    sim_b = SchedSim(spec_b, seed, 60.0)
    for pd in _policy_sequence(seed):
        ta = sim_a.run_window(sim_pfd(pd))
        tb = sim_b.run_window(sched_pfd(pd))
        assert ta == tb


def test_known_of_exposes_known_parameters():
    k = adapters.known_of(adapters.make_spec(0))
    assert "a_service_ms" in k and "a_interference" in k and "thermal_ambient" in k


def _windows(seed: int, n: int):
    from sim.policy import policy_from_dict
    sim = adapters.make_sim(seed)
    cur = adapters.initial_policy()
    tel = sim.run_window(policy_from_dict(cur))
    for _ in range(n):
        yield tel, cur
        tel = sim.run_window(policy_from_dict(cur))


def test_base_greedy_matches_student_controller():
    from pis.config import sibling_paths
    from sched.controllers import StudentController
    from sched.dataset import policy_dict
    from sim.policy import policy_from_dict

    spec = adapters.make_spec(3)
    base = adapters.BaseScheduler(CKPT, spec)
    ctrl = StudentController(spec, str(sibling_paths()["sched"] / CKPT))
    cur = policy_dict(ctrl.initial_policy())
    rng = np.random.default_rng(0)
    sim = adapters.make_sim(3)
    tel = sim.run_window(policy_from_dict(cur))
    for _ in range(5):
        greedy, _ = base.propose(tel, cur, 0, rng)
        want = policy_dict(ctrl.decide(tel))
        # version / valid_for_ms are controller bookkeeping, not policy content
        assert greedy["workloads"] == want["workloads"]
        cur = want
        tel = sim.run_window(policy_from_dict(cur))


def test_samples_are_reproducible():
    spec = adapters.make_spec(4)
    tel, cur = next(iter(_windows(4, 1)))
    outs = []
    for _ in range(2):
        base = adapters.BaseScheduler(CKPT, spec)
        outs.append(base.propose(tel, cur, 4, np.random.default_rng(7)))
    assert outs[0] == outs[1]
    assert len(outs[0][1]) == 4


def _next_policies(n):
    return _policy_sequence(5, n)


def test_clone_continues_identically():
    from sim.policy import policy_from_dict
    sim = adapters.make_sim(5)
    pols = _next_policies(6)
    for pd in pols[:2]:
        sim.run_window(policy_from_dict(pd))
    clone = adapters.clone_sim(sim)
    for pd in pols[2:]:
        assert sim.run_window(policy_from_dict(pd)) == clone.run_window(policy_from_dict(pd))


def test_reseed_changes_outcome():
    from sim.policy import policy_from_dict
    sim = adapters.make_sim(5)
    pols = _next_policies(6)
    sim.run_window(policy_from_dict(pols[0]))
    clone = adapters.clone_sim(sim, reseed=123)
    diverged = any(sim.run_window(policy_from_dict(pd)) != clone.run_window(policy_from_dict(pd))
                   for pd in pols[1:])
    assert diverged
