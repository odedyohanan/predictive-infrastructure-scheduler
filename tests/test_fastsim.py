"""FastSimulator must produce exactly the Python simulator's telemetry and generator state.

Comparisons are exact (`==` on telemetry dicts, `np.array_equal` on arrays, equal generator states). To keep
the suite short, long stretches run on the fast simulator alone and a Python copy is forked from its state
("fork" below: a deep copy whose class is switched to the Python `Simulator`); both then run the same
windows side by side. The full side-by-side check over hundreds of episodes is experiments/fastsim_check.py.
"""
import copy
import dataclasses

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.fastsim import FastSimulator, _pairwise  # noqa: E402
from pis.candidates import random_valid_policies  # noqa: E402
from pis.phase_b import scenarios, seeds  # noqa: E402
from pis.phase_b.episode import run_episode  # noqa: E402
from pis.phase_b.scenarios import DegradedSim  # noqa: E402
from sim.controllers import example_policy, isolated_policy  # noqa: E402
from sim.policy import LATENCY_CLASSES, Policy, WorkloadPolicy  # noqa: E402
from sim.simulator import Simulator  # noqa: E402


def _random_policy(rng) -> Policy:
    """Any feasible policy: random cpusets (empty = all CPUs), shares, latency classes, GPU owners."""
    gpus = [[], [], []]
    for g in range(2):
        owner = int(rng.integers(4))           # 3 = no owner
        if owner < 3:
            gpus[owner].append(g)
    wl = []
    for w in range(3):
        k = int(rng.integers(0, 9))
        cpus = sorted(int(c) for c in rng.choice(8, k, replace=False)) if k < 8 else []
        wl.append(WorkloadPolicy(cpus, int(rng.integers(1, 100)), LATENCY_CLASSES[int(rng.integers(4))],
                                 gpus[w]))
    return Policy(1, 500, wl)


def _fork_python(sim):
    """A Python `Simulator` in exactly the state of the FastSimulator `sim`."""
    c = copy.deepcopy(sim)
    py = Simulator.__new__(Simulator)
    py.__dict__.update({k: v for k, v in c.__dict__.items() if not k.startswith("_q")})
    py.queue = c.queue
    return py


def _fork_dsim(ds):
    c = copy.deepcopy(ds)
    c.sim = _fork_python(ds.sim)
    return c


def _assert_same_state(a, b):
    assert a.t == b.t
    assert a.b_buffer == b.b_buffer
    assert np.array_equal(a.temps, b.temps)
    assert np.array_equal(a.migrating_until, b.migrating_until)
    assert np.array_equal(a.prev_usage, b.prev_usage)
    assert np.array_equal(a.prev_elig, b.prev_elig)
    assert np.array_equal(a.bursts, b.bursts)
    assert [list(r) for r in a.queue] == [list(r) for r in b.queue]
    assert len(a.all_latencies) == len(b.all_latencies)
    assert all(np.array_equal(x, y) for x, y in zip(a.all_latencies, b.all_latencies))
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert a.backlog() == b.backlog()


def _lockstep(py, fast, policies):
    for pol in policies:
        assert type(py) is Simulator and type(fast) is FastSimulator
        assert py.run_window(pol) == fast.run_window(pol)
    _assert_same_state(py, fast)


def test_pairwise_matches_numpy_sum():
    rng = np.random.default_rng(0)
    for n in list(range(0, 140)) + [255, 256, 1000, 1027, 5000]:
        for _ in range(5):
            a = rng.random(n) * rng.choice([1.0, 1e-9, 1e9], n)
            assert _pairwise(a, 0, n) == a.sum()


@pytest.mark.parametrize("seed", range(20))
def test_fast_matches_python_fixed_policies(seed):
    spec = adapters.make_spec(seed)
    mode = seed % 3
    prng = np.random.default_rng(1000 + seed)

    def policy():
        return example_policy() if mode == 0 else isolated_policy() if mode == 1 else _random_policy(prng)

    py, fast = Simulator(spec, seed, 20.0), FastSimulator(spec, seed, 20.0)
    _lockstep(py, fast, [policy() for _ in range(4)])          # from the start
    stop = 8.0 + seed % 10                                     # fast alone up to a warm state
    while fast.t < stop - 1e-9:
        fast.run_window(policy())
    py = _fork_python(fast)
    _lockstep(py, fast, [policy() for _ in range(3)])


def _seed_of_kind(kind: str) -> int:
    return next(s for s in seeds.HISTORY if scenarios.scenario_for(s).kind == kind)


class _Walk:
    """A random walk over valid policies (the Phase B "random" source's perturbations), a new one per window."""

    def __init__(self, spec, seed):
        self.spec, self.rng, self.cur = spec, np.random.default_rng(seed), adapters.initial_policy()

    def __call__(self) -> dict:
        self.cur = random_valid_policies(self.spec, self.cur, 1, float(self.rng.random()), self.rng)[0]
        return self.cur


def _advance(ds, until, walk):
    while ds.sim.t < until - 1e-9 and not ds.done():
        ds.run_window(walk())


def _lockstep_dsim(py, fast, n, walk):
    for _ in range(n):
        p = walk()
        assert py.run_window(p) == fast.run_window(p)
        assert py.last_applied == fast.last_applied
    _assert_same_state(py.sim, fast.sim)


@pytest.mark.parametrize("kind", ("none",) + scenarios.DEGRADED_KINDS)
def test_fast_matches_python_degraded(kind, monkeypatch):
    monkeypatch.setenv("PIS_FAST_SIM", "1")
    seed = _seed_of_kind(kind)
    sc = scenarios.scenario_for(seed)
    d = sc.degradation
    walk = _Walk(sc.spec, seed)
    fast = DegradedSim(sc)
    assert type(fast.sim) is FastSimulator
    onset = d.onset_s if kind != "none" else 100.0
    ramp = d.ramp_s if kind != "none" else 60.0

    _advance(fast, onset - 1.0, walk)                          # across the onset
    py = _fork_dsim(fast)
    _lockstep_dsim(py, fast, 6, walk)

    _advance(fast, onset + ramp / 2, walk)                     # mid-ramp
    py = _fork_dsim(fast)
    _lockstep_dsim(py, fast, 3, walk)

    # A reseeded clone (future arrivals and bursts redrawn) continues identically on both classes.
    c_py, c_fast = py.clone(reseed=seed + 7), fast.clone(reseed=seed + 7)
    assert type(c_py.sim) is Simulator and type(c_fast.sim) is FastSimulator
    assert np.array_equal(c_py.sim.bursts, c_fast.sim.bursts)
    _lockstep_dsim(c_py, c_fast, 3, walk)

    _advance(fast, sc.duration_s, walk)                        # the rest of the 300 s episode
    assert fast.done()
    py = _fork_dsim(fast)
    _assert_same_state(py.sim, fast.sim)


def test_fast_matches_python_r6_episode(monkeypatch):
    from pis.phase_b import policies
    seed = _seed_of_kind("cpu_loss")
    base = scenarios.scenario_for(seed)
    deg = dataclasses.replace(base.degradation, onset_s=3.0)
    sc = dataclasses.replace(base, duration_s=8.0, degradation=deg)
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("PIS_FAST_SIM", flag)
        out[flag] = run_episode(sc, policies.make("r6", sc.spec, seed), "r6")
    assert [_untimed(r) for r in out["0"]] == [_untimed(r) for r in out["1"]]


def _untimed(rec: dict) -> dict:
    """A record without the controller's wall-clock decision time."""
    return {**rec, "info": {k: v for k, v in rec["info"].items() if k != "decide_ms"}}


def test_switch_default_is_python(monkeypatch):
    monkeypatch.delenv("PIS_FAST_SIM", raising=False)
    assert adapters.simulator_class() is Simulator
    assert type(adapters.make_sim(1100, 1.0)) is Simulator
    assert type(DegradedSim(scenarios.scenario_for(3100)).sim) is Simulator


def test_switch_env_selects_fast(monkeypatch):
    monkeypatch.setenv("PIS_FAST_SIM", "1")
    assert adapters.simulator_class() is FastSimulator
    sim = adapters.make_sim(1100, 1.0)
    assert type(sim) is FastSimulator
    assert type(adapters.clone_sim(sim, reseed=3)) is FastSimulator
    assert type(DegradedSim(scenarios.scenario_for(3100)).sim) is FastSimulator
