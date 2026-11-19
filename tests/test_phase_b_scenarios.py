import dataclasses
from collections import Counter

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.phase_b import scenarios, seeds  # noqa: E402
from pis.phase_b.episode import run_episode  # noqa: E402
from pis.phase_b.scenarios import DEAD_CPUS, DEAD_GPU, MAGNITUDES, Degradation, DegradedSim  # noqa: E402
from sim.policy import policy_from_dict  # noqa: E402
from sim.simulator import Simulator  # noqa: E402

SEED = 3101


def _scenario(kind, onset=4.0, ramp=4.0, duration=20.0, seed=SEED):
    base = scenarios.scenario_for(seed)
    mag = MAGNITUDES.get(kind, 0.0)
    return dataclasses.replace(base, duration_s=duration,
                               degradation=Degradation(kind, onset, ramp, mag))


def _policies(n):
    """A fixed, varied policy sequence (no controller randomness involved)."""
    rng = np.random.default_rng(5)
    out, cur = [], adapters.initial_policy()
    for _ in range(n):
        p = {"workloads": {k: dict(v) for k, v in cur["workloads"].items()}}
        p["workloads"]["A"]["cpu_share"] = int(rng.choice([30, 40, 50]))
        p["workloads"]["C"]["preferred_cpus"] = [] if rng.random() < 0.3 else [4, 5, 6, 7]
        out.append(p)
    return out


def _strip(tel):
    return {k: v for k, v in tel.items() if k != "phase_b"}


class Fixed:
    def __init__(self, policy):
        self.policy = policy
        self.seen = []

    def decide(self, tel, current, history):
        self.seen.append(tel)
        assert all("phase_b" not in h for h, _ in history)
        return self.policy, {"decide_ms": 0.0}


def test_degradation_factor():
    d = Degradation("leak", 10.0, 20.0, 0.8)
    assert d.factor(0.0) == 0.0 and d.factor(9.99) == 0.0
    assert d.factor(10.0) == 0.0
    assert d.factor(20.0) == pytest.approx(0.5)
    assert d.factor(30.0) == 1.0 and d.factor(100.0) == 1.0
    step = Degradation("cpu_loss", 10.0, 20.0, 2.0)
    assert step.factor(9.99) == 0.0 and step.factor(10.0) == 1.0
    assert Degradation("none", 0.0, 0.0, 0.0).factor(50.0) == 0.0


def test_onset_window_is_active():
    sc = _scenario("leak", onset=2.0, ramp=2.0, duration=4.0)
    ds = DegradedSim(sc)
    flags = [ds.run_window(adapters.initial_policy())["phase_b"]["active"] for _ in range(8)]
    assert flags == [False] * 4 + [True] * 4               # the window starting at 2.0 s is active


def test_no_degradation_is_identical():
    sc = _scenario("none", duration=20.0)
    ds = DegradedSim(sc)
    plain = Simulator(sc.spec, sc.seed, sc.duration_s)
    for p in _policies(40):
        a = ds.run_window(p)
        b = plain.run_window(policy_from_dict(p))
        assert _strip(a) == b
        assert a["phase_b"] == {"kind": "none", "active": False, "factor": 0.0,
                                "alive_cpus": list(range(8))}
    assert ds.sim.spec is sc.spec
    assert ds.done() and plain.done()


def test_degradation_before_onset_is_identical():
    sc = _scenario("leak", onset=15.0, duration=20.0)
    ds = DegradedSim(sc)
    plain = Simulator(sc.spec, sc.seed, sc.duration_s)
    for p in _policies(30):                              # windows up to t = 15 s
        assert _strip(ds.run_window(p)) == plain.run_window(policy_from_dict(p))


def test_load_ramp_raises_rate():
    sc = _scenario("load_ramp", onset=2.0, ramp=4.0)
    ds = DegradedSim(sc)
    pol = adapters.initial_policy()
    while ds.sim.t < 8.0:
        tel = ds.run_window(pol)
    base = sc.spec.a.base_rate
    assert ds.sim.spec.a.base_rate == pytest.approx(base * (1 + MAGNITUDES["load_ramp"]))
    plain = Simulator(sc.spec, sc.seed, sc.duration_s)
    t = ds.sim.t
    assert ds.sim.rate_a(t) - plain.rate_a(t) == pytest.approx(base * MAGNITUDES["load_ramp"])
    assert tel["phase_b"]["active"] and tel["phase_b"]["factor"] == 1.0


@pytest.mark.parametrize("kind,field,expected", [
    ("leak", ("a", "service_ms"), lambda s: s.a.service_ms * (1 + MAGNITUDES["leak"])),
    ("interference", ("a", "interference"), lambda s: s.a.interference + MAGNITUDES["interference"]),
    ("ambient", ("thermal", "ambient"), lambda s: s.thermal.ambient + MAGNITUDES["ambient"]),
])
def test_spec_degradations_reach_full_magnitude(kind, field, expected):
    sc = _scenario(kind, onset=1.0, ramp=2.0, duration=5.0)
    ds = DegradedSim(sc)
    pol = adapters.initial_policy()
    while not ds.done():
        ds.run_window(pol)
    got = getattr(getattr(ds.sim.spec, field[0]), field[1])
    assert got == pytest.approx(expected(sc.spec))


def test_known_params_unchanged():
    sc = _scenario("leak", onset=2.0, ramp=2.0, duration=10.0)
    before = adapters.known_of(sc.spec)
    fresh = adapters.known_of(adapters.make_spec(sc.seed))
    ds = DegradedSim(sc)
    pol = adapters.initial_policy()
    while not ds.done():
        ds.run_window(pol)
        assert adapters.known_of(sc.spec) == before
    assert ds.sim.spec.a.service_ms > sc.spec.a.service_ms
    assert before == fresh


def test_cpu_loss_dead_cpus_idle():
    sc = _scenario("cpu_loss", onset=3.0, duration=10.0)
    ds = DegradedSim(sc)
    pol = {"workloads": {k: dict(v) for k, v in adapters.initial_policy()["workloads"].items()}}
    pol["workloads"]["C"]["preferred_cpus"] = []              # "all CPUs" must exclude dead ones too
    used_before = False
    while not ds.done():
        tel = ds.run_window(pol)
        dead_use = [tel["cpu_util"][k] for k in DEAD_CPUS]
        if tel["phase_b"]["active"]:
            assert dead_use == [0.0] * len(DEAD_CPUS)
            assert tel["phase_b"]["alive_cpus"] == [k for k in range(8) if k not in DEAD_CPUS]
        else:
            used_before = used_before or sum(dead_use) > 0
    assert used_before


def test_mask_policy():
    pol = {"workloads": {
        "A": {"preferred_cpus": [2, 3], "cpu_share": 40, "latency_class": "critical", "gpus": []},
        "B": {"preferred_cpus": [], "cpu_share": 50, "latency_class": "batch", "gpus": [0, 1]},
        "C": {"preferred_cpus": [1, 3, 4], "cpu_share": 10, "latency_class": "best_effort", "gpus": []}}}
    m = scenarios.mask_policy(pol, dead_cpus=(2, 3), dead_gpus=(0,), n_cpus=8)
    assert m["workloads"]["A"]["preferred_cpus"] == [0, 1, 4, 5, 6, 7]   # nothing left -> all alive
    assert m["workloads"]["B"]["preferred_cpus"] == [0, 1, 4, 5, 6, 7]
    assert m["workloads"]["C"]["preferred_cpus"] == [1, 4]
    assert m["workloads"]["B"]["gpus"] == [1]
    assert pol["workloads"]["A"]["preferred_cpus"] == [2, 3]              # input untouched


def test_gpu_loss_gpu_unused():
    sc = _scenario("gpu_loss", onset=2.0, duration=6.0)
    ds = DegradedSim(sc)
    pol = adapters.initial_policy()                            # B owns both GPUs
    while not ds.done():
        tel = ds.run_window(pol)
        if ds.sim.t > 2.6:
            assert tel["gpu_power"][DEAD_GPU] == 0.0
            assert tel["gpu_power"][1 - DEAD_GPU] > 0.0


def test_clone_keeps_degradation():
    sc = _scenario("load_ramp", onset=2.0, ramp=10.0, duration=16.0)
    ds = DegradedSim(sc)
    pol = adapters.initial_policy()
    while ds.sim.t < 6.0:                                       # mid-ramp
        ds.run_window(pol)
    twin = ds.clone()
    other = ds.clone(reseed=123)
    while not ds.done():
        a, b, c = ds.run_window(pol), twin.run_window(pol), other.run_window(pol)
        assert a == b
        assert a["phase_b"] == c["phase_b"]
        assert ds.sim.spec == other.sim.spec
    assert ds.sim.spec.a.base_rate == pytest.approx(sc.spec.a.base_rate * (1 + MAGNITUDES["load_ramp"]))


def test_clone_keeps_dead_cpus():
    sc = _scenario("cpu_loss", onset=2.0, duration=8.0)
    ds = DegradedSim(sc)
    pol = adapters.initial_policy()
    while ds.sim.t < 4.0:
        ds.run_window(pol)
    twin = ds.clone(reseed=7)
    while not twin.done():
        tel = twin.run_window(pol)
        assert [tel["cpu_util"][k] for k in DEAD_CPUS] == [0.0] * len(DEAD_CPUS)


def test_scenario_for_is_deterministic():
    a, b = scenarios.scenario_for(3123), scenarios.scenario_for(3123)
    assert a == b
    assert a.duration_s == 300.0 and a.seed == 3123
    assert a.spec == adapters.make_spec(3123)
    assert a.kind == a.degradation.kind


@pytest.mark.parametrize("seed_range", [seeds.HISTORY, seeds.CALIBRATION, seeds.SELECTION, seeds.HELD_OUT])
def test_scenario_mix(seed_range):
    kinds = Counter(scenarios.scenario_for(s).kind for s in seed_range)
    n = len(seed_range)
    assert 0.4 * n <= kinds["none"] <= 0.6 * n
    degraded = n - kinds["none"]
    for k in scenarios.DEGRADED_KINDS:
        assert abs(kinds[k] - degraded / 6) <= max(0.3 * degraded / 6, 1.5), (k, kinds)


def test_scenario_mix_not_tied_to_split():
    """Validation (seed % 10 == 8) and test (9) seeds of HISTORY get both ordinary and degraded runs."""
    for split in (8, 9):
        kinds = Counter(scenarios.scenario_for(s).kind for s in seeds.HISTORY if s % 10 == split)
        assert 8 <= kinds["none"] <= 22 and len(kinds) >= 5, kinds


def test_degradation_parameters_over_history():
    for s in seeds.HISTORY:
        d = scenarios.scenario_for(s).degradation
        if d.kind != "none":
            assert 60.0 <= d.onset_s <= 180.0 and 30.0 <= d.ramp_s <= 120.0
            assert d.magnitude == MAGNITUDES[d.kind]


def test_run_episode_records():
    sc = _scenario("cpu_loss", onset=3.0, duration=6.0)
    pol = adapters.initial_policy()
    seen = []
    ctl = Fixed(pol)
    recs = run_episode(sc, ctl, "fixed", on_window=lambda ds, rec: seen.append(rec["t"]))
    assert len(recs) == 12 and seen == [r["t"] for r in recs]
    assert all("phase_b" not in t for t in ctl.seen)        # the degradation is not announced
    for r in recs:
        assert set(r) == {"t", "tel", "policy", "info", "applied", "source"}
        assert r["source"] == "fixed" and r["policy"] == pol
    late = [r for r in recs if r["tel"]["phase_b"]["active"]]
    assert late and all(r["applied"]["workloads"]["A"]["preferred_cpus"] == [0, 1] for r in late)
    early = [r for r in recs if not r["tel"]["phase_b"]["active"]]
    assert early and all(r["applied"] == pol for r in early)


def test_run_episode_matches_harness_without_degradation():
    from pis.harness import BaseGreedyController, run_scenario
    seed = 3102
    sc = dataclasses.replace(_scenario("none", duration=10.0, seed=seed), spec=adapters.make_spec(seed))
    mine = run_episode(sc, BaseGreedyController(sc.spec), "r6")
    ref = run_scenario(seed, BaseGreedyController(sc.spec), duration_s=10.0)
    assert [(_strip(r["tel"]), r["policy"]) for r in mine] == [(r["tel"], r["policy"]) for r in ref]


class WithDsim(Fixed):
    def __init__(self, policy):
        super().__init__(policy)
        self.dsims, self.sims = [], []

    def attach_dsim(self, dsim):
        self.dsims.append(dsim)

    def attach_sim(self, sim):
        self.sims.append(sim)


def test_run_episode_attaches_degraded_sim():
    sc = _scenario("cpu_loss", onset=1.0, duration=3.0)
    pol = adapters.initial_policy()
    ctl = WithDsim(pol)
    recs = run_episode(sc, ctl, "fixed")
    assert len(ctl.dsims) == 1 and isinstance(ctl.dsims[0], DegradedSim)
    assert ctl.sims == [ctl.dsims[0].sim]                     # the same running simulator
    assert ctl.dsims[0].scenario is sc and ctl.dsims[0].done()
    plain = run_episode(sc, Fixed(pol), "fixed")             # behaviour without the hook is unchanged
    assert [r["tel"] for r in plain] == [r["tel"] for r in recs]


def _mid_burst_sim():
    """A full-length scenario whose clock sits inside a burst (the rest of the state is irrelevant)."""
    ds = DegradedSim(scenarios.scenario_for(3400))
    b = ds.sim.bursts
    i = int(np.argmax(b > 20.0))
    assert b[i] > 20.0 and i + 1 < len(b)
    ds.sim.t = float(b[i]) + 0.5 * ds.sim.spec.a.burst_len      # burst i is ongoing
    return ds, i


def test_reseeded_clone_redraws_future_bursts():
    ds, i = _mid_burst_sim()
    a, b = ds.clone(reseed=11), ds.clone(reseed=12)
    t = ds.sim.t
    for c in (a, b):
        assert np.array_equal(c.sim.bursts[c.sim.bursts <= t], ds.sim.bursts[:i + 1])   # past and ongoing kept
        assert np.all(np.diff(c.sim.bursts) > 0) and c.sim.bursts[-1] < c.sim.duration
    assert not np.array_equal(a.sim.bursts[a.sim.bursts > t], b.sim.bursts[b.sim.bursts > t])
    assert not np.array_equal(a.sim.bursts[a.sim.bursts > t], ds.sim.bursts[i + 1:])
    assert np.array_equal(ds.clone(reseed=11).sim.bursts, a.sim.bursts)                # same reseed, same bursts


def test_clone_without_reseed_keeps_bursts():
    ds, _ = _mid_burst_sim()
    assert np.array_equal(ds.clone().sim.bursts, ds.sim.bursts)


def test_redrawn_burst_gaps_follow_the_spec():
    ds, _ = _mid_burst_sim()
    t = ds.sim.t
    gaps = []
    for r in range(300):
        f = ds.clone(reseed=r).sim.bursts
        f = f[f > t]
        if len(f):
            gaps += [f[0] - t] + list(np.diff(f))
    assert np.mean(gaps) == pytest.approx(ds.sim.spec.a.burst_every, rel=0.15)
