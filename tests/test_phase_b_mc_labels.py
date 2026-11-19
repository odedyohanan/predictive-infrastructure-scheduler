import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis.phase_b import mc_labels, scenarios  # noqa: E402
from pis.phase_b.incidents import IncidentRule  # noqa: E402
from pis.phase_b.scenarios import Degradation  # noqa: E402
from tests._load import load_script  # noqa: E402
from worldmodel.features import encode_policy, encode_state  # noqa: E402

script = load_script("b_mc_labels")
RULE = IncidentRule(k=2, interval=8)
PROBE = Path(__file__).resolve().parent.parent / "data" / "phase_b" / "timing_probe"


def _scenario(seed=3400, duration=8.0, degradation=None):
    sc = scenarios.scenario_for(seed)
    if degradation is not None:
        sc = dataclasses.replace(sc, degradation=degradation)
    return dataclasses.replace(sc, duration_s=duration)


# ------------------------------------------------------------------ labels over a joined series
def test_new_onset_within_horizon():
    viol = np.zeros(40, dtype=bool)
    viol[30:33] = True
    act = np.zeros(40, dtype=bool)
    assert mc_labels.new_onset(viol, act, RULE, t=25, horizon=10, warmup=20)
    assert not mc_labels.new_onset(viol, act, RULE, t=25, horizon=4, warmup=20)   # onset 30 > 29
    assert not mc_labels.new_onset(viol, act, RULE, t=30, horizon=9, warmup=20)   # already started


def test_state_inside_incident_counts_only_new_onsets():
    viol = np.zeros(60, dtype=bool)
    viol[22:30] = True                   # incident in force at t = 25
    act = np.zeros(60, dtype=bool)
    assert not mc_labels.new_onset(viol, act, RULE, t=25, horizon=20, warmup=20)
    viol[40:43] = True                   # a second incident after the first one ended
    assert mc_labels.new_onset(viol, act, RULE, t=25, horizon=20, warmup=20)
    assert not mc_labels.new_onset(viol, act, RULE, t=25, horizon=10, warmup=20)


def test_onset_near_horizon_end_needs_the_tail():
    # k = 3: a run starting at t+h-1 reaches length 3 only after t+h; its onset (t+h-1) is within h
    rule = IncidentRule(k=3, interval=100)
    viol = np.zeros(60, dtype=bool)
    act = np.zeros(60, dtype=bool)
    t, h = 30, 10
    viol[t + h - 1:t + h + 2] = True
    assert not mc_labels.new_onset(viol[:t + h + 1], act[:t + h + 1], rule, t=t, horizon=h, warmup=20)
    n = t + 1 + mc_labels.branch_windows(h, rule)
    assert mc_labels.new_onset(viol[:n], act[:n], rule, t=t, horizon=h, warmup=20)


def test_new_onset_ignores_warmup():
    viol = np.zeros(40, dtype=bool)
    viol[5:8] = True
    act = np.zeros(40, dtype=bool)
    assert not mc_labels.new_onset(viol, act, RULE, t=2, horizon=10, warmup=20)


# ------------------------------------------------------------------ replay and branches
@pytest.fixture(scope="module")
def cpu_state():
    """A state two windows before CPU loss begins (onset 2.5 s = window 5)."""
    sc = _scenario(degradation=Degradation("cpu_loss", 2.5, 0.0, 2.0))
    return mc_labels.state_at(sc, 2)


def test_branch_inherits_schedule(cpu_state):
    recs = mc_labels.continue_branch(cpu_state, 6, reseed=mc_labels.branch_reseed(3400, 0))
    assert [r["tel"]["phase_b"]["active"] for r in recs] == [False, False, True, True, True, True]
    for r in recs[2:]:
        assert r["tel"]["phase_b"]["alive_cpus"] == [0, 1, 4, 5, 6, 7]
        for w in r["applied"]["workloads"].values():
            assert not set(w["preferred_cpus"]) & {2, 3}


def test_branch_keeps_active_degradation():
    sc = _scenario(degradation=Degradation("leak", 1.0, 4.0, 0.8))
    st = mc_labels.state_at(sc, 4)       # 2.5 s: the leak is ramping
    recs = mc_labels.continue_branch(st, 3, reseed=mc_labels.branch_reseed(3400, 1))
    d = sc.degradation
    for i, r in enumerate(recs):
        start = round(2.5 + 0.5 * i, 3)
        assert r["tel"]["phase_b"]["active"]
        assert r["tel"]["phase_b"]["factor"] == pytest.approx(d.factor(start))


def test_continuation_without_reseed_is_the_episode():
    sc = _scenario()
    full = mc_labels.state_at(sc, 9)
    st = mc_labels.state_at(sc, 4)
    recs = mc_labels.continue_branch(st, 5, reseed=None)
    for a, b in zip(recs, full.records[5:10]):
        assert np.array_equal(encode_state(a["tel"]), encode_state(b["tel"]))
        assert a["policy"] == b["policy"]


def test_branch_leaves_state_untouched(cpu_state):
    t0 = cpu_state.dsim.sim.t
    hist = list(cpu_state.controller.base.history)
    mc_labels.continue_branch(cpu_state, 3, reseed=mc_labels.branch_reseed(3400, 2))
    assert cpu_state.dsim.sim.t == t0
    assert cpu_state.controller.base.history == hist
    assert len(cpu_state.records) == 3


def test_branch_deterministic_for_fixed_reseeds(cpu_state):
    a = mc_labels.continue_branch(cpu_state, 4, reseed=mc_labels.branch_reseed(3400, 3))
    b = mc_labels.continue_branch(cpu_state, 4, reseed=mc_labels.branch_reseed(3400, 3))
    assert all(np.array_equal(encode_state(x["tel"]), encode_state(y["tel"])) for x, y in zip(a, b))
    r1 = mc_labels.branch_risk(cpu_state, RULE, horizon=4, n_branches=3, reseed_base=mc_labels.branch_reseed(3400, 0),
                               warmup=0)
    r2 = mc_labels.branch_risk(cpu_state, RULE, horizon=4, n_branches=3, reseed_base=mc_labels.branch_reseed(3400, 0),
                               warmup=0)
    assert r1 == r2 and 0.0 <= r1 <= 1.0


def test_different_reseeds_give_different_futures(cpu_state):
    a = mc_labels.continue_branch(cpu_state, 4, reseed=mc_labels.branch_reseed(3400, 5))
    b = mc_labels.continue_branch(cpu_state, 4, reseed=mc_labels.branch_reseed(3400, 6))
    assert any(x["tel"]["a"]["arrivals"] != y["tel"]["a"]["arrivals"] for x, y in zip(a, b))


def test_dead_cpus_stay_dead_in_branches():
    sc = _scenario(degradation=Degradation("cpu_loss", 1.0, 0.0, 2.0))
    st = mc_labels.state_at(sc, 5)       # CPUs lost since window 2
    assert st.records[-1]["tel"]["phase_b"]["alive_cpus"] == [0, 1, 4, 5, 6, 7]
    for j in range(2):
        for r in mc_labels.continue_branch(st, 3, reseed=mc_labels.branch_reseed(3400, 7 + j)):
            assert r["tel"]["phase_b"]["active"] and r["tel"]["phase_b"]["alive_cpus"] == [0, 1, 4, 5, 6, 7]
            for w in r["applied"]["workloads"].values():
                assert not set(w["preferred_cpus"]) & {2, 3}


def test_branch_controllers_share_no_mutable_state(cpu_state):
    a = mc_labels.clone_controller(cpu_state.controller)
    b = mc_labels.clone_controller(cpu_state.controller)
    assert a.base.history is not b.base.history is not cpu_state.controller.base.history
    assert a.rng is not b.rng
    assert a.base.model is b.base.model is cpu_state.controller.base.model     # read-only weights shared
    a.base.history.append(-1.0)
    assert b.base.history == cpu_state.controller.base.history


def test_branch_risk_runs_the_tail(cpu_state, monkeypatch):
    seen = []
    real = mc_labels.continue_branch
    monkeypatch.setattr(mc_labels, "continue_branch", lambda st, n, r, hold=None: seen.append(n) or real(st, n, r))
    rule = IncidentRule(k=3, interval=12)
    mc_labels.branch_risk(cpu_state, rule, horizon=4, n_branches=1, reseed_base=1, warmup=0)
    assert seen == [4 + 3 - 1]


def test_branch_hold_policy(cpu_state):
    hold = cpu_state.records[-1]["policy"]
    recs = mc_labels.continue_branch(cpu_state, 3, reseed=mc_labels.branch_reseed(3400, 4), hold=hold)
    assert all(r["policy"] == hold for r in recs)


def test_branch_refuses_past_episode_end(cpu_state):
    with pytest.raises(ValueError):
        mc_labels.continue_branch(cpu_state, 20, reseed=1)


def test_replay_refuses_non_r6_and_other_ranges():
    with pytest.raises(ValueError):
        mc_labels.replay_to(3100, 5)     # a loop_a history episode
    with pytest.raises(ValueError):
        mc_labels.replay_to(3000, 5)     # held out
    with pytest.raises(ValueError):
        mc_labels.replay_to(3400, 600)   # past the episode


@pytest.mark.skipif(not PROBE.is_dir(), reason="stored probe episodes not available")
def test_replay_reproduces_stored_telemetry():
    from pis.phase_b.history import POLICY_COLUMNS, STATE_COLUMNS, Store
    store = Store(PROBE)
    seed = int(store.episodes("source = 'r6' AND kind <> 'none'").seed.iloc[0])
    st = mc_labels.replay_to(seed, 24)
    rows = store.window_rows(seed, 0, 25)
    assert len(st.records) == 25 == len(rows)
    got = np.stack([encode_state(r["tel"]) for r in st.records]).astype(np.float32)
    assert np.array_equal(got, rows[list(STATE_COLUMNS.values())].to_numpy(dtype=np.float32))
    pol = np.stack([encode_policy(r["applied"]) for r in st.records]).astype(np.float32)
    assert np.array_equal(pol, rows[list(POLICY_COLUMNS.values())].to_numpy(dtype=np.float32))


# ------------------------------------------------------------------ multilevel splitting
def _walk_exact(p_up, start, target, steps):
    """P(max of a +-1 walk reaches `target` within `steps` steps)."""
    dist = {start: 1.0}
    hit = 0.0
    for _ in range(steps):
        nxt = {}
        for x, pr in dist.items():
            for y, q in ((x + 1, p_up), (x - 1, 1 - p_up)):
                if y >= target:
                    hit += pr * q
                else:
                    nxt[y] = nxt.get(y, 0.0) + pr * q
        dist = nxt
    return hit


class _Walk:
    def __init__(self, rng):
        self.x, self.top, self.rng = 0, 0, rng


def test_splitting_estimate_random_walk():
    p_up, target, steps = 0.4, 14, 60
    exact = _walk_exact(p_up, 0, target, steps)
    assert 1e-4 < exact < 1e-2

    def clone(w, reseed):
        c = _Walk(np.random.default_rng(reseed))
        c.x, c.top = w.x, w.top
        return c

    def step(w):
        w.x += 1 if w.rng.uniform() < p_up else -1
        w.top = max(w.top, w.x)

    est, n_paths = mc_labels.splitting_estimate(
        root=_Walk(np.random.default_rng(0)), clone=clone, step=step, score=lambda w: w.x,
        hit=lambda w: w.top >= target, levels=[3, 6, 9, 12], n_roots=400, split=3, horizon=steps,
        reseed_base=1000)
    assert est == pytest.approx(exact, rel=0.3)
    assert n_paths > 400
    again, _ = mc_labels.splitting_estimate(
        root=_Walk(np.random.default_rng(0)), clone=clone, step=step, score=lambda w: w.x,
        hit=lambda w: w.top >= target, levels=[3, 6, 9, 12], n_roots=400, split=3, horizon=steps,
        reseed_base=1000)
    assert again == est


def test_splitting_without_levels_is_plain_monte_carlo():
    def clone(w, reseed):
        return _Walk(np.random.default_rng(reseed))

    def step(w):
        w.x += 1 if w.rng.uniform() < 0.5 else -1
        w.top = max(w.top, w.x)

    est, n = mc_labels.splitting_estimate(root=_Walk(None), clone=clone, step=step, score=lambda w: w.x,
                                          hit=lambda w: w.top >= 3, levels=[], n_roots=2000, split=3,
                                          horizon=10, reseed_base=0)
    assert n == 2000
    assert est == pytest.approx(_walk_exact(0.5, 0, 3, 10), abs=0.03)


def test_restart_risk_on_simulator(cpu_state):
    r, n = mc_labels.restart_risk(cpu_state, RULE, horizon=3, levels=[1, 5], n_roots=2, split=2,
                                  reseed_base=mc_labels.restart_reseed(3400, 0), warmup=0)
    assert 0.0 <= r <= 1.0 and n >= 2
    assert mc_labels.restart_risk(cpu_state, RULE, horizon=3, levels=[1, 5], n_roots=2, split=2,
                                  reseed_base=mc_labels.restart_reseed(3400, 0), warmup=0) == (r, n)


# ------------------------------------------------------------------ state sampling
def _episodes():
    return [{"episode": 3400 + i, "degraded": i % 2 == 1, "onsets": [300] if i % 2 == 1 else [],
             "windows": 600} for i in range(10)]


def test_sample_states_stratified():
    states = mc_labels.sample_states(_episodes(), n=100, horizon=20, warmup=20)
    assert len(states) == 100 == len({(s[0], s[1]) for s in states})
    pre = [s for s in states if s[2] == "pre_onset"]
    assert len(pre) == 50
    assert all(s[0] % 2 == 1 and 300 - 240 <= s[1] < 300 for s in pre)
    assert all(20 <= s[1] <= 600 - 1 - 20 for s in states)
    assert states == sorted(states)
    assert mc_labels.sample_states(_episodes(), n=100, horizon=20, warmup=20) == states


def test_sample_states_fills_from_uniform_when_few_onsets():
    eps = _episodes()
    for e in eps:
        e["onsets"] = []
    eps[1]["onsets"] = [100]             # only 80 pre-onset windows after the warm-up
    states = mc_labels.sample_states(eps, n=400, horizon=20, warmup=20)
    assert len(states) == 400
    assert sum(s[2] == "pre_onset" for s in states) == 80


def test_onset_near_end_keeps_room_for_horizon():
    eps = [{"episode": 3401, "degraded": True, "onsets": [590], "windows": 600}]
    states = mc_labels.sample_states(eps, n=10, horizon=20, warmup=20)
    assert all(s[1] <= 579 for s in states)
    states = mc_labels.sample_states(eps, n=10, horizon=20, warmup=20, tail=6)
    assert all(s[1] <= 573 for s in states)


# ------------------------------------------------------------------ script guards
def test_script_seed_guard():
    script.check_seeds([3400, 3499], smoke=False)
    with pytest.raises(ValueError):
        script.check_seeds([3100], smoke=False)           # history seeds only in smoke runs
    script.check_seeds([3100], smoke=True)
    for bad in ([3000], [1100], [2050]):
        with pytest.raises(ValueError):
            script.check_seeds(bad, smoke=True)


def test_script_refuses_smoke_outside_smoke_dir(tmp_path):
    with pytest.raises(SystemExit):
        script.main(["--label", "smoke", "--seeds", "3104", "--store", str(tmp_path)])


def test_script_refuses_smoke_options_in_real_runs():
    for extra in (["--horizon", "5"], ["--method", "plain"], ["--smoke-dir", "data/phase_b/smoke_t"]):
        with pytest.raises(SystemExit, match="need --label smoke"):
            script.main(extra)
    with pytest.raises(SystemExit):
        script.main(["--seeds", "3500"])                   # outside calibration and selection


def test_sample_states_refuses_too_many():
    with pytest.raises(ValueError):
        mc_labels.sample_states(_episodes()[:1], n=1000, horizon=20, warmup=20)
