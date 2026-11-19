import dataclasses

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis.candidates import policy_key  # noqa: E402
from pis.evaluator import predicted_rewards  # noqa: E402
from pis.phase_b import advisor, features, scenarios  # noqa: E402
from pis.phase_b.episode import run_episode  # noqa: E402
from pis.phase_b.incidents import IncidentRule  # noqa: E402
from pis.phase_b.mc_labels import new_onset  # noqa: E402

RULE = IncidentRule(3, 12, 0.5)
SEED = 3452
I = features.IDX


def _scenario(duration=6.0, kind="none", seed=SEED):
    base = scenarios.scenario_for(seed)
    deg = scenarios.Degradation(kind, 1.0, 1.0, scenarios.MAGNITUDES.get(kind, 0.0)) if kind != "none" \
        else scenarios.Degradation("none", 0.0, 0.0, 0.0)
    return dataclasses.replace(base, duration_s=duration, degradation=deg)


def _state(b_rate=1.0, violate=False):
    """An encoded 43-number state: B's rate sets the value; `violate` makes A's window late."""
    s = np.zeros(features.N_STATE, np.float32)
    s[I["b.rate"]] = b_rate
    s[I["a.completed_ontime"]] = np.log1p(1000.0)
    s[I["a.late_new"]] = np.log1p(500.0) if violate else 0.0
    return s


class StubModel:
    """Rollouts from a function of the candidate index: fn(i) -> per-future list of (b_rate, violate)."""
    context, room = 4, 64

    def __init__(self, fn, bounds=None):
        self.fn, self.bounds, self.calls = fn, bounds, []
        self.meta = {"sha256": "stub"}

    def sample_rollouts_batch(self, history, policies, known, horizon, n, seed=None):
        self.calls.append({"k": len(policies), "horizon": horizon, "n": n, "seed": seed,
                           "history": len(history.states)})
        out = np.zeros((len(policies), n, horizon, features.N_STATE), np.float32)
        for i in range(len(policies)):
            for j, (b, v) in enumerate(self.fn(i, n)):
                out[i, j] = _state(b, v)
        return out


class ScriptedRisk:
    """Predictor stub: the risk for the t-th window comes from a list (by the number of rows seen)."""
    name = "scripted"

    def __init__(self, risks):
        self.risks, self.seen = risks, []

    def risk(self, rows, H, episode=None):
        t = len(rows) - 1
        self.seen.append((t, int(rows["window"].iloc[-1]), episode, H))
        return self.risks[t] if t < len(self.risks) else self.risks[-1]


def _params(**kw):
    base = dict(m=2, n=4, h=3, eps=0.25, slo_lam=40.0, mu=8.0, change_cost=0.0)
    return advisor.AdvisorParams(**{**base, **kw})


def _alarms(thr=0.5, low=0.35, H=6):
    return {"predictor": "scripted", "threshold": thr, "low": low, "H": H}


def _ctrl(sc, risks, model=None, selection="model", alarms=None, **kw):
    model = model or StubModel(lambda i, n: [(1.0, False)] * n)
    return advisor.AdvisorController(sc.spec, ScriptedRisk(risks), alarms or _alarms(), model,
                                     kw.pop("params", _params()), selection=selection, rule=RULE, warmup=2,
                                     seed=sc.seed, **kw)


# ------------------------------------------------------------------ pure parts
def test_cvar_is_mean_of_worst_tail():
    x = np.array([[5.0, 1.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                  [0.0] * 10])
    np.testing.assert_allclose(advisor.cvar(x, 0.10), [1.0, 0.0])
    np.testing.assert_allclose(advisor.cvar(x, 0.30), [2.0, 0.0])
    np.testing.assert_allclose(advisor.cvar(np.array([[4.0, 2.0]]), 0.10), [2.0])   # at least one value


def test_choose_candidate_chance_constraint():
    mean = np.array([3.0, 2.0, 1.0])
    cv = np.zeros(3)
    assert advisor.choose_candidate(mean, cv, np.array([0.5, 0.1, 0.0]), eps=0.1) == 1
    assert advisor.choose_candidate(mean, cv, np.array([0.1, 0.1, 0.0]), eps=0.1) == 0
    assert advisor.choose_candidate(mean, cv, np.array([0.5, 0.5, 0.2]), eps=0.1) == -1


def test_choose_candidate_cvar_tie_break():
    mean = np.array([2.0, 2.0 + 5e-10, 2.0, 1.0])
    cv = np.array([0.5, 0.2, 0.9, 5.0])
    p = np.array([0.0, 0.0, 0.0, 0.0])
    assert advisor.choose_candidate(mean, cv, p, eps=0.1) == 2           # within 1e-9: higher CVaR wins
    assert advisor.choose_candidate(mean, np.array([0.5, 0.5, 0.5, 5.0]), p, eps=0.1) == 0   # then index
    assert advisor.choose_candidate(np.array([2.0, 2.0 + 1e-6]), np.array([9.0, 0.0]), p[:2], 0.1) == 1


def test_score_rollouts_uses_phase_a_reward_on_the_first_39_numbers():
    spec = scenarios.scenario_for(SEED).spec
    rng = np.random.default_rng(0)
    roll = rng.uniform(0, 3, (2, 5, 7, features.N_STATE)).astype(np.float32)
    mean, total = advisor.score_rollouts(spec, roll, h=4, slo_lam=40.0, mu=8.0, change=np.array([0.0, 0.3]))
    r = predicted_rewards(spec, roll[:, :, :4, :39], 40.0, 8.0)
    np.testing.assert_allclose(total, r.sum(axis=2) - np.array([0.0, 0.3])[:, None])
    np.testing.assert_allclose(mean, r.mean(axis=(1, 2)) - np.array([0.0, 0.3]))


def test_onset_share_matches_full_series_judgement():
    rng = np.random.default_rng(1)
    rule = IncidentRule(4, 10, 0.5)
    for _ in range(40):
        T = int(rng.integers(5, 80))
        v0 = rng.random(T) < rng.uniform(0, 0.7)
        pv = rng.random((6, 9)) < rng.uniform(0, 0.8)
        t, h, warm = T - 1, 6, int(rng.integers(0, 30))
        want = np.mean([new_onset(np.concatenate([v0, pv[i]]), np.zeros(T + 9, bool), rule, t, h, warm)
                        for i in range(6)])
        assert advisor.onset_share(v0, pv, rule, h, warm) == pytest.approx(want)


# ------------------------------------------------------------------ the controller in episodes
def test_below_threshold_returns_r6_greedy_unchanged():
    from pis.harness import BaseGreedyController
    sc = _scenario(5.0)
    ref = run_episode(sc, BaseGreedyController(sc.spec), "r6")
    model = StubModel(lambda i, n: [(1.0, False)] * n)
    ctl = _ctrl(sc, [0.1, 0.2, 0.49], model=model)
    mine = run_episode(sc, ctl, "advisor")
    assert [r["policy"] for r in mine] == [r["policy"] for r in ref]
    assert not model.calls
    for r in mine:
        i = r["info"]
        assert set(advisor.INFO_KEYS) <= set(i)
        assert i["mode"] is False and i["fallback"] is False and i["pred_incident_p"] is None
    assert [r["info"]["risk"] for r in mine[:2]] == [None, None]                  # warm-up: no risk
    assert mine[2]["info"]["risk"] == pytest.approx(0.49)
    seen = ctl.predictor.seen
    assert [s[0] for s in seen] == list(range(2, len(mine)))                    # rows up to t only
    assert all(s[1] == s[0] and s[2] == sc.seed and s[3] == 6 for s in seen)


def test_no_threshold_never_intervenes():
    sc = _scenario(3.0)
    model = StubModel(lambda i, n: [(1.0, False)] * n)
    recs = run_episode(sc, _ctrl(sc, [0.99], model=model, alarms=_alarms(None, None)), "advisor")
    assert not any(r["info"]["mode"] for r in recs) and not model.calls


def test_hysteresis_keeps_mode_until_low():
    sc = _scenario(5.0)
    risks = [0, 0, 0.4, 0.6, 0.4, 0.36, 0.34, 0.45, 0.5, 0.1]
    recs = run_episode(sc, _ctrl(sc, risks), "advisor")
    assert [r["info"]["mode"] for r in recs] == [False, False, False, True, True, True, False, False, True,
                                                False]
    on = [r["info"] for r in recs if r["info"]["mode"]]
    assert all(i["pred_incident_p"] is not None and i["n_candidates"] >= 1 for i in on)


def test_chance_constrained_choice_with_known_rollouts():
    sc = _scenario(3.0)

    def fn(i, n):          # greedy (0): best value but always violates; 1: good and safe; others safe, worse
        if i == 0:
            return [(5.0, True)] * n
        if i == 1:
            return [(3.0, False)] * (n - 1) + [(3.0, True)]          # p = 1/4 <= eps 0.25
        return [(1.0, False)] * n
    model = StubModel(fn)
    ctl = _ctrl(sc, [0.9], model=model)
    recs = run_episode(sc, ctl, "advisor")
    on = [r for r in recs if r["info"]["mode"]]
    assert on
    for r in on:
        i = r["info"]
        assert i["chosen"] == 1 and i["fallback"] is False
        assert i["pred_incident_p"] == pytest.approx(0.25)
    call = model.calls[-1]
    assert call["horizon"] == 3 + RULE.k - 1 and call["n"] == 4                # k - 1 tail windows
    last = ctl.last
    assert last["kinds"][0] == "greedy" and len(last["candidates"]) == call["k"]
    assert recs[-1]["policy"] == last["candidates"][1]
    assert len({policy_key(p) for p in last["candidates"]}) == len(last["candidates"])


def test_infeasible_candidates_fall_back_to_greedy():
    from pis.harness import BaseGreedyController
    sc = _scenario(3.0)
    ref = run_episode(sc, BaseGreedyController(sc.spec), "r6")
    model = StubModel(lambda i, n: [(1.0 + i, True)] * n)
    recs = run_episode(sc, _ctrl(sc, [0.9], model=model), "advisor")
    assert [r["policy"] for r in recs] == [r["policy"] for r in ref]
    on = [r["info"] for r in recs if r["info"]["mode"]]
    assert on and all(i["fallback"] and i["chosen"] == -1 and i["reason"] == "infeasible" for i in on)
    assert all(i["pred_incident_p"] == pytest.approx(1.0) for i in on)


def test_out_of_training_bounds_falls_back():
    sc = _scenario(3.0)
    lo, hi = np.zeros(features.N_STATE, np.float32), np.full(features.N_STATE, 1e-3, np.float32)
    model = StubModel(lambda i, n: [(1.0, False)] * n, bounds=(lo, hi))
    recs = run_episode(sc, _ctrl(sc, [0.9], model=model), "advisor")
    on = [r["info"] for r in recs if r["info"]["mode"]]
    assert on and all(i["fallback"] and i["reason"] == "out_of_bounds" for i in on)
    assert not model.calls


def test_decide_time_with_stub_is_small():
    sc = _scenario(5.0)
    recs = run_episode(sc, _ctrl(sc, [0.9]), "advisor")
    ms = [r["info"]["decide_ms"] for r in recs if r["info"]["mode"]]
    assert ms and float(np.median(ms)) < 50.0
    i = recs[-1]["info"]
    assert i["risk_ms"] >= 0 and i["select_ms"] >= 0 and i["risk_ms"] + i["select_ms"] <= i["decide_ms"] + 1e-6


def test_random_selection_is_seeded_and_uses_the_candidates():
    sc = _scenario(4.0)
    a = run_episode(sc, _ctrl(sc, [0.9], selection="random"), "advisor")
    b = run_episode(sc, _ctrl(sc, [0.9], selection="random"), "advisor")
    assert [r["policy"] for r in a] == [r["policy"] for r in b]
    on = [r["info"] for r in a if r["info"]["mode"]]
    assert on and all(0 <= i["chosen"] < i["n_candidates"] and i["pred_incident_p"] is None for i in on)


def test_always_selection_ignores_the_alarm():
    sc = _scenario(3.0)
    recs = run_episode(sc, _ctrl(sc, [0.0], selection="always"), "advisor")
    assert [r["info"]["mode"] for r in recs] == [False, False] + [True] * (len(recs) - 2)


def test_oracle_selection_branches_the_degraded_sim():
    sc = _scenario(4.0, kind="cpu_loss")
    ctl = _ctrl(sc, [0.9], selection="oracle", params=_params(h=2), oracle_branches=2)
    recs = run_episode(sc, ctl, "advisor")
    assert len(recs) == 8
    assert np.allclose(np.diff([r["t"] for r in recs]), 0.5)                    # branches never advance it
    on = [r["info"] for r in recs if r["info"]["mode"]]
    assert on and all(i["pred_incident_p"] is None or 0.0 <= i["pred_incident_p"] <= 1.0 for i in on)
    assert any(not i["fallback"] for i in on)


def test_oracle_needs_the_degraded_sim():
    sc = _scenario(3.0)
    ctl = _ctrl(sc, [0.9], selection="oracle")
    from pis.harness import run_scenario
    with pytest.raises(RuntimeError, match="attach_dsim"):
        run_scenario(sc.seed, ctl, duration_s=3.0)


def test_unknown_selection_refused():
    sc = _scenario(3.0)
    with pytest.raises(ValueError):
        _ctrl(sc, [0.0], selection="greedy")


def test_params_from_step_a():
    p = advisor.AdvisorParams.from_step_a({"mu": 8.0, "slo_lam": 40.0, "change_cost": 0.0, "lam": 1.0},
                                          H=30, n=8, eps=0.05)
    assert (p.m, p.n, p.h, p.eps, p.mu, p.slo_lam) == (4, 8, 20, 0.05, 8.0, 40.0)
    assert advisor.AdvisorParams.from_step_a({"mu": 1, "slo_lam": 2, "change_cost": 0}, H=12).h == 12
