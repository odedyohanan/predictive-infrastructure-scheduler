import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.candidates import policy_key  # noqa: E402
from pis.ensemble import Ensemble  # noqa: E402
from pis.evaluator import Params, predicted_rewards  # noqa: E402
from pis.harness import BaseGreedyController, run_scenario, scenario_metrics  # noqa: E402
from pis.loop import LoopController  # noqa: E402

from sim.policy import policy_from_dict  # noqa: E402
from worldmodel.features import STATE_NAMES, encode_state  # noqa: E402

SEED = 1100
CKPT = "models/rl_r6/final.pt"
B = STATE_NAMES.index("b.rate")


def P(**kw):
    base = dict(n_candidates_m=4, horizon_h=2, lam=0.0, mu=2.0, change_cost=0.0, tau=1.0, slo_lam=10.0)
    return Params(**{**base, **kw})


class StubEnsemble:
    """history 1 (always full); every member predicts the newest state, candidate i gets B rate b(i)."""
    history = 1
    train_std = np.ones(39, np.float32)

    def __init__(self, n_members=3, spread_on=None, b_of=lambda i: float(i)):
        self.n, self.spread_on, self.b_of = n_members, spread_on, b_of
        self.calls = 0

    def __len__(self):
        return self.n

    def rollout_batch(self, history, policies, known, horizon):
        self.calls += 1
        s = encode_state(history[-1][0])
        r = np.broadcast_to(s, (self.n, len(policies), horizon, 39)).copy()
        for i in range(len(policies)):
            r[:, i, :, B] = self.b_of(i)
        if self.spread_on is not None:                 # members disagree on every candidate
            r[:, :, :, B] += np.arange(self.n)[:, None, None] * self.spread_on
        return r


def _views(n_windows=12):
    """Real (tel, current, view) triples from a base-greedy run, as run_scenario builds them."""
    recs = run_scenario(SEED, BaseGreedyController(adapters.make_spec(SEED)), duration_s=n_windows * 0.5)
    out, hist, cur = [], [], adapters.initial_policy()
    for r in recs:
        out.append((r["tel"], cur, hist + [(r["tel"], cur)]))
        hist.append((r["tel"], r["policy"]))
        cur = r["policy"]
    return out


@pytest.fixture(scope="module")
def views():
    return _views()


def _ctl(ens, **kw):
    params = kw.pop("params", P())
    return LoopController(adapters.make_spec(SEED), ens, params, base_ckpt=CKPT, **kw)


def _greedy(tel, cur):
    base = adapters.BaseScheduler(CKPT, adapters.make_spec(SEED))
    g, _ = base.propose(tel, cur, 0, np.random.default_rng(0))
    return g


def test_picks_model_best_and_records_info(views):
    tel, cur, view = views[0]
    ctl = _ctl(StubEnsemble(b_of=lambda i: 10.0 * i))     # the last candidate is predicted best
    pol, info = ctl.decide(tel, cur, view)
    assert not info["fallback"] and info["reason"] == "model"
    assert info["chosen"] == info["n_candidates"] - 1
    assert policy_key(pol) == policy_key(ctl.last_candidates[-1])
    assert len(info["scores"]) == len(info["disagreement"]) == info["n_candidates"]


def test_high_disagreement_falls_back_to_base_greedy(views):
    tel, cur, view = views[0]
    ctl = _ctl(StubEnsemble(spread_on=5.0, b_of=lambda i: 10.0 * i), params=P(tau=0.1))
    pol, info = ctl.decide(tel, cur, view)
    assert info["fallback"] and info["reason"] == "untrusted"
    assert pol == _greedy(tel, cur)
    # the same disagreement without the fallback gate picks the model's best
    ctl2 = _ctl(StubEnsemble(spread_on=5.0, b_of=lambda i: 10.0 * i), params=P(tau=0.1), use_fallback=False)
    _, info2 = ctl2.decide(tel, cur, view)
    assert not info2["fallback"]


def test_use_uncertainty_off_removes_penalty(views):
    tel, cur, view = views[0]
    on = _ctl(StubEnsemble(spread_on=1.0), params=P(lam=3.0, tau=np.inf))
    off = _ctl(StubEnsemble(spread_on=1.0), params=P(lam=3.0, tau=np.inf), use_uncertainty=False)
    _, i_on = on.decide(tel, cur, view)
    _, i_off = off.decide(tel, cur, view)
    assert np.allclose(np.array(i_off["scores"]) - np.array(i_on["scores"]),
                       3.0 * np.array(i_on["disagreement"]))


def test_short_history_uses_base(views):
    tel, cur, view = views[0]                    # a single window of history
    stub = StubEnsemble(b_of=lambda i: 10.0 * i)
    stub.history = 8
    ctl = _ctl(stub)
    pol, info = ctl.decide(tel, cur, view)
    assert info["fallback"] and info["reason"] == "short_history"
    assert stub.calls == 0
    assert pol == _greedy(tel, cur)


def test_single_identical_candidate_no_crash(views):
    tel, cur, view = views[0]
    g = _greedy(tel, cur)
    ctl = _ctl(StubEnsemble(n_members=1), params=P(n_candidates_m=0))
    pol, info = ctl.decide(tel, g, view[:-1] + [(tel, g)])     # current == greedy: one candidate
    assert info["n_candidates"] == 1 and info["chosen"] == 0
    assert policy_key(pol) == policy_key(g)


def test_random_selection_reproducible_with_seed(views):
    def picks(select_seed):
        ctl = _ctl(StubEnsemble(), selection="random", select_seed=select_seed, params=P(n_candidates_m=6))
        return [ctl.decide(t, c, v)[1]["chosen"] for t, c, v in views]
    a, b = picks(3), picks(3)
    assert a == b
    assert len(set(a)) > 1


def test_oracle_selection_picks_true_best():
    from pis.adapters import clone_sim
    from pis.evaluator import change_penalty
    spec = adapters.make_spec(SEED)
    sim = adapters.make_sim(SEED)
    base = BaseGreedyController(spec)
    cur, hist = adapters.initial_policy(), []
    for _ in range(10):
        tel = sim.run_window(policy_from_dict(cur))
        pol, _ = base.decide(tel, cur, hist)
        hist.append((tel, pol))
        cur = pol
    tel = sim.run_window(policy_from_dict(cur))
    params = P(horizon_h=2, n_candidates_m=6, change_cost=0.1)
    ctl = _ctl(StubEnsemble(), selection="oracle", params=params)
    ctl.attach_sim(sim)
    t_before = sim.t if hasattr(sim, "t") else None
    pol, info = ctl.decide(tel, cur, hist + [(tel, cur)])
    if t_before is not None:
        assert sim.t == t_before                       # the real run is untouched
    cands = ctl.last_candidates
    assert len(cands) > 1 and not info["fallback"] and info["reason"] == "oracle"
    true = []
    for p in cands:
        c = clone_sim(sim)
        enc = np.stack([encode_state(c.run_window(policy_from_dict(p))) for _ in range(2)])
        true.append(predicted_rewards(spec, enc[None, None], params.slo_lam, params.mu).mean()
                    - change_penalty([p], cur, params.change_cost)[0])
    assert info["chosen"] == int(np.argmax(true))
    assert policy_key(pol) == policy_key(cands[int(np.argmax(true))])


def test_oracle_without_sim_raises(views):
    tel, cur, view = views[0]
    with pytest.raises(RuntimeError):
        _ctl(StubEnsemble(), selection="oracle").decide(tel, cur, view)


def test_decision_time_recorded(views):
    tel, cur, view = views[0]
    _, info = _ctl(StubEnsemble()).decide(tel, cur, view)
    assert info["decide_ms"] > 0


def test_always_fallback_equals_base_run():
    """tau below any disagreement: the loop behaves exactly like the base scheduler (real ensemble)."""
    spec = adapters.make_spec(SEED)
    ens = Ensemble.from_paths(["worldmodel:models/gru_mix_bounded.pt"])
    loop = run_scenario(SEED, LoopController(spec, ens, P(tau=-1.0), base_ckpt=CKPT), duration_s=6.0)
    base = run_scenario(SEED, BaseGreedyController(spec), duration_s=6.0)
    assert [r["policy"] for r in loop] == [r["policy"] for r in base]
    reasons = [r["info"]["reason"] for r in loop]
    assert reasons[:7] == ["short_history"] * 7 and set(reasons[7:]) == {"untrusted"}
    m = scenario_metrics(loop, spec)
    assert m["fallback_rate"] == 1.0
    assert m["value_per_s"] == pytest.approx(scenario_metrics(base, spec)["value_per_s"])


def test_scenario_metrics_keys(views):
    spec = adapters.make_spec(SEED)
    recs = run_scenario(SEED, BaseGreedyController(spec), duration_s=4.0)
    m = scenario_metrics(recs, spec)
    for k in ("value_per_s", "slo_v1_pct", "slo_v2_pct", "a_late_pct", "policy_changes", "decide_ms_mean",
              "decide_ms_p95", "a_queue_overdue_end", "fallback_rate", "windows"):
        assert k in m
    assert m["windows"] == 8 and m["value_per_s"] > 0
