import copy

import numpy as np
import pytest
import torch
from torch import nn

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.candidates import is_valid, policy_key  # noqa: E402
from pis.ensemble import Ensemble  # noqa: E402
from pis.evaluator import Params, score_candidates  # noqa: E402
from pis.harness import run_scenario  # noqa: E402
from pis.reviser import (ReviserLoopController, consequences, neighbours, project, revise,  # noqa: E402
                         revise_detail, torch_rollout)

from sim.policy import policy_from_dict  # noqa: E402
from worldmodel.features import INPUT_NAMES, POLICY_NAMES, STATE_NAMES  # noqa: E402
from worldmodel.model import Normalizer, WorldModel  # noqa: E402

SEED = 1100
CKPT = "models/rl_r6/final.pt"
GATE = "worldmodel:models/gru_mix_bounded.pt"
NS, NP = len(STATE_NAMES), len(POLICY_NAMES)
B, P99 = STATE_NAMES.index("b.rate"), STATE_NAMES.index("a.p99_ms")
TARGET_B_CPUS = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)


def P(**kw):
    base = dict(n_candidates_m=4, horizon_h=2, lam=1.0, mu=2.0, change_cost=0.0, tau=np.inf, slo_lam=10.0)
    return Params(**{**base, **kw})


class ToyNet(nn.Module):
    """Uses the newest input row only; sets chosen state fields to fn(policy encoding) (dict index -> value)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        last = x[:, -1] if x.dim() == 3 else x
        s, pol = last[:, :NS], last[:, NS:NS + NP]
        delta = torch.zeros_like(s)
        for idx, val in self.fn(pol).items():
            delta[:, idx] = val - s[:, idx]
        return delta


def stub_member(fn) -> WorldModel:
    n_in = len(INPUT_NAMES)
    return WorldModel(ToyNet(fn), Normalizer(np.zeros(n_in, np.float32), np.ones(n_in, np.float32)),
                      Normalizer(np.zeros(NS, np.float32), np.ones(NS, np.float32)),
                      meta={"arch": "gru", "history": 1})


def stub_ensemble(fns) -> Ensemble:
    return Ensemble([stub_member(f) for f in fns], train_std=np.ones(NS, np.float32))


def _elig_b(pol):
    return pol[:, 8:16]


def _share_b(pol):
    return pol[:, 25]


def toy_fn(pol):
    """Known optimum: B share 50 % and B on CPUs 0-3 (B rate 2000); start (B 20 %, CPUs 4-7) gives 640."""
    b = 1000.0 * (2.0 - 4.0 * (_share_b(pol) - 0.5) ** 2 - ((_elig_b(pol) - TARGET_B_CPUS) ** 2).mean(1))
    return {B: b}


def start_policy():
    p = adapters.initial_policy()       # A 0-3, B 4-7 (GPUs 0, 1), C 4-7
    for w, s in zip("ABC", (40, 20, 40)):
        p["workloads"][w]["cpu_share"] = s
    return p


def optimum_policy():
    p = start_policy()
    p["workloads"]["B"]["preferred_cpus"] = [0, 1, 2, 3]
    p["workloads"]["B"]["cpu_share"], p["workloads"]["A"]["cpu_share"] = 50, 25
    p["workloads"]["C"]["cpu_share"] = 25
    return p


@pytest.fixture(scope="module")
def setting():
    spec = adapters.make_spec(SEED)
    sim = adapters.make_sim(SEED)
    tel = sim.run_window(policy_from_dict(adapters.initial_policy()))
    return spec, [(tel, start_policy())]


def _score(spec, hist, ens, pols, params, lam=None):
    roll = ens.rollout_batch(hist, pols, adapters.known_of(spec), params.horizon_h)
    return score_candidates(spec, roll, pols, pols[0], params, train_std=ens.train_std, lam=lam)


# ------------------------------------------------------------------ brief tests
def test_revise_never_returns_invalid_policy(setting):
    spec, hist = setting
    rng = np.random.default_rng(0)
    fns = [lambda pol, i=i: {B: 500.0 * (1 + i) * torch.sin(3.0 * pol[:, :24]).sum(1)} for i in range(3)]
    ens = stub_ensemble(fns)
    from pis.candidates import random_valid_policies
    starts = random_valid_policies(spec, start_policy(), 6, 1.0, rng)
    for start in starts:
        for mech in ("gradient", "neighbourhood"):
            p = revise(spec, hist, start, ens, P(), steps=5, mechanism=mech, lr=2.0)
            assert is_valid(p)
            policy_from_dict(p)                                  # the simulator accepts it
            assert sorted(p["workloads"]) == ["A", "B", "C"]
            shares = [p["workloads"][w]["cpu_share"] for w in "ABC"]
            assert sum(shares) == 100 and min(shares) >= 10 and all(s % 5 == 0 for s in shares)
            assert p["workloads"]["A"]["gpus"] == []
            assert [p["workloads"][w]["latency_class"] for w in "ABC"] == \
                [start["workloads"][w]["latency_class"] for w in "ABC"]


def test_revise_improves_or_keeps_predicted_score(setting):
    spec, hist = setting
    ens = stub_ensemble([toy_fn, toy_fn])
    params = P()
    for mech in ("gradient", "neighbourhood"):
        r = revise_detail(spec, hist, start_policy(), ens, params, steps=8, mechanism=mech, lr=0.5)
        s, _ = _score(spec, hist, ens, [start_policy(), r.policy], params)
        assert s[1] >= s[0]
        assert r.score == pytest.approx(s[1]) and r.start_score == pytest.approx(s[0])
        assert r.score > r.start_score                           # the toy has room to improve
    # at the optimum nothing better exists: the start is kept
    for mech in ("gradient", "neighbourhood"):
        p = revise(spec, hist, optimum_policy(), ens, params, steps=4, mechanism=mech)
        assert policy_key(p) == policy_key(optimum_policy())


def test_uncertainty_penalty_stops_revision_toward_high_disagreement_region(setting):
    """Members agree at B share 20 % and disagree more the more share B gets, while their mean says more
    share for B is better: a strong penalty keeps the start, a weak one follows the mean."""
    spec, hist = setting
    fns = [lambda pol, i=i: {B: 1000.0 * (1.0 + (1.0 + 2.0 * i) * (_share_b(pol) - 0.2))} for i in range(3)]
    ens = stub_ensemble(fns)
    for mech in ("gradient", "neighbourhood"):
        strong = revise_detail(spec, hist, start_policy(), ens, P(), steps=6, mechanism=mech, lam=1.0, lr=0.5)
        weak = revise_detail(spec, hist, start_policy(), ens, P(), steps=6, mechanism=mech, lam=0.01, lr=0.5)
        assert strong.policy["workloads"]["B"]["cpu_share"] <= 20, mech
        assert strong.disagreement <= strong.start_disagreement + 1e-9
        assert weak.policy["workloads"]["B"]["cpu_share"] > 20, mech
        assert weak.disagreement > weak.start_disagreement
    with pytest.raises(ValueError):
        revise(spec, hist, start_policy(), ens, P(lam=0.0))      # the penalty may never be off
    with pytest.raises(ValueError):
        revise(spec, hist, start_policy(), ens, P(), lam=0.0)


def test_consequences_signs(setting):
    """Stub: B rate and A p99 both grow with B's CPU count; less CPU for B lowers both."""
    spec, hist = setting
    def fn(pol, k=1.0):
        n = _elig_b(pol).sum(1)
        return {B: 10.0 * k * n, P99: torch.log1p(20.0 + 5.0 * k * n)}
    ens = stub_ensemble([fn, lambda pol: fn(pol, 1.2)])
    more = start_policy()                                         # B on CPUs 4-7
    less = copy.deepcopy(more)
    less["workloads"]["B"]["preferred_cpus"] = [6, 7]
    roll = ens.rollout_batch(hist, [less, more], adapters.known_of(spec), 3)
    c = consequences(roll[:, 0], roll[:, 1])
    assert c["b_rate"] < 0 and c["a_p99_ms"] < 0
    assert c["c_work_core_ms"] == 0.0 and c["a_queue_work_ms"] == 0.0
    assert "B rate falls" in c["text"] and "A p99 falls" in c["text"] and "C work unchanged" in c["text"]
    # reference vs candidate swapped: signs flip; [N, 1, H, 39] input accepted
    c2 = consequences(roll[:, 1:2], roll[:, 0:1])
    assert c2["b_rate"] == pytest.approx(-c["b_rate"]) and c2["a_p99_ms"] == pytest.approx(-c["a_p99_ms"])


def test_neighbourhood_search_matches_gradient_on_toy_problem_within_tolerance(setting):
    spec, hist = setting
    ens = stub_ensemble([toy_fn, toy_fn, toy_fn])
    params = P()
    g = revise_detail(spec, hist, start_policy(), ens, params, steps=30, mechanism="gradient", lr=0.5)
    n = revise_detail(spec, hist, start_policy(), ens, params, steps=30, mechanism="neighbourhood")
    s, _ = _score(spec, hist, ens, [start_policy(), optimum_policy()], params)
    gain = s[1] - s[0]
    assert gain > 0
    for r in (g, n):
        assert r.score >= s[1] - 0.05 * gain                     # within 5 % of the optimum's gain
    assert abs(g.score - n.score) <= 0.05 * gain
    assert n.policy["workloads"]["B"]["preferred_cpus"] == [0, 1, 2, 3]
    assert n.policy["workloads"]["B"]["cpu_share"] == 50


# ------------------------------------------------------------------ helpers and controller
def test_project_repairs_and_snaps():
    tmpl = start_policy()
    elig = np.zeros((3, 8))
    elig[0, :4] = 0.9                          # A wants 0-3; B wants nothing; nobody wants 4-7
    elig[1, 6] = 0.3                           # B's most likely CPU is 6
    elig[2, 4:] = 0.2
    p = project(tmpl, elig, np.array([0.31, 0.62, 0.07]), np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]))
    assert is_valid(p)
    wl = p["workloads"]
    assert wl["A"]["preferred_cpus"] == [0, 1, 2, 3] and wl["B"]["preferred_cpus"] == [6]
    assert wl["C"]["preferred_cpus"] == [4, 5, 7] or set(wl["C"]["preferred_cpus"]) >= {4, 5, 7}
    assert [wl[w]["cpu_share"] for w in "ABC"] == [30, 60, 10]
    assert wl["C"]["gpus"] == [0] and wl["B"]["gpus"] == []
    full = project(tmpl, np.ones((3, 8)), np.ones(3) / 3, np.array([[0, 1, 0], [0, 1, 0]]))
    assert all(full["workloads"][w]["preferred_cpus"] == [] for w in "ABC")


def test_neighbours_are_valid_single_edits():
    p = start_policy()
    nb = neighbours(p)
    assert nb and all(is_valid(q) for q in nb)
    assert len({policy_key(q) for q in nb}) == len(nb) and policy_key(p) not in {policy_key(q) for q in nb}


def test_torch_rollout_matches_numpy_rollout():
    """The differentiable rollout reproduces Ensemble.rollout_batch for discrete policies (real GRU)."""
    from worldmodel.features import encode_policy
    spec = adapters.make_spec(SEED)
    sim = adapters.make_sim(SEED)
    hist, cur = [], adapters.initial_policy()
    for _ in range(10):
        tel = sim.run_window(policy_from_dict(cur))
        hist.append((tel, cur))
    ens = Ensemble.from_paths([GATE])
    pols = [start_policy(), optimum_policy()]
    known = adapters.known_of(spec)
    for h in (hist, hist[:1], hist[:4]):
        ref = ens.rollout_batch(h, pols, known, 3)
        with torch.no_grad():
            got = torch_rollout(ens.models, h, torch.as_tensor(np.stack([encode_policy(p) for p in pols])),
                                known, 3).numpy()
        assert np.allclose(got, ref, atol=1e-4)




def test_reviser_controller_revises_untrusted_and_calls_propose_once():
    """Members disagree on everything except at the toy optimum; tau gates Step A's best candidate, the
    reviser finds a trusted improvement. propose runs exactly once per window."""
    spec = adapters.make_spec(SEED)
    fns = [lambda pol, i=i: toy_fn(pol) | {P99: torch.tensor(0.0).expand(pol.shape[0])
                                            + 0.5 * i * ((_elig_b(pol) - TARGET_B_CPUS) ** 2).mean(1)}
           for i in range(3)]
    ens = stub_ensemble(fns)
    params = P(tau=0.05, lam=0.1, horizon_h=1)
    ctl = ReviserLoopController(spec, ens, params, mechanism="neighbourhood", rev_steps=12, rev_lam=1.0,
                                probe_real=True, base_ckpt=CKPT)
    calls = []
    orig = ctl.base.propose
    ctl.base.propose = lambda *a, **k: calls.append(1) or orig(*a, **k)
    recs = run_scenario(SEED, ctl, duration_s=3.0)
    assert len(calls) == len(recs)
    infos = [r["info"] for r in recs]
    revised = [i for i in infos if i["revised"]]
    assert revised, [i["reason"] for i in infos]
    for i in revised:
        assert i["reason"] == "revised" and not i["fallback"]
        assert i["rev_dis"] <= params.tau and i["rev_pred_gain"] > 0
        assert set(i["consequences"]) >= {"a_p99_ms", "b_rate", "c_work_core_ms", "a_queue_work_ms", "text"}
        assert np.isfinite(i["rev_real_gain"])
    for r in recs:
        assert is_valid(r["policy"])


def test_reviser_controller_rejects_inactive_penalty():
    spec = adapters.make_spec(SEED)
    with pytest.raises(ValueError):
        ReviserLoopController(spec, stub_ensemble([toy_fn]), P(), rev_lam=0.0, base_ckpt=CKPT)


def test_gradient_decision_time_five_grus(setting):
    """Gradient revision at the default steps through 5 GRU members (H=3): per-revision time recorded.
    The budget (p95 <= 50 ms for the whole decision) is checked on the tuning runs; here a loose ceiling
    guards against accidental blow-ups on a loaded machine."""
    import time
    spec = adapters.make_spec(SEED)
    sim = adapters.make_sim(SEED)
    hist, cur = [], adapters.initial_policy()
    for _ in range(10):
        hist.append((sim.run_window(policy_from_dict(cur)), cur))
    ens = Ensemble.from_paths([GATE] * 5)
    torch.set_num_threads(1)
    ts = []
    for _ in range(12):
        t0 = time.perf_counter()
        revise(spec, hist, start_policy(), ens, P(horizon_h=3))
        ts.append((time.perf_counter() - t0) * 1000.0)
    assert np.percentile(ts[2:], 95) < 500.0
