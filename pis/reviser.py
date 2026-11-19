"""Step B: revise a policy with its predicted consequences.

Two mechanisms improve a start policy; both maximise the same objective, the ensemble score of
`pis.evaluator.score_candidates` with an uncertainty penalty that is always active (lam > 0, enforced):
gradients and local search both find regions where the world model is wrong faster than sampling does, and
the penalty keeps them where the members agree.

  gradient       (primary) a relaxed policy - CPU eligibility sigmoid(logits), CPU shares softmax(logits),
                 GPU owner softmax(logits) per GPU - is fed to every member's differentiable rollout (same
                 conventions as `Ensemble.rollout_batch`); normalised gradient ascent on predicted reward
                 - lam * disagreement: each step moves the logit with the largest gradient by `lr` and every
                 other logit in proportion, so with the default start logit (+-1) and step size (0.6) the
                 most promising CPU bit flips in two steps while weakly pulled bits stay put.
                 After every step the relaxed policy is projected to a valid discrete policy.
  neighbourhood  (fallback and ablation) best-improvement hill climbing over single edits: flip one CPU
                 bit, move one share step between two workloads, change one GPU owner.

Whatever the mechanism, the start policy and every policy visited are rescored discretely by the ensemble
(`score_candidates`, same lam) and the best is returned, so a revision never lowers the predicted score and
never returns an invalid policy.

`consequences` turns two rollouts into per-workload predicted deltas (A p99 and backlog, B rate, C work), a
diagnostic for humans. `ReviserLoopController` is the Step A loop plus a revision of its best candidate when
that candidate is untrusted (disagreement above tau) or predicted to violate A's SLO.
"""
from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from pis.adapters import clone_sim, known_of
from pis.candidates import MIN_SHARE, N_CPUS, N_GPUS, SHARE_STEP, WL, is_valid, policy_key
from pis.config import setup_imports
from pis.ensemble import DISAGREEMENT_KEYS, N_STATE
from pis.evaluator import Decision, Params, predicted_rewards, score_candidates
from pis.loop import LoopController

setup_imports()

from sim.policy import policy_from_dict  # noqa: E402
from worldmodel.features import (STATE_NAMES, encode_input, encode_known, encode_policy,  # noqa: E402
                                 encode_state)

MECHANISMS = ("gradient", "neighbourhood")
DEFAULT_STEPS = {"gradient": 3, "neighbourhood": 2}   # ~26 ms and ~17 ms at H=3, 5 GRU members
DEFAULT_LR = 0.6           # largest logit move per gradient step
INIT_LOGIT = 1.0           # |logit| of a CPU bit or GPU owner at the start policy (sigmoid(1) = 0.73)
_GPU_OWNERS = (None, "B", "C")
_B, _C = STATE_NAMES.index("b.rate"), STATE_NAMES.index("c.work_core_ms")
_P99, _Q = STATE_NAMES.index("a.p99_ms"), STATE_NAMES.index("a.queue_work_ms")
_DIS_IDX = [STATE_NAMES.index(k) for k in DISAGREEMENT_KEYS]
SHARE_COMBOS = [c for c in itertools.product(range(MIN_SHARE // SHARE_STEP, 100 // SHARE_STEP + 1), repeat=3)
                if sum(c) == 100 // SHARE_STEP]          # in units of SHARE_STEP, each >= MIN_SHARE


# ------------------------------------------------------------------ consequences
CONSEQUENCE_KEYS = (("a_p99_ms", _P99, True), ("a_queue_work_ms", _Q, True), ("b_rate", _B, False),
                    ("c_work_core_ms", _C, False))


def consequences(ens_rollouts_candidate: np.ndarray, ens_rollouts_reference: np.ndarray) -> dict:
    """Predicted (candidate - reference) per workload, in real units, mean over members and horizon.

    Inputs are one policy's ensemble rollouts, [N, H, 39] (or [N, 1, H, 39]). Returns the four deltas and a
    one-line `text` such as "A p99 falls by 3.1 ms; A backlog rises by 0.4 ms; B rate falls by 2.0; ...".
    """
    def decoded(r):
        r = np.asarray(r, dtype=np.float64)
        r = r.reshape(-1, r.shape[-2], r.shape[-1]) if r.ndim == 4 else r
        return r
    cand, ref = decoded(ens_rollouts_candidate), decoded(ens_rollouts_reference)
    out: dict = {}
    for name, idx, logged in CONSEQUENCE_KEYS:
        a, b = cand[..., idx], ref[..., idx]
        if logged:
            a, b = np.expm1(a), np.expm1(b)
        out[name] = float(np.maximum(a, 0.0).mean() - np.maximum(b, 0.0).mean())
    labels = {"a_p99_ms": ("A p99", " ms"), "a_queue_work_ms": ("A backlog", " ms"), "b_rate": ("B rate", ""),
              "c_work_core_ms": ("C work", " core-ms")}
    parts = []
    for k, (label, unit) in labels.items():
        d = out[k]
        word = "unchanged" if abs(d) < 1e-9 else ("rises by" if d > 0 else "falls by")
        parts.append(f"{label} {word}" + ("" if word == "unchanged" else f" {abs(d):.3g}{unit}"))
    out["text"] = "; ".join(parts)
    return out


# ------------------------------------------------------------------ discrete policy helpers
def _cpu_rows(policy: dict) -> np.ndarray:
    return np.stack([encode_policy(policy)[i * N_CPUS:(i + 1) * N_CPUS] > 0.5 for i in range(len(WL))])


def _build(template: dict, elig: np.ndarray, combo: tuple, owner: list) -> dict:
    """A policy dict with the template's other fields (latency classes) and the given decisions."""
    p = copy.deepcopy(template)
    for w, name in enumerate(WL):
        wp = p["workloads"][name]
        cpus = [int(k) for k in np.flatnonzero(elig[w])]
        wp["preferred_cpus"] = [] if len(cpus) == N_CPUS else cpus
        wp["cpu_share"] = int(combo[w] * SHARE_STEP)
        wp["gpus"] = [g for g in range(N_GPUS) if owner[g] == name]
    return p


def _decisions(policy: dict) -> tuple[np.ndarray, tuple, list]:
    wl = policy["workloads"]
    combo = tuple(int(wl[w]["cpu_share"]) // SHARE_STEP for w in WL)
    owner: list = [None] * N_GPUS
    for w in WL:
        for g in wl[w]["gpus"]:
            owner[g] = w
    return _cpu_rows(policy), combo, owner


def project(template: dict, elig_p: np.ndarray, share_frac: np.ndarray, gpu_p: np.ndarray) -> dict:
    """Relaxed policy -> nearest valid discrete policy.

    CPU bits: probability > 0.5, then repaired (a workload without a CPU gets its most likely CPU; an unused
    CPU goes to the workload most likely to want it). Shares: the valid split (multiples of SHARE_STEP, each
    >= MIN_SHARE, sum 100) nearest to the relaxed fractions. GPUs: most likely owner (none, B or C).
    """
    elig_p = np.asarray(elig_p, dtype=np.float64).reshape(len(WL), N_CPUS)
    e = elig_p > 0.5
    for w in range(len(WL)):
        if not e[w].any():
            e[w, int(np.argmax(elig_p[w]))] = True
    for k in range(N_CPUS):
        if not e[:, k].any():
            e[int(np.argmax(elig_p[:, k])), k] = True
    target = np.asarray(share_frac, dtype=np.float64) * (100 // SHARE_STEP)
    combo = min(SHARE_COMBOS, key=lambda c: float(((np.array(c) - target) ** 2).sum()))
    owner = [_GPU_OWNERS[int(np.argmax(np.asarray(gpu_p)[g]))] for g in range(N_GPUS)]
    return _build(template, e, combo, owner)


def neighbours(policy: dict) -> list[dict]:
    """Valid single-edit neighbours: one CPU bit flipped, one share step moved, one GPU owner changed."""
    e, combo, owner = _decisions(policy)
    out = []
    for w in range(len(WL)):
        for k in range(N_CPUS):
            if e[w, k] and e[w].sum() == 1:         # would leave the workload without a CPU
                continue
            e2 = e.copy()
            e2[w, k] = not e2[w, k]
            out.append(_build(policy, e2, combo, owner))
    for i, j in itertools.permutations(range(len(WL)), 2):
        c = list(combo)
        c[i] -= 1
        c[j] += 1
        if c[i] >= MIN_SHARE // SHARE_STEP:
            out.append(_build(policy, e, tuple(c), owner))
    for g in range(N_GPUS):
        for o in _GPU_OWNERS:
            if o != owner[g]:
                ow = list(owner)
                ow[g] = o
                out.append(_build(policy, e, combo, ow))
    return [p for p in out if is_valid(p)]


# ------------------------------------------------------------------ differentiable rollout
def _member_tensors(model) -> dict:
    cache = getattr(model, "_pis_torch", None)
    if cache is None:
        b = model.bounds
        cache = {"xm": torch.as_tensor(np.asarray(model.x_norm.mean, np.float32)),
                 "xs": torch.as_tensor(np.asarray(model.x_norm.std, np.float32)),
                 "ym": torch.as_tensor(np.asarray(model.y_norm.mean, np.float32)),
                 "ys": torch.as_tensor(np.asarray(model.y_norm.std, np.float32)),
                 "lo": None if b is None else torch.as_tensor(b[0]),
                 "hi": None if b is None else torch.as_tensor(b[1])}
        model._pis_torch = cache
    return cache


def _prepare(model, hist: torch.Tensor) -> torch.Tensor:
    """Torch twin of `WorldModel.prepare`."""
    hist = hist[:, -model.history:]
    if model.arch == "gru":
        return hist
    return hist[:, -1] if model.history == 1 else hist.reshape(hist.shape[0], -1)


def _past_rows(history: list[tuple[dict, dict]], known: dict, k: int) -> np.ndarray | None:
    real = [encode_input(encode_state(t), p, known) for t, p in history[:-1]][-(k - 1):] if k > 1 else []
    while real and len(real) < k - 1:
        real.insert(0, real[0])
    return np.stack(real).astype(np.float32) if real else None


def torch_rollout(models: list, history: list[tuple[dict, dict]], pol: torch.Tensor, known: dict,
                  horizon: int) -> torch.Tensor:
    """Differentiable ensemble rollout of policy encodings `pol` [n, 33]: [N, n, horizon, 39].

    Same conventions as `Ensemble.rollout_batch` (history newest last, the newest pair's policy slot
    replaced by the candidate, policy held for the horizon, states clipped at 0 and to the member's bounds);
    for a discrete policy encoding it reproduces that function's numbers.
    """
    n = pol.shape[0]
    k = models[0].history
    past_np = _past_rows(history, known, k)
    s0 = torch.as_tensor(encode_state(history[-1][0])).expand(n, N_STATE)
    kn = torch.as_tensor(encode_known(known).astype(np.float32)).expand(n, -1)
    outs = []
    for model in models:
        t = _member_tensors(model)
        past = None if past_np is None else torch.as_tensor(past_np).expand(n, *past_np.shape)
        s, steps = s0, []
        for _ in range(horizon):
            x = torch.cat([s, pol, kn], dim=1)
            if past is None and k > 1:
                past = x[:, None].expand(n, k - 1, x.shape[1])
            hist = x[:, None] if past is None else torch.cat([past, x[:, None]], dim=1)
            delta = model.net((_prepare(model, hist) - t["xm"]) / t["xs"]) * t["ys"] + t["ym"]
            s = torch.clamp(s + delta, min=0.0)
            if t["lo"] is not None:
                s = torch.maximum(torch.minimum(s, t["hi"]), t["lo"])
            if past is not None:
                past = hist[:, 1:]
            steps.append(s)
        outs.append(torch.stack(steps, dim=1))
    return torch.stack(outs)


def _objective(spec, roll: torch.Tensor, params: Params, lam: float, train_std: torch.Tensor) -> torch.Tensor:
    """Torch twin of the score (without the change cost): [n] predicted reward - lam * disagreement."""
    ws = float(spec.policy_period_s)
    b = torch.clamp(roll[..., _B], min=0.0)
    c = torch.clamp(roll[..., _C], min=0.0)
    p99 = torch.clamp(torch.expm1(roll[..., _P99]), min=0.0)
    q = torch.clamp(torch.expm1(roll[..., _Q]), min=0.0)
    slo = float(spec.a.slo_p99_ms)
    value = (spec.b.value_per_batch * b * ws + spec.c.value_per_core_ms * c) / ws
    rew = (value / 1000.0 - params.slo_lam * torch.clamp((p99 - slo) / slo, 0.0, 1.0)
           - params.mu * torch.clamp(q / slo, 0.0, 5.0)).mean(dim=(0, 2))
    sub = roll[..., _DIS_IDX]
    var = sub.var(dim=0, unbiased=False) if roll.shape[0] > 1 else torch.zeros_like(sub[0])
    dis = (torch.sqrt(var + 1e-12) / train_std[_DIS_IDX]).mean(dim=(1, 2))
    return rew - lam * dis


def _relaxed(policy: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e, combo, owner = _decisions(policy)
    ze = torch.tensor(np.where(e, INIT_LOGIT, -INIT_LOGIT), dtype=torch.float32)
    zs = torch.tensor(np.log(np.array(combo, np.float64) / sum(combo)), dtype=torch.float32)
    zg = torch.tensor([[INIT_LOGIT if o == own else 0.0 for o in _GPU_OWNERS] for own in owner],
                      dtype=torch.float32)
    return ze.requires_grad_(), zs.requires_grad_(), zg.requires_grad_()


def _encode_relaxed(ze, zs, zg) -> torch.Tensor:
    return torch.cat([torch.sigmoid(ze).reshape(-1), torch.softmax(zs, 0),
                      torch.softmax(zg, 1).reshape(-1)])[None]


# ------------------------------------------------------------------ revision
@dataclass
class Revision:
    policy: dict
    score: float                    # discrete ensemble score of `policy` (with the revision's lam)
    start_score: float
    disagreement: float
    start_disagreement: float
    mechanism: str
    visited: int                    # distinct policies scored discretely
    rollout: np.ndarray = field(repr=False, default=None)        # [N, H, 39] of `policy`
    start_rollout: np.ndarray = field(repr=False, default=None)  # [N, H, 39] of the start policy

    @property
    def changed(self) -> bool:
        return self.score > self.start_score


def _score(spec, ens, history, policies, known, params, current, lam):
    roll = ens.rollout_batch(history, policies, known, params.horizon_h)
    s, d = score_candidates(spec, roll, policies, current, params, train_std=ens.train_std, lam=lam)
    return roll, s, d


def revise_detail(spec, history: list[tuple[dict, dict]], start_policy: dict, ens, params: Params,
                  steps: int | None = None, mechanism: str = "gradient", lam: float | None = None,
                  current: dict | None = None, lr: float = DEFAULT_LR) -> Revision:
    """Revise `start_policy`; see `revise`. Returns the policy with its scores and rollouts."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    lam = params.lam if lam is None else float(lam)
    if not lam > 0:
        raise ValueError("the uncertainty penalty must be active during revision (lam > 0)")
    if not is_valid(start_policy):
        raise ValueError("the start policy is not valid")
    steps = DEFAULT_STEPS[mechanism] if steps is None else int(steps)
    if current is None:                         # no policy in force given: no change cost
        params, current = replace(params, change_cost=0.0), start_policy
    known = known_of(spec)
    seen = {policy_key(start_policy): start_policy}
    if mechanism == "gradient":
        models = list(ens.models)
        tstd = torch.as_tensor(np.asarray(ens.train_std, np.float32))
        ze, zs, zg = _relaxed(start_policy)
        for _ in range(steps):
            obj = _objective(spec, torch_rollout(models, history, _encode_relaxed(ze, zs, zg), known,
                                                 params.horizon_h), params, lam, tstd)[0]
            g = [torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                 for x in torch.autograd.grad(obj, [ze, zs, zg])]
            scale = max(float(x.abs().max()) for x in g)
            if not scale > 0:                       # flat objective: nothing to follow
                break
            with torch.no_grad():
                for z, gz in zip((ze, zs, zg), g):
                    z += lr * gz / scale
                p = project(start_policy, torch.sigmoid(ze).numpy(), torch.softmax(zs, 0).numpy(),
                            torch.softmax(zg, 1).numpy())
            seen.setdefault(policy_key(p), p)
        pols = list(seen.values())
        roll, s, d = _score(spec, ens, history, pols, known, params, current, lam)
        best = int(np.argmax(s))                    # ties: the start policy (index 0)
        return Revision(pols[best], float(s[best]), float(s[0]), float(d[best]), float(d[0]), mechanism,
                        len(pols), roll[:, best], roll[:, 0])
    # neighbourhood: best-improvement hill climbing
    roll, s, d = _score(spec, ens, history, [start_policy], known, params, current, lam)
    cur, cur_s, cur_d, cur_r = start_policy, float(s[0]), float(d[0]), roll[:, 0]
    start = (cur_s, cur_d, cur_r)
    for _ in range(steps):
        nb = [p for p in neighbours(cur) if policy_key(p) not in seen]
        if not nb:
            break
        for p in nb:
            seen[policy_key(p)] = p
        roll, s, d = _score(spec, ens, history, nb, known, params, current, lam)
        i = int(np.argmax(s))
        if not s[i] > cur_s:
            break
        cur, cur_s, cur_d, cur_r = nb[i], float(s[i]), float(d[i]), roll[:, i]
    return Revision(cur, cur_s, start[0], cur_d, start[1], mechanism, len(seen), cur_r, start[2])


def revise(spec, history: list[tuple[dict, dict]], start_policy: dict, ens, params: Params,
           steps: int | None = None, **kw) -> dict:
    """A valid policy whose predicted score (ensemble reward - lam * disagreement, lam > 0 always) is at
    least the start policy's. `mechanism` "gradient" (default) or "neighbourhood"; `lam` overrides
    `params.lam`; `current` (the policy in force) adds the change cost; `lr` is the gradient step size."""
    return revise_detail(spec, history, start_policy, ens, params, steps, **kw).policy


def warm_up(ens, seed: int) -> None:
    """One throwaway revision of each mechanism, so torch's one-off first-call costs (autograd set-up,
    allocator) are paid at start-up instead of by the first timed decision."""
    from pis.adapters import initial_policy, make_sim, make_spec
    sim, pol = make_sim(seed), initial_policy()
    hist = [(sim.run_window(policy_from_dict(pol)), pol) for _ in range(ens.history)]
    params = Params(n_candidates_m=1, horizon_h=1, lam=1.0, mu=0.0, change_cost=0.0, tau=float("inf"))
    for mech in MECHANISMS:
        revise_detail(make_spec(seed), hist, pol, ens, params, 1, mechanism=mech)


def predicted_violation(spec, rollout: np.ndarray) -> bool:
    """One policy's rollouts [N, H, 39]: does the members' mean A p99 exceed the SLO in any window?"""
    p99 = np.maximum(np.expm1(np.asarray(rollout, np.float64)[..., _P99]), 0.0).mean(axis=0)
    return bool((p99 > spec.a.slo_p99_ms).any())


# ------------------------------------------------------------------ controller
class ReviserLoopController(LoopController):
    """Step A loop; its best candidate is revised when it is untrusted or predicted to violate A's SLO.

    The revised policy replaces Step A's decision only when its revision score beats the start's and, with
    the fallback gate on, its disagreement is within tau; otherwise Step A's decision stands (greedy
    fallback when untrusted). `probe_real=True` (needs attach_sim) also runs both the start and the revised
    policy (and, when the start was untrusted, the greedy fallback) for `horizon_h` windows in copies of the real simulator after the decision is timed, to detect
    model exploitation (predicted gain, real loss).
    """

    def __init__(self, spec, ensemble, params: Params, mechanism: str = "gradient", rev_steps: int | None = None,
                 rev_lam: float = 1.0, rev_lr: float = DEFAULT_LR, probe_real: bool = False, **kw):
        super().__init__(spec, ensemble, params, **kw)
        if mechanism not in MECHANISMS:
            raise ValueError(f"mechanism must be one of {MECHANISMS}")
        if not rev_lam > 0:
            raise ValueError("the uncertainty penalty must be active during revision (rev_lam > 0)")
        self.mechanism, self.rev_lam, self.rev_lr = mechanism, float(rev_lam), float(rev_lr)
        self.rev_steps = DEFAULT_STEPS[mechanism] if rev_steps is None else int(rev_steps)
        self.probe_real = probe_real
        self._probe = None

    def _after_choice(self, d: Decision, cands, greedy, policy, current, history) -> tuple[dict, dict]:
        self._probe = None
        info = {"revised": False, "rev_trigger": None}
        if self.selection != "model" or d.reason not in ("model", "untrusted"):
            return policy, info
        best = int(np.argmax(np.where(np.isfinite(d.scores), d.scores, -np.inf)))
        trigger = "untrusted" if d.reason == "untrusted" else (
            "slo" if predicted_violation(self.spec, self.last_rollouts[:, best]) else None)
        info["rev_trigger"] = trigger
        if trigger is None:
            return policy, info
        t0 = time.perf_counter()
        r = revise_detail(self.spec, history, cands[best], self.ensemble, self.params, self.rev_steps,
                          mechanism=self.mechanism, lam=self.rev_lam, current=current, lr=self.rev_lr)
        trusted = (not self.use_fallback) or r.disagreement <= self.params.tau
        accept = r.changed and trusted and policy_key(r.policy) != policy_key(cands[best])
        info.update(rev_ms=(time.perf_counter() - t0) * 1000.0, rev_visited=r.visited,
                    rev_pred_gain=r.score - r.start_score,
                    rev_pred_reward_gain=(r.score + self.rev_lam * r.disagreement)
                    - (r.start_score + self.rev_lam * r.start_disagreement),
                    rev_dis=r.disagreement, rev_start_dis=r.start_disagreement)
        if not accept:
            return policy, info
        d.fallback, d.reason, d.chosen = False, "revised", best
        info.update(revised=True, consequences=consequences(r.rollout, r.start_rollout))
        if self.probe_real:
            self._probe = (cands[best], r.policy, current, greedy if trigger == "untrusted" else None)
        return r.policy, info

    def _real_score(self, policy: dict, current: dict) -> float:
        from pis.evaluator import change_penalty
        p = self.params
        sim = clone_sim(self.sim)
        pol = policy_from_dict(policy)
        enc = np.stack([encode_state(sim.run_window(pol)) for _ in range(p.horizon_h)])
        return float(predicted_rewards(self.spec, enc, p.slo_lam, p.mu).mean()
                     - change_penalty([policy], current, p.change_cost)[0])

    def decide(self, tel, current, history):
        policy, info = super().decide(tel, current, history)
        if self._probe is not None:
            if self.sim is None:
                raise RuntimeError("probe_real needs attach_sim(sim) (run_scenario does this)")
            start, revised, cur, greedy = self._probe
            real_rev = self._real_score(revised, cur)
            info["rev_real_gain"] = real_rev - self._real_score(start, cur)
            if greedy is not None:                  # untrusted: what Step A would have applied instead
                info["rev_real_gain_vs_greedy"] = real_rev - self._real_score(greedy, cur)
            self._probe = None
        return policy, info
