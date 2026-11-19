"""The incident advisor: R6 runs the system until the incident predictor raises an alarm; then candidate
policies are compared by sampled world-model futures under a chance constraint on a new incident.

Every window the advisor appends the telemetry it sees (never the degradation block) and the policy in
force to its own history, builds predictor rows from it (`features.rows_from_records`) and asks the
predictor for the risk of a new incident within H (from the warm-up on; before it there is no risk). The
intervention mode turns on when the risk reaches the alarm threshold and stays on until the risk falls
below `low` (the alarm's hysteresis). Outside the mode the advisor returns R6's greedy decision unchanged.

In the mode the candidates are R6's greedy decision, `m` R6 samples and the "after" policies of the named
interventions that differ from the policy in force, de-duplicated (first occurrence kept, greedy first).
Each candidate is held for h windows in `n` sampled futures (all candidates in one batch). Its value is the
Phase A reward of the predicted states (the 39 world-model numbers; `slo_lam`, `mu`; no disagreement term;
`change_cost` when it differs from the policy in force), averaged over windows and futures; its incident
probability is the share of futures with a new counted onset in (t, t + h], judged on the observed
violations joined with the future's (futures run k - 1 windows past h, as the Monte Carlo labels do).
Choice: among candidates with probability <= eps the highest mean value; values within 1e-9 are tied and
the higher CVaR (10 %) of the per-future total reward wins; then the lower index. No feasible candidate, or
a current state outside the model's training range: R6's greedy decision, flagged as a fallback.

Per window `info` holds decide_ms (risk_ms + select_ms and bookkeeping), risk (None before the warm-up),
mode, chosen (candidate index; -1 on a fallback; None outside the mode), fallback, reason, chosen_kind,
pred_incident_p (the chosen candidate's predicted probability, R6 greedy's on a fallback; None outside the
mode, for random selection and when the model was not consulted) and n_candidates.

Selection arms on the same candidates: "model" (the above), "random" (uniform choice in the mode),
"oracle" (the real degraded simulator, branched `oracle_branches` times per candidate, true values and
true onset shares, same rule; needs `attach_dsim`), "always" (the mode on in every window from the
warm-up, without the alarm).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pis.adapters import BaseScheduler
from pis.candidates import is_valid, policy_key
from pis.config import setup_imports
from pis.evaluator import change_penalty, predicted_rewards
from pis.phase_b import features, incidents
from pis.probe import named_interventions

setup_imports()

from sched.randomize import known_params  # noqa: E402
from worldmodel.features import encode_policy, encode_state  # noqa: E402

BASE_CKPT = "models/rl_r6/final.pt"
SELECTIONS = ("model", "random", "oracle", "always")
INFO_KEYS = ("decide_ms", "risk", "mode", "chosen", "fallback", "pred_incident_p", "n_candidates")
ROLL_TAG = 18_000_000       # model futures at window t of episode s: ROLL_TAG + s*1000 + t
RANDOM_TAG = 15_000         # random selection of episode s: RANDOM_TAG + s
ORACLE_TAG = 16_000_000     # oracle branch j at any window of episode s: ORACLE_TAG + s*1000 + j
ORACLE_BRANCHES = 8
CVAR_LEVEL = 0.10
TIE = 1e-9
BOUND_TOL = 1e-6
N_WM = 39                   # the world-model state numbers lead the 43-number state


@dataclass(frozen=True)
class AdvisorParams:
    m: int = 4               # R6 samples per decision in the mode
    n: int = 16              # sampled futures per candidate
    h: int = 20              # windows each candidate is held and judged
    eps: float = 0.1         # bound on the predicted probability of a new incident
    slo_lam: float = 40.0
    mu: float = 8.0
    change_cost: float = 0.0

    @classmethod
    def from_step_a(cls, step_a_params: dict, H: int, m: int = 4, n: int = 16, eps: float = 0.1,
                    h_max: int = 20) -> "AdvisorParams":
        """Reward weights of the Step A loop; horizon min(H, h_max)."""
        return cls(m=int(m), n=int(n), h=int(min(H, h_max)), eps=float(eps),
                   slo_lam=float(step_a_params["slo_lam"]), mu=float(step_a_params["mu"]),
                   change_cost=float(step_a_params.get("change_cost", 0.0)))

    def to_dict(self) -> dict:
        return dict(m=self.m, n=self.n, h=self.h, eps=self.eps, slo_lam=self.slo_lam, mu=self.mu,
                    change_cost=self.change_cost)


# ------------------------------------------------------------------ scoring
def cvar(x: np.ndarray, level: float = CVAR_LEVEL) -> np.ndarray:
    """Mean of the lowest ceil(level x n) values (at least one) of each row of x [K, n]."""
    x = np.sort(np.asarray(x, dtype=np.float64), axis=-1)
    k = max(1, int(math.ceil(level * x.shape[-1] - 1e-12)))
    return x[..., :k].mean(axis=-1)


def score_rollouts(spec, roll: np.ndarray, h: int, slo_lam: float, mu: float,
                   change: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Futures [K, n, >= h, 43] -> mean value per candidate [K] (mean over futures and the first h windows,
    minus the change cost) and the total reward per future [K, n] (sum over the h windows, minus it)."""
    r = predicted_rewards(spec, np.asarray(roll)[:, :, :h, :N_WM], slo_lam, mu)
    change = np.asarray(change, dtype=np.float64)
    return r.mean(axis=(1, 2)) - change, r.sum(axis=2) - change[:, None]


def choose_candidate(mean_value: np.ndarray, cvar_value: np.ndarray, p_incident: np.ndarray, eps: float,
                     tie: float = TIE) -> int:
    """Index of the chosen candidate, -1 when none has p_incident <= eps."""
    mv, cv, p = (np.asarray(a, dtype=np.float64) for a in (mean_value, cvar_value, p_incident))
    ok = (p <= eps) & np.isfinite(mv)
    if not ok.any():
        return -1
    best = mv[ok].max()
    tied = np.flatnonzero(ok & (mv >= best - tie))
    return int(tied[np.argmax(cv[tied])])          # argmax keeps the lowest index among equal CVaRs


def onset_share(observed: np.ndarray, future: np.ndarray, rule: incidents.IncidentRule, horizon: int,
                warmup: int = incidents.WARMUP) -> float:
    """Share of the futures (rows of `future` [R, m], violation flags) with a new counted onset in
    (t, t + horizon], t = the last observed window, judged on the observed violations joined with each.

    Only the last interval + k + 1 observed windows can change the verdict (runs and the trailing majority
    look back that far at most), so the series is cut there; the warm-up is shifted with the cut.
    """
    v0 = np.asarray(observed, dtype=bool)
    fut = np.asarray(future, dtype=bool)
    if not len(fut):
        return float("nan")
    cut = max(len(v0) - (rule.interval + rule.k + 1), 0)
    v0 = v0[cut:]
    t, wu = len(v0) - 1, max(warmup - cut, 0)
    idle = np.zeros(len(v0) + fut.shape[1], dtype=bool)
    hits = 0
    for row in fut:
        lab = incidents.label_series(np.concatenate([v0, row]), idle, rule, wu)
        hits += incidents.onset_within(lab, t, horizon)
    return hits / len(fut)


# ------------------------------------------------------------------ history kept by the advisor
class _Rows:
    """The advisor's own window history: predictor rows (as `features.rows_from_records` builds them) and
    the encoded states and policies, grown one window at a time."""

    def __init__(self, known: dict, episode: int, capacity: int):
        self.known, self.episode, self.cap = known, int(episode), max(int(capacity), 1)
        self.cols: dict[str, np.ndarray] = {}
        self.states: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.viol: list[bool] = []
        self.n = 0

    def append(self, tel: dict, applied: dict) -> None:
        row = features.rows_from_records([{"tel": tel, "applied": applied}], self.known, episode=self.episode)
        if self.n == self.cap or not self.cols:
            self.cap = max(self.cap, 2 * self.n, 1)
            old = self.cols
            self.cols = {c: np.empty(self.cap, dtype=row[c].dtype) for c in row.columns}
            for c, v in old.items():
                self.cols[c][:self.n] = v[:self.n]
        for c in row.columns:
            self.cols[c][self.n] = row[c].iloc[0]
        self.cols["window"][self.n] = self.n
        s = features.encode_state_b(tel)
        self.states.append(s)
        self.policies.append(encode_policy(applied).astype(np.float32))
        self.viol.append(bool(row["slo_v2_violation"].iloc[0]))
        self.n += 1

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({c: v[:self.n] for c, v in self.cols.items()})

    def window(self, context: int) -> features.Window:
        return features.Window(np.stack(self.states[-context:]), np.stack(self.policies[-context:]),
                               features.known_vector(self.known))


# ------------------------------------------------------------------ the controller
class AdvisorController:
    def __init__(self, spec, predictor, alarms_cfg: dict, model, params: AdvisorParams, selection: str = "model",
                 *, rule: incidents.IncidentRule, warmup: int = incidents.WARMUP, seed: int = 0,
                 base_ckpt: str = BASE_CKPT, oracle_branches: int = ORACLE_BRANCHES, duration_s: float | None = None):
        if selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        self.spec, self.predictor, self.model, self.params = spec, predictor, model, params
        self.selection, self.rule, self.warmup, self.seed = selection, rule, int(warmup), int(seed)
        self.H = int(alarms_cfg["H"])
        thr, low = alarms_cfg.get("threshold"), alarms_cfg.get("low")
        self.threshold = None if thr is None else float(thr)
        self.low = None if low is None else float(low)
        if self.threshold is not None and (self.low is None or self.low > self.threshold):
            raise ValueError("the alarm needs low <= threshold")
        self.oracle_branches = int(oracle_branches)
        self.base = BaseScheduler(base_ckpt, spec)
        self.known = known_params(spec)
        self.rng = np.random.default_rng(seed)                     # R6 samples
        self.select_rng = np.random.default_rng(RANDOM_TAG + self.seed)
        cap = int(round((duration_s or 300.0) / spec.policy_period_s))
        self.rows = _Rows(self.known, self.seed, cap)
        self.mode = False
        self.sim = self.dsim = None
        self.last: dict = {}

    def attach_sim(self, sim) -> None:
        self.sim = sim

    def attach_dsim(self, dsim) -> None:
        """The degraded simulator of the running episode (read through reseeded clones only)."""
        self.dsim = dsim

    # ------------------------------------------------------------------ risk and mode
    def _risk(self) -> float | None:
        t = self.rows.n - 1
        if t < self.warmup:
            return None
        r = float(self.predictor.risk(self.rows.frame(), self.H, episode=self.seed))
        return r if math.isfinite(r) else None

    def _update_mode(self, risk: float | None) -> bool:
        if self.selection == "always":
            return self.rows.n - 1 >= self.warmup
        if risk is None or self.threshold is None:
            return False
        if not self.mode:
            return risk >= self.threshold
        return risk >= self.low

    # ------------------------------------------------------------------ candidates
    def _candidates(self, greedy: dict, samples: list[dict], current: dict) -> tuple[list[dict], list[str]]:
        cur = policy_key(current)
        pool = [(greedy, "greedy")] + [(s, "sample") for s in samples]
        pool += [(after, name) for name, (_, after) in named_interventions(current).items()
                 if policy_key(after) != cur]
        seen, cands, kinds = set(), [], []
        for p, kind in pool:
            k = policy_key(p)
            if k in seen or (kind != "greedy" and not is_valid(p)):
                continue
            seen.add(k)
            cands.append(p)
            kinds.append(kind)
        return cands, kinds

    # ------------------------------------------------------------------ selections
    def _out_of_bounds(self) -> bool:
        b = getattr(self.model, "bounds", None)
        if b is None:
            return False
        lo, hi = b
        s = self.rows.states[-1]
        return bool((s < np.asarray(lo) - BOUND_TOL).any() or (s > np.asarray(hi) + BOUND_TOL).any())

    def _judge(self, roll_values, totals, p) -> tuple[int, str]:
        i = choose_candidate(roll_values, cvar(totals), p, self.params.eps)
        return (i, "model") if i >= 0 else (-1, "infeasible")

    def _model(self, cands: list[dict], current: dict) -> tuple[int, str, np.ndarray | None]:
        if self._out_of_bounds():
            return -1, "out_of_bounds", None
        p = self.params
        t = self.rows.n - 1
        m = p.h + self.rule.k - 1
        roll = self.model.sample_rollouts_batch(self.rows.window(self.model.context),
                                                [encode_policy(c) for c in cands], self.known, m, p.n,
                                                seed=ROLL_TAG + self.seed * 1000 + t)
        mean, total = score_rollouts(self.spec, roll, p.h, p.slo_lam, p.mu,
                                     change_penalty(cands, current, p.change_cost))
        pv = features.predicted_violation(features.decode(roll))
        v0 = np.array(self.rows.viol, dtype=bool)
        prob = np.array([onset_share(v0, pv[i], self.rule, p.h, self.warmup) for i in range(len(cands))])
        self.last.update(mean_value=mean, cvar=cvar(total), p_incident=prob)
        i, reason = self._judge(mean, total, prob)
        return i, reason, prob

    def _oracle(self, cands: list[dict], current: dict) -> tuple[int, str, np.ndarray | None]:
        if self.dsim is None:
            raise RuntimeError("oracle selection needs attach_dsim(dsim) (run_episode does this)")
        p = self.params
        sc = self.dsim.scenario
        left = int(round(sc.duration_s / sc.spec.policy_period_s)) - self.rows.n
        if left < 1:
            return -1, "episode_end", None
        h = min(p.h, left)
        m = min(h + self.rule.k - 1, left)
        v0 = np.array(self.rows.viol, dtype=bool)
        states = np.empty((len(cands), self.oracle_branches, h, N_WM), dtype=np.float32)
        viol = np.empty((len(cands), self.oracle_branches, m), dtype=bool)
        slo = self.spec.a.slo_p99_ms
        for i, c in enumerate(cands):
            for j in range(self.oracle_branches):
                b = self.dsim.clone(reseed=ORACLE_TAG + self.seed * 1000 + j)
                recs = [{"tel": b.run_window(c)} for _ in range(m)]
                viol[i, j] = incidents.violations(recs, slo)
                states[i, j] = np.stack([encode_state(r["tel"]) for r in recs[:h]])
        mean, total = score_rollouts(self.spec, states, h, p.slo_lam, p.mu,
                                     change_penalty(cands, current, p.change_cost))
        prob = np.array([onset_share(v0, viol[i], self.rule, h, self.warmup) for i in range(len(cands))])
        self.last.update(mean_value=mean, cvar=cvar(total), p_incident=prob)
        i, reason = self._judge(mean, total, prob)
        return i, ("oracle" if i >= 0 else reason), prob

    # ------------------------------------------------------------------ controller protocol
    def decide(self, tel: dict, current: dict, history) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        self.rows.append(tel, current)
        risk = self._risk()
        t_risk = time.perf_counter()
        self.mode = self._update_mode(risk)
        info = {"risk": risk, "mode": self.mode, "chosen": None, "fallback": False, "pred_incident_p": None,
                "n_candidates": 0, "reason": "no_alarm", "chosen_kind": "greedy", "selection": self.selection}
        if not self.mode:
            greedy, _ = self.base.propose(tel, current, 0, self.rng)
            policy = greedy
        else:
            greedy, samples = self.base.propose(tel, current, self.params.m, self.rng)
            cands, kinds = self._candidates(greedy, samples, current)
            self.last = {"candidates": cands, "kinds": kinds}
            prob = None
            if self.selection == "random":
                i, reason = int(self.select_rng.integers(len(cands))), "random"
            elif self.selection == "oracle":
                i, reason, prob = self._oracle(cands, current)
            else:
                i, reason, prob = self._model(cands, current)
            fallback = i < 0
            policy = greedy if fallback else cands[i]
            shown = 0 if fallback else i
            info.update(chosen=i, fallback=fallback, n_candidates=len(cands), reason=reason,
                        chosen_kind=kinds[shown] if not fallback else "greedy",
                        pred_incident_p=None if prob is None else float(prob[shown]))
        t1 = time.perf_counter()
        info.update(decide_ms=(t1 - t0) * 1000.0, risk_ms=(t_risk - t0) * 1000.0,
                    select_ms=(t1 - t_risk) * 1000.0)
        return policy, info
