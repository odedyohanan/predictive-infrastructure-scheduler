"""Score candidate policies from ensemble rollouts and decide: accept the best or fall back.

score(candidate) = mean over members and horizon of the predicted reward
                   (`pis.reward.reward_from_state` with weight `slo_lam`, minus `pis.reward.queue_penalty`
                   with weight `mu`: A's p99 reads 0 when nothing completed, so a growing backlog must cost)
                 - lam * ensemble disagreement (normalised, see `pis.ensemble.disagreement`)
                 - change_cost * [candidate differs from the policy in force]

A candidate with any non-finite forecast gets score -inf and disagreement +inf (never trusted).
`choose` takes the best finite score and falls back (to the base scheduler's own greedy decision, applied by
the caller) when nothing is finite or, with the fallback gate on, when the best candidate's disagreement
exceeds `tau`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from pis.candidates import policy_key
from pis.config import setup_imports
from pis.ensemble import DISAGREEMENT_KEYS, disagreement

setup_imports()

from worldmodel.features import STATE_NAMES  # noqa: E402

_B, _C = STATE_NAMES.index("b.rate"), STATE_NAMES.index("c.work_core_ms")
_P99, _Q = STATE_NAMES.index("a.p99_ms"), STATE_NAMES.index("a.queue_work_ms")


@dataclass(frozen=True)
class Params:
    n_candidates_m: int      # scheduler samples per decision (plus greedy and keep-current)
    horizon_h: int           # windows each candidate is rolled out, policy held fixed
    lam: float               # weight of the disagreement penalty
    mu: float                # weight of the A queue penalty
    change_cost: float       # cost of changing the policy in force (reward units)
    tau: float               # trust threshold on the chosen candidate's disagreement
    slo_lam: float = 10.0    # SLO weight of the reward (the one the RL scheduler was trained with)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tau"] = "inf" if math.isinf(self.tau) and self.tau > 0 else self.tau
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        d = dict(d)
        d["tau"] = float(d["tau"])
        return cls(**d)


@dataclass
class Decision:
    policy: dict | None
    chosen: int                          # index into the candidates; -1 = fallback
    fallback: bool
    scores: np.ndarray = field(repr=False)
    disagreement: np.ndarray = field(repr=False)
    reason: str = ""


def predicted_rewards(spec, enc: np.ndarray, slo_lam: float, mu: float) -> np.ndarray:
    """reward_from_state - queue_penalty for every encoded state, vectorised; shape enc.shape[:-1]."""
    e = np.asarray(enc, dtype=np.float64)
    ws = float(spec.policy_period_s)
    b = np.maximum(e[..., _B], 0.0)
    c = np.maximum(e[..., _C], 0.0)
    p99 = np.maximum(np.expm1(e[..., _P99]), 0.0)
    q = np.maximum(np.expm1(e[..., _Q]), 0.0)
    value_rate = (spec.b.value_per_batch * b * ws + spec.c.value_per_core_ms * c) / ws
    slo = spec.a.slo_p99_ms
    over = np.clip((p99 - slo) / slo, 0.0, 1.0)
    return value_rate / 1000.0 - slo_lam * over - mu * np.clip(q / slo, 0.0, 5.0)


def change_penalty(policies: list[dict], current: dict, change_cost: float) -> np.ndarray:
    cur = policy_key(current)
    return np.array([change_cost * (policy_key(p) != cur) for p in policies], dtype=np.float64)


def score_candidates(spec, ens_rollouts: np.ndarray, policies: list[dict], current: dict, params: Params,
                     train_std: np.ndarray | None = None, keys: list[str] | None = None,
                     lam: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """ens_rollouts [N, n_cand, H, 39] -> (scores [n_cand], disagreements [n_cand]).

    `train_std` normalises the disagreement (None = unit scale, for tests and stubs); `lam` overrides
    `params.lam` (0 for the no-uncertainty ablation and for oracle branches).
    """
    r = np.asarray(ens_rollouts)
    if train_std is None:
        train_std = np.ones(r.shape[-1], np.float32)
    dis = disagreement(r, keys or DISAGREEMENT_KEYS, train_std)
    finite = np.isfinite(r).all(axis=(0, 2, 3))
    with np.errstate(invalid="ignore", over="ignore"):
        rew = predicted_rewards(spec, np.where(np.isfinite(r), r, 0.0), params.slo_lam, params.mu)
    w = params.lam if lam is None else lam
    scores = rew.mean(axis=(0, 2)) - change_penalty(policies, current, params.change_cost)
    if w:
        scores = scores - w * np.where(finite, dis, 0.0)
    scores = np.where(finite, scores, -np.inf)
    return scores, dis


def choose(scores: np.ndarray, disagreement: np.ndarray, params: Params,
           use_fallback: bool = True) -> Decision:
    """Best finite score; fallback when none is finite or (gate on) its disagreement exceeds tau."""
    scores, disagreement = np.asarray(scores, float), np.asarray(disagreement, float)
    finite = np.isfinite(scores)
    if not finite.any():
        return Decision(None, -1, True, scores, disagreement, "no_finite")
    best = int(np.argmax(np.where(finite, scores, -np.inf)))       # ties: lowest index (greedy first)
    if use_fallback and not disagreement[best] <= params.tau:      # NaN disagreement is untrusted too
        return Decision(None, -1, True, scores, disagreement, "untrusted")
    return Decision(None, best, False, scores, disagreement, "model")
