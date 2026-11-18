"""An ensemble of world models: batched rollouts of every member and their disagreement.

Every member rolls out every candidate with the conventions of `pis.forecaster.rollout_batch` (history
newest last, the newest pair's policy slot replaced by each candidate, policy held for the horizon, bounded
states); member i's output equals `rollout_batch(member_i, ...)` exactly. Policy and known-parameter
encodings are computed once per call instead of once per step and candidate, which keeps the per-decision
cost low.

Disagreement = standard deviation across members of each key quantity of the encoded state, divided by
that quantity's training standard deviation, averaged over keys and horizon. The training standard
deviation is the per-window input normaliser of member 0 (`x_norm.std`, fitted on its training inputs),
restricted to the state columns, so it is in encoded units (log1p for latency and queue fields), the same
units as the rollouts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from pis.config import resolve, setup_imports
from pis.privacy import safe_path

setup_imports()

from worldmodel.features import (STATE_NAMES, clip_state, encode_input, encode_known,  # noqa: E402
                                 encode_policy, encode_state)
from worldmodel.model import WorldModel  # noqa: E402

# Quantities the reward depends on (A latency and backlog, B rate, C work).
DISAGREEMENT_KEYS = ["a.p99_ms", "a.queue_work_ms", "b.rate", "c.work_core_ms"]
N_STATE = len(STATE_NAMES)


def disagreement(rollouts: np.ndarray, keys: list[str], train_std: np.ndarray) -> np.ndarray:
    """rollouts [N, n_cand, H, 39] -> [n_cand]; +inf for a candidate with any non-finite forecast."""
    idx = [STATE_NAMES.index(k) for k in keys]
    full = np.asarray(rollouts, dtype=np.float64)
    bad = ~np.isfinite(full).all(axis=(0, 2, 3))      # any broken number, key or not
    sub = full[..., idx]
    with np.errstate(invalid="ignore"):
        d = (sub.std(axis=0) / np.asarray(train_std, dtype=np.float64)[idx]).mean(axis=(1, 2))
    d[bad] = np.inf
    return d


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Ensemble:
    def __init__(self, models: list[WorldModel], train_std: np.ndarray | None = None,
                 refs: list[str] | None = None, shas: list[str] | None = None):
        if not models:
            raise ValueError("an ensemble needs at least one member")
        hs = {m.history for m in models}
        if len(hs) != 1:
            raise ValueError(f"members disagree on history length: {sorted(hs)}")
        self.models = models
        if train_std is None:
            train_std = models[0].x_norm.std[:N_STATE]
        self.train_std = np.asarray(train_std, dtype=np.float32)
        self.refs = refs or [f"member_{i}" for i in range(len(models))]
        self.shas = shas or [""] * len(models)

    @classmethod
    def from_paths(cls, paths: list[str]) -> "Ensemble":
        """Paths may carry a sibling prefix ('worldmodel:models/x.pt') or be relative to this repo."""
        files = [resolve(p) for p in paths]
        models = [WorldModel.load(str(f)) for f in files]
        return cls(models, refs=list(paths), shas=[_sha256(f) for f in files])

    def __len__(self) -> int:
        return len(self.models)

    @property
    def history(self) -> int:
        return self.models[0].history

    def describe(self) -> list[dict]:
        """Provenance of each member: path as given (absolute paths reduced) and checkpoint sha256."""
        return [{"path": safe_path(r), "sha256": s} for r, s in zip(self.refs, self.shas)]

    def rollout_batch(self, history: list[tuple[dict, dict]], policies: list[dict], known: dict,
                      horizon: int) -> np.ndarray:
        """Encoded predicted states, shape [N, n_candidates, horizon, 39]."""
        k = self.history
        n = len(policies)
        pol = np.stack([encode_policy(p) for p in policies]).astype(np.float32)
        kn = np.tile(encode_known(known).astype(np.float32), (n, 1))
        real = [encode_input(encode_state(t), p, known) for t, p in history[:-1]][-(k - 1):] if k > 1 else []
        while real and len(real) < k - 1:
            real.insert(0, real[0])
        s0 = encode_state(history[-1][0])
        out = np.empty((len(self.models), n, horizon, N_STATE), dtype=np.float32)
        for mi, model in enumerate(self.models):
            bounds = model.bounds
            past = np.broadcast_to(np.stack(real), (n, k - 1, len(real[0]))).copy() if real else None
            s = np.tile(s0, (n, 1))
            for t in range(horizon):
                x = np.concatenate([s, pol, kn], axis=1).astype(np.float32)
                if past is None and k > 1:   # no past at all: each candidate pads with its own first input
                    past = np.repeat(x[:, None], k - 1, axis=1)
                hist = x[:, None] if past is None else np.concatenate([past, x[:, None]], axis=1)
                s = clip_state(model.predict_next(model.prepare(hist), s))
                if bounds is not None:
                    s = np.clip(s, bounds[0], bounds[1])
                if past is not None:
                    past = hist[:, 1:]
                out[mi, :, t] = s
        return out

    def disagreement(self, rollouts: np.ndarray, keys: list[str] | None = None) -> np.ndarray:
        return disagreement(rollouts, keys or DISAGREEMENT_KEYS, self.train_std)
