"""Batched open-loop GRU rollouts: many candidate policies, one forward pass per step.

History format: `list[(telemetry, policy)]`, newest last. Every older pair is a real recorded
window and the policy that was applied to it. The newest pair carries the current telemetry; its
policy slot is replaced by each candidate (held fixed for the whole horizon), so only the telemetry
of the newest pair is used. Conventions follow `worldmodel.rollout` / `worldmodel.history`.
"""
from __future__ import annotations

import numpy as np

from pis.config import setup_imports, sibling_paths

setup_imports()

from worldmodel.features import clip_state, encode_input, encode_state  # noqa: E402
from worldmodel.model import WorldModel  # noqa: E402

DEFAULT_MODEL = "models/gru_mix_bounded.pt"


def load_world_model(rel_path: str = DEFAULT_MODEL) -> WorldModel:
    return WorldModel.load(str(sibling_paths()["worldmodel"] / rel_path))


def have_full_history(model: WorldModel, history: list[tuple[dict, dict]]) -> bool:
    return len(history) >= model.history


def rollout_batch(model: WorldModel, history: list[tuple[dict, dict]], policies: list[dict],
                  known: dict, horizon: int) -> np.ndarray:
    """Encoded predicted states, shape [n_candidates, horizon, 39]."""
    k, bounds = model.history, model.bounds
    # Real rows before the newest window; a short history is padded with its first row.
    real = [encode_input(encode_state(t), p, known) for t, p in history[:-1]][-(k - 1):] if k > 1 else []
    s0 = encode_state(history[-1][0])
    if k > 1 and not real:
        real = [encode_input(s0, policies[0], known)]      # no past at all: copy of the first input
    while len(real) < k - 1:
        real.insert(0, real[0])
    n = len(policies)
    past = np.broadcast_to(np.stack(real), (n, k - 1, len(real[0]))).copy() if k > 1 else None
    s = np.tile(s0, (n, 1))
    out = np.empty((n, horizon, s0.shape[0]), dtype=np.float32)
    for t in range(horizon):
        x = np.stack([encode_input(s[i], policies[i], known) for i in range(n)])      # [n, n_in]
        hist = x[:, None] if past is None else np.concatenate([past, x[:, None]], axis=1)
        s = clip_state(model.predict_next(model.prepare(hist), s))
        if bounds is not None:
            s = np.clip(s, bounds[0], bounds[1])
        if past is not None:
            past = hist[:, 1:]
        out[:, t] = s
    return out
