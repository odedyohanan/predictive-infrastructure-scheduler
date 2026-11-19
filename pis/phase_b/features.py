"""The 43-number state of the probabilistic world model and its inputs.

State = the 39 world-model state numbers (`worldmodel.features.STATE_NAMES`, encoded as there) plus four
raw counts of A's window that the SLO v2 verdict is made of: queued requests overdue at the window end,
requests that became late, requests completed on time, arrivals (log1p on all four). Input = state +
33 policy numbers + 3 known parameters = 79.

A `Window` is an episode's history in this encoding: `states[j]` is window j's state and `policies[j]` the
(masked) policy that ran during window j and produced it, exactly as a store row holds them. The model's
input at window j pairs state j with the policy of window j+1 (the decision taken after seeing j), so
`steps` builds input j from row j's state and row j+1's policy, target = state j+1 - state j. Episodes are
never joined.

`encode_rows` reads store rows (`pis.phase_b.history`), `encode_records` live episode records
(`pis.phase_b.episode.run_episode`); both give the same numbers. `rows_from_records` turns live records
into rows shaped like the store's, for the predictors.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from pis.config import setup_imports
from pis.phase_b import history, incidents

setup_imports()

from worldmodel.features import (KNOWN_NAMES, LOG_FIELDS, POLICY_NAMES, STATE_NAMES,  # noqa: E402
                                 encode_known, encode_policy, encode_state)

RAW_FIELDS = [f"a.{k}" for k in history.RAW_COUNTS]       # queue_overdue, late_new, completed_ontime, arrivals
STATE_B: list[str] = STATE_NAMES + RAW_FIELDS
INPUT_B: list[str] = STATE_B + POLICY_NAMES + KNOWN_NAMES
N_STATE, N_POLICY, N_KNOWN = len(STATE_B), len(POLICY_NAMES), len(KNOWN_NAMES)
N_INPUT = len(INPUT_B)
LOG_B = set(LOG_FIELDS) | set(RAW_FIELDS)
_LOG_MASK = np.array([n in LOG_B for n in STATE_B])
IDX = {n: i for i, n in enumerate(STATE_B)}
_P50, _P99 = IDX["a.p50_ms"], IDX["a.p99_ms"]
_OVERDUE, _LATE, _ONTIME = IDX["a.queue_overdue"], IDX["a.late_new"], IDX["a.completed_ontime"]

STATE_COLS = [history.STATE_COLUMNS[n] for n in STATE_NAMES] + [history.RAW_COLUMNS[n] for n in RAW_FIELDS]
POLICY_COLS = [history.POLICY_COLUMNS[n] for n in POLICY_NAMES]
KNOWN_COLS = [history.KNOWN_COLUMNS[n] for n in KNOWN_NAMES]


class Window(NamedTuple):
    states: np.ndarray            # [T, 43] encoded
    policies: np.ndarray          # [T, 33] the policy that ran in each window
    known: np.ndarray | None      # [3] start-of-episode known parameters


def known_vector(known) -> np.ndarray:
    """Known parameters as the 3-vector: from a dict (`known_params`) or an array."""
    if isinstance(known, dict):
        return encode_known(known)
    k = np.asarray(known, dtype=np.float32).reshape(-1)
    if k.shape != (N_KNOWN,):
        raise ValueError(f"known parameters need {N_KNOWN} numbers, got {k.shape}")
    return k


def _log_raw(counts: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(counts, dtype=np.float64), 0.0)).astype(np.float32)


def encode_rows(df: pd.DataFrame, known=None) -> Window:
    """Store rows (one episode, by window) -> Window. Known values from `known` or the rows' known_* columns."""
    s39 = df[STATE_COLS[:39]].to_numpy(dtype=np.float32)
    raw = _log_raw(df[STATE_COLS[39:]].to_numpy())
    pol = df[POLICY_COLS].to_numpy(dtype=np.float32)
    if known is None and all(c in df.columns for c in KNOWN_COLS) and len(df):
        known = df[KNOWN_COLS].iloc[-1].to_numpy(dtype=np.float32)
    return Window(np.concatenate([s39, raw], axis=1), pol, None if known is None else known_vector(known))


def encode_state_b(tel: dict) -> np.ndarray:
    raw = _log_raw(np.array([tel["a"][k] for k in history.RAW_COUNTS]))
    return np.concatenate([encode_state(tel), raw]).astype(np.float32)


def encode_records(records: list[dict], known) -> Window:
    """Live episode records (with "tel" and "applied") -> Window."""
    if not records:
        raise ValueError("no records")
    s = np.stack([encode_state_b(r["tel"]) for r in records])
    p = np.stack([encode_policy(r["applied"]) for r in records]).astype(np.float32)
    return Window(s, p, known_vector(known))


def rows_from_records(records: list[dict], known, slo_ms: float = incidents.SLO_MS,
                      episode: int | None = None) -> pd.DataFrame:
    """Live records as rows with the store's state, raw-count, policy, violation and known columns (and an
    `episode` column when `episode` is given).

    `deg_active` comes from the records' degradation block when present (False otherwise); predictors
    never read it, labels do.
    """
    w = encode_records(records, known)
    cols: dict[str, object] = {"window": np.arange(len(records), dtype=np.int64)}
    if episode is not None:
        cols["episode"] = np.full(len(records), int(episode), dtype=np.int64)
    for i, c in enumerate(STATE_COLS[:39]):
        cols[c] = w.states[:, i]
    for c, k in zip(STATE_COLS[39:], history.RAW_COUNTS):
        cols[c] = np.array([r["tel"]["a"][k] for r in records], dtype=np.int64)
    for i, c in enumerate(POLICY_COLS):
        cols[c] = w.policies[:, i]
    cols["slo_v2_violation"] = incidents.violations(records, slo_ms)
    cols["deg_active"] = np.array([bool(r["tel"].get("phase_b", {}).get("active", False)) for r in records])
    # the exact known values, as the store writes them (the model input rounds them to float32 separately)
    exact = [float(known[n]) for n in KNOWN_NAMES] if isinstance(known, dict) else [float(v) for v in w.known]
    for c, v in zip(KNOWN_COLS, exact):
        cols[c] = np.full(len(records), v)
    return pd.DataFrame(cols)


def join_known(rows: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Store window rows with their episode's known_* columns added (episodes = `Store.episodes()`)."""
    k = episodes[["seed"] + KNOWN_COLS].rename(columns={"seed": "episode"})
    return rows.drop(columns=[c for c in KNOWN_COLS if c in rows.columns]).merge(k, on="episode", how="left")


def decode(states: np.ndarray) -> np.ndarray:
    """Encoded states [..., 43] -> real units (counts, ms, cores, ...), clipped at 0."""
    v = np.asarray(states, dtype=np.float64).copy()
    v[..., _LOG_MASK] = np.expm1(v[..., _LOG_MASK])
    return np.maximum(v, 0.0)


def predicted_violation(decoded):
    """SLO v2 verdict (`sim.slo.window_violation` v2) of decoded states: [..., 43] -> bool [...].

    resolved = completed on time + late; violated if late > 1 % of resolved, or (nothing resolved) if any
    request is overdue in the queue. Counts are rounded to whole requests first.
    """
    d = np.asarray(decoded, dtype=np.float64)
    late, ontime, overdue = (np.rint(d[..., i]) for i in (_LATE, _ONTIME, _OVERDUE))
    resolved = ontime + late
    out = np.where(resolved > 0, late > 0.01 * resolved, overdue > 0)
    return bool(out) if out.ndim == 0 else out.astype(bool)


def target_mask(states: np.ndarray) -> np.ndarray:
    """1 = use this target number. p50/p99 are reported as 0 (and meaningless) when no A request completed."""
    s = np.asarray(states)
    m = np.ones(s.shape, dtype=np.float32)
    none = s[..., _P99] <= 0.0
    m[..., _P50] = np.where(none, 0.0, 1.0)
    m[..., _P99] = np.where(none, 0.0, 1.0)
    return m


def steps(w: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training pairs of one episode: X [T-1, 79] = (state j, policy of window j+1, known), Y = state j+1 -
    state j, M = target mask of state j+1."""
    if w.known is None:
        raise ValueError("the window has no known parameters")
    T = len(w.states)
    x = np.concatenate([w.states[:-1], w.policies[1:], np.broadcast_to(w.known, (T - 1, N_KNOWN))], axis=1)
    y = w.states[1:] - w.states[:-1]
    return x.astype(np.float32), y.astype(np.float32), target_mask(w.states[1:])


def episode_data(store, seeds) -> list[Window]:
    """One Window per history episode (training query: history seeds only), ordered by seed."""
    rows = store.training_windows(seeds)
    eps = store.episodes().set_index("seed")
    out = []
    for ep, part in rows.groupby("episode", sort=True):
        out.append(encode_rows(part, eps.loc[int(ep), KNOWN_COLS].to_numpy(dtype=np.float32)))
    return out
