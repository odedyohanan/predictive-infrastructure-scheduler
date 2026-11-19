"""Incident labels from per-window SLO v2 violations.

A window is in incident when it lies inside a run of at least `k` consecutive violated windows, or
when more than `majority` of the trailing `interval` windows (ending at it) are violated. An
incident is a maximal stretch of incident windows; `end` is exclusive. Incidents whose onset falls
in the warm-up are not counted as incidents, but their windows still count as time in incident.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from pis.config import setup_imports

setup_imports()

from sim.config import WorkloadA  # noqa: E402
from sim.slo import window_violation  # noqa: E402

SLO_MS = next(f.default for f in dataclasses.fields(WorkloadA) if f.name == "slo_p99_ms")
WINDOW_S = 0.5
WARMUP = 20


@dataclass(frozen=True)
class IncidentRule:
    k: int
    interval: int
    majority: float = 0.5


@dataclass(frozen=True)
class Incident:
    onset: int
    end: int            # exclusive
    cause: str          # "building_up" (onset while a degradation is active) or "sudden"


@dataclass
class Labels:
    in_incident: np.ndarray
    incidents: list[Incident] = field(default_factory=list)
    time_in_incident_s: float = 0.0


def violations(records: list[dict], slo_ms: float = SLO_MS) -> np.ndarray:
    return np.array([window_violation(r["tel"], slo_ms, "v2") for r in records], dtype=bool)


def _in_runs(viol: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(len(viol), dtype=bool)
    i, n = 0, len(viol)
    while i < n:
        if not viol[i]:
            i += 1
            continue
        j = i
        while j < n and viol[j]:
            j += 1
        if j - i >= k:
            out[i:j] = True
        i = j
    return out


def _majority(viol: np.ndarray, interval: int, majority: float) -> np.ndarray:
    """Trailing-window share; the first windows are judged against the full interval length."""
    c = np.concatenate([[0], np.cumsum(viol.astype(int))])
    idx = np.arange(1, len(viol) + 1)
    counts = c[idx] - c[np.maximum(idx - interval, 0)]
    return counts > majority * interval


def label_series(viol: np.ndarray, active: np.ndarray, rule: IncidentRule, warmup: int = WARMUP,
                 window_s: float = WINDOW_S) -> Labels:
    viol = np.asarray(viol, dtype=bool)
    active = np.asarray(active, dtype=bool)
    if viol.shape != active.shape or viol.ndim != 1:
        raise ValueError(f"violation and active series differ: {viol.shape} vs {active.shape}")
    inc = _in_runs(viol, rule.k) | _majority(viol, rule.interval, rule.majority)
    found: list[Incident] = []
    i, n = 0, len(inc)
    while i < n:
        if not inc[i]:
            i += 1
            continue
        j = i
        while j < n and inc[j]:
            j += 1
        if i >= warmup:
            found.append(Incident(i, j, "building_up" if active[i] else "sudden"))
        i = j
    return Labels(in_incident=inc, incidents=found, time_in_incident_s=float(inc.sum()) * window_s)


def label(records: list[dict], rule: IncidentRule, warmup: int = WARMUP) -> Labels:
    """`label_series` over an episode's records; records without a "phase_b" block are not active."""
    active = np.array([bool(r["tel"].get("phase_b", {}).get("active", False)) for r in records], dtype=bool)
    return label_series(violations(records), active, rule, warmup)


def is_chronic(labels: Labels, n_windows: int) -> bool:
    """More than half of the episode's windows in incident."""
    return int(labels.in_incident.sum()) > n_windows / 2


def early_signal(records: list[dict], w: int = 4) -> np.ndarray:
    """Least-squares slope (per window) of `a.queue_overdue` over the last `w` windows, current included.

    An input feature only. 0 where fewer than two windows are available.
    """
    q = np.array([r["tel"]["a"]["queue_overdue"] for r in records], dtype=float)
    out = np.zeros(len(q), dtype=float)
    for i in range(1, len(q)):
        y = q[max(0, i - w + 1):i + 1]
        x = np.arange(len(y), dtype=float)
        x -= x.mean()
        out[i] = float((x * (y - y.mean())).sum() / (x * x).sum())
    return out


def onset_within(labels: Labels, t: int, h: int) -> bool:
    """A counted incident starts in windows (t, t + h]: the prediction target."""
    return any(t < inc.onset <= t + h for inc in labels.incidents)
