"""Reward computed from a (predicted) decoded state instead of raw telemetry."""
from __future__ import annotations

import numpy as np


def reward_from_state(spec, state: dict[str, float], window_s: float, lam: float) -> float:
    """Same quantity as `sched.rl.reward(...)[0]`, from `worldmodel.features.decode_state` output.

    The state stores batch *rates* (b.rate = batches / window), so batches = b.rate * window_s.
    """
    batches = state["b.rate"] * window_s
    value_rate = (spec.b.value_per_batch * batches
                  + spec.c.value_per_core_ms * state["c.work_core_ms"]) / window_s
    slo = spec.a.slo_p99_ms
    over = float(np.clip((state["a.p99_ms"] - slo) / slo, 0.0, 1.0))
    return value_rate / 1000.0 - lam * over


def queue_penalty(spec, state: dict[str, float], mu: float) -> float:
    """Penalty on queued work; a.p99_ms is 0 when nothing completed, so p99 alone hides a backlog."""
    return mu * float(np.clip(state["a.queue_work_ms"] / spec.a.slo_p99_ms, 0.0, 5.0))
