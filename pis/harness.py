"""Closed-loop episode runner and the stage 1 check: world-model forecast vs simulated fact."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np

from pis.adapters import BaseScheduler, initial_policy, make_sim
from pis.config import setup_imports
from pis.privacy import scrub
from pis.reward import reward_from_state

setup_imports()

from sim.policy import policy_from_dict  # noqa: E402
from worldmodel.features import KEY_TARGETS, decode_state, encode_state  # noqa: E402
from worldmodel.rollout import rollout  # noqa: E402

BASE_CKPT = "models/rl_r6/final.pt"


class Controller(Protocol):
    def decide(self, tel: dict, current: dict,
               history: list[tuple[dict, dict]]) -> tuple[dict, dict]:
        """Policy to apply next and an info dict (at least `decide_ms`).

        `history` is newest last: older pairs are (telemetry, policy applied after it); the newest
        pair is (current telemetry, policy currently in force).
        """


class BaseGreedyController:
    """The RL student acting greedily (the R6 baseline)."""

    def __init__(self, spec, ckpt_rel_path: str = BASE_CKPT):
        self.base = BaseScheduler(ckpt_rel_path, spec)
        self.rng = np.random.default_rng(0)

    def decide(self, tel, current, history):
        t0 = time.perf_counter()
        greedy, _ = self.base.propose(tel, current, 0, self.rng)
        return greedy, {"decide_ms": (time.perf_counter() - t0) * 1000.0}


def run_scenario(seed: int, controller: Controller, duration_s: float = 60.0) -> list[dict]:
    """One record per window: {"t", "tel", "policy", "info"}.

    `policy` is what the controller chose after seeing `tel`; it shapes the next window. The last
    record's policy is never applied (the run is over).
    """
    sim = make_sim(seed, duration_s)
    if hasattr(controller, "attach_sim"):      # oracle controllers branch the real simulator
        controller.attach_sim(sim)
    current = initial_policy()
    records: list[dict] = []
    history: list[tuple[dict, dict]] = []
    tel = sim.run_window(policy_from_dict(current))
    while True:
        view = history + [(tel, current)]
        policy, info = controller.decide(tel, current, view)
        records.append({"t": tel["t"], "tel": tel, "policy": policy, "info": info})
        history.append((tel, policy))
        if sim.done():
            break
        current = policy
        tel = sim.run_window(policy_from_dict(current))
    return records


def scenario_metrics(records: list[dict], spec) -> dict:
    """Per-scenario metrics of a `run_scenario` result.

    value_per_s as in `sched.rl.reward` (B batches x value + C core-ms x value, per second); SLO v1/v2 window
    violation % (`sim.slo.window_violation`); A late % = late requests / arrivals; policy changes between
    the policies actually in force; the overdue A queue at the end of the last window; decision time; and,
    for loop controllers, the fallback rate and how often the base scheduler's greedy policy was applied.
    """
    from pis.candidates import policy_key
    from sim.slo import window_violation
    tels = [r["tel"] for r in records]
    dur = sum(t["window_s"] for t in tels)
    b = sum(t["b"]["batches"] for t in tels)
    c = sum(t["c"]["work_core_ms"] for t in tels)
    slo = spec.a.slo_p99_ms
    applied = [initial_policy()] + [r["policy"] for r in records[:-1]]
    keys = [policy_key(p) for p in applied]
    ms = np.array([r["info"].get("decide_ms", np.nan) for r in records], dtype=float)
    fb = [r["info"]["fallback"] for r in records if "fallback" in r["info"]]
    greedy = [r["info"]["chose_greedy"] for r in records if "chose_greedy" in r["info"]]
    return {
        "windows": len(records),
        "value_per_s": float((spec.b.value_per_batch * b + spec.c.value_per_core_ms * c) / dur),
        "slo_v1_pct": 100.0 * float(np.mean([window_violation(t, slo, "v1") for t in tels])),
        "slo_v2_pct": 100.0 * float(np.mean([window_violation(t, slo, "v2") for t in tels])),
        "a_late_pct": 100.0 * sum(t["a"]["late_new"] for t in tels) / max(sum(t["a"]["arrivals"] for t in tels), 1),
        "a_queue_overdue_end": int(tels[-1]["a"]["queue_overdue"]),
        "policy_changes": int(sum(a != b for a, b in zip(keys, keys[1:]))),
        "decide_ms_mean": float(np.nanmean(ms)),
        "decide_ms_p95": float(np.nanpercentile(ms, 95)),
        "decide_ms_max": float(np.nanmax(ms)),
        "fallback_rate": float(np.mean(fb)) if fb else float("nan"),
        "greedy_rate": float(np.mean(greedy)) if greedy else float("nan"),
    }


def _sibling_records(records: list[dict], known: dict) -> list[dict]:
    """Records in the world-model repo's format; the outcome of window i is window i+1's telemetry."""
    return [{"input": {"telemetry": r["tel"], "known": known}, "applied": r["policy"],
             "outcome": nxt["tel"]} for r, nxt in zip(records, records[1:])]


def forecast_vs_fact(records: list[dict], model, spec, horizons=(1, 5, 10)) -> dict:
    """Mean absolute error (real units) of open-loop forecasts against the simulated windows.

    Every window is a start point; the forecast uses the policies that were actually applied.
    Returns {"model": {h: {target: mae, "reward": mae}}, "persistence": same (state stays as at
    the start), "n_starts": {h: count}}.
    """
    from pis.adapters import known_of
    run = _sibling_records(records, known_of(spec))
    h_max = max(horizons)
    errs = {h: {"model": [], "persistence": []} for h in horizons}
    for start in range(len(run)):
        steps = min(h_max, len(run) - start)
        pred, true = rollout(model, run, start, h_max)
        s0 = encode_state(run[start]["input"]["telemetry"])
        for h in horizons:
            if h > steps:
                continue
            truth = decode_state(true[h - 1])
            win = run[start]["input"]["telemetry"]["window_s"]
            for name, vec in (("model", pred[h - 1]), ("persistence", s0)):
                got = decode_state(vec)
                e = {k: abs(got[k] - truth[k]) for k in KEY_TARGETS}
                e["reward"] = abs(reward_from_state(spec, got, win, 1.0)
                                  - reward_from_state(spec, truth, win, 1.0))
                errs[h][name].append(e)
    out = {"model": {}, "persistence": {}, "n_starts": {}}
    for h in horizons:
        for name in ("model", "persistence"):
            rows = errs[h][name]
            out[name][h] = {k: float(np.mean([r[k] for r in rows])) if rows else float("nan")
                            for k in [*KEY_TARGETS, "reward"]}
        out["n_starts"][h] = len(errs[h]["model"])
    return out


def merge_results(results: list[dict]) -> dict:
    """Average per-scenario `forecast_vs_fact` results, weighting by number of start points."""
    out = {"model": {}, "persistence": {}, "n_starts": {}}
    for h in results[0]["n_starts"]:
        w = np.array([r["n_starts"][h] for r in results], dtype=float)
        out["n_starts"][h] = int(w.sum())
        for name in ("model", "persistence"):
            out[name][h] = {k: float(np.average([r[name][h][k] for r in results], weights=w))
                            for k in results[0][name][h]}
    return out


def render_report(results: dict, meta: dict) -> str:
    """Markdown text of a stage 1 result (not yet scrubbed)."""
    lines = ["# Stage 1: forecast vs fact", ""]
    for k, v in meta.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "Mean absolute error in real units; persistence = state stays as at the start.", ""]
    names = list(next(iter(results["model"].values())))
    for h in results["model"]:
        lines += [f"## Horizon {h} windows (n = {results['n_starts'][h]} starts)", "",
                  "| target | model MAE | persistence MAE | model / persistence |",
                  "|---|---|---|---|"]
        for k in names:
            m, p = results["model"][h][k], results["persistence"][h][k]
            ratio = m / p if p > 0 else float("nan")
            lines.append(f"| {k} | {m:.4g} | {p:.4g} | {ratio:.2f} |")
        lines.append("")
    return "\n".join(lines)


def write_report(text: str, path: str | Path) -> None:
    """Write scrubbed text only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scrub(text))
