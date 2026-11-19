"""Monte Carlo risk labels: the share of real-simulator futures in which a new incident starts soon.

A state is (episode, window t): the degraded simulator at the end of window t, the records so far and the
R6 controller that drove them. States are rebuilt by replaying the episode: the scenario and R6 are both
deterministic, so re-running seed `s` with a fresh R6 reproduces the stored telemetry exactly. Only
R6-driven episodes can be replayed (the stored policy encoding cannot be decoded back to a policy).

A branch clones the simulator (with its degradation schedule and dead CPUs) and the controller (R6 keeps
a history of rates), reseeds the simulator's future randomness (arrivals, service and A's future bursts)
and lets R6 continue. The label is whether a counted incident starts in (t, t + horizon] on the joined
series (records so far + branch), so an incident already in force at t never counts; a new one after it
ends does. Branches run k - 1 windows past the horizon: a violation run starting near t + horizon is only
an incident once it reaches length k.

`restart_risk` estimates the same probability by multilevel splitting on A's overdue backlog, for states
where plain branching almost never sees an incident.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pis.phase_b import incidents
from pis.phase_b import seeds as seed_ranges
from pis.phase_b.episode import run_episode
from pis.phase_b.scenarios import DegradedSim, Scenario, scenario_for

BRANCH_TAG = 12_000_000     # branch j of a state of episode s: BRANCH_TAG + s*1000 + j
RESTART_TAG = 14_000_000    # splitting copies: RESTART_TAG + s*10000 + counter
SAMPLE_TAG = 64_000         # state sampling
PRE_ONSET_WINDOWS = 240     # 120 s of 500 ms windows before an incident onset
SOURCE = "r6"


def branch_reseed(seed: int, j: int = 0) -> int:
    return BRANCH_TAG + seed * 1000 + j


def restart_reseed(seed: int, j: int = 0) -> int:
    return RESTART_TAG + seed * 10_000 + j


@dataclass
class State:
    """The simulator at the end of window `t`, the records of windows 0..t and the controller after them.

    Branches never modify it.
    """
    seed: int
    t: int
    dsim: DegradedSim
    records: list[dict]
    controller: object
    _history: list | None = field(default=None, repr=False)

    def history(self) -> list[tuple[dict, dict]]:
        """The (seen telemetry, policy decided after it) pairs the episode loop had built."""
        if self._history is None:
            self._history = [(_hidden(r["tel"]), r["policy"]) for r in self.records]
        return self._history


# ------------------------------------------------------------------ replay
def check_replayable(seed: int) -> None:
    """Raise ValueError unless the seed's episode is driven by R6 (history, calibration or selection)."""
    from pis.phase_b.policies import source_for
    if seed in seed_ranges.HELD_OUT:
        raise ValueError(f"seed {seed} is held out and is never replayed")
    source = source_for(seed)           # raises for seeds outside history, calibration and selection
    if source != SOURCE:
        raise ValueError(f"seed {seed} is driven by {source!r}; only R6 episodes can be replayed")


class _Stop(Exception):
    pass


def run_to(scenario: Scenario, ts, visit: Callable[[State], None]) -> None:
    """Run the scenario with a fresh R6 and call `visit(state)` at the end of each window in `ts`.

    The state is live: `visit` must branch from it (clones) and not keep it. The run stops after the last
    requested window.
    """
    from pis.phase_b import policies
    want = sorted({int(t) for t in ts})
    n = int(round(scenario.duration_s / scenario.spec.policy_period_s))
    if not want or want[0] < 0 or want[-1] >= n:
        raise ValueError(f"windows {want[:3]}... outside the episode's {n} windows")
    ctrl = policies.make(SOURCE, scenario.spec, scenario.seed)
    recs: list[dict] = []
    todo = set(want)

    def on_window(ds, rec):
        recs.append(rec)
        i = len(recs) - 1
        if i in todo:
            visit(State(scenario.seed, i, ds, list(recs), ctrl))
            if i == want[-1]:
                raise _Stop

    try:
        run_episode(scenario, ctrl, SOURCE, on_window)
    except _Stop:
        return


def state_at(scenario: Scenario, t: int) -> State:
    """The state at the end of window `t` of a scenario run by R6."""
    out: list[State] = []
    run_to(scenario, [t], out.append)
    return out[0]


def replay_visit(seed: int, ts, visit: Callable[[State], None]) -> None:
    check_replayable(seed)
    run_to(scenario_for(seed), ts, visit)


def replay_to(seed: int, t: int) -> State:
    """Deterministic re-run of an R6 episode up to and including window `t`."""
    check_replayable(seed)
    return state_at(scenario_for(seed), t)


# ------------------------------------------------------------------ branches
def clone_controller(ctrl):
    """Deep copy of a controller; the network weights (read-only) are shared, not copied."""
    memo: dict = {}
    base = getattr(ctrl, "base", None)
    for name in ("model", "norm"):
        obj = getattr(base, name, None)
        if obj is not None:
            memo[id(obj)] = obj
    return copy.deepcopy(ctrl, memo)


@dataclass
class Branch:
    """One continuation: its own simulator copy, controller copy, history and records."""
    dsim: DegradedSim
    controller: object
    history: list
    current: dict
    hold: dict | None
    records: list = field(default_factory=list)

    def step(self) -> None:
        tel = self.dsim.run_window(self.current)
        if self.hold is None:
            seen = _hidden(tel)
            policy, info = self.controller.decide(seen, self.current, self.history + [(seen, self.current)])
            self.history.append((seen, policy))
        else:
            policy, info = self.hold, {}
        self.records.append({"t": tel["t"], "tel": tel, "policy": policy, "info": info,
                             "applied": self.dsim.last_applied, "source": SOURCE if self.hold is None else "hold"})
        self.current = policy

    def clone(self, reseed: int | None) -> "Branch":
        return Branch(self.dsim.clone(reseed), clone_controller(self.controller), list(self.history),
                      self.current, self.hold, list(self.records))


def branch_windows(horizon: int, rule: incidents.IncidentRule) -> int:
    """Windows a branch runs to judge onsets within `horizon`: runs need k windows to become incidents
    (the majority rule only looks back)."""
    return horizon + rule.k - 1


def windows_left(state: State) -> int:
    sc = state.dsim.scenario
    return int(round(sc.duration_s / sc.spec.policy_period_s)) - (state.t + 1)


def start_branch(state: State, reseed: int | None, hold: dict | None = None) -> Branch:
    """A branch at the state: R6 continues (or `hold` is applied every window)."""
    return Branch(state.dsim.clone(reseed), clone_controller(state.controller) if hold is None else None,
                  list(state.history()) if hold is None else [], state.records[-1]["policy"] if hold is None
                  else hold, hold)


def continue_branch(state: State, n: int, reseed: int | None, hold: dict | None = None) -> list[dict]:
    """Records of `n` more windows from the state. `reseed=None` keeps the episode's own randomness.

    Without `hold`, R6 continues from the policy it chose after window t; with `hold`, that policy runs
    in every window instead.
    """
    if n > windows_left(state):
        raise ValueError(f"{n} windows requested, {windows_left(state)} left in the episode")
    b = start_branch(state, reseed, hold)
    for _ in range(n):
        b.step()
    return b.records


# ------------------------------------------------------------------ labels
def series(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-window SLO v2 violation and degradation-active flags of records."""
    active = np.array([bool(r["tel"].get("phase_b", {}).get("active", False)) for r in records], dtype=bool)
    return incidents.violations(records), active


def new_onset(viol: np.ndarray, active: np.ndarray, rule: incidents.IncidentRule, t: int, horizon: int,
              warmup: int) -> bool:
    """A counted incident starts in (t, t + horizon] of the series."""
    return incidents.onset_within(incidents.label_series(viol, active, rule, warmup), t, horizon)


def _hit(state_series, branch_records, rule, t, horizon, warmup) -> bool:
    v0, a0 = state_series
    v1, a1 = series(branch_records)
    return new_onset(np.concatenate([v0, v1]), np.concatenate([a0, a1]), rule, t, horizon, warmup)


def branch_risk(state: State, rule: incidents.IncidentRule, horizon: int, n_branches: int, reseed_base: int,
                warmup: int = incidents.WARMUP) -> float:
    """Share of `n_branches` R6 continuations (reseed `reseed_base + j`) with a new incident onset."""
    if n_branches < 1:
        raise ValueError("n_branches must be >= 1")
    base = series(state.records)
    n = branch_windows(horizon, rule)
    hits = sum(_hit(base, continue_branch(state, n, reseed_base + j), rule, state.t, horizon, warmup)
               for j in range(n_branches))
    return hits / n_branches


# ------------------------------------------------------------------ multilevel splitting
def splitting_estimate(root, clone, step, score, hit, levels, n_roots: int, split: int, horizon: int,
                       reseed_base: int, max_paths: int = 100_000) -> tuple[float, int]:
    """P(hit) after `horizon` steps by multilevel splitting with weights.

    `n_roots` paths start as `clone(root, reseed)`, weight 1/n_roots each. A path whose `score` first
    reaches one or more new `levels` is split into split**m copies (m = levels crossed), each with the
    path's weight divided by that factor; copies get fresh reseeds and run to the horizon. Copies are
    never discarded, so the estimate (sum of weights of paths with `hit`) is unbiased for any event; the
    levels only decide where effort goes. No more splits once `max_paths` paths exist. Deterministic for
    fixed reseeds. Returns (estimate, number of paths run).
    """
    levels = sorted(levels)
    counter = [0]
    paths = [0]

    def fresh(p):
        counter[0] += 1
        return clone(p, reseed_base + counter[0])

    def run(p, left: int, level: int, weight: float) -> float:
        total = 0.0
        while left > 0:
            step(p)
            left -= 1
            m = 0
            while level + m < len(levels) and score(p) >= levels[level + m]:
                m += 1
            if m and paths[0] < max_paths:
                level += m
                factor = split ** m
                weight /= factor
                for _ in range(factor - 1):
                    paths[0] += 1
                    total += run(fresh(p), left, level, weight)
            elif m:
                level += m
        return total + (weight if hit(p) else 0.0)

    est = 0.0
    for _ in range(n_roots):
        paths[0] += 1
        est += run(fresh(root), horizon, 0, 1.0 / n_roots)
    return est, paths[0]


def restart_risk(state: State, rule: incidents.IncidentRule, horizon: int, levels, n_roots: int, split: int,
                 reseed_base: int, warmup: int = incidents.WARMUP, max_paths: int = 2000) -> tuple[float, int]:
    """`branch_risk` by multilevel splitting on A's overdue backlog (`a.queue_overdue` levels)."""
    n = branch_windows(horizon, rule)
    if n > windows_left(state):
        raise ValueError(f"{n} branch windows exceed the {windows_left(state)} windows left")
    base = series(state.records)
    root = start_branch(state, None)
    return splitting_estimate(
        root=root, clone=lambda b, r: b.clone(r), step=lambda b: b.step(),
        score=lambda b: b.records[-1]["tel"]["a"]["queue_overdue"],
        hit=lambda b: _hit(base, b.records, rule, state.t, horizon, warmup),
        levels=levels, n_roots=n_roots, split=split, horizon=n, reseed_base=reseed_base,
        max_paths=max_paths)


# ------------------------------------------------------------------ state sampling
def sample_states(episodes: list[dict], n: int, horizon: int, warmup: int, tail: int = 0,
                  tag: int = SAMPLE_TAG) -> list[tuple[int, int, str]]:
    """`n` distinct states (episode, window, stratum), sorted; deterministic.

    `episodes`: dicts {"episode", "degraded", "onsets" (counted incident onsets), "windows"}. Half the
    states ("pre_onset") come from degraded episodes in the PRE_ONSET_WINDOWS windows before an onset, the
    rest ("uniform") uniformly from all windows; when there are too few pre-onset windows, all are taken
    and the uniform part fills up. Every state leaves room for `horizon` + `tail` windows (the branch length,
    see `branch_windows`) and lies after the warm-up.
    """
    def ok(e, w):
        return warmup <= w <= e["windows"] - 1 - horizon - tail

    pre = sorted({(e["episode"], w) for e in episodes if e["degraded"] for o in e["onsets"]
                  for w in range(o - PRE_ONSET_WINDOWS, o) if ok(e, w)})
    every = [(e["episode"], w) for e in episodes for w in range(e["windows"]) if ok(e, w)]
    if n > len(every):
        raise ValueError(f"{n} states requested, only {len(every)} windows available")
    rng = np.random.default_rng(tag)
    k = min(n // 2, len(pre))
    chosen = [pre[i] for i in rng.choice(len(pre), size=k, replace=False)] if k else []
    taken = set(chosen)
    rest = [s for s in every if s not in taken]
    uni = [rest[i] for i in rng.choice(len(rest), size=n - k, replace=False)]
    out = [(e, w, "pre_onset") for e, w in chosen] + [(e, w, "uniform") for e, w in uni]
    return sorted((int(e), int(w), s) for e, w, s in out)


def _hidden(tel: dict) -> dict:
    return {k: v for k, v in tel.items() if k != "phase_b"}
