"""Policy sources that drive the history episodes.

History seeds draw one source each (weights in SOURCES), so the stored history covers varied control
behaviour, not only the reference scheduler's. Calibration and selection seeds are always driven by R6:
their labels must describe the system the advisor will sit next to.

  r6                R6 RL student, greedy (the reference scheduler)
  r3                the earlier R3 RL student, greedy
  random            a random valid perturbation of the current policy every RANDOM_EVERY windows
  adaptive          the non-ML feedback controller with the adaptive_c9 parameters
  loop_a            the Step A loop with its chosen members and parameters
  r6_interventions  R6, plus at random times (about one start per minute, never overlapping) one of the
                    named interventions applied and held for INTERVENTION_WINDOWS windows

Controller definitions come from the frozen evaluation protocol (configs/final_eval.json) and the Step A
config (configs/step_a.json), resolved by `pis.runner.arm_definitions`.
"""
from __future__ import annotations

import copy
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from pis.candidates import policy_key, random_valid_policies
from pis.phase_b import seeds as seed_ranges

SOURCES = {"r6": 0.30, "r3": 0.10, "random": 0.15, "adaptive": 0.15, "loop_a": 0.10,
           "r6_interventions": 0.20}
SOURCE_TAG = 61_000          # per-seed source draw
INTERVENTION_TAG = 62_000    # per-seed intervention schedule and choice
RANDOM_TAG = 63_000          # per-seed random policies
RANDOM_EVERY = 10            # windows between two random policies
INTERVENTION_WINDOWS = 10    # windows an intervention is held
INTERVENTION_START_P = 1.0 / 120.0   # per free window: about one start per minute of 500 ms windows

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/final_eval.json"
STEP_A = "configs/step_a.json"
ARM_OF = {"r6": "base_r6", "r3": "base_r3", "adaptive": "adaptive_c9", "loop_a": "loop_a"}


def source_for(seed: int) -> str:
    """The policy source of a seed: a weighted draw on history seeds, R6 on calibration and selection."""
    if seed in seed_ranges.HISTORY:
        names = list(SOURCES)
        p = np.array([SOURCES[n] for n in names], dtype=float)
        rng = np.random.default_rng(SOURCE_TAG + seed)
        return names[int(rng.choice(len(names), p=p / p.sum()))]
    if seed in seed_ranges.CALIBRATION or seed in seed_ranges.SELECTION:
        return "r6"
    raise ValueError(f"seed {seed} is not a history, calibration or selection seed")


@lru_cache(maxsize=1)
def arm_definitions() -> dict:
    """Controller definitions of the arms the sources use, from the protocol and the Step A config."""
    from pis.runner import arm_definitions as resolve_arms
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    step_a = json.loads((ROOT / STEP_A).read_text())
    arms = {"arms": {a: protocol["arms"][a] for a in ARM_OF.values()}}
    return resolve_arms(arms, step_a, {})


def make(source: str, spec, seed: int):
    """A fresh controller of `source` for one episode with `spec` (the spec known at the start)."""
    if source not in SOURCES:
        raise ValueError(f"unknown policy source {source!r}; expected one of {sorted(SOURCES)}")
    from pis.runner import make_controller
    if source in ARM_OF:
        return make_controller(arm_definitions()[ARM_OF[source]], spec)
    if source == "random":
        return RandomPolicies(spec, seed)
    return R6Interventions(spec, seed)


class RandomPolicies:
    """A random valid perturbation of the policy in force every RANDOM_EVERY windows (held in between)."""

    def __init__(self, spec, seed: int):
        self.spec = spec
        self.rng = np.random.default_rng(RANDOM_TAG + seed)
        self.n = 0

    def decide(self, tel, current, history):
        t0 = time.perf_counter()
        self.n += 1
        if self.n % RANDOM_EVERY == 0:
            strength = float(self.rng.uniform())
            policy = random_valid_policies(self.spec, current, 1, strength, self.rng)[0]
        else:
            policy = copy.deepcopy(current)
        return policy, {"decide_ms": (time.perf_counter() - t0) * 1000.0}


class R6Interventions:
    """R6 greedy, with named interventions held for INTERVENTION_WINDOWS windows at random times.

    R6 still sees every window (its rate history stays complete); while an intervention is held its
    answer is overridden. An intervention whose "after" policy equals the policy in force is never
    chosen.
    """

    def __init__(self, spec, seed: int, start_p: float = INTERVENTION_START_P):
        from pis.runner import make_controller
        self.r6 = make_controller(arm_definitions()[ARM_OF["r6"]], spec)
        when, which = np.random.SeedSequence(INTERVENTION_TAG + seed).spawn(2)
        self.rng = np.random.default_rng(when)          # the start schedule
        self.choice_rng = np.random.default_rng(which)  # which intervention
        self.start_p = start_p
        self.held: dict | None = None
        self.name: str | None = None
        self.left = 0                                   # windows still held after this one

    def _next_window(self) -> str:
        """Advance the start schedule by one window: "start", "hold" (an intervention is in force) or "free".

        The only place the schedule lives; `decide` calls it once per window.
        """
        if self.left > 0:
            self.left -= 1
            return "hold"
        if self.rng.uniform() < self.start_p:
            self.left = INTERVENTION_WINDOWS - 1
            return "start"
        return "free"

    def decide(self, tel, current, history):
        from pis.probe import named_interventions
        t0 = time.perf_counter()
        policy, info = self.r6.decide(tel, current, history)
        step = self._next_window()
        if step == "start":
            cur = policy_key(current)
            options = {n: after for n, (_, after) in named_interventions(current).items()
                       if policy_key(after) != cur}
            if options:
                names = sorted(options)
                self.name = names[int(self.choice_rng.integers(len(names)))]
                self.held = options[self.name]
            else:                               # nothing to apply: the window stays free
                self.left, step = 0, "free"
        if step != "free":
            policy = copy.deepcopy(self.held)
            info = {**info, "intervention": self.name, "intervention_start": step == "start"}
        info["decide_ms"] = (time.perf_counter() - t0) * 1000.0
        return policy, info
