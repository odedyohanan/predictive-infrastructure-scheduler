"""Degradation scenarios over the simulator.

A scenario is an ordinary randomised spec plus at most one degradation that starts at `onset_s` and
grows linearly to full strength over `ramp_s` (CPU and GPU loss are steps). The degradation is applied
by `DegradedSim`, which replaces `sim.spec` (frozen dataclass) before each window and masks the policy
for lost CPUs and GPUs. The scenario's own `spec` is never touched, so the parameters controllers know
in advance stay at the episode's start values: drift is not announced.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pis.adapters import clone_sim, make_spec
from pis.config import setup_imports

setup_imports()

from sim.policy import policy_from_dict  # noqa: E402
from sim.simulator import Simulator  # noqa: E402

Kind = Literal["none", "load_ramp", "leak", "interference", "cpu_loss", "ambient", "gpu_loss"]
CAUSES = ("load_ramp", "leak", "interference", "cpu_loss")
DISTRACTORS = ("ambient", "gpu_loss")
DEGRADED_KINDS = CAUSES + DISTRACTORS
STEP_KINDS = ("cpu_loss", "gpu_loss")

# Full-strength effect of each kind:
#   load_ramp     A's base rate x (1 + m)
#   leak          A's mean service time x (1 + m)
#   interference  A's interference coefficient + m (absolute)
#   ambient       thermal ambient + m degrees C
#   cpu_loss      the first m CPUs of DEAD_CPUS stop
#   gpu_loss      GPU DEAD_GPU stops (m is unused, kept at 1)
MAGNITUDES = {"load_ramp": 0.6, "leak": 0.8, "interference": 0.4, "cpu_loss": 2.0,
              "ambient": 15.0, "gpu_loss": 1.0}
DEAD_CPUS = (2, 3)        # two of A's four CPUs in the default isolated layout
DEAD_GPU = 0
ONSET_S = (60.0, 180.0)
RAMP_S = (30.0, 120.0)
DURATION_S = 300.0
RNG_TAG = 60_000          # per-seed onset and ramp
BLOCK_TAG = 61_000        # per-block shuffle of the kinds
# Every block of BLOCK consecutive seeds holds each degraded kind once and as many ordinary seeds,
# shuffled, so even short seed ranges are close to half ordinary with equal shares of the kinds.
BLOCK = 2 * len(DEGRADED_KINDS)


@dataclass(frozen=True)
class Degradation:
    kind: Kind
    onset_s: float
    ramp_s: float
    magnitude: float

    def active(self, t: float) -> bool:
        return self.kind != "none" and t >= self.onset_s

    def factor(self, t: float) -> float:
        """Share of full strength at time t: 0 before onset, linear to 1 over the ramp (steps jump)."""
        if not self.active(t):
            return 0.0
        if self.kind in STEP_KINDS or self.ramp_s <= 0:
            return 1.0
        return min((t - self.onset_s) / self.ramp_s, 1.0)

    def dead_cpus(self, t: float) -> tuple[int, ...]:
        if self.kind != "cpu_loss" or not self.active(t):
            return ()
        return DEAD_CPUS[:int(self.magnitude)]

    def dead_gpus(self, t: float) -> tuple[int, ...]:
        return (DEAD_GPU,) if self.kind == "gpu_loss" and self.active(t) else ()


@dataclass(frozen=True)
class Scenario:
    seed: int
    spec: object
    degradation: Degradation
    duration_s: float = DURATION_S

    @property
    def kind(self) -> str:
        return self.degradation.kind


def scenario_for(seed: int) -> Scenario:
    """About half ordinary, half with one degradation kind in equal shares; fixed by the seed."""
    slots = ("none",) * len(DEGRADED_KINDS) + DEGRADED_KINDS
    perm = np.random.default_rng(BLOCK_TAG + seed // BLOCK).permutation(BLOCK)
    kind = slots[int(perm[seed % BLOCK])]
    rng = np.random.default_rng(RNG_TAG + seed)
    onset = float(rng.uniform(*ONSET_S))
    ramp = float(rng.uniform(*RAMP_S))
    deg = Degradation("none", 0.0, 0.0, 0.0) if kind == "none" else \
        Degradation(kind, onset, ramp, MAGNITUDES[kind])
    return Scenario(seed=seed, spec=make_spec(seed), degradation=deg)


def degraded_spec(spec, kind: str, amount: float):
    """`spec` with `amount` (= magnitude x factor) of a spec-level degradation applied."""
    r = dataclasses.replace
    if kind == "load_ramp":
        return r(spec, a=r(spec.a, base_rate=spec.a.base_rate * (1.0 + amount)))
    if kind == "leak":
        return r(spec, a=r(spec.a, service_ms=spec.a.service_ms * (1.0 + amount)))
    if kind == "interference":
        return r(spec, a=r(spec.a, interference=spec.a.interference + amount))
    if kind == "ambient":
        return r(spec, thermal=r(spec.thermal, ambient=spec.thermal.ambient + amount))
    raise ValueError(f"{kind!r} does not change the spec")


def mask_policy(policy: dict, dead_cpus=(), dead_gpus=(), n_cpus: int = 8) -> dict:
    """Copy of a policy dict that never uses dead CPUs or GPUs.

    An empty cpuset ("all CPUs") becomes the alive CPUs; a cpuset left empty by the removal also
    becomes the alive CPUs.
    """
    out = copy.deepcopy(policy)
    if not dead_cpus and not dead_gpus:
        return out
    alive = [k for k in range(n_cpus) if k not in dead_cpus]
    for w in out["workloads"].values():
        cpus = [k for k in (w["preferred_cpus"] or alive) if k in alive]
        w["preferred_cpus"] = cpus or list(alive)
        w["gpus"] = [g for g in w["gpus"] if g not in dead_gpus]
    return out


@dataclass
class DegradedSim:
    """A simulator whose physics follow the scenario's degradation schedule (keyed by `sim.t`)."""
    scenario: Scenario
    sim: Simulator = field(init=False)
    last_applied: dict | None = field(init=False, default=None)

    def __post_init__(self):
        sc = self.scenario
        self.sim = Simulator(sc.spec, sc.seed, sc.duration_s)

    def run_window(self, policy: dict) -> dict:
        d = self.scenario.degradation
        t = round(self.sim.t, 3)      # window start; sim.t accumulates 1 ms steps and drifts
        f = d.factor(t)
        if d.kind != "none" and d.kind not in STEP_KINDS:
            self.sim.spec = degraded_spec(self.scenario.spec, d.kind, d.magnitude * f) if f > 0 \
                else self.scenario.spec
        dead = d.dead_cpus(t)
        applied = mask_policy(policy, dead, d.dead_gpus(t), self.sim.n_cpus)
        self.last_applied = applied
        tel = self.sim.run_window(policy_from_dict(applied))
        tel["phase_b"] = {"kind": d.kind, "active": d.active(t), "factor": f,
                          "alive_cpus": [k for k in range(self.sim.n_cpus) if k not in dead]}
        return tel

    def done(self) -> bool:
        return self.sim.done()

    def clone(self, reseed: int | None = None) -> "DegradedSim":
        """Independent copy that keeps the schedule; `reseed` makes its future randomness differ."""
        c = copy.copy(self)
        c.sim = clone_sim(self.sim, reseed)
        c.last_applied = copy.deepcopy(self.last_applied)
        return c

