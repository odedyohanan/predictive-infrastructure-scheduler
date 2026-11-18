"""Adapters over the sibling repos: reality (sim), the RL student (sched)."""
from __future__ import annotations

import copy

import numpy as np

from pis.config import setup_imports, sibling_paths

setup_imports()

import torch  # noqa: E402
from sched.controllers import isolated_policy  # noqa: E402
from sched.dataset import policy_dict  # noqa: E402
from sched.randomize import known_params  # noqa: E402
from sched.rl import apply_action, greedy_action, sample_action  # noqa: E402
from sched.student.data import normalise_one  # noqa: E402
from sched.student.features import encode  # noqa: E402
from sched.student.model import load  # noqa: E402
from sim.randomize import random_spec  # noqa: E402
from sim.simulator import Simulator  # noqa: E402


def make_spec(seed: int):
    """Scenario spec for a seed (reality = the `sim` package)."""
    return random_spec(np.random.default_rng(900000 + seed))


def make_sim(seed: int, duration_s: float = 60.0) -> Simulator:
    return Simulator(make_spec(seed), seed, duration_s)


def known_of(spec) -> dict:
    """The parameters a real scheduler knows in advance."""
    return known_params(spec)


def initial_policy() -> dict:
    return policy_dict(isolated_policy())


def clone_sim(sim: Simulator, reseed: int | None = None) -> Simulator:
    """Deep copy of a running simulator; `reseed` makes its future randomness differ."""
    c = copy.deepcopy(sim)
    if reseed is not None:
        c.rng = np.random.default_rng(reseed)
    return c


class BaseScheduler:
    """The RL student: greedy policy plus sampled alternatives from one forward pass."""

    def __init__(self, ckpt_rel_path: str, spec):
        torch.set_num_threads(1)
        self.model, self.norm, _ = load(str(sibling_paths()["sched"] / ckpt_rel_path))
        self.model.eval()
        self.known = known_params(spec)
        self.history: list[float] = []

    def propose(self, tel: dict, current: dict, m: int,
                rng: np.random.Generator) -> tuple[dict, list[dict]]:
        x = encode(tel, current, self.known, self.history)
        self.history.append(tel["a"]["rate"])
        with torch.no_grad():
            out = self.model(normalise_one(x, self.norm, current))
        greedy = apply_action(greedy_action(out), current, repair="model")
        samples = [apply_action(sample_action(out, rng), current, repair="model")
                   for _ in range(m)]
        return greedy, samples
