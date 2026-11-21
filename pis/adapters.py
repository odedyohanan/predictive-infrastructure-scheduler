"""Adapters over the sibling repos: reality (sim), the RL student (sched)."""
from __future__ import annotations

import copy
import os

import numpy as np

from pis.config import setup_imports, sibling_paths

setup_imports()

import torch  # noqa: E402
from sched.controllers import isolated_policy  # noqa: E402
from sched.dataset import policy_dict  # noqa: E402
from sched.randomize import known_params  # noqa: E402
from sched.rl import apply_action, greedy_action, log_prob, sample_action  # noqa: E402
from sched.student.data import normalise_one  # noqa: E402
from sched.student.features import encode, encode_target  # noqa: E402
from sched.student.model import load  # noqa: E402
from sim.randomize import random_spec  # noqa: E402
from sim.simulator import Simulator  # noqa: E402


def make_spec(seed: int):
    """Scenario spec for a seed (reality = the `sim` package)."""
    return random_spec(np.random.default_rng(900000 + seed))


def simulator_class() -> type:
    """The simulator to build: the compiled `FastSimulator` when PIS_FAST_SIM=1, else the Python `Simulator`.

    Read at call time. Both produce identical telemetry; the Python one stays the default.
    """
    if os.environ.get("PIS_FAST_SIM") == "1":
        from pis.fastsim import FastSimulator
        return FastSimulator
    return Simulator


def make_sim(seed: int, duration_s: float = 60.0) -> Simulator:
    return simulator_class()(make_spec(seed), seed, duration_s)


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

    def log_probs(self, tel: dict, current: dict, policies: list[dict]) -> np.ndarray:
        """log pi(policy | state) of each policy under the scheduler, without advancing its history.

        A policy equal to `current` can be reached by "no change" or by a change to the same fields; both
        are summed. A policy outside the scheduler's answer space (e.g. off-grid shares) gets -inf.
        """
        x = encode(tel, current, self.known, self.history)
        with torch.no_grad():
            out = self.model(normalise_one(x, self.norm, current))
            cur_key = _fields_key(current)
            res = []
            for p in policies:
                t = encode_target(p)
                lp = -np.inf
                if t is not None:
                    act = {"gate": torch.ones(1), "elig": torch.from_numpy(t["elig"])[None],
                           "share": torch.tensor([int(t["share"])]),
                           "gpu": torch.from_numpy(t["gpu"])[None]}
                    lp = float(log_prob(out, act)[0])
                if _fields_key(p) == cur_key:
                    keep = {"gate": torch.zeros(1), "elig": torch.zeros(1, 3, 8),
                            "share": torch.zeros(1, dtype=torch.long), "gpu": torch.zeros(1, 2, dtype=torch.long)}
                    lp = float(np.logaddexp(lp, float(log_prob(out, keep)[0])))
                res.append(lp)
        return np.array(res)


def _fields_key(p: dict) -> tuple:
    wl = p["workloads"]
    return tuple((tuple(sorted(wl[w]["preferred_cpus"])) or tuple(range(8)), wl[w]["cpu_share"],
                  tuple(sorted(wl[w]["gpus"]))) for w in ("A", "B", "C"))
