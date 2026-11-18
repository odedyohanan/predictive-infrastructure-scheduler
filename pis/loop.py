"""Step A loop: candidates from the base scheduler -> ensemble rollouts -> score -> accept or fall back.

Each window the base scheduler's `propose` runs exactly once; its greedy decision is both a candidate and
the fallback, so a loop that always falls back behaves exactly like the base scheduler.

Selection modes (same candidates, same fallback rules for short history):
  model   ensemble scores (the Step A loop)
  random  uniform choice among the candidates (lower bound: no ranking)
  oracle  each candidate run in a deep copy of the real simulator for `horizon_h` windows and scored with
          the same reward terms (upper bound: the headroom the candidate set offers); needs `attach_sim`
Ablation switches (model selection only):
  use_uncertainty=False  no disagreement penalty in the score (lam = 0); the tau gate still applies
  use_fallback=False     no tau gate (the best finite candidate is always taken)
Short history (fewer windows than the members' history length) always uses the base greedy decision, in
every mode, so all arms share the start of a run.
"""
from __future__ import annotations

import time

import numpy as np

from pis.adapters import BaseScheduler, clone_sim, known_of
from pis.candidates import candidates_from, policy_key
from pis.config import setup_imports
from pis.evaluator import Decision, Params, choose, score_candidates

setup_imports()

from sim.policy import policy_from_dict  # noqa: E402
from worldmodel.features import encode_state  # noqa: E402

BASE_CKPT = "models/rl_r6/final.pt"
SELECTIONS = ("model", "random", "oracle")


class LoopController:
    def __init__(self, spec, ensemble, params: Params, base_ckpt: str = BASE_CKPT, selection: str = "model",
                 use_uncertainty: bool = True, use_fallback: bool = True, seed: int = 0,
                 select_seed: int = 0, oracle_reseeds: tuple = (None,)):
        if selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        self.spec, self.ensemble, self.params = spec, ensemble, params
        self.selection, self.use_uncertainty, self.use_fallback = selection, use_uncertainty, use_fallback
        self.base = BaseScheduler(base_ckpt, spec)
        self.known = known_of(spec)
        self.rng = np.random.default_rng(seed)               # scheduler samples
        self.select_rng = np.random.default_rng(select_seed)  # random selection only
        self.oracle_reseeds = tuple(oracle_reseeds)
        self.sim = None
        self.last_candidates: list[dict] = []

    def attach_sim(self, sim) -> None:
        """The real simulator of the running episode (read through deep copies only)."""
        self.sim = sim

    # ------------------------------------------------------------------ selection
    def _model(self, history, cands, current) -> Decision:
        p = self.params
        roll = self.ensemble.rollout_batch(history, cands, self.known, p.horizon_h)
        scores, dis = score_candidates(self.spec, roll, cands, current, p, train_std=self.ensemble.train_std,
                                       lam=p.lam if self.use_uncertainty else 0.0)
        return choose(scores, dis, p, use_fallback=self.use_fallback)

    def _random(self, cands) -> Decision:
        i = int(self.select_rng.integers(len(cands)))
        z = np.zeros(len(cands))
        return Decision(None, i, False, z, z, "random")

    def _oracle(self, cands, current) -> Decision:
        if self.sim is None:
            raise RuntimeError("oracle selection needs attach_sim(sim) (run_scenario does this)")
        p = self.params
        real = np.empty((len(self.oracle_reseeds), len(cands), p.horizon_h, 39), dtype=np.float32)
        for i, c in enumerate(cands):
            pol = policy_from_dict(c)
            for j, rs in enumerate(self.oracle_reseeds):
                sim = clone_sim(self.sim, reseed=rs)
                for h in range(p.horizon_h):
                    real[j, i, h] = encode_state(sim.run_window(pol))
        scores, dis = score_candidates(self.spec, real, cands, current, p, lam=0.0)
        d = choose(scores, dis, p, use_fallback=False)
        d.reason = "oracle" if not d.fallback else d.reason
        return d

    # ------------------------------------------------------------------ controller protocol
    def decide(self, tel: dict, current: dict, history: list[tuple[dict, dict]]) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        greedy, samples = self.base.propose(tel, current, self.params.n_candidates_m, self.rng)
        cands = candidates_from(greedy, samples, current)
        self.last_candidates = cands
        if len(history) < self.ensemble.history or not cands:
            z = np.zeros(len(cands))
            d = Decision(None, -1, True, z, z, "short_history" if cands else "no_candidates")
        elif self.selection == "random":
            d = self._random(cands)
        elif self.selection == "oracle":
            d = self._oracle(cands, current)
        else:
            d = self._model(history, cands, current)
        policy = greedy if d.fallback else cands[d.chosen]
        info = {"decide_ms": (time.perf_counter() - t0) * 1000.0, "fallback": d.fallback, "reason": d.reason,
                "chosen": d.chosen, "n_candidates": len(cands), "selection": self.selection,
                "chose_greedy": policy_key(policy) == policy_key(greedy),
                "scores": [float(x) for x in d.scores], "disagreement": [float(x) for x in d.disagreement]}
        return policy, info
