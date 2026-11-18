"""Counterfactual training data: branch the real simulator from real states with many varied policies.

Records use the world-model repo's format (`worldmodel.generate`): input.telemetry, input.current_policy,
input.known, applied, outcome, plus run / t. Every branch is its own run:

  7 *context* records   the real transitions before the branch state (so a history-8 model sees the real
                        past); flagged `"context": true` and dropped from the training rows after the
                        history arrays are built, because the same real transitions repeat in every branch;
  `branch_len` records  the branch: one candidate policy held fixed from the real state, simulated with its
                        own reseed (`clone_sim`).

Run id = branch_index * 10000 + scenario seed, so `run % 10 == seed % 10` and the sibling's split
(`worldmodel.data.split_of`: 8 = val, 9 = test) follows the scenario: no scenario is in two splits.
Only development scenarios 1150-1199 are allowed (1100-1149 are the Step 0 probe states).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from pis.adapters import clone_sim
from pis.candidates import is_valid, policy_key, random_valid_policies
from pis.config import setup_imports
from pis.probe import DRIVERS, MIN_WINDOW, drive, n_windows

setup_imports()

from sim.policy import policy_from_dict  # noqa: E402
from worldmodel.data import load_records  # noqa: E402
from worldmodel.history import build_history_arrays  # noqa: E402

CF_SEEDS = range(1150, 1200)
CF_RESEED = 8_000_000
_POLICY_RNG = 53_000


def _plain(x):
    """numpy -> python, so records are JSON-identical before and after saving."""
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else o.item()))


def _check_seed(seed: int) -> None:
    if seed not in CF_SEEDS:
        raise ValueError(f"seed {seed} is not a counterfactual development scenario "
                         f"({CF_SEEDS.start}-{CF_SEEDS.stop - 1})")


def _record(run_id, seed, window, driver, tel, current, known, applied, outcome, context):
    return _plain({"run": run_id, "t": tel["t"], "driver": f"counterfactual:{driver}",
                   "scenario_seed": seed, "window": window, "context": context, "explored": not context,
                   "input": {"telemetry": tel, "current_policy": current, "known": known},
                   "applied": applied, "outcome": outcome})


def branch_policies(state, n_policies: int, rng: np.random.Generator) -> list[dict]:
    """Keep-current plus `n_policies` random valid perturbations, strengths uniform in [0, 1]; distinct."""
    pool = [state.current] + [random_valid_policies(state.spec, state.current, 1, float(rng.uniform()), rng)[0]
                              for _ in range(n_policies)]
    out, seen = [], set()
    for p in pool:
        if is_valid(p) and policy_key(p) not in seen:
            seen.add(policy_key(p))
            out.append(p)
    return out


def make_seed_records(seed: int, drivers=DRIVERS, every: int = 8, n_policies: int = 8,
                      branch_len: int = 4, history: int = 8, duration_s: float = 60.0) -> list[dict]:
    """All counterfactual records of one scenario (every driver, a branch state every `every` windows)."""
    _check_seed(seed)
    last = n_windows(seed, duration_s) - 1 - branch_len
    windows = list(range(max(MIN_WINDOW, history), last + 1, every))
    rng = np.random.default_rng([_POLICY_RNG, seed])
    out, idx = [], 0
    for drv in drivers:
        for st in drive(seed, drv, windows, duration_s):
            w, hist = st.ref.window, st.history     # hist[i] = (telemetry i, policy applied after i)
            for pol in branch_policies(st, n_policies, rng):
                run_id = idx * 10000 + seed
                for i in range(w - (history - 1), w):   # i >= 1: the policy that ran during i is hist[i-1][1]
                    out.append(_record(run_id, seed, i, drv, hist[i][0], hist[i - 1][1],
                                       st.known, hist[i][1], hist[i + 1][0], True))
                sim = clone_sim(st.sim, reseed=CF_RESEED + seed * 10000 + idx)
                tel, cur = st.tel, st.current
                for h in range(branch_len):
                    nxt = sim.run_window(policy_from_dict(pol))
                    out.append(_record(run_id, seed, w + h, drv, tel, cur, st.known, pol, nxt, False))
                    tel, cur = nxt, pol
                idx += 1
    return out


def write_records(records: list[dict], out_dir, seed: int) -> Path:
    """One gzipped JSONL file per scenario, written atomically (load_records globs run_*.jsonl.gz)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{seed:05d}.jsonl.gz"
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)
    return path


def read_records(dirs: list[str], split: str | None = None) -> list[dict]:
    return load_records(dirs, split)


def history_arrays_without_context(records: list[dict], k: int):
    """`build_history_arrays`, then context rows dropped: they only provide history to the branch rows."""
    H, S, Y, M, ordered = build_history_arrays(records, k)
    keep = np.array([not r.get("context") for r in ordered])
    return H[keep], S[keep], Y[keep], M[keep], [r for r, kp in zip(ordered, keep) if kp]
