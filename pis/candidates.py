"""Candidate policies: the base scheduler's proposals and random valid perturbations."""
from __future__ import annotations

import copy

import numpy as np

N_CPUS, N_GPUS, WL = 8, 2, ("A", "B", "C")
SHARE_STEP, MIN_SHARE = 5, 10


def _cpus(wp: dict) -> list[int]:
    return sorted(wp["preferred_cpus"]) if wp["preferred_cpus"] else list(range(N_CPUS))


def is_valid(policy: dict) -> bool:
    """Every CPU is used by someone and every workload has a CPU (empty list = all CPUs)."""
    wl = policy["workloads"]
    used = set()
    for w in WL:
        cpus = _cpus(wl[w])
        if not cpus:
            return False
        used.update(cpus)
    return used == set(range(N_CPUS))


def policy_key(policy: dict) -> tuple:
    """Canonical form for de-duplication (empty CPU list == all CPUs)."""
    wl = policy["workloads"]
    return tuple((tuple(_cpus(wl[w])), wl[w]["cpu_share"], tuple(sorted(wl[w]["gpus"]))) for w in WL)


def _dedup_valid(policies: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in policies:
        k = policy_key(p)
        if k not in seen and is_valid(p):
            seen.add(k)
            out.append(p)
    return out


def scheduler_candidates(base, tel: dict, current: dict, m: int, rng: np.random.Generator) -> list[dict]:
    """Greedy first, then `m` samples, then keep-current; valid and de-duplicated, order kept.

    `base.propose` already repairs raw actions with the sibling's `apply_action`; policies that are
    still invalid are dropped here.
    """
    greedy, samples = base.propose(tel, current, m, rng)
    return candidates_from(greedy, samples, current)


def candidates_from(greedy: dict, samples: list[dict], current: dict) -> list[dict]:
    """Candidate list from one `propose` result: greedy, samples, keep-current; valid, de-duplicated."""
    return _dedup_valid([greedy, *samples, copy.deepcopy(current)])


def _mutate_cpus(p: dict, rng: np.random.Generator) -> None:
    w = WL[int(rng.integers(3))]
    cpus = set(_cpus(p["workloads"][w]))
    for k in rng.choice(N_CPUS, size=int(rng.integers(1, 3)), replace=False):
        cpus.symmetric_difference_update({int(k)})
    p["workloads"][w]["preferred_cpus"] = [] if len(cpus) == N_CPUS else sorted(cpus)


def _mutate_shares(p: dict, rng: np.random.Generator) -> None:
    i, j = rng.choice(3, size=2, replace=False)
    wi, wj = p["workloads"][WL[i]], p["workloads"][WL[j]]
    room = (wi["cpu_share"] - MIN_SHARE) // SHARE_STEP
    if room > 0:
        d = int(rng.integers(1, room + 1)) * SHARE_STEP
        wi["cpu_share"] -= d
        wj["cpu_share"] += d


def _mutate_gpu(p: dict, rng: np.random.Generator) -> None:
    g = int(rng.integers(N_GPUS))
    for w in WL:
        if g in p["workloads"][w]["gpus"]:
            p["workloads"][w]["gpus"] = [x for x in p["workloads"][w]["gpus"] if x != g]
    owner = (None, "B", "C")[int(rng.integers(3))]
    if owner:
        p["workloads"][owner]["gpus"] = sorted(p["workloads"][owner]["gpus"] + [g])


def random_valid_policies(spec, current: dict, k: int, strength: float,
                          rng: np.random.Generator) -> list[dict]:
    """`k` valid random perturbations of `current`; strength in [0, 1] sets how many edits (1 to 5)."""
    ops = (_mutate_cpus, _mutate_shares, _mutate_gpu)
    n_edits = 1 + int(round(float(np.clip(strength, 0.0, 1.0)) * 4))
    out = []
    for _ in range(k):
        p = copy.deepcopy(current)
        for _ in range(n_edits):
            trial = copy.deepcopy(p)
            ops[int(rng.integers(len(ops)))](trial, rng)
            if is_valid(trial):                  # an edit that breaks validity is skipped
                p = trial
        out.append(p)
    return out
