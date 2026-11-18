"""Counterfactual world-model data from development scenarios 1150-1199 (see pis/counterfactual.py).

  .venv/bin/python experiments/make_counterfactual_data.py --out data/counterfactual_v1 --procs 5
Finished scenarios are skipped, so a stopped job can be restarted with the same command.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pis.counterfactual import CF_SEEDS, make_seed_records, write_records  # noqa: E402
from pis.probe import DRIVERS  # noqa: E402


def _one(job):
    import torch
    torch.set_num_threads(1)
    seed, a = job
    path = Path(a["out"]) / f"run_{seed:05d}.jsonl.gz"
    if path.exists():
        return seed, 0, 0.0
    t0 = time.time()
    recs = make_seed_records(seed, drivers=a["drivers"], every=a["every"], n_policies=a["policies"],
                             branch_len=a["branch_len"], history=a["history"], duration_s=a["duration"])
    write_records(recs, a["out"], seed)
    return seed, len(recs), time.time() - t0


def parse_seeds(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=f"{CF_SEEDS.start}-{CF_SEEDS.stop - 1}")
    ap.add_argument("--drivers", default=",".join(DRIVERS))
    ap.add_argument("--every", type=int, default=8, help="a branch state every N windows")
    ap.add_argument("--policies", type=int, default=8, help="random policies per branch state (+ keep-current)")
    ap.add_argument("--branch-len", type=int, default=4, help="windows each candidate is held")
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--out", default="data/counterfactual_v1")
    a = ap.parse_args()
    cfg = {**vars(a), "drivers": a.drivers.split(","), "branch_len": a.branch_len}
    seeds = parse_seeds(a.seeds)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "manifest.json").write_text(json.dumps({**cfg, "seeds": seeds,
                                                           "generator": "pis.counterfactual"}, indent=1))
    jobs = [(s, cfg) for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(a.procs) as pool:
        for i, (seed, n, sec) in enumerate(pool.imap_unordered(_one, jobs), 1):
            print(f"[{i}/{len(jobs)}] seed {seed}: {n} records {sec:.1f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
