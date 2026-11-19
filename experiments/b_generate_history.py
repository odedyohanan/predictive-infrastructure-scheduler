"""Generate the episode history: one degradation scenario per seed, driven by its policy source.

  .venv/bin/python experiments/b_generate_history.py --seeds 3100-3399 --procs 4 --out data/phase_b/history
  .venv/bin/python experiments/b_generate_history.py --seeds 3100-3103 --label smoke --duration 60 \\
      --out data/phase_b/smoke_history/<name>

History, calibration and selection seeds only (held-out and earlier seeds are refused). Shards of 10 seeds
run through `pis.shards.run_shards`: rerunning the same command skips finished shards, and a changed
configuration in the same output directory is refused. A dirty working tree is refused unless the run is
labelled "smoke"; `--duration` (shorter episodes) needs the smoke label, and a smoke run never writes into
the default output directory. Long runs go through scripts/gentle.sh.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pis.phase_b import runs  # noqa: E402
from pis.phase_b import seeds as seed_ranges  # noqa: E402
from pis.privacy import safe_path  # noqa: E402

DEFAULT_OUT = "data/phase_b/history"
SHARD_SIZE = 10
GENERATED = ("HISTORY", "CALIBRATION", "SELECTION")


def check_seeds(seeds: list[int]) -> None:
    """Every seed must be a history, calibration or selection seed (held-out and earlier seeds refused)."""
    runs.check_seeds(seeds, GENERATED)


def run_shard(shard, shard_dir: Path, duration_s: float | None) -> None:
    """Worker side: every seed of the shard, one episode each, written into the shard's directory."""
    import dataclasses

    import torch
    torch.set_num_threads(1)
    from pis.phase_b import history, policies
    from pis.phase_b.episode import run_episode
    from pis.phase_b.scenarios import scenario_for
    for seed in shard.items:
        t0 = time.time()
        sc = scenario_for(seed)
        if duration_s is not None:
            sc = dataclasses.replace(sc, duration_s=duration_s)
        source = policies.source_for(seed)
        records = run_episode(sc, policies.make(source, sc.spec, seed), source)
        history.write_episode(shard_dir, sc, records, source, wall_s=time.time() - t0)


def make_shards(seeds: list[int]):
    from pis.shards import Shard
    return [Shard(f"s{chunk[0]}-{chunk[-1]}", chunk)
            for chunk in (seeds[i:i + SHARD_SIZE] for i in range(0, len(seeds), SHARD_SIZE))]


def summary(out: Path) -> str:
    from pis.phase_b import history
    try:
        ep = history.Store(out).episodes()
    except ValueError:
        return "no finished episodes"
    by = defaultdict(list)
    for src, w in zip(ep.source, ep.wall_s):
        by[src].append(w)
    lines = [f"{len(ep)} episodes, {int(ep.windows.sum())} windows"]
    for src in sorted(by):
        v = by[src]
        lines.append(f"  {src:17s} {len(v):4d} episodes, wall s/episode mean {sum(v) / len(v):.1f} max {max(v):.1f}")
    for kind, n in ep.kind.value_counts().sort_index().items():
        lines.append(f"  kind {kind:13s} {n:4d}")
    return "\n".join(lines)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", required=True, help='e.g. "3100-3399"')
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--label", default="", help='"smoke" for a test run (allows a dirty tree and --duration)')
    ap.add_argument("--duration", type=float, default=None, help="episode length in s (smoke runs only)")
    a = ap.parse_args(argv)
    try:
        seeds = seed_ranges.parse(a.seeds)
        check_seeds(seeds)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("refusing to run: repeated seeds")
    if a.label not in runs.LABELS:
        raise SystemExit(f"refusing to run: unknown label {a.label!r}")
    smoke = a.label == "smoke"
    if a.duration is not None and not smoke:
        raise SystemExit("refusing to run: --duration needs --label smoke")
    if a.duration is not None and a.duration <= 0:
        raise SystemExit("refusing to run: --duration must be positive")
    if smoke and Path(a.out).resolve() == (ROOT / DEFAULT_OUT).resolve():
        raise SystemExit("refusing to run: a smoke run never writes into the default output directory")
    if not smoke and runs.tree_dirty():
        raise SystemExit("refusing to run: the working tree has uncommitted changes (commit, or --label smoke)")

    from pis.phase_b import history, policies
    from pis.phase_b.scenarios import DURATION_S
    from pis.shards import run_shards
    config = {"seeds": seeds, "label": a.label, "duration_s": a.duration or DURATION_S,
              "shard_size": SHARD_SIZE, "generator_version": history.GENERATOR_VERSION,
              "sources": policies.SOURCES}
    out = Path(a.out)
    t0 = time.time()
    print(f"generating {len(seeds)} episodes into {safe_path(str(out))} with {a.procs} processes", flush=True)
    try:
        res = run_shards(out, make_shards(seeds), partial(run_shard, duration_s=a.duration), config, a.procs)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    print(f"shards done {len(res['done'])}, skipped {len(res['skipped'])}, failed {len(res['failed'])} "
          f"{res['failed'] or ''} in {time.time() - t0:.0f} s", flush=True)
    print(summary(out), flush=True)
    return res


if __name__ == "__main__":
    main()
