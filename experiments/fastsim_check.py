"""Equivalence check: FastSimulator against the Python simulator, every record compared exactly.

  .venv/bin/python experiments/fastsim_check.py --seeds 3100-3399 --phase-a-seeds 1100-1199 --procs 4
  .venv/bin/python experiments/fastsim_check.py --seeds 3100-3101 --phase-a-seeds 1100 --label smoke \\
      --duration 5 --out data/fastsim/smoke_check

Two kinds of closed-loop episodes, each run once with the Python simulator and once with FastSimulator
(PIS_FAST_SIM=1), all driven by the R6 student (greedy):
  b  Phase B degradation scenarios (`scenario_for`, 300 s) on HISTORY seeds, through `run_episode`;
  a  Phase A scenarios (`make_sim`, 60 s) through `pis.harness.run_scenario`.
Every record must be equal (`==`, the controller's wall-clock `decide_ms` aside), and after the episode
the generator state and the backlog too. A seed's result is the number of equal windows and, on a
mismatch, the first window and key that differ.

Shards of SHARD_SIZE episodes run through `pis.shards.run_shards` (rerunning skips finished shards). A
dirty working tree is refused unless the run is labelled "smoke"; `--duration` needs the smoke label, and a
smoke run writes its note into its own output directory, never into results/. Long runs go through
scripts/gentle.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pis.phase_b import runs  # noqa: E402
from pis.phase_b import seeds as seed_ranges  # noqa: E402
from pis.privacy import safe_path  # noqa: E402

DEFAULT_OUT = "data/fastsim/check"
NOTE = "results/fastsim/equivalence.md"
SHARD_SIZE = 5
PHASE_A_SEEDS = range(1100, 1200)       # Phase A development scenarios
PHASE_A_DURATION_S = 60.0
CHECK_VERSION = 1


def first_diff(a, b, path: str = "") -> str | None:
    """Path of the first difference between two nested records, None when equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            d = first_diff(a[k], b[k], f"{path}/{k}")
            if d is not None:
                return d
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d is not None:
                return d
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def untimed(rec: dict) -> dict:
    """A record without the controller's wall-clock decision time."""
    return {**rec, "info": {k: v for k, v in rec["info"].items() if k != "decide_ms"}}


class _Keep:
    """Wraps a controller and keeps the simulator the episode runs on."""

    def __init__(self, inner):
        self.inner, self.sim = inner, None

    def attach_sim(self, sim):
        self.sim = sim

    def decide(self, tel, current, history):
        return self.inner.decide(tel, current, history)


def run_one(kind: str, seed: int, fast: bool, duration_s: float | None) -> tuple[list[dict], dict, float]:
    """(records without timing, final simulator state, wall s) of one episode on one simulator class."""
    import dataclasses
    os.environ["PIS_FAST_SIM"] = "1" if fast else "0"
    t0 = time.time()
    if kind == "b":
        from pis.phase_b import policies
        from pis.phase_b.episode import run_episode
        from pis.phase_b.scenarios import scenario_for
        sc = scenario_for(seed)
        if duration_s is not None:
            sc = dataclasses.replace(sc, duration_s=duration_s)
        ctl = _Keep(policies.make("r6", sc.spec, seed))
        records = run_episode(sc, ctl, "r6")
    else:
        from pis.adapters import make_spec
        from pis.harness import BaseGreedyController, run_scenario
        ctl = _Keep(BaseGreedyController(make_spec(seed)))
        records = run_scenario(seed, ctl, duration_s or PHASE_A_DURATION_S)
    wall = time.time() - t0
    sim = ctl.sim
    state = {"rng": sim.rng.bit_generator.state, "backlog": sim.backlog(), "t": sim.t,
             "queue": [list(r) for r in sim.queue]}
    return [untimed(r) for r in records], state, wall


def check_one(kind: str, seed: int, duration_s: float | None) -> dict:
    py, py_state, py_wall = run_one(kind, seed, False, duration_s)
    fa, fa_state, fa_wall = run_one(kind, seed, True, duration_s)
    res = {"kind": kind, "seed": seed, "windows": len(py), "equal_windows": 0, "mismatch": None,
           "state_equal": py_state == fa_state, "wall_python_s": py_wall, "wall_fast_s": fa_wall}
    if len(py) != len(fa):
        res["mismatch"] = {"window": min(len(py), len(fa)), "key": f"episode length {len(py)} vs {len(fa)}"}
    for i, (a, b) in enumerate(zip(py, fa)):
        d = first_diff(a, b)
        if d is not None:
            res["mismatch"] = res["mismatch"] or {"window": i, "t": a["t"], "key": d[:300]}
            break
        res["equal_windows"] += 1
    if res["mismatch"] is None and not res["state_equal"]:
        res["mismatch"] = {"window": len(py), "key": "final simulator state"}
    res["equal"] = res["mismatch"] is None
    return res


def run_shard(shard, shard_dir: Path, duration_s: float | None) -> None:
    """Worker side: every (kind, seed) of the shard, both simulators, one JSON line each."""
    import torch
    torch.set_num_threads(1)
    with open(shard_dir / "results.jsonl", "w") as f:
        for kind, seed in shard.items:
            f.write(json.dumps(check_one(kind, seed, duration_s)) + "\n")


def make_shards(items: list[tuple[str, int]]):
    from pis.shards import Shard
    return [Shard(f"{chunk[0][0]}{chunk[0][1]}-{chunk[-1][1]}", chunk)
            for chunk in (items[i:i + SHARD_SIZE] for i in range(0, len(items), SHARD_SIZE))]


def collect(out: Path) -> list[dict]:
    rows = []
    for f in sorted(out.glob("*/results.jsonl")):
        if (f.parent / "done").exists():
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows


def note(rows: list[dict], config: dict, elapsed_s: float) -> str:
    lines = ["# FastSimulator equivalence check", "",
             f"Commit {runs.git_head()}. Label {config['label'] or '(none)'}; episodes driven by R6 (greedy), "
             "each run on the Python simulator and on FastSimulator; every record compared with `==` "
             "(controller `decide_ms` aside), then the generator state, backlog and queue.", ""]
    for kind, name in (("b", "Phase B degradation scenarios"), ("a", "Phase A scenarios")):
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            continue
        n_eq = sum(x["equal"] for x in r)
        win = sum(x["windows"] for x in r)
        py = sum(x["wall_python_s"] for x in r)
        fa = sum(x["wall_fast_s"] for x in r)
        seeds = sorted(x["seed"] for x in r)
        lines += [f"## {name}", "",
                  f"- seeds {seeds[0]} to {seeds[-1]} ({len(r)} episodes, {win} windows)",
                  f"- identical episodes: **{n_eq} / {len(r)}**",
                  f"- episode wall time (with R6): Python {py:.0f} s, fast {fa:.0f} s "
                  f"(x{py / max(fa, 1e-9):.1f})", ""]
        bad = sorted((x for x in r if not x["equal"]), key=lambda x: x["seed"])
        if bad:
            first = bad[0]
            lines += [f"First mismatch: seed {first['seed']}, window {first['mismatch']['window']}, "
                      f"key `{first['mismatch']['key']}`", "",
                      "Seeds with a mismatch: " + ", ".join(str(x["seed"]) for x in bad), ""]
    lines += [f"Run time {elapsed_s / 60:.1f} min. Configuration: `{json.dumps(config, sort_keys=True)}`", ""]
    return "\n".join(lines)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", required=True, help='Phase B HISTORY seeds, e.g. "3100-3399"')
    ap.add_argument("--phase-a-seeds", default="1100-1199", help='Phase A seeds within 1100-1199, "" for none')
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--label", default="", help='"smoke" for a test run (allows a dirty tree and --duration)')
    ap.add_argument("--duration", type=float, default=None, help="episode length in s (smoke runs only)")
    a = ap.parse_args(argv)
    try:
        b_seeds = seed_ranges.parse(a.seeds)
        seed_ranges.check(b_seeds, "HISTORY")
        a_seeds = seed_ranges.parse(a.phase_a_seeds) if a.phase_a_seeds else []
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    if any(s not in PHASE_A_SEEDS for s in a_seeds):
        raise SystemExit("refusing to run: Phase A seeds must lie in 1100-1199")
    if len(set(b_seeds)) != len(b_seeds) or len(set(a_seeds)) != len(a_seeds):
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

    from pis.shards import run_shards
    items = [("b", s) for s in b_seeds] + [("a", s) for s in a_seeds]
    config = {"b_seeds": b_seeds, "a_seeds": a_seeds, "label": a.label, "duration_s": a.duration,
              "shard_size": SHARD_SIZE, "version": CHECK_VERSION}
    out = Path(a.out)
    t0 = time.time()
    print(f"checking {len(items)} episodes into {safe_path(str(out))} with {a.procs} processes", flush=True)
    try:
        res = run_shards(out, make_shards(items), partial(run_shard, duration_s=a.duration), config, a.procs)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    rows = collect(out)
    text = note(rows, config, time.time() - t0)
    runs.write_note(out / "equivalence.md" if smoke else ROOT / NOTE, text)
    print(f"shards done {len(res['done'])}, skipped {len(res['skipped'])}, failed {len(res['failed'])} "
          f"{res['failed'] or ''}", flush=True)
    print(text, flush=True)
    return res


if __name__ == "__main__":
    main()
