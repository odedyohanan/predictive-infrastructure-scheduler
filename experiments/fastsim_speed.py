"""Speed of FastSimulator against the Python simulator on Phase B R6 episodes.

  .venv/bin/python experiments/fastsim_speed.py --seeds 3100-3102              # normal priority
  scripts/gentle.sh fastsim_speed .venv/bin/python experiments/fastsim_speed.py --seeds 3100-3102
  .venv/bin/python experiments/fastsim_speed.py --seeds 3100 --label smoke --duration 5 --out data/fastsim/smoke_speed

For every seed (HISTORY seeds, scenario of the seed, 300 s, driven by R6 greedy) and both simulator classes:
  episode   wall time of the closed-loop episode (`run_episode`: simulator + R6 + bookkeeping);
  sim only  wall time of replaying the same policies on a fresh `DegradedSim` (the simulator alone).
Runs in one process, one after another, so the numbers depend on the priority the process runs at (normal,
or background QoS under scripts/gentle.sh); it is detected and recorded. Results of each priority are kept
in results/fastsim/speed.json and the note results/fastsim/speed.md shows all of them. A smoke run writes
into its own output directory instead.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pis.phase_b import runs  # noqa: E402
from pis.phase_b import seeds as seed_ranges  # noqa: E402

OUT_DIR = "results/fastsim"


def episode(sc, fast: bool) -> tuple[list[dict], float]:
    from pis.phase_b import policies
    from pis.phase_b.episode import run_episode
    os.environ["PIS_FAST_SIM"] = "1" if fast else "0"
    t0 = time.perf_counter()
    records = run_episode(sc, policies.make("r6", sc.spec, sc.seed), "r6")
    return records, time.perf_counter() - t0


def replay(sc, policies: list[dict], fast: bool) -> float:
    from pis.phase_b.scenarios import DegradedSim
    os.environ["PIS_FAST_SIM"] = "1" if fast else "0"
    ds = DegradedSim(sc)
    t0 = time.perf_counter()
    for p in policies:
        ds.run_window(p)
    return time.perf_counter() - t0


def warm_up() -> float:
    """Seconds to get the compiled kernel ready in this process (cache load, or compile on a cold cache)."""
    from pis.adapters import initial_policy
    from pis.fastsim import FastSimulator
    from pis.phase_b.scenarios import scenario_for
    from sim.policy import policy_from_dict
    sc = scenario_for(3100)
    t0 = time.perf_counter()
    FastSimulator(sc.spec, sc.seed, 1.0).run_window(policy_from_dict(initial_policy()))
    return time.perf_counter() - t0


def measure(seeds: list[int], duration_s: float | None) -> dict:
    import torch
    torch.set_num_threads(1)
    from pis.adapters import initial_policy
    from pis.phase_b.scenarios import scenario_for
    rows = []
    warm = warm_up()
    for seed in seeds:
        sc = scenario_for(seed)
        if duration_s is not None:
            sc = dataclasses.replace(sc, duration_s=duration_s)
        rec_py, ep_py = episode(sc, False)
        rec_fa, ep_fa = episode(sc, True)
        pols = [initial_policy()] + [r["policy"] for r in rec_py[:-1]]
        sim_py, sim_fa = replay(sc, pols, False), replay(sc, pols, True)
        rows.append({"seed": seed, "kind": sc.kind, "windows": len(rec_py),
                     "queue_len_max": max(r["tel"]["a"]["queue_len"] for r in rec_py),
                     "same_policies": [r["policy"] for r in rec_py] == [r["policy"] for r in rec_fa],
                     "episode_python_s": ep_py, "episode_fast_s": ep_fa,
                     "sim_python_s": sim_py, "sim_fast_s": sim_fa})
        print(f"seed {seed} ({sc.kind}): episode {ep_py:.1f} s -> {ep_fa:.2f} s, "
              f"sim only {sim_py:.1f} s -> {sim_fa:.2f} s", flush=True)
    background, how = runs.background_priority()
    return {"priority": "background" if background else "normal", "priority_detail": how,
            "machine": f"{platform.system()} {platform.machine()}", "commit": runs.git_head(),
            "duration_s": duration_s, "warm_up_s": warm, "rows": rows}


def render(results: dict) -> str:
    lines = ["# FastSimulator speed", "",
             "Phase B scenarios driven by R6 (greedy). *episode*: the closed-loop episode (simulator + R6 + "
             "bookkeeping); *sim only*: the same policies replayed on the simulator alone. Wall seconds per "
             "episode; speed-up = Python / fast.", ""]
    for prio in sorted(results):
        r = results[prio]
        rows = r["rows"]
        ep_py = sum(x["episode_python_s"] for x in rows)
        ep_fa = sum(x["episode_fast_s"] for x in rows)
        si_py = sum(x["sim_python_s"] for x in rows)
        si_fa = sum(x["sim_fast_s"] for x in rows)
        n = max(len(rows), 1)
        lines += [f"## {prio} priority ({r['priority_detail']})", "",
                  f"{r['machine']}, commit {r['commit'][:7]}, episode length "
                  f"{r['duration_s'] or 300:g} s, kernel warm-up {r['warm_up_s']:.2f} s (once per process).", "",
                  "| seed | kind | windows | max A queue | episode Python s | episode fast s | x | "
                  "sim only Python s | sim only fast s | x |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for x in rows:
            lines.append(f"| {x['seed']} | {x['kind']} | {x['windows']} | {x['queue_len_max']} | "
                         f"{x['episode_python_s']:.1f} | {x['episode_fast_s']:.3f} | "
                         f"{x['episode_python_s'] / max(x['episode_fast_s'], 1e-9):.0f} | "
                         f"{x['sim_python_s']:.1f} | {x['sim_fast_s']:.3f} | "
                         f"{x['sim_python_s'] / max(x['sim_fast_s'], 1e-9):.0f} |")
        lines += [f"| mean | | | | {ep_py / n:.1f} | {ep_fa / n:.3f} | {ep_py / max(ep_fa, 1e-9):.0f} | "
                  f"{si_py / n:.1f} | {si_fa / n:.3f} | {si_py / max(si_fa, 1e-9):.0f} |", ""]
        if not all(x["same_policies"] for x in rows):
            lines += ["**Warning:** R6 chose different policies on the two simulators in some episode.", ""]
    return "\n".join(lines)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="3100-3102", help='HISTORY seeds, e.g. "3100-3102"')
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--label", default="", help='"smoke" for a test run (allows a dirty tree and --duration)')
    ap.add_argument("--duration", type=float, default=None, help="episode length in s (smoke runs only)")
    a = ap.parse_args(argv)
    try:
        seeds = seed_ranges.parse(a.seeds)
        seed_ranges.check(seeds, "HISTORY")
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    if a.label not in runs.LABELS:
        raise SystemExit(f"refusing to run: unknown label {a.label!r}")
    smoke = a.label == "smoke"
    if a.duration is not None and not smoke:
        raise SystemExit("refusing to run: --duration needs --label smoke")
    if a.duration is not None and a.duration <= 0:
        raise SystemExit("refusing to run: --duration must be positive")
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    if smoke and out.resolve() == (ROOT / OUT_DIR).resolve():
        raise SystemExit("refusing to run: a smoke run never writes into results/")
    if not smoke and runs.tree_dirty():
        raise SystemExit("refusing to run: the working tree has uncommitted changes (commit, or --label smoke)")

    res = measure(seeds, a.duration)
    store = out / "speed.json"
    results = runs.read_json(store)
    results[res["priority"]] = res
    runs.write_json(store, results)
    text = render(results)
    runs.write_note(out / "speed.md", text)
    print(text, flush=True)
    return res


if __name__ == "__main__":
    main()
