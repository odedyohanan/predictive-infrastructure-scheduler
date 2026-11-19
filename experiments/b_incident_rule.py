"""Fix the incident rule from ordinary R6 history episodes, then describe the incidents it finds.

  .venv/bin/python experiments/b_incident_rule.py
  .venv/bin/python experiments/b_incident_rule.py --label smoke --store data/phase_b/timing_probe \\
      --smoke-dir data/phase_b/smoke_<name>

The quiet minutes of ordinary (no degradation) R6 history episodes set the run length: after the warm-up,
each episode is cut into 1-minute blocks (120 windows, a partial last block is dropped; a run belongs to
the block it starts in) and `k` is the
smallest k >= 2 such that at most 5 % of blocks hold a violation run of length >= k; `interval` = 4k. With
fewer than 20 such episodes, ordinary R6-with-interventions episodes are added (recorded in the note).
With that rule every history episode is labelled: incidents per degradation kind, building-up vs sudden,
chronic episodes. Fewer than 30 building-up incidents means MAGNITUDES should be raised once; the note
says so, the constants are not changed here.

Writes configs/phase_b/incident.json (keeping the lead-time keys already there) and
results/phase_b/incident_rule.md; a smoke run writes both into `--smoke-dir`. Analysis only: seconds, no
shards.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pis.phase_b import runs  # noqa: E402
from pis.phase_b import seeds as seed_ranges  # noqa: E402

DEFAULT_STORE = "data/phase_b/history"
CONFIG = "configs/phase_b/incident.json"
NOTE = "results/phase_b/incident_rule.md"
BLOCK = 120                  # 1 minute of 500 ms windows
LIMIT = 0.05
MIN_EPISODES = 20
MIN_BUILDING_UP = 30
MAJORITY = 0.5
CHRONIC = 0.5
WARMUP = 20
RULE_KEYS = ("k", "interval", "majority", "warmup", "chronic")


def block_run_lengths(viol, warmup: int = WARMUP, block: int = BLOCK) -> list[list[int]]:
    """Lengths of the violation runs per full block after the warm-up. A run belongs, whole, to the block it
    starts in (as an incident run would be judged); runs starting in the warm-up or in the partial last
    block are dropped."""
    v = np.asarray(viol, dtype=bool)
    n_blocks = (len(v) - warmup) // block
    out: list[list[int]] = [[] for _ in range(max(n_blocks, 0))]
    i = 0
    while i < len(v):
        if not v[i]:
            i += 1
            continue
        j = i
        while j < len(v) and v[j]:
            j += 1
        b = (i - warmup) // block
        if i >= warmup and b < n_blocks:
            out[b].append(j - i)
        i = j
    return out


def choose_k(run_lengths_per_block, limit: float = LIMIT) -> int:
    """Smallest k >= 2 such that the share of blocks holding a run of length >= k is at most `limit`."""
    tops = np.array([max(r, default=0) for r in run_lengths_per_block], dtype=int)
    if not len(tops):
        raise ValueError("no blocks to choose k from")
    k = 2
    while (tops >= k).mean() > limit:
        k += 1
    return k


def rule_config(k: int) -> dict:
    return {"k": int(k), "interval": 4 * int(k), "majority": MAJORITY, "warmup": WARMUP, "chronic": CHRONIC}


def write_config(path: Path, rule: dict) -> None:
    """Write the rule keys; other keys already in the file (the lead time) are kept."""
    cfg = runs.read_json(path)
    cfg.update(rule)
    runs.write_json(Path(path), cfg)


def _viol_active(store, seed: int, n: int):
    w = store.window_rows(seed, 0, n)
    return w.slo_v2_violation.to_numpy(bool), w.deg_active.to_numpy(bool)


def analyse(store, seeds: list[int], limit: float = LIMIT) -> dict:
    from pis.phase_b import incidents
    eps = store.episodes()
    eps = eps[eps.seed.isin(seeds)]
    if eps.empty:
        raise ValueError("no stored episodes among the seeds")
    quiet = eps[(eps.kind == "none") & (eps.source == "r6")]
    sources = ["r6"]
    if len(quiet) < MIN_EPISODES:
        quiet = eps[(eps.kind == "none") & eps.source.isin(["r6", "r6_interventions"])]
        sources.append("r6_interventions")
    blocks: list[list[int]] = []
    for e in quiet.itertuples():
        blocks += block_run_lengths(_viol_active(store, int(e.seed), int(e.windows))[0])
    k = choose_k(blocks, limit)
    tops = np.array([max(r, default=0) for r in blocks])
    share_by_k = {j: float((tops >= j).mean()) for j in range(1, max(k, int(tops.max(initial=0))) + 2)}
    rule = incidents.IncidentRule(k, 4 * k, MAJORITY)
    per_kind = defaultdict(lambda: Counter())
    chronic = Counter()
    n_kind = Counter()
    for e in eps.itertuples():
        v, a = _viol_active(store, int(e.seed), int(e.windows))
        lab = incidents.label_series(v, a, rule, WARMUP)
        n_kind[e.kind] += 1
        chronic[e.kind] += incidents.is_chronic(lab, int(e.windows))
        for inc in lab.incidents:
            per_kind[e.kind][inc.cause] += 1
    building = sum(c["building_up"] for c in per_kind.values())
    return {"k": k, "rule": rule_config(k), "quiet_episodes": len(quiet), "quiet_sources": sources,
            "blocks": len(blocks), "share_by_k": share_by_k, "per_kind": {k_: dict(c) for k_, c in per_kind.items()},
            "chronic": dict(chronic), "episodes_by_kind": dict(n_kind), "episodes": len(eps),
            "building_up": building, "raise_magnitudes": building < MIN_BUILDING_UP}


def note_text(r: dict, limit: float) -> str:
    rule = r["rule"]
    lines = ["# Incident rule", "",
             f"Quiet minutes: {r['blocks']} one-minute blocks (after a {WARMUP}-window warm-up) from "
             f"{r['quiet_episodes']} ordinary episodes driven by {' + '.join(r['quiet_sources'])}"
             + (" (fewer than 20 ordinary R6 episodes: R6-with-interventions episodes added)"
                if len(r["quiet_sources"]) > 1 else "") + ".", "",
             "| k | blocks with a run >= k |", "|---|---|"]
    lines += [f"| {j} | {s:.1%} |" for j, s in r["share_by_k"].items()]
    lines += ["", f"Chosen: k = {rule['k']} (smallest k >= 2 with at most {limit:.0%} of quiet blocks), "
              f"interval = {rule['interval']}, majority {rule['majority']}, warm-up {rule['warmup']}, "
              f"chronic share {rule['chronic']}.", "",
              f"## Incidents with this rule ({r['episodes']} history episodes)", "",
              "| kind | episodes | building-up | sudden | chronic episodes |", "|---|---|---|---|---|"]
    tot_b = tot_s = 0
    for kind in sorted(r["episodes_by_kind"]):
        c = r["per_kind"].get(kind, {})
        b, s = c.get("building_up", 0), c.get("sudden", 0)
        tot_b, tot_s = tot_b + b, tot_s + s
        lines.append(f"| {kind} | {r['episodes_by_kind'][kind]} | {b} | {s} | {r['chronic'].get(kind, 0)} |")
    tot = tot_b + tot_s
    n_chronic = sum(r["chronic"].values())
    lines += ["", f"Incidents: {tot}; building-up {tot_b} ({tot_b / tot:.0%} of incidents)" if tot else
              "", f"Chronic episodes: {n_chronic} of {r['episodes']} ({n_chronic / r['episodes']:.0%}).", "",
              f"Raise MAGNITUDES once (fewer than {MIN_BUILDING_UP} building-up incidents): "
              f"{'YES' if r['raise_magnitudes'] else 'no'} ({r['building_up']} building-up incidents)."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="3100-3399")
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--label", default="")
    ap.add_argument("--smoke-dir", default=None)
    a = ap.parse_args(argv)
    if a.label not in runs.LABELS:
        raise SystemExit(f"refusing to run: unknown label {a.label!r}")
    smoke = a.label == "smoke"
    try:
        seeds = seed_ranges.parse(a.seeds)
        seed_ranges.check(seeds, "HISTORY")
        sdir = runs.smoke_dir(a.smoke_dir) if smoke else None
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    if not smoke:
        if a.smoke_dir is not None:
            raise SystemExit("refusing to run: --smoke-dir needs --label smoke")
        if runs.tree_dirty(ignore=(CONFIG, NOTE)):
            raise SystemExit("refusing to run: the working tree has uncommitted changes (commit, or --label smoke)")
    from pis.phase_b.history import Store
    try:
        r = analyse(Store(a.store), seeds)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    config = (sdir / "incident.json") if smoke else ROOT / CONFIG
    note = (sdir / "incident_rule.md") if smoke else ROOT / NOTE
    write_config(config, r["rule"])
    runs.write_note(note, note_text(r, LIMIT))
    print(f"k = {r['k']}, interval = {4 * r['k']} from {r['blocks']} quiet blocks of {r['quiet_episodes']} "
          f"episodes; building-up incidents {r['building_up']}; raise MAGNITUDES: "
          f"{'yes' if r['raise_magnitudes'] else 'no'}", flush=True)
    return r


if __name__ == "__main__":
    main()
