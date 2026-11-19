"""Minimum useful lead time: how many windows an intervention needs before it measurably lowers A's backlog.

  .venv/bin/python experiments/b_lead_time.py --procs 4
  .venv/bin/python experiments/b_lead_time.py --label smoke --seeds 3104,3111 --states 1 --reseeds 3 \\
      --horizon 6 --store data/phase_b/timing_probe --smoke-dir data/phase_b/smoke_<name> \\
      --incident data/phase_b/smoke_<name>/incident.json

States: in each calibration episode the first window (after the warm-up, with room for the horizon) where
the early signal (slope of A's overdue backlog) has been positive for 3 windows; at most 40 states, one
per episode, in seed order. At each state every named intervention whose "after" policy differs from the
policy in force is held for the horizon, against R6 continuing, with paired reseeds
(13_000_000 + seed*1000 + j) on the real simulator. The effect window of an intervention is the first
window where the paired 95 % CI of A's overdue backlog (intervention - R6) lies entirely below zero; a
state's effect window is the earliest over its interventions (states without any effect are reported, not
used). min_lead_windows = median over states, rounded up; H = max(20, 2 * min_lead_windows). Both are
added to configs/phase_b/incident.json; note results/phase_b/lead_time.md. Shards of 5 states.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pis.phase_b import runs  # noqa: E402
from pis.phase_b import seeds as seed_ranges  # noqa: E402

DEFAULT_STORE = "data/phase_b/history"
CONFIG = "configs/phase_b/incident.json"
NOTE = "results/phase_b/lead_time.md"
DEFAULT_OUT = "results/raw/phase_b/lead_time"
LEAD_TAG = 13_000_000
MAX_STATES = 40
RESEEDS = 8
HORIZON = 40
RISING = 3                  # windows of positive early signal
SHARD_SIZE = 5
MIN_H = 20
RESULT_FILE = "states.json"


def early_signal_series(q) -> np.ndarray:
    from pis.phase_b import incidents
    return incidents.early_signal([{"tel": {"a": {"queue_overdue": float(v)}}} for v in q])


def lead_state(q, warmup: int, last: int) -> int | None:
    """First window t with the early signal > 0 in t-2..t, all after the warm-up, and t <= last."""
    es = early_signal_series(q)
    for t in range(warmup + RISING - 1, min(last, len(es) - 1) + 1):
        if np.all(es[t - RISING + 1:t + 1] > 0):
            return t
    return None


def effect_window(diffs, persist: int = 1) -> int | None:
    """First window (1-based, counted from the intervention's first window) whose paired 95 % CI of
    `diffs[w, j]` (intervention - R6, reseed j) lies entirely below zero, and stays so for `persist`
    consecutive windows; None if there is none."""
    from pis.stats import paired_diff
    diffs = np.asarray(diffs, dtype=float)
    zero = np.zeros(diffs.shape[1])
    sig = [paired_diff(row, zero)["hi"] < 0 for row in diffs]       # NaN (one reseed) is never below
    for w in range(len(sig) - persist + 1):
        if all(sig[w:w + persist]):
            return w + 1
    return None


def min_lead(effects) -> int:
    if not len(effects):
        raise ValueError("no state with an effect")
    return int(math.ceil(float(np.median(effects))))


def horizon_for(lead: int) -> int:
    return max(MIN_H, 2 * int(lead))


# ------------------------------------------------------------------ worker
def run_shard(shard, shard_dir: Path, params: dict) -> None:
    import torch
    torch.set_num_threads(1)
    from pis.adapters import initial_policy
    from pis.candidates import policy_key
    from pis.phase_b import mc_labels
    from pis.probe import named_interventions
    out = []
    h, n = params["horizon"], params["reseeds"]
    for seed, t in shard.items:
        t0 = time.time()
        st = mc_labels.replay_to(int(seed), int(t))
        replay_s = time.time() - t0
        current = st.records[-2]["policy"] if t > 0 else initial_policy()
        arms = {name: after for name, (_, after) in named_interventions(current).items()
                if policy_key(after) != policy_key(current)}
        reseeds = [LEAD_TAG + int(seed) * 1000 + j for j in range(n)]

        def overdue(hold, r):
            return [rec["tel"]["a"]["queue_overdue"] for rec in mc_labels.continue_branch(st, h, r, hold)]

        base = np.array([overdue(None, r) for r in reseeds], dtype=float)          # [j, w]
        effects = {}
        for name, after in sorted(arms.items()):
            arm = np.array([overdue(after, r) for r in reseeds], dtype=float)
            effects[name] = effect_window((arm - base).T)
        found = [w for w in effects.values() if w is not None]
        out.append({"episode": int(seed), "window": int(t), "effects": effects,
                    "skipped": sorted(set(named_interventions(current)) - set(arms)),
                    "earliest": min(found) if found else None, "replay_s": replay_s,
                    "branch_windows": (len(arms) + 1) * n * h, "branch_s": time.time() - t0 - replay_s})
    (shard_dir / RESULT_FILE).write_text(json.dumps(out, indent=2))


def pick_states(store, seeds: list[int], warmup: int, horizon: int, max_states: int = MAX_STATES):
    from pis.phase_b import mc_labels
    eps = store.episodes()
    eps = eps[eps.seed.isin(seeds) & (eps.source == mc_labels.SOURCE)]
    out = []
    for e in eps.sort_values("seed").itertuples():
        q = store.window_rows(int(e.seed), 0, int(e.windows)).a_queue_overdue.to_numpy(float)
        t = lead_state(q, warmup, int(e.windows) - 1 - horizon)
        if t is not None:
            out.append([int(e.seed), t])
        if len(out) == max_states:
            break
    return out, len(eps)


def collect(out: Path) -> list[dict]:
    from pis.shards import collect as done_dirs
    return [r for d in done_dirs(out) for r in json.loads((d / RESULT_FILE).read_text())]


def note_text(cfg: dict, results: list[dict], lead: int | None, h: int | None) -> str:
    lines = ["# Minimum useful lead time", "",
             f"States: {len(results)} (first window with a rising overdue backlog for {RISING} windows, one per "
             f"episode, {cfg['episodes']} R6 episodes of seeds {cfg['seeds'][0]}-{cfg['seeds'][-1]}). "
             f"Interventions held for {cfg['horizon']} windows vs R6 continuing, {cfg['reseeds']} paired reseeds; "
             "effect = first window whose paired 95 % CI of A's overdue backlog (intervention - R6) is below 0.",
             "", "| episode | window | " + " | ".join(cfg["interventions"]) + " | earliest |",
             "|---|---|" + "---|" * (len(cfg["interventions"]) + 1)]
    for r in sorted(results, key=lambda r: r["episode"]):
        cells = ["skipped" if n not in r["effects"] else "-" if r["effects"][n] is None else str(r["effects"][n])
                 for n in cfg["interventions"]]
        lines.append(f"| {r['episode']} | {r['window']} | " + " | ".join(cells) + f" | {r['earliest'] or '-'} |")
    none = [r for r in results if r["earliest"] is None]
    lines += ["", f"States without any effect: {len(none)} (reported, not used)."]
    if lead is None:
        lines.append("No state showed an effect: the minimum lead time is not set.")
    else:
        lines += [f"min_lead_windows = {lead} (median over {len(results) - len(none)} states, rounded up); "
                  f"H = max({MIN_H}, 2 x {lead}) = {h} windows."]
    bw = sum(r["branch_windows"] for r in results)
    if bw:
        lines.append(f"Timing: {1000 * sum(r['branch_s'] for r in results) / bw:.1f} ms per branch-window, "
                     f"replay {np.mean([r['replay_s'] for r in results]):.1f} s per state.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="3400-3449")
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--incident", default=CONFIG)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--states", type=int, default=MAX_STATES, help="smoke runs only")
    ap.add_argument("--reseeds", type=int, default=RESEEDS, help="smoke runs only")
    ap.add_argument("--horizon", type=int, default=HORIZON, help="smoke runs only")
    ap.add_argument("--label", default="")
    ap.add_argument("--smoke-dir", default=None)
    a = ap.parse_args(argv)
    if a.label not in runs.LABELS:
        raise SystemExit(f"refusing to run: unknown label {a.label!r}")
    smoke = a.label == "smoke"
    try:
        seeds = seed_ranges.parse(a.seeds)
        runs.check_seeds(seeds, ("CALIBRATION",) + (("HISTORY",) if smoke else ()))
        sdir = runs.smoke_dir(a.smoke_dir) if smoke else None
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    if not smoke:
        if (a.states, a.reseeds, a.horizon) != (MAX_STATES, RESEEDS, HORIZON) or a.smoke_dir is not None:
            raise SystemExit("refusing to run: --states, --reseeds, --horizon and --smoke-dir need --label smoke")
        if Path(a.incident).resolve() != (ROOT / CONFIG).resolve():
            raise SystemExit(f"refusing to run: a real run updates {CONFIG}")
        if runs.tree_dirty(ignore=(CONFIG, NOTE, "results/phase_b/incident_rule.md")):
            raise SystemExit("refusing to run: the working tree has uncommitted changes (commit, or --label smoke)")
    inc_path = Path(a.incident)
    if smoke and sdir not in inc_path.resolve().parents:
        raise SystemExit("refusing to run: a smoke run updates an --incident file inside --smoke-dir only")
    inc = runs.read_json(inc_path)
    if "warmup" not in inc:
        raise SystemExit(f"refusing to run: {a.incident} has no incident rule yet (run b_incident_rule.py first)")
    from pis.phase_b.history import Store
    from pis.probe import INTERVENTIONS
    from pis.shards import Shard, run_shards
    states, n_eps = pick_states(Store(a.store), seeds, int(inc["warmup"]), a.horizon, a.states)
    if not states:
        raise SystemExit("refusing to run: no state with a rising backlog among the seeds")
    out = (sdir / "lead_time") if smoke else ROOT / DEFAULT_OUT
    note = (sdir / "lead_time.md") if smoke else ROOT / NOTE
    cfg = {"seeds": seeds, "label": a.label, "states": states, "reseeds": a.reseeds, "horizon": a.horizon,
           "warmup": int(inc["warmup"]), "episodes": n_eps, "interventions": list(INTERVENTIONS)}
    shards = [Shard(f"s{i // SHARD_SIZE:04d}", states[i:i + SHARD_SIZE]) for i in range(0, len(states), SHARD_SIZE)]
    t0 = time.time()
    print(f"{len(states)} states from {n_eps} episodes, {len(shards)} shards, {a.procs} processes", flush=True)
    try:
        res = run_shards(out, shards, partial(run_shard, params=cfg), cfg, a.procs)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    print(f"shards done {len(res['done'])}, skipped {len(res['skipped'])}, failed {res['failed']} "
          f"in {time.time() - t0:.0f} s", flush=True)
    results = collect(out)
    if len(results) != len(states):
        raise SystemExit("unfinished shards: rerun the same command")
    effects = [r["earliest"] for r in results if r["earliest"] is not None]
    lead = min_lead(effects) if effects else None
    h = horizon_for(lead) if lead is not None else None
    if lead is not None:
        cfg_now = runs.read_json(inc_path)
        cfg_now.update({"min_lead_windows": lead, "H": h})
        runs.write_json(inc_path, cfg_now)
    runs.write_note(note, note_text(cfg, results, lead, h))
    bw = sum(r["branch_windows"] for r in results)
    per_bw = sum(r["branch_s"] for r in results) / bw
    print(f"min_lead_windows = {lead}, H = {h}; {len(effects)} of {len(results)} states with an effect; "
          f"{1000 * per_bw:.1f} ms per branch-window", flush=True)
    return {"min_lead_windows": lead, "H": h, "results": results, "s_per_branch_window": per_bw}


if __name__ == "__main__":
    main()
