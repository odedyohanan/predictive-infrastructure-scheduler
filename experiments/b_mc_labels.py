"""Monte Carlo risk labels on calibration and selection states, by branching the real simulator.

  .venv/bin/python experiments/b_mc_labels.py --procs 4
  .venv/bin/python experiments/b_mc_labels.py --label smoke --seeds 3104,3111 --states 4 --branches 2 \\
      --horizon 6 --smoke-dir data/phase_b/smoke_<name> --incident data/phase_b/smoke_<name>/incident.json

States are sampled from the stored R6 episodes of the seeds (half from degraded episodes in the 120 s
before an incident onset, half uniform), replayed deterministically and branched: R6 continues for H
windows in each branch, and the risk is the share of branches where a new incident starts. Shards of 20
states (`pis.shards.run_shards`): rerunning skips finished shards. When the plain estimate is 0 in more than
80 % of the states, the states are labelled again by multilevel splitting on A's overdue backlog and those
labels are the final ones (recorded in the note).

Real runs: calibration and selection seeds, clean tree, the frozen rule and H from
configs/phase_b/incident.json; output data/phase_b/mc_labels/ and the note results/phase_b/mc_labels.md.
Smoke runs (`--label smoke`) may use R6 history episodes, smaller sizes and `--horizon`, allow a dirty tree,
and write only into `--smoke-dir`. Long runs go through scripts/gentle.sh.
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pis.phase_b import runs  # noqa: E402
from pis.phase_b import seeds as seed_ranges  # noqa: E402

DEFAULT_STORE = "data/phase_b/history"
DEFAULT_INCIDENT = "configs/phase_b/incident.json"
DEFAULT_OUT = "data/phase_b/mc_labels"
NOTE = "results/phase_b/mc_labels.md"
SHARD_SIZE = 20
ZERO_SHARE_FOR_RESTART = 0.8
LEVEL_QUANTILES = (0.5, 0.8, 0.95)   # of positive overdue backlogs: the splitting levels
RESTART = {"n_roots": 16, "split": 2, "max_paths": 128}
LABELS_FILE = "labels.parquet"


def check_seeds(seeds: list[int], smoke: bool) -> None:
    """Calibration and selection seeds; a smoke run may also use history seeds."""
    runs.check_seeds(seeds, ("CALIBRATION", "SELECTION") + (("HISTORY",) if smoke else ()))


# ------------------------------------------------------------------ worker
def run_shard(shard, shard_dir: Path, params: dict) -> None:
    """Label every state of the shard; one replay per episode, branches at each of its states."""
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    from pis.phase_b import incidents, mc_labels
    rule = incidents.IncidentRule(params["k"], params["interval"], params["majority"])
    h, warmup = params["horizon"], params["warmup"]
    rows: list[dict] = []
    by_ep: dict[int, list] = {}
    for ep, w, stratum in shard.items:
        by_ep.setdefault(int(ep), []).append((int(w), stratum))
    for ep, items in sorted(by_ep.items()):
        stratum_of = dict(items)
        t_rep = time.time()
        clock = {"branch_s": 0.0}

        def visit(state, ep=ep, stratum_of=stratum_of, clock=clock):
            t0 = time.time()
            if params["method"] == "plain":
                risk = mc_labels.branch_risk(state, rule, h, params["branches"],
                                             mc_labels.branch_reseed(ep), warmup)
                n = params["branches"]
            else:
                risk, n = mc_labels.restart_risk(state, rule, h, params["levels"], RESTART["n_roots"],
                                                 RESTART["split"], mc_labels.restart_reseed(ep), warmup,
                                                 RESTART["max_paths"])
            dt = time.time() - t0
            clock["branch_s"] += dt
            v, a = mc_labels.series(state.records)
            now = bool(incidents.label_series(v, a, rule, warmup).in_incident[-1])
            rows.append({"episode": ep, "window": state.t, "risk": float(risk), "n": int(n),
                         "method": params["method"], "stratum": stratum_of[state.t],
                         "kind": state.dsim.scenario.kind, "in_incident": now,
                         # upper bound for splitting: copies start part-way through the branch
                         "branch_windows": int(n) * mc_labels.branch_windows(h, rule), "branch_s": dt})

        mc_labels.replay_visit(ep, list(stratum_of), visit)
        replay_s = time.time() - t_rep - clock["branch_s"]
        for r in rows:
            if r["episode"] == ep:
                r["replay_s"] = replay_s / len(items)
    pd.DataFrame(rows).to_parquet(shard_dir / LABELS_FILE, index=False)


# Label columns: episode, window (t), risk, n (branches, or paths for splitting), method, stratum, kind,
# in_incident (in incident by the rule applied to windows 0..t only: a run in progress at t that becomes an
# incident later is not marked), branch_windows, branch_s, replay_s (timing).


def make_shards(states):
    from pis.shards import Shard
    return [Shard(f"s{i // SHARD_SIZE:04d}", [list(s) for s in states[i:i + SHARD_SIZE]])
            for i in range(0, len(states), SHARD_SIZE)]


def collect_labels(out: Path):
    import pandas as pd
    from pis.shards import collect
    parts = [pd.read_parquet(d / LABELS_FILE) for d in collect(out)]
    return pd.concat(parts, ignore_index=True).sort_values(["episode", "window"]) if parts else None


# ------------------------------------------------------------------ inputs
def episode_table(store, seeds: list[int], rule, warmup: int) -> list[dict]:
    """Stored R6 episodes of the seeds with their counted incident onsets."""
    from pis.phase_b import incidents, mc_labels
    eps = store.episodes()
    eps = eps[eps.seed.isin(seeds)]
    missing = sorted(set(seeds) - set(int(s) for s in eps.seed))
    if missing:
        raise ValueError(f"no stored episode for seeds {missing[:10]}")
    out = []
    for e in eps.itertuples():
        if e.source != mc_labels.SOURCE:
            continue
        mc_labels.check_replayable(int(e.seed))
        w = store.window_rows(int(e.seed), 0, int(e.windows))
        lab = incidents.label_series(w.slo_v2_violation.to_numpy(bool), w.deg_active.to_numpy(bool), rule, warmup)
        out.append({"episode": int(e.seed), "degraded": e.kind != "none", "kind": e.kind,
                    "onsets": [i.onset for i in lab.incidents], "windows": int(e.windows)})
    if not out:
        raise ValueError("no R6 episodes among the seeds")
    return out


def overdue_levels(store, seeds: list[int]) -> list[int]:
    import numpy as np
    q = store.state_matrix(seeds, ["a_queue_overdue"])[:, 0]
    q = q[q > 0]
    if not len(q):
        return [1]
    return sorted({max(1, int(np.quantile(q, p))) for p in LEVEL_QUANTILES})


# ------------------------------------------------------------------ report
def summarize(df, horizon: int, full_states: int = 1200, full_branches: int = 32) -> dict:
    """Label statistics and timing; the projection is for `full_states` x `full_branches` at `horizon`."""
    zero = float((df.risk == 0).mean())
    bw = float(df.branch_windows.sum())
    per_bw = float(df.branch_s.sum()) / bw if bw else float("nan")
    per_replay_state = float(df.replay_s.mean())
    full_s = full_states * (full_branches * horizon * per_bw + per_replay_state)
    return {"states": len(df), "zero_share": zero, "mean_risk": float(df.risk.mean()),
            "s_per_branch_window": per_bw, "replay_s_per_state": per_replay_state,
            "projected_full_s_one_proc": full_s}


def note_text(cfg: dict, plain: dict, chosen: str, final, restart: dict | None, forced: bool = False) -> str:
    lines = ["# Monte Carlo risk labels", "",
             f"Seeds: {cfg['seeds'][0]}-{cfg['seeds'][-1]} ({len(cfg['seeds'])} seeds), R6 episodes only. "
             f"Rule k={cfg['k']}, interval={cfg['interval']}, majority={cfg['majority']}, warm-up {cfg['warmup']}; "
             f"horizon H={cfg['horizon']} windows.",
             f"States: {plain['states']} (shards of {SHARD_SIZE}); {cfg['branches']} branches each, reseed "
             f"12_000_000 + seed*1000 + j; R6 greedy continues in every branch for H + k - 1 windows (onsets "
             "counted in (t, t + H]); reseeding redraws arrivals, service times and A's future bursts.",
             "All states of one episode reuse the same reseed streams, so labels within an episode are slightly "
             "correlated.", "",
             "## Plain branching", "",
             f"- risk = 0 in {plain['zero_share']:.1%} of states; mean risk {plain['mean_risk']:.3f}",
             f"- {plain['s_per_branch_window'] * 1000:.1f} ms per branch-window, replay "
             f"{plain['replay_s_per_state']:.1f} s per state", "",
             "## Estimator", "",
             f"Multilevel splitting is used when the plain estimate is 0 in more than "
             f"{ZERO_SHARE_FOR_RESTART:.0%} of states: {'yes' if chosen == 'restart' else 'no'} "
             f"(final labels: {chosen}{'; forced by --method, no plain pass' if forced else ''})."]
    if restart is not None:
        lines += [f"- levels on A's overdue backlog {cfg['levels']}, {RESTART['n_roots']} roots, split "
                  f"{RESTART['split']}, at most {RESTART['max_paths']} paths; risk = 0 in "
                  f"{restart['zero_share']:.1%} of states, mean risk {restart['mean_risk']:.3f}"]
    lines += ["", "## Labels by stratum and kind", "", "| stratum | kind | states | mean risk | risk 0 |",
              "|---|---|---|---|---|"]
    for (s, k), g in final.groupby(["stratum", "kind"]):
        lines.append(f"| {s} | {k} | {len(g)} | {g.risk.mean():.3f} | {(g.risk == 0).mean():.0%} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="3400-3499")
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--incident", default=DEFAULT_INCIDENT)
    ap.add_argument("--states", type=int, default=1200)
    ap.add_argument("--branches", type=int, default=32)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--method", choices=("auto", "plain", "restart"), default="auto")
    ap.add_argument("--horizon", type=int, default=None, help="windows; smoke runs only (real runs use H)")
    ap.add_argument("--label", default="")
    ap.add_argument("--smoke-dir", default=None)
    a = ap.parse_args(argv)
    if a.label not in runs.LABELS:
        raise SystemExit(f"refusing to run: unknown label {a.label!r}")
    smoke = a.label == "smoke"
    try:
        seeds = seed_ranges.parse(a.seeds)
        check_seeds(seeds, smoke)
        sdir = runs.smoke_dir(a.smoke_dir) if smoke else None
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("refusing to run: repeated seeds")
    if not smoke:
        if a.horizon is not None or a.smoke_dir is not None or a.method != "auto":
            raise SystemExit("refusing to run: --horizon, --method and --smoke-dir need --label smoke")
        if Path(a.incident).resolve() != (ROOT / DEFAULT_INCIDENT).resolve():
            raise SystemExit(f"refusing to run: a real run reads the frozen {DEFAULT_INCIDENT}")
        if runs.tree_dirty():
            raise SystemExit("refusing to run: the working tree has uncommitted changes (commit, or --label smoke)")
    inc = runs.read_json(Path(a.incident))
    need = ["k", "interval", "majority", "warmup"] + ([] if a.horizon else ["H"])
    if any(k not in inc for k in need):
        raise SystemExit(f"refusing to run: {a.incident} lacks {[k for k in need if k not in inc]}")
    horizon = a.horizon or int(inc["H"])

    from pis.phase_b import incidents, mc_labels
    from pis.phase_b.history import Store
    from pis.shards import run_shards
    rule = incidents.IncidentRule(int(inc["k"]), int(inc["interval"]), float(inc["majority"]))
    store = Store(a.store)
    try:
        eps = episode_table(store, seeds, rule, int(inc["warmup"]))
        states = mc_labels.sample_states(eps, a.states, horizon, int(inc["warmup"]), tail=rule.k - 1)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    out = (sdir / "mc_labels") if smoke else ROOT / DEFAULT_OUT
    note = (sdir / "mc_labels.md") if smoke else ROOT / NOTE
    cfg = {"seeds": seeds, "label": a.label, "k": rule.k, "interval": rule.interval, "majority": rule.majority,
           "warmup": int(inc["warmup"]), "horizon": horizon, "branches": a.branches, "states": a.states,
           "shard_size": SHARD_SIZE, "store_episodes": sorted(e["episode"] for e in eps)}
    shards = make_shards(states)
    t0 = time.time()
    print(f"{len(states)} states from {len(eps)} episodes, {len(shards)} shards, {a.procs} processes", flush=True)
    method = "restart" if a.method == "restart" else "plain"
    try:
        res = run_shards(out, shards, partial(run_shard, params={**cfg, "method": "plain"}),
                         {**cfg, "method": "plain"}, a.procs) if method == "plain" else None
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    restart_sum = None
    plain_df = collect_labels(out) if method == "plain" else None
    if method == "plain":
        print(f"plain: done {len(res['done'])}, skipped {len(res['skipped'])}, failed {res['failed']} "
              f"in {time.time() - t0:.0f} s", flush=True)
        if plain_df is None or len(plain_df) != len(states):
            raise SystemExit("unfinished shards: rerun the same command")
        plain = summarize(plain_df, horizon)
        if a.method == "auto" and plain["zero_share"] > ZERO_SHARE_FOR_RESTART:
            method = "restart"
    final = plain_df
    if method == "restart":
        levels = overdue_levels(store, sorted({s[0] for s in states}))
        rcfg = {**cfg, "method": "restart", "levels": levels, "restart": RESTART}
        cfg["levels"] = levels
        rres = run_shards(out / "restart", shards, partial(run_shard, params=rcfg), rcfg, a.procs)
        final = collect_labels(out / "restart")
        if final is None or len(final) != len(states):
            raise SystemExit(f"unfinished restart shards {rres['failed']}: rerun the same command")
        restart_sum = summarize(final, horizon)
        if plain_df is None:
            plain_df, plain = final, restart_sum
    final.to_parquet(out / LABELS_FILE, index=False)
    runs.write_note(note, note_text(cfg, plain, method, final, restart_sum, a.method != "auto"))
    print(f"labels: {len(final)} states, method {method}; risk 0 in {(final.risk == 0).mean():.1%}", flush=True)
    print(f"timing: {plain['s_per_branch_window'] * 1000:.1f} ms per branch-window, "
          f"replay {plain['replay_s_per_state']:.1f} s per state; full 1200 x 32 x H={horizon} projected "
          f"{plain['projected_full_s_one_proc'] / 3600:.1f} h on one process", flush=True)
    return {"plain": plain, "restart": restart_sum, "method": method, "out": str(out)}


if __name__ == "__main__":
    main()
