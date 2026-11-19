"""Step 0 gate run: does the world model rank the consequences of candidate policies like the simulator?

  .venv/bin/python experiments/step0_run.py --model worldmodel:models/gru_mix_bounded.pt --attempt 1 --procs 5

Defaults are the frozen run sizes (DEFAULTS); the thresholds are `pis.probe.GATE`. Both, at the recorded git
HEAD, are the frozen record. A run refuses to start when pis/ or experiments/ have uncommitted changes, when
the attempt is not 1 or 2, when that attempt already has a section in the output, or when the scenario seeds
leave 1100-1149 (label "smoke" lifts the dirty-code, attempt and seed checks). Any size other than the
defaults is stamped PROTOCOL DEVIATION. Appends a section to `--out` (default results/step0/step0.md) and
writes the per-state numbers to results/raw/step0/attempt<N>.json (gitignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pis.config import resolve, setup_imports  # noqa: E402
from pis.privacy import safe_path, scrub  # noqa: E402
from pis.probe import (GATE, INTERVENTIONS, GRUPredictor, HeuristicPredictor,  # noqa: E402
                       LogProbPredictor, PersistencePredictor, gate, probe_many, sample_states,
                       summarise)

setup_imports()

from worldmodel.model import WorldModel  # noqa: E402

NAMES = ("model", "heuristic", "logprob", "action_blind", "persistence")
SCORE_ONLY = ("logprob",)
DEFAULTS = {"states": 500, "k": 10, "n_seeds": 8, "h2": 4, "seeds": "1100-1149", "sample_seed": 0}
PROBE_SEEDS = range(1100, 1150)
MAX_ATTEMPTS = 2


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def parse_seeds(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def check_run(args: dict, existing: str, dirty: bool) -> tuple[list[str], list[str]]:
    """(errors that refuse the run, names of sizes that deviate from DEFAULTS)."""
    smoke = args.get("label") == "smoke"
    errors = []
    if not smoke:
        if dirty:
            errors.append("pis/ or experiments/ have uncommitted changes")
        if not 1 <= args["attempt"] <= MAX_ATTEMPTS:
            errors.append(f"attempt must be 1..{MAX_ATTEMPTS}")
        if not set(parse_seeds(args["seeds"])) <= set(PROBE_SEEDS):
            errors.append(f"seeds must lie in {PROBE_SEEDS.start}-{PROBE_SEEDS.stop - 1}")
    if re.search(rf"^## Attempt {args['attempt']}\b", existing, flags=re.M):
        errors.append(f"attempt {args['attempt']} already has a section in the output")
    deviations = [k for k in ("states", "k", "n_seeds", "h2") if args[k] != DEFAULTS[k]]
    return errors, deviations


def _f(x, nd=3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(summary: dict, verdict: dict, meta: dict, horizons) -> str:
    stamp = " (PROTOCOL DEVIATION: " + ", ".join(meta["deviations"]) + ")" if meta.get("deviations") else ""
    L = [f"## Attempt {meta['attempt']}{' (' + meta['label'] + ')' if meta.get('label') else ''}: "
         f"**{verdict['verdict']}{stamp}**", ""]
    L += [f"- {k}: {v}" for k, v in meta.items() if k not in ("attempt", "label", "deviations")]
    L += ["", "### Gate (h = 1)", "", "| criterion | value | pass |", "|---|---|---|"]
    for name, (ok, val) in verdict["checks"].items():
        L.append(f"| {name} | {val} | {'yes' if ok else 'no'} |")
    L += ["", "Bootstrap 95% lower bound of (model - control) median Spearman, reported, not gated: "
          + ", ".join(f"{c} {_f(v)}" for c, v in verdict["lower_bounds_vs_controls"].items()), "",
          "Controls: heuristic = resource-ratio heuristic (sees the action); logprob = RL scheduler "
          "log-probability ranking (score only, no interventions); action_blind and persistence are leak "
          "checks that cannot see the action (0 by construction).", ""]
    for hh in horizons:
        h = f"h{hh}"
        L += [f"### Horizon {hh}", "",
              "| predictor | median Spearman (resolvable) | median Spearman (all) | resolvable states "
              "| pairwise sign acc | resolvable pairs | reward MAE |", "|---|---|---|---|---|---|---|"]
        for n in NAMES:
            m = summary[n][h]
            L.append(f"| {n} | {_f(m['median_spearman'])} | {_f(m['median_spearman_all'])} | "
                     f"{m['n_resolvable_states']}/{m['n_states']} | {_f(m['pairwise_sign_acc'])} | "
                     f"{m['n_resolvable_pairs']} | {'n/a' if n in SCORE_ONLY else _f(m['reward_mae'], 4)} |")
        L.append("")
    L += ["### Named interventions (h = 1)", "",
          "| predictor | intervention | sign acc | cases | state majority acc | A p99 | B rate | C work |",
          "|---|---|---|---|---|---|---|---|"]
    for n in NAMES:
        if n in SCORE_ONLY:
            continue
        for i in INTERVENTIONS:
            m = summary[n]["h1"]["interventions"][i]
            pq = " | ".join(f"{_f(a)} ({c})" for a, c in m["per_quantity"].values())
            L.append(f"| {n} | {i} | {_f(m['sign_acc'])} | {m['n_cases']} | "
                     f"{_f(m['state_majority_acc'])} | {pq} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="worldmodel:models/gru_mix_bounded.pt")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--label", default="", help='"smoke" for test runs')
    ap.add_argument("--states", type=int, default=DEFAULTS["states"])
    ap.add_argument("--k", type=int, default=DEFAULTS["k"])
    ap.add_argument("--n-seeds", type=int, default=DEFAULTS["n_seeds"])
    ap.add_argument("--seeds", default=DEFAULTS["seeds"])
    ap.add_argument("--h2", type=int, default=DEFAULTS["h2"])
    ap.add_argument("--sample-seed", type=int, default=DEFAULTS["sample_seed"])
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--out", default="results/step0/step0.md")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--protocol", default=None, help="optional protocol file; its sha256 is recorded")
    ap.add_argument("--protocol-sha256", default=None, help="optional protocol hash to record")
    a = ap.parse_args()
    out = Path(a.out)
    existing = out.read_text() if out.exists() else ""
    dirty = bool(_git("status", "--porcelain", "--", "pis", "experiments"))
    errors, deviations = check_run(vars(a), existing, dirty)
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))
    raw = Path(a.raw or f"results/raw/step0/attempt{a.attempt}{'_' + a.label if a.label else ''}.json")
    proto_sha = a.protocol_sha256
    if a.protocol:
        proto_sha = hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest()
    path = resolve(a.model)
    model = WorldModel.load(str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    predictors = {"model": GRUPredictor(model), "heuristic": HeuristicPredictor(),
                  "logprob": LogProbPredictor(),
                  "action_blind": GRUPredictor(model, action_blind=True),
                  "persistence": PersistencePredictor()}
    horizons = (1, a.h2) if a.h2 > 1 else (1,)
    refs = sample_states(parse_seeds(a.seeds), a.states, np.random.default_rng(a.sample_seed),
                         horizon=max(horizons))
    t0 = time.time()

    def progress(i, n):
        print(f"[{i}/{n}] trajectories, {time.time() - t0:.0f}s", flush=True)

    records, dropped = probe_many(predictors, refs, a.k, a.n_seeds, horizons, procs=a.procs,
                                  progress=progress)
    summary = summarise(records, list(NAMES), horizons)
    verdict = gate(summary, records, controls=[n for n in NAMES if n != "model"])
    meta = {"attempt": a.attempt, "label": a.label, "deviations": deviations,
            "code": f"git HEAD {_git('rev-parse', 'HEAD') or 'unknown'}{' (uncommitted changes)' if dirty else ''}",
            "protocol sha256": proto_sha or "not given",
            "model": f"{safe_path(a.model)} (sha256 {sha[:16]})",
            "states": f"{len(records)} evaluated, {len(dropped)} dropped (requested {a.states}; "
                      f"seeds {a.seeds}, sampling seed {a.sample_seed})",
            "K / n_seeds / horizons": f"{a.k} / {a.n_seeds} / {', '.join(map(str, horizons))}",
            "gate thresholds": json.dumps(GATE),
            "runtime": f"{time.time() - t0:.0f} s with {a.procs} processes"}
    text = render(summary, verdict, meta, horizons)
    out.parent.mkdir(parents=True, exist_ok=True)
    head = "" if existing else "# Step 0: counterfactual gate results\n\nOne section per attempt; never edited.\n\n"
    with out.open("a") as f:
        f.write(scrub(head + text + "\n"))
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(scrub(json.dumps({"meta": meta, "verdict": {**verdict, "checks": {
        k: list(v) for k, v in verdict["checks"].items()}}, "summary": summary,
        "records": records, "dropped": dropped})))
    print(f"verdict {verdict['verdict']} -> {out} (raw {raw})")


if __name__ == "__main__":
    main()
