"""Step 0 gate run, exactly per docs/protocols/step0-gate.md (defaults = protocol values).

  .venv/bin/python experiments/step0_run.py --model worldmodel:models/gru_mix_bounded.pt --attempt 1 --procs 5

Appends a verdict section to `--out` (default docs/results/step0.md) and writes the raw per-state numbers to
`--raw` (default results/raw/step0_attempt<N>.json, untracked). Refuses to run while the protocol file has
uncommitted changes, and records the commit the protocol was frozen in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pis.config import resolve, setup_imports  # noqa: E402
from pis.privacy import scrub  # noqa: E402
from pis.probe import (GATE, INTERVENTIONS, GRUPredictor, PersistencePredictor,  # noqa: E402
                       gate, probe_many, sample_states, summarise)

setup_imports()

from worldmodel.model import WorldModel  # noqa: E402

PROTOCOL = "docs/protocols/step0-gate.md"
NAMES = ("model", "action_blind", "persistence")


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def protocol_commit(allow_dirty: bool) -> str:
    if _git("status", "--porcelain", "--", PROTOCOL) and not allow_dirty:
        raise SystemExit(f"{PROTOCOL} has uncommitted changes; the protocol must be frozen first")
    return _git("log", "-1", "--format=%h", "--", PROTOCOL) or "uncommitted"


def _f(x, nd=3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(summary: dict, verdict: dict, meta: dict, horizons) -> str:
    L = [f"## Attempt {meta['attempt']}{' (' + meta['label'] + ')' if meta.get('label') else ''}: "
         f"**{verdict['verdict']}**", ""]
    L += [f"- {k}: {v}" for k, v in meta.items() if k not in ("attempt", "label")]
    L += ["", "### Gate (h = 1)", "", "| criterion | value | pass |", "|---|---|---|"]
    for name, (ok, val) in verdict["checks"].items():
        L.append(f"| {name} | {val} | {'yes' if ok else 'no'} |")
    L += ["", f"Random-ranking control (95th percentile of the median Spearman, not gated): "
              f"{_f(verdict['random_ranking_p95'])}", ""]
    for hh in horizons:
        h = f"h{hh}"
        L += [f"### Horizon {hh}", "",
              "| predictor | median Spearman (resolvable) | median Spearman (all) | resolvable states "
              "| pairwise sign acc | resolvable pairs | reward MAE |", "|---|---|---|---|---|---|---|"]
        for n in NAMES:
            m = summary[n][h]
            L.append(f"| {n} | {_f(m['median_spearman'])} | {_f(m['median_spearman_all'])} | "
                     f"{m['n_resolvable_states']}/{m['n_states']} | {_f(m['pairwise_sign_acc'])} | "
                     f"{m['n_resolvable_pairs']} | {_f(m['reward_mae'], 4)} |")
        L.append("")
    L += ["### Named interventions (h = 1)", "",
          "| predictor | intervention | sign acc | cases | state majority acc | A p99 | B rate | C work |",
          "|---|---|---|---|---|---|---|---|"]
    for n in NAMES:
        for i in INTERVENTIONS:
            m = summary[n]["h1"]["interventions"][i]
            pq = " | ".join(f"{_f(a)} ({c})" for a, c in m["per_quantity"].values())
            L.append(f"| {n} | {i} | {_f(m['sign_acc'])} | {m['n_cases']} | "
                     f"{_f(m['state_majority_acc'])} | {pq} |")
    return "\n".join(L) + "\n"


def parse_seeds(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="worldmodel:models/gru_mix_bounded.pt")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--states", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seeds", default="1100-1149")
    ap.add_argument("--h2", type=int, default=4)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--out", default="docs/results/step0.md")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--allow-dirty-protocol", action="store_true", help="smoke tests only")
    a = ap.parse_args()
    raw = Path(a.raw or f"results/raw/step0_attempt{a.attempt}.json")
    frozen = protocol_commit(a.allow_dirty_protocol)
    path = resolve(a.model)
    model = WorldModel.load(str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    predictors = {"model": GRUPredictor(model), "action_blind": GRUPredictor(model, action_blind=True),
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
    verdict = gate(summary, records)
    meta = {"attempt": a.attempt, "label": a.label,
            "model": f"{a.model} (sha256 {sha[:16]})",
            "protocol": f"{PROTOCOL} at commit {frozen}",
            "states": f"{len(records)} evaluated, {len(dropped)} dropped (requested {a.states}; "
                      f"seeds {a.seeds}, sampling seed {a.sample_seed})",
            "K / n_seeds / horizons": f"{a.k} / {a.n_seeds} / {', '.join(map(str, horizons))}",
            "gate thresholds": json.dumps(GATE),
            "runtime": f"{time.time() - t0:.0f} s with {a.procs} processes"}
    text = render(summary, verdict, meta, horizons)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    head = "" if out.exists() else ("# Step 0: counterfactual gate results\n\n"
                                    f"Protocol: `{PROTOCOL}`. One section per attempt; never edited.\n\n")
    with out.open("a") as f:
        f.write(scrub(head + text + "\n"))
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(scrub(json.dumps({"meta": meta, "verdict": {**verdict, "checks": {
        k: list(v) for k, v in verdict["checks"].items()}}, "summary": summary,
        "records": records, "dropped": dropped})))
    print(f"verdict {verdict['verdict']} -> {out} (raw {raw})")


if __name__ == "__main__":
    main()
