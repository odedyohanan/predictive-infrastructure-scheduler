"""Decision time of a chosen Step A or Step B controller at normal priority, against the official limits.

  .venv/bin/python experiments/time_decisions.py --step a        # reads configs/step_a.json
  .venv/bin/python experiments/time_decisions.py --step b        # reads configs/step_b.json

Run it directly (not through scripts/gentle.sh): one process at normal priority, so the numbers are not
inflated by background QoS on efficiency cores the way the tuning runs are (their gate is 2x looser). It runs
the controller of the config on a few development seeds (default 1100-1102, 60 s each; takes a few minutes at
most) and reports decision time mean / p95 / max per scenario and pooled, against the official limits
p95 <= 50 ms and max <= 500 ms (per scenario, worst case).

Config formats: Step A {"members", "params"}; Step B {"members", "step_a_params", "reviser": {mechanism,
rev_steps, rev_lam, rev_lr}}. Only development seeds 1100-1199 are accepted (smoke runs included). A run refuses
uncommitted changes in pis/ or experiments/ and an existing note unless `--label smoke`; a smoke run may not
write the official note. Output: results/step_<a|b>/timing.md (or --out).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pis.privacy import safe_path, scrub  # noqa: E402

DEV_SEEDS = range(1100, 1200)
DECIDE_P95_MS, DECIDE_MAX_MS = 50.0, 500.0
CONFIGS = {"a": "configs/step_a.json", "b": "configs/step_b.json"}
NOTES = {"a": "results/step_a/timing.md", "b": "results/step_b/timing.md"}


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def check_run(a: dict, dirty: bool, note_exists: bool) -> list[str]:
    errors = []
    outside = sorted(s for s in set(parse_seeds(a["seeds"])) if s not in DEV_SEEDS)
    if outside:
        errors.append(f"seeds outside the development range {DEV_SEEDS.start}-{DEV_SEEDS.stop - 1}: "
                      f"{outside[:5]}")
    if a.get("label") == "smoke":
        if Path(a["out"]).resolve() in {Path(p).resolve() for p in NOTES.values()}:
            errors.append("a smoke run must not write an official timing note")
    else:
        if dirty:
            errors.append("pis/ or experiments/ have uncommitted changes")
        if a["duration"] != 60.0:
            errors.append("duration differs from 60 s outside a smoke run")
        if note_exists:
            errors.append("the timing note already exists (never overwritten)")
    return errors


def load_config(path: str) -> dict:
    def _no_nan(x):
        raise ValueError(f"non-strict JSON constant {x}")
    cfg = json.loads(Path(path).read_text(), parse_constant=_no_nan)
    if not cfg.get("members"):
        raise ValueError("the config needs 'members'")
    if "reviser" in cfg:
        if not isinstance(cfg.get("step_a_params"), dict):
            raise ValueError("a Step B config needs 'step_a_params'")
        cfg["kind"] = "b"
    elif isinstance(cfg.get("params"), dict):
        cfg["kind"] = "a"
    else:
        raise ValueError("neither a Step A ('params') nor a Step B ('reviser') config")
    return cfg


def make_controller(cfg: dict, spec, ens):
    from pis.evaluator import Params
    if cfg["kind"] == "a":
        from pis.loop import LoopController
        return LoopController(spec, ens, Params.from_dict(cfg["params"]))
    from pis.reviser import ReviserLoopController
    r = cfg["reviser"]
    return ReviserLoopController(spec, ens, Params.from_dict(cfg["step_a_params"]), mechanism=r["mechanism"],
                                 rev_steps=r["rev_steps"], rev_lam=r["rev_lam"], rev_lr=r["rev_lr"])


def summarise(ms: np.ndarray) -> dict:
    ms = np.asarray(ms, float)
    return {"n": int(len(ms)), "mean": float(ms.mean()), "p95": float(np.percentile(ms, 95)),
            "max": float(ms.max())}


def verdict(per_seed: dict[int, dict]) -> dict:
    worst_p95 = max(s["p95"] for s in per_seed.values())
    worst_max = max(s["max"] for s in per_seed.values())
    return {"worst_p95": worst_p95, "worst_max": worst_max,
            "p95_ok": worst_p95 <= DECIDE_P95_MS, "max_ok": worst_max <= DECIDE_MAX_MS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("a", "b"), default="b")
    ap.add_argument("--config", default=None, help="default: configs/step_<step>.json")
    ap.add_argument("--seeds", default="1100-1102")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--label", default="", help='"smoke" for test runs')
    ap.add_argument("--out", default=None, help="default: results/step_<step>/timing.md")
    a = ap.parse_args(argv)
    a.config = a.config or CONFIGS[a.step]
    a.out = a.out or NOTES[a.step]
    dirty = bool(_git("status", "--porcelain", "--", "pis", "experiments"))
    errors = check_run(vars(a), dirty, Path(a.out).exists())
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))
    try:
        cfg = load_config(a.config)
    except (OSError, ValueError) as e:
        raise SystemExit(scrub(f"cannot read the config {safe_path(a.config)}: {e}"))
    import torch
    torch.set_num_threads(1)
    from pis.adapters import make_spec
    from pis.config import resolve
    from pis.ensemble import Ensemble, _sha256
    from pis.harness import run_scenario
    members = list(cfg["members"])
    missing = [m for m in members if not resolve(m).is_file()]
    if missing:
        raise SystemExit("missing ensemble members: " + ", ".join(safe_path(m) for m in missing))
    ens = Ensemble.from_paths(members)
    if cfg["kind"] == "b":
        from pis.reviser import warm_up
        warm_up(ens, DEV_SEEDS.start)
    seeds = parse_seeds(a.seeds)
    t0 = time.time()
    per_seed, pooled, revised = {}, [], {}
    for s in seeds:
        recs = run_scenario(s, make_controller(cfg, make_spec(s), ens), a.duration)
        ms = np.array([r["info"]["decide_ms"] for r in recs], float)
        per_seed[s] = summarise(ms)
        pooled += ms.tolist()
        revised[s] = sum(bool(r["info"].get("revised")) for r in recs)
        print(f"seed {s}: p95 {per_seed[s]['p95']:.1f} ms, max {per_seed[s]['max']:.1f} ms", flush=True)
    runtime = time.time() - t0
    v = verdict(per_seed)
    allp = summarise(np.array(pooled))
    head = _git("rev-parse", "HEAD") or "unknown"
    nice = os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else math.nan
    L = [f"# Decision time, Step {a.step.upper()} at normal priority{' (' + a.label + ')' if a.label else ''}", "",
         f"- code: git HEAD {head}{' (uncommitted changes)' if dirty else ''}",
         f"- config: {safe_path(a.config)}",
         f"- members: " + ", ".join(f"{safe_path(m)} (sha256 {_sha256(resolve(m))[:16]})" for m in members),
         f"- controller params: {json.dumps(cfg.get('params') or cfg.get('step_a_params'))}"
         + (f"; reviser {json.dumps(cfg['reviser'])}" if cfg["kind"] == "b" else ""),
         f"- seeds: {a.seeds}; scenario {a.duration:g} s; 1 process, 1 torch thread, nice {nice}",
         f"- runtime: {runtime:.0f} s", "",
         f"Official limits: p95 <= {DECIDE_P95_MS:g} ms and max <= {DECIDE_MAX_MS:g} ms in every scenario.", "",
         "| seed | decisions | mean ms | p95 ms | max ms |" + (" revised |" if cfg["kind"] == "b" else ""),
         "|---|---|---|---|---|" + ("---|" if cfg["kind"] == "b" else "")]
    for s, m in per_seed.items():
        L.append(f"| {s} | {m['n']} | {m['mean']:.1f} | {m['p95']:.1f} | {m['max']:.1f} |"
                 + (f" {revised[s]} |" if cfg["kind"] == "b" else ""))
    L.append(f"| all | {allp['n']} | {allp['mean']:.1f} | {allp['p95']:.1f} | {allp['max']:.1f} |"
             + (f" {sum(revised.values())} |" if cfg["kind"] == "b" else ""))
    L += ["", f"- worst per-scenario p95 {v['worst_p95']:.1f} ms: "
          f"{'within' if v['p95_ok'] else '**over**'} the {DECIDE_P95_MS:g} ms limit",
          f"- worst max {v['worst_max']:.1f} ms: {'within' if v['max_ok'] else '**over**'} the "
          f"{DECIDE_MAX_MS:g} ms limit"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(scrub("\n".join(L) + "\n"))
    print(f"p95 ok: {v['p95_ok']}, max ok: {v['max_ok']} -> {safe_path(str(out))}, runtime {runtime:.0f} s")
    return {"per_seed": per_seed, "all": allp, **v}


if __name__ == "__main__":
    main()
