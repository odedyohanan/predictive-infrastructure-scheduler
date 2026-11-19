"""Decision time of the tuned advisor at normal priority, against the official limits.

  .venv/bin/python experiments/b_time_advisor.py                    # reads configs/phase_b/advisor.json
  .venv/bin/python experiments/b_time_advisor.py --label smoke --seeds 3450 --duration 60 \\
      --config data/phase_b/smoke_<name>/tune/advisor.json --alarms data/phase_b/smoke_<name>/alarms.json \\
      --incident data/phase_b/smoke_g3/lt/incident.json --world-model models/smoke_g4/pgru_smoke.pt \\
      --fitted data/phase_b/smoke_g5/shadow/predictors.pkl --smoke-dir data/phase_b/smoke_<name>

Run it directly (not through scripts/gentle.sh): one process, one torch thread, normal priority. It runs the
advisor (model selection) of the config on three SELECTION seeds, whole 300 s episodes: the three with the highest
share of windows in intervention mode in the tuning run (its raw results, the chosen config), and reports the
decision time per window (p50 / p95 / max) per seed and pooled, also over the windows in intervention mode
alone (in-mode p95 reported alongside the verdict), with the share spent on the alarm risk and on the candidate selection. Verdict against p95 <= 50 ms
and max <= 500 ms in every episode. Output: results/phase_b/advisor_timing.md with the config and the
world model's sha256 (never overwritten). Smoke runs (`--label smoke`) may use history seeds, a shorter
`--duration` and `--selection always` (every window from the warm-up on in intervention mode, the warm-up
itself never: the worst case), allow a dirty tree, and write only into `--smoke-dir`. A real run reads the
frozen configs/phase_b/{incident,alarms}.json, and they must match the rule and alarm the advisor config was
tuned with.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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

DECIDE_P95_MS, DECIDE_MAX_MS = 50.0, 500.0
DEFAULT_CONFIG = "configs/phase_b/advisor.json"
NOTE = "results/phase_b/advisor_timing.md"
N_SEEDS = 3
DEFAULT_TUNE_DIR = "results/raw/phase_b/advisor_tune"
DURATION_S = 300.0


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def summarise(ms) -> dict:
    ms = np.asarray(ms, float)
    if not len(ms):
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"n": int(len(ms)), "mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)),
            "p95": float(np.percentile(ms, 95)), "max": float(ms.max())}


def top_mode_seeds(tune_dir, arm: str | None, n: int = N_SEEDS) -> list[int]:
    """The `n` SELECTION seeds on which the tuning run's chosen config spent the largest share of windows in
    intervention mode (ties: lower seed), sorted; [] without tuning results for that config."""
    from pis.shards import collect
    d = Path(tune_dir) if Path(tune_dir).is_absolute() else ROOT / tune_dir
    eps = [e for x in collect(d) if (x / "episodes.json").is_file()
           for e in json.loads((x / "episodes.json").read_text())]
    mine = [e for e in eps if arm is not None and e.get("arm") == arm and int(e["seed"]) in seed_ranges.SELECTION]
    best = sorted(mine, key=lambda e: (-float(e["mode_share"]), int(e["seed"])))[:n]
    return sorted(int(e["seed"]) for e in best)


def check_run(a: dict, dirty: bool, note_exists: bool) -> list[str]:
    errors = []
    smoke = a.get("label") == "smoke"
    try:
        seeds = seed_ranges.parse(a["seeds"])
        runs.check_seeds(seeds, ("SELECTION",) + (("HISTORY",) if smoke else ()))
    except ValueError as e:
        return [str(e)]
    if smoke:
        try:
            runs.smoke_dir(a.get("smoke_dir"))
        except ValueError as e:
            errors.append(str(e))
    else:
        if a.get("smoke_dir") is not None:
            errors.append("--smoke-dir needs --label smoke")
        if dirty:
            errors.append("pis/ or experiments/ have uncommitted changes")
        if a["duration"] != DURATION_S:
            errors.append(f"duration differs from {DURATION_S:g} s outside a smoke run")
        if len(set(seeds)) != N_SEEDS:
            errors.append(f"a real run times {N_SEEDS} seeds")
        if a.get("selection", "model") != "model":
            errors.append("a real run times the model selection")
        if note_exists:
            errors.append("the timing note already exists (never overwritten)")
    return errors


FROZEN = {"incident": "configs/phase_b/incident.json", "alarms": "configs/phase_b/alarms.json"}
ALARM_KEYS = ("predictor", "threshold", "low", "H")
RULE_KEYS = ("k", "interval", "majority", "warmup")


def frozen_problems(a: dict, cfg: dict, inc: dict, alarms: dict) -> list[str]:
    """Why a real run's incident and alarms configs are not the frozen ones the advisor was tuned with."""
    out = []
    for key, frozen in FROZEN.items():
        if Path(a[key]).resolve() != (ROOT / frozen).resolve():
            out.append(f"a real run reads the frozen {frozen} (--{key})")
    if cfg.get("alarms") != {k: alarms.get(k) for k in ALARM_KEYS}:
        out.append("the alarms config differs from the one in the advisor config")
    if cfg.get("rule") != {k: inc.get(k) for k in RULE_KEYS}:
        out.append("the incident rule differs from the one in the advisor config")
    return out


def measure(seeds, duration_s: float, cfg: dict, inc: dict, alarms: dict, world_model, fitted,
            selection: str = "model") -> dict:
    """Decision times of whole episodes of the advisor; per seed, pooled and in intervention mode."""
    import torch
    torch.set_num_threads(1)
    from pis.phase_b.advisor import AdvisorParams
    from pis.phase_b.episode import run_episode
    tune = runs.load_script("b_tune_advisor")
    params = AdvisorParams(**cfg["params"])
    stack = tune.load_stack(alarms, inc, world_model, fitted, int(cfg.get("wm_n", 32)))
    per_seed, pooled, on, risk_ms, sel_ms, sel_on, wall = {}, [], [], [], [], [], {}
    for s in seeds:
        sc = tune.scenario(int(s), duration_s)
        t0 = time.perf_counter()
        recs = run_episode(sc, tune.make_advisor(stack, sc.spec, int(s), alarms, inc, params, selection,
                                                 duration_s), "advisor")
        wall[int(s)] = time.perf_counter() - t0
        info = [r["info"] for r in recs]
        ms = np.array([i["decide_ms"] for i in info], float)
        in_mode = [i["decide_ms"] for i in info if i["mode"]]
        per_seed[int(s)] = {**summarise(ms), "mode_windows": int(sum(i["mode"] for i in info)),
                            "fallbacks": int(sum(bool(i["fallback"]) for i in info)),
                            "in_mode_p95": float(np.percentile(in_mode, 95)) if in_mode else None}
        pooled += ms.tolist()
        on += [i["decide_ms"] for i in info if i["mode"]]
        risk_ms += [i["risk_ms"] for i in info]
        sel_ms += [i["select_ms"] for i in info]
        sel_on += [i["select_ms"] for i in info if i["mode"]]
        print(f"seed {s}: p95 {per_seed[int(s)]['p95']:.1f} ms, max {per_seed[int(s)]['max']:.1f} ms, "
              f"{per_seed[int(s)]['mode_windows']} windows in mode, {wall[int(s)]:.0f} s", flush=True)
    worst_p95 = max(v["p95"] for v in per_seed.values())
    worst_max = max(v["max"] for v in per_seed.values())
    breakdown = {"risk_ms_mean": float(np.mean(risk_ms)), "risk_ms_p95": float(np.percentile(risk_ms, 95)),
                 "select_ms_mean": float(np.mean(sel_ms)),
                 "select_ms_mean_in_mode": float(np.mean(sel_on)) if sel_on else None,
                 "select_ms_p95_in_mode": float(np.percentile(sel_on, 95)) if sel_on else None}
    in_mode_p95 = [v["in_mode_p95"] for v in per_seed.values() if v["in_mode_p95"] is not None]
    return {"per_seed": per_seed, "all": summarise(pooled), "in_mode": summarise(on), "breakdown": breakdown,
            "worst_in_mode_p95": max(in_mode_p95) if in_mode_p95 else None,
            "wall_s": wall, "worst_p95": worst_p95, "worst_max": worst_max,
            "p95_ok": worst_p95 <= DECIDE_P95_MS, "max_ok": worst_max <= DECIDE_MAX_MS}


_f = partial(runs.fmt, default="{:.1f}")


def note_text(res: dict, run: dict, cfg: dict, sha: str, alarms: dict) -> str:
    from pis.privacy import safe_path
    head = _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--", "pis", "experiments"))
    nice = os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else math.nan
    b = res["breakdown"]
    L = [f"# Advisor decision time at normal priority{' (' + run['label'] + ')' if run.get('label') else ''}", "",
         f"- code: git HEAD {head}{' (uncommitted changes)' if dirty else ''}",
         f"- config: {safe_path(run['config'])}; params {json.dumps(cfg['params'], sort_keys=True)}; selection "
         f"{run['selection']}",
         f"- alarm: predictor {alarms.get('predictor')}, threshold {alarms.get('threshold')}, low "
         f"{alarms.get('low')}, H {alarms.get('H')}",
         f"- world model sha256 {sha}",
         f"- seeds: {run['seeds']}; episodes {run['duration']:g} s; 1 process, 1 torch thread, nice {nice}",
         f"- wall time per episode: " + ", ".join(f"{s} {w:.0f} s" for s, w in res["wall_s"].items()), "",
         f"Official limits: p95 <= {DECIDE_P95_MS:g} ms and max <= {DECIDE_MAX_MS:g} ms in every episode.", "",
         "| seed | decisions | in mode | fallbacks | mean ms | p50 ms | p95 ms | max ms | p95 in mode ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s, m in res["per_seed"].items():
        L.append(f"| {s} | {m['n']} | {m['mode_windows']} | {m['fallbacks']} | {_f(m['mean'])} | {_f(m['p50'])} | "
                 f"{_f(m['p95'])} | {_f(m['max'])} | {_f(m.get('in_mode_p95'))} |")
    a, o = res["all"], res["in_mode"]
    L.append(f"| all | {a['n']} | {o['n']} | - | {_f(a['mean'])} | {_f(a['p50'])} | {_f(a['p95'])} | "
             f"{_f(a['max'])} | {_f(o['p95'])} |")
    L += ["", f"Windows in intervention mode only: n {o['n']}, p50 {_f(o['p50'])} ms, p95 {_f(o['p95'])} ms, "
          f"max {_f(o['max'])} ms.",
          f"Breakdown: alarm risk mean {_f(b['risk_ms_mean'])} ms (p95 {_f(b['risk_ms_p95'])}); candidate "
          f"selection in mode mean {_f(b['select_ms_mean_in_mode'])} ms (p95 {_f(b['select_ms_p95_in_mode'])}).",
          "", f"- worst per-episode p95 {res['worst_p95']:.1f} ms: "
          f"{'within' if res['p95_ok'] else '**over**'} the {DECIDE_P95_MS:g} ms limit",
          f"- worst max {res['worst_max']:.1f} ms: {'within' if res['max_ok'] else '**over**'} the "
          f"{DECIDE_MAX_MS:g} ms limit",
          f"- in-mode p95 (worst episode) {_f(res.get('worst_in_mode_p95'))} ms: reported alongside the verdict, which "
          "is per episode over all windows"]
    return "\n".join(L) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seeds", default=None, help="default: the 3 SELECTION seeds with the highest intervention "
                                                   "share in the tuning run (a real run uses exactly those)")
    ap.add_argument("--tune-dir", default=DEFAULT_TUNE_DIR)
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--incident", default="configs/phase_b/incident.json")
    ap.add_argument("--alarms", default="configs/phase_b/alarms.json")
    ap.add_argument("--world-model", default="models/pgru_b1.pt")
    ap.add_argument("--fitted", default="results/raw/phase_b/shadow/predictors.pkl")
    ap.add_argument("--selection", default="model", choices=("model", "always"))
    ap.add_argument("--label", default="")
    ap.add_argument("--smoke-dir", default=None)
    a = ap.parse_args(argv)
    if a.label not in runs.LABELS:
        raise SystemExit(f"refusing to run: unknown label {a.label!r}")
    smoke = a.label == "smoke"
    cfg0 = runs.read_json(Path(a.config))
    derived = top_mode_seeds(a.tune_dir, cfg0.get("chosen"))
    if a.seeds is None:
        a.seeds = ",".join(map(str, derived)) if derived else ("3450-3452" if smoke else "")
    if not smoke:
        try:
            given = seed_ranges.parse(a.seeds) if a.seeds else []
        except ValueError as e:
            raise SystemExit(f"refusing to run: {e}")
        if not derived or given != derived:
            raise SystemExit("refusing to run: a real run times the 3 SELECTION seeds with the highest intervention "
                             f"share in the tuning run ({derived or 'no tuning results for the chosen config'})")
    dirty = bool(_git("status", "--porcelain", "--", "pis", "experiments"))
    errors = check_run(vars(a), dirty, (ROOT / NOTE).exists())
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))
    out = (runs.smoke_dir(a.smoke_dir) / "advisor_timing.md") if smoke else ROOT / NOTE
    cfg, inc, alarms = (runs.read_json(Path(p)) for p in (a.config, a.incident, a.alarms))
    if not cfg.get("params"):
        raise SystemExit("refusing to run: the advisor config has no chosen params")
    if not smoke:
        errors = frozen_problems(vars(a), cfg, inc, alarms)
        if errors:
            raise SystemExit("refusing to run: " + "; ".join(errors))
    from pis.phase_b.wm_prob import sha256
    if not Path(a.world_model).exists():
        raise SystemExit("refusing to run: no world-model checkpoint")
    sha = sha256(a.world_model)
    if cfg.get("world_model_sha256") not in (None, sha) and not smoke:
        raise SystemExit("refusing to run: the world model differs from the one the advisor was tuned with")
    seeds = seed_ranges.parse(a.seeds)
    res = measure(seeds, a.duration, cfg, inc, alarms, a.world_model, a.fitted, a.selection)
    runs.write_note(out, note_text(res, {"label": a.label, "seeds": a.seeds, "duration": a.duration,
                                         "config": a.config, "selection": a.selection}, cfg, sha, alarms))
    print(f"p95 ok: {res['p95_ok']} (worst {res['worst_p95']:.1f} ms), max ok: {res['max_ok']} "
          f"(worst {res['worst_max']:.1f} ms)", flush=True)
    return res


if __name__ == "__main__":
    main()
