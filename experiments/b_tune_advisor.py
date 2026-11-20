"""Advisor tuning on development seeds: R6 and the advisor (model selection) with every (eps, n) of the grid,
paired per seed; the config with the least time in incident that keeps the value per second.

  .venv/bin/python experiments/b_tune_advisor.py --procs 4
  .venv/bin/python experiments/b_tune_advisor.py --label smoke --seeds 3450-3451 --duration 30 \\
      --alarms data/phase_b/smoke_<name>/alarms.json --incident data/phase_b/smoke_g3/lt/incident.json \\
      --world-model models/smoke_g4/pgru_smoke.pt --fitted data/phase_b/smoke_g5/shadow/predictors.pkl \\
      --smoke-dir data/phase_b/smoke_<name>/tune

Every seed runs R6 (the reference) and the advisor with eps in EPS_GRID x n in N_GRID (m = 4, h = min(H, 20)) on
the same scenario. The alarm predictor, its threshold and `low` come from the alarms config; the predictor is
loaded as the shadow run used it (its fitted pickle, or the world model with the shadow run's number of futures;
a real run refuses a world model other than the shadow run's); the advisor's futures come from `--world-model`.
`--world-model` and `--fitted` default to the world model and the shadow directory the alarms config records
(models/pgru_b1.pt and results/raw/phase_b/shadow/ for a config that records none). Reward weights (slo_lam, mu,
change cost) are the Step A loop's (configs/step_a.json).

Per episode: time in incident (s), counted incidents, value per second, share of windows in intervention
mode, fallback share, decision time; chronic = R6's episode spends more than half its windows in incident.
Choice: the config with the lowest mean time in incident over the non-chronic episodes, among those whose
paired 95 % lower bound of (advisor - R6) value per second over all episodes is at least -1 % of R6's mean
value per second; ties -> smaller n, then smaller eps. Shards of SHARD_SIZE seeds hold all arms of their seeds
(pairing stays inside a shard); rerunning skips finished shards and refuses a changed configuration.

Real runs: all SELECTION seeds, 300 s, a clean tree, configs/phase_b/{incident,alarms}.json with a predictor
and threshold, the shadow run's predictors, and the current code reproducing the shadow run's risks on two
calibration episodes (re-scored from the store; a pis/phase_b diff since its commit is a warning); writes configs/phase_b/advisor.json and
results/phase_b/advisor_dev.md, raw output under results/raw/phase_b/advisor_tune/. Smoke runs (`--label
smoke`) may use history seeds and a shorter `--duration`, allow a dirty tree and write only into
`--smoke-dir`. Long runs go through scripts/gentle.sh.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
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

DEFAULT_INCIDENT = "configs/phase_b/incident.json"
DEFAULT_ALARMS = "configs/phase_b/alarms.json"
FITTED_FILE = "predictors.pkl"
STEP_A = "configs/step_a.json"
CONFIG = "configs/phase_b/advisor.json"
NOTE = "results/phase_b/advisor_dev.md"
DEFAULT_OUT = "results/raw/phase_b/advisor_tune"
EPS_GRID = (0.05, 0.1, 0.2)
N_GRID = (8, 16)
M = 4
H_MAX = 20
SHARD_SIZE = 5
VALUE_TOL = 0.01
REFERENCE = "r6"
EPISODES_FILE = "episodes.json"
DURATION_S = 300.0


def grid() -> list[dict]:
    return [{"eps": e, "n": n} for n in N_GRID for e in EPS_GRID]


def arm_name(cfg: dict) -> str:
    return f"eps{cfg['eps']:g}_n{cfg['n']}"


# ------------------------------------------------------------------ the advisor stack
def rule_of(inc: dict):
    from pis.phase_b import incidents
    return incidents.IncidentRule(int(inc["k"]), int(inc["interval"]), float(inc["majority"]))


def load_stack(alarms: dict, inc: dict, world_model, fitted, wm_n: int) -> dict:
    """Predictor (as the shadow run used it) and the advisor's world model."""
    from pis.phase_b.world_models import load_world_model
    shadow = runs.load_script("b_shadow")
    name = alarms.get("predictor")
    pred = None
    if name is not None:
        pred = shadow.load_predictors(None if name == "world_model" else fitted, [name], inc, world_model,
                                      wm_n)[name]
    return {"predictor": pred, "predictor_name": name, "for_episode": shadow.for_episode,
            "model": load_world_model(world_model)}


class _NoRisk:
    """Stand-in predictor when the alarms config names none (no feasible threshold): never consulted."""
    name = "none"

    def risk(self, rows, H, episode=None):
        return float("nan")


def make_advisor(stack: dict, spec, seed: int, alarms: dict, inc: dict, params, selection: str = "model",
                 duration_s: float = DURATION_S, oracle_branches: int | None = None,
                 oracle_horizon: int | None = None):
    from pis.phase_b.advisor import AdvisorController
    pred = stack["predictor"] or _NoRisk()
    if stack["predictor"] is not None:
        stack["for_episode"]({stack["predictor_name"]: pred}, seed)
    from pis.phase_b.advisor import ORACLE_BRANCHES
    return AdvisorController(spec, pred, alarms, stack["model"], params, selection, rule=rule_of(inc),
                             warmup=int(inc["warmup"]), seed=seed, duration_s=duration_s,
                             oracle_branches=ORACLE_BRANCHES if oracle_branches is None else oracle_branches,
                             oracle_horizon=oracle_horizon)


def scenario(seed: int, duration_s: float):
    from pis.phase_b.scenarios import scenario_for
    sc = scenario_for(seed)
    return sc if duration_s == sc.duration_s else dataclasses.replace(sc, duration_s=duration_s)


def shadow_wm_n(fitted, world_model_sha: str | None, wm_n: int | None, smoke: bool) -> int:
    """Number of world-model futures of the alarm predictor: the shadow run's (its config.json next to the
    fitted predictors). A real run refuses a missing shadow config, a world model other than the shadow run's
    and a --wm-n other than its; a smoke run takes --wm-n, else the shadow config's, else 32."""
    path = Path(fitted).parent / "config.json" if fitted else None
    shadow_cfg = runs.read_json(path) if path is not None else {}
    if smoke:
        return int(wm_n) if wm_n is not None else int(shadow_cfg.get("wm_n", 32))
    if not shadow_cfg:
        raise ValueError("a real run needs the shadow run's config.json next to --fitted")
    if shadow_cfg.get("world_model_sha256") != world_model_sha:
        raise ValueError("the world model differs from the one the shadow run scored")
    if "wm_n" not in shadow_cfg:
        raise ValueError("the shadow run's config.json lacks wm_n")
    if wm_n is not None and int(wm_n) != int(shadow_cfg["wm_n"]):
        raise ValueError(f"--wm-n {wm_n} differs from the shadow run's wm_n {shadow_cfg['wm_n']}")
    return int(shadow_cfg["wm_n"])


# ------------------------------------------------------------------ metrics
def episode_metrics(records: list[dict], spec, inc: dict) -> dict:
    from pis.harness import scenario_metrics
    from pis.phase_b import incidents
    lab = incidents.label(records, rule_of(inc), int(inc["warmup"]))
    m = scenario_metrics(records, spec)
    info = [r["info"] for r in records]
    mode = [bool(i.get("mode", False)) for i in info]
    fb = [bool(i.get("fallback", False)) for i, on in zip(info, mode) if on]
    ms = np.array([i.get("decide_ms", np.nan) for i in info], float)
    return {"windows": len(records), "time_in_incident_s": lab.time_in_incident_s,
            "incidents": len(lab.incidents), "value_per_s": m["value_per_s"],
            "chronic": incidents.is_chronic(lab, len(records)),
            "mode_share": float(np.mean(mode)) if mode else 0.0,
            "fallback_share": float(np.mean(fb)) if fb else None,
            "decide_ms_p95": float(np.nanpercentile(ms, 95)), "decide_ms_max": float(np.nanmax(ms))}


# ------------------------------------------------------------------ worker
def run_shard(shard, shard_dir: Path, params: dict) -> None:
    """R6 and every advisor config on each seed of the shard."""
    import torch
    torch.set_num_threads(1)
    from pis.phase_b import policies
    from pis.phase_b.advisor import AdvisorParams
    from pis.phase_b.episode import run_episode
    inc, alarms, dur = params["incident"], params["alarms"], float(params["duration_s"])
    stack = load_stack(alarms, inc, params["world_model"], params["fitted"], params["wm_n"])
    out = []
    for seed in shard.items:
        seed = int(seed)
        sc = scenario(seed, dur)
        t0 = time.perf_counter()
        recs = run_episode(sc, policies.make("r6", sc.spec, seed), REFERENCE)
        out.append({"seed": seed, "arm": REFERENCE, "seconds": time.perf_counter() - t0,
                    **episode_metrics(recs, sc.spec, inc)})
        for cfg in params["grid"]:
            p = AdvisorParams.from_step_a(params["step_a"], int(alarms["H"]), m=params["m"], n=cfg["n"],
                                          eps=cfg["eps"], h_max=params["h_max"])
            t0 = time.perf_counter()
            recs = run_episode(sc, make_advisor(stack, sc.spec, seed, alarms, inc, p, "model", dur), "advisor")
            out.append({"seed": seed, "arm": arm_name(cfg), "seconds": time.perf_counter() - t0,
                        **episode_metrics(recs, sc.spec, inc)})
    (shard_dir / EPISODES_FILE).write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ choice
def summarise(episodes: list[dict], configs: list[dict]) -> dict:
    """Per config: paired statistics against R6 (time in incident over non-chronic seeds, value over all)."""
    from pis.stats import paired_diff
    by = {(e["seed"], e["arm"]): e for e in episodes}
    seeds = sorted({e["seed"] for e in episodes if e["arm"] == REFERENCE})
    ref = [by[(s, REFERENCE)] for s in seeds]
    calm = [s for s, r in zip(seeds, ref) if not r["chronic"]]
    r6_value = float(np.mean([r["value_per_s"] for r in ref])) if ref else float("nan")
    out = {}
    for cfg in configs:
        a = arm_name(cfg)
        miss = [s for s in seeds if (s, a) not in by]
        if miss:
            raise ValueError(f"arm {a} lacks seeds {miss[:5]}")
        adv = [by[(s, a)] for s in seeds]
        t_adv = np.array([by[(s, a)]["time_in_incident_s"] for s in calm], float)
        t_ref = np.array([by[(s, REFERENCE)]["time_in_incident_s"] for s in calm], float)
        v = paired_diff([x["value_per_s"] for x in adv], [x["value_per_s"] for x in ref])
        t = paired_diff(t_adv, t_ref) if len(calm) else {"mean": float("nan"), "lo": float("nan"),
                                                          "hi": float("nan"), "n": 0}
        lo = v["lo"] if math.isfinite(v["lo"]) else v["mean"]          # a single seed: the point estimate
        fb = [x["fallback_share"] for x in adv if x["fallback_share"] is not None]
        out[a] = {**cfg, "seeds": len(seeds), "non_chronic": len(calm),
                  "time_advisor": float(t_adv.mean()) if len(calm) else float("nan"),
                  "time_r6": float(t_ref.mean()) if len(calm) else float("nan"),
                  "time_diff": t, "value_diff": v, "value_lo": lo, "r6_value": r6_value,
                  "value_ok": bool(lo >= -VALUE_TOL * abs(r6_value)),
                  "incidents_advisor": int(sum(x["incidents"] for x in adv)),
                  "incidents_r6": int(sum(x["incidents"] for x in ref)),
                  "mode_share": float(np.mean([x["mode_share"] for x in adv])),
                  "fallback_share": float(np.mean(fb)) if fb else None,
                  "decide_ms_p95": float(max(x["decide_ms_p95"] for x in adv)),
                  "decide_ms_max": float(max(x["decide_ms_max"] for x in adv))}
    return out


def choose_config(summary: dict) -> str | None:
    ok = [a for a, s in summary.items() if s["value_ok"] and math.isfinite(s["time_advisor"])]
    if not ok:
        return None
    return min(ok, key=lambda a: (summary[a]["time_advisor"], summary[a]["n"], summary[a]["eps"], a))


# ------------------------------------------------------------------ note
_f = partial(runs.fmt, default="{:.2f}")


def note_text(cfg: dict, summary: dict, chosen: str | None, timing: dict) -> str:
    L = ["# Advisor tuning on development seeds", "",
         f"Label: {cfg['label'] or 'real'}. Seeds {len(cfg['seeds'])} ({cfg['seeds'][0]}-{cfg['seeds'][-1]}), "
         f"{cfg['duration_s']:g} s episodes. Alarm predictor {cfg['alarms'].get('predictor')}, threshold "
         f"{_f(cfg['alarms'].get('threshold'), '{:.4f}')}, low {_f(cfg['alarms'].get('low'), '{:.4f}')}, "
         f"H {cfg['alarms']['H']}. Advisor m = {cfg['m']}, h = {min(int(cfg['alarms']['H']), cfg['h_max'])}; "
         f"reward slo_lam {cfg['step_a']['slo_lam']}, mu {cfg['step_a']['mu']}, change cost "
         f"{cfg['step_a'].get('change_cost', 0.0)}. World model sha256 {cfg['world_model_sha256']}.", "",
         "Time in incident: mean over the episodes that are not chronic for R6 (more than half the windows in "
         "incident). Value per second: paired over all episodes; a config is admissible when the 95 % lower bound "
         f"of (advisor - R6) is at least -{VALUE_TOL:.0%} of |R6's mean value per second|.", "",
         "The incident probability of a candidate is a share of n futures, so a bound eps admits only multiples of "
         "1/n below it: with n = 8, eps 0.05 and 0.1 both mean no future may show a new incident (p = 0), and so "
         "does eps 0.05 with n = 16; the grid is kept as tuned.", "",
         "| config | time in incident advisor / R6 (s) | diff [95 % CI] | value diff [95 % CI] | value ok | "
         "incidents advisor / R6 | mode share | fallback share | decide p95 / max ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a, s in summary.items():
        t, v = s["time_diff"], s["value_diff"]
        L.append(f"| {a} | {_f(s['time_advisor'])} / {_f(s['time_r6'])} | {_f(t['mean'], '{:+.2f}')} "
                 f"[{_f(t['lo'], '{:+.2f}')}, {_f(t['hi'], '{:+.2f}')}] | {_f(v['mean'], '{:+.2f}')} "
                 f"[{_f(v['lo'], '{:+.2f}')}, {_f(v['hi'], '{:+.2f}')}] | {'yes' if s['value_ok'] else 'no'} | "
                 f"{s['incidents_advisor']} / {s['incidents_r6']} | {_f(s['mode_share'])} | "
                 f"{_f(s['fallback_share'])} | {_f(s['decide_ms_p95'], '{:.1f}')} / "
                 f"{_f(s['decide_ms_max'], '{:.1f}')} |")
    any_s = next(iter(summary.values()), None)
    if any_s:
        L += ["", f"Non-chronic episodes: {any_s['non_chronic']} of {any_s['seeds']}; R6 mean value per second "
              f"{_f(any_s['r6_value'])}."]
    L += ["", "## Timing", "",
          f"Wall time per episode: R6 {_f(timing['r6_s'], '{:.1f}')} s, advisor "
          f"{_f(timing['advisor_s'], '{:.1f}')} s (mean over configs); a full run (50 seeds x "
          f"{1 + len(cfg['grid'])} arms x {DURATION_S:g} s) projected {_f(timing['projected_h'], '{:.1f}')} h "
          "on one process.", "", "## Decision", ""]
    if chosen:
        s = summary[chosen]
        L.append(f"Chosen: {chosen} (eps {s['eps']:g}, n {s['n']}): time in incident {_f(s['time_advisor'])} s "
                 f"vs R6 {_f(s['time_r6'])} s on non-chronic episodes.")
    else:
        L.append("No config keeps the value per second within the tolerance: none chosen.")
    return "\n".join(L) + "\n"


def resolve_inputs(a, alarms: dict) -> None:
    """Fill `a.world_model` and `a.fitted` left unset from the shadow run the alarms config came from;
    explicit values stay."""
    src = runs.shadow_inputs(alarms)
    if a.world_model is None:
        a.world_model = src["world_model"]
    if a.fitted is None:
        a.fitted = str(Path(src["shadow_dir"]) / FITTED_FILE)


# ------------------------------------------------------------------ run
def _git_head() -> str:
    p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() or "unknown"


def tune(seeds, duration_s: float, inc: dict, alarms: dict, world_model, fitted, wm_n: int, step_a: dict,
         out, note, config_path, procs: int, label: str = "") -> dict:
    from pis.phase_b.advisor import AdvisorParams
    from pis.phase_b.wm_prob import sha256
    from pis.shards import collect, make_shards, run_shards
    out = Path(out)
    seeds = sorted(int(s) for s in seeds)
    if alarms.get("predictor") is not None and alarms.get("threshold") is None:
        raise ValueError("the alarms config names a predictor without a threshold")
    if alarms.get("predictor") not in (None, "world_model") and (fitted is None or not Path(fitted).exists()):
        raise ValueError("the alarm predictor needs the shadow run's fitted predictors (--fitted)")
    if world_model is None or not Path(world_model).exists():
        raise ValueError("the advisor needs an existing --world-model checkpoint")
    cfg = {"label": label, "seeds": seeds, "duration_s": float(duration_s), "incident": inc, "alarms": alarms,
           "world_model_sha256": sha256(world_model),
           "fitted_sha256": sha256(fitted) if fitted is not None and Path(fitted).exists() else None,
           "wm_n": int(wm_n), "step_a": step_a, "grid": grid(), "m": M, "h_max": H_MAX, "shard_size": SHARD_SIZE}
    params = {"incident": inc, "alarms": alarms, "duration_s": float(duration_s), "world_model": str(world_model),
              "fitted": None if fitted is None else str(fitted), "wm_n": int(wm_n), "step_a": step_a,
              "grid": grid(), "m": M, "h_max": H_MAX}
    shards = make_shards(seeds, SHARD_SIZE)
    t0 = time.time()
    res = run_shards(out, shards, partial(run_shard, params=params), cfg, procs)
    print(f"shards done {len(res['done'])}, skipped {len(res['skipped'])}, failed {res['failed']} "
          f"in {time.time() - t0:.0f} s", flush=True)
    dirs = collect(out)
    if len(dirs) != len(shards):
        raise RuntimeError("unfinished shards: rerun the same command")
    episodes = [e for d in dirs for e in json.loads((d / EPISODES_FILE).read_text())]
    summary = summarise(episodes, grid())
    chosen = choose_config(summary)
    r6_s = float(np.mean([e["seconds"] for e in episodes if e["arm"] == REFERENCE]))
    adv_s = float(np.mean([e["seconds"] for e in episodes if e["arm"] != REFERENCE]))
    scale = DURATION_S / float(duration_s)
    timing = {"r6_s": r6_s, "advisor_s": adv_s,
              "projected_h": len(seed_ranges.SELECTION) * (r6_s + len(grid()) * adv_s) * scale / 3600}
    params_out = None
    if chosen:
        s = summary[chosen]
        params_out = AdvisorParams.from_step_a(step_a, int(alarms["H"]), m=M, n=s["n"], eps=s["eps"],
                                               h_max=H_MAX).to_dict()
    runs.write_json(Path(config_path), {
        "params": params_out, "chosen": chosen, "predictor": alarms.get("predictor"),
        "alarms": {k: alarms.get(k) for k in ("predictor", "threshold", "low", "H")},
        "rule": {k: inc[k] for k in ("k", "interval", "majority", "warmup")}, "wm_n": int(wm_n),
        "world_model_sha256": cfg["world_model_sha256"], "fitted_sha256": cfg["fitted_sha256"],
        "selected_on_seeds": f"{seeds[0]}-{seeds[-1]}", "duration_s": float(duration_s),
        "value_tol": VALUE_TOL, "code": _git_head(), "label": label})
    runs.write_note(Path(note), note_text(cfg, summary, chosen, timing))
    return {"summary": summary, "chosen": chosen, "shards": res, "timing": timing, "episodes": episodes}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="3450-3499")
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--incident", default=DEFAULT_INCIDENT)
    ap.add_argument("--alarms", default=DEFAULT_ALARMS)
    ap.add_argument("--world-model", default=None, help="default: the alarms config's world model")
    ap.add_argument("--fitted", default=None, help="the shadow run's predictors.pkl (default: in the alarms "
                                                   "config's shadow directory)")
    ap.add_argument("--wm-n", type=int, default=None, help="default: the shadow run's (32)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--label", default="")
    ap.add_argument("--smoke-dir", default=None)
    a = ap.parse_args(argv)
    if a.label not in runs.LABELS:
        raise SystemExit(f"refusing to run: unknown label {a.label!r}")
    smoke = a.label == "smoke"
    try:
        seeds = seed_ranges.parse(a.seeds)
        runs.check_seeds(seeds, ("SELECTION",) + (("HISTORY",) if smoke else ()))
        if len(set(seeds)) != len(seeds):
            raise ValueError("repeated seeds")
        sdir = runs.smoke_dir(a.smoke_dir) if smoke else None
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    if not smoke:
        if a.smoke_dir is not None:
            raise SystemExit("refusing to run: --smoke-dir needs --label smoke")
        if seeds != list(seed_ranges.SELECTION):
            raise SystemExit("refusing to run: a real run uses the whole selection range")
        if a.duration != DURATION_S:
            raise SystemExit(f"refusing to run: a real run uses {DURATION_S:g} s episodes")
        for given, frozen in ((a.incident, DEFAULT_INCIDENT), (a.alarms, DEFAULT_ALARMS)):
            if Path(given).resolve() != (ROOT / frozen).resolve():
                raise SystemExit(f"refusing to run: a real run reads the frozen {frozen}")
        if runs.tree_dirty(ignore=(CONFIG, NOTE)):
            raise SystemExit("refusing to run: the working tree has uncommitted changes (commit, or --label smoke)")
    inc, alarms = runs.read_json(Path(a.incident)), runs.read_json(Path(a.alarms))
    need = ["k", "interval", "majority", "warmup", "H"]
    if any(k not in inc for k in need):
        raise SystemExit(f"refusing to run: {a.incident} lacks {[k for k in need if k not in inc]}")
    if "H" not in alarms or "predictor" not in alarms:
        raise SystemExit(f"refusing to run: {a.alarms} is not an alarms config")
    resolve_inputs(a, alarms)
    if not smoke and (alarms.get("predictor") is None or alarms.get("threshold") is None):
        raise SystemExit("refusing to run: the alarms config has no predictor with a threshold")
    step_a = json.loads((ROOT / STEP_A).read_text())["params"]
    from pis.phase_b.wm_prob import sha256
    wm_sha = sha256(a.world_model) if a.world_model and Path(a.world_model).exists() else None
    try:
        wm_n = shadow_wm_n(a.fitted, wm_sha, a.wm_n, smoke)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    if not smoke:
        refusals, warnings = runs.load_script("b_shadow").identity_check(Path(a.fitted).parent, a.world_model)
        for w in warnings:
            print(f"warning: {w}", flush=True)
        if refusals:
            raise SystemExit("refusing to run: " + "; ".join(refusals))
    out = (sdir / "advisor_tune") if smoke else ROOT / DEFAULT_OUT
    note = (sdir / "advisor_dev.md") if smoke else ROOT / NOTE
    config = (sdir / "advisor.json") if smoke else ROOT / CONFIG
    try:
        res = tune(seeds, a.duration, inc, alarms, a.world_model, a.fitted, wm_n, step_a, out, note, config,
                   a.procs, a.label)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    print(f"chosen {res['chosen']}; full run projected {res['timing']['projected_h']:.1f} h on one process",
          flush=True)
    return res


if __name__ == "__main__":
    main()
