"""Step B tuning on development seeds: the reviser over mechanism x steps x penalty, paired against the base
scheduler and the Step A loop.

  .venv/bin/python experiments/tune_step_b.py --procs 4

Step A's chosen members and Params are read from its config (`configs/step_a.json`, or --step-a-config);
the reviser runs on top of exactly that loop. Arms, each on every evaluation seed:
  base     the base scheduler (R6 greedy)
  step_a   the Step A loop with its chosen Params
  reviser  `pis.reviser.ReviserLoopController` for every grid config (mechanism, steps, revision lam)
Every reviser run also probes reality after each decision is timed: when a revision is applied, the start
and the revised policy are each run for H windows in copies of the real simulator. Model exploitation = an
applied revision whose predicted reward rises while its real score falls; its rate is reported per config.

Objective: the largest mean paired value per second (reviser - base), among configs with paired results on
every seed, paired upper 95% bound of the (reviser - base) SLO v2 window-violation % at most +`slo_margin`
percentage points, and worst per-scenario decision time p95 <= 100 ms and max <= 1000 ms
(background-QoS limits, 2x the official 50 / 500 ms; see experiments/time_decisions.py). `beats_base` and
`beats_step_a`: the chosen config's mean paired value difference is above zero with its 95% CI excluding zero.

Seeds: only development seeds 1100-1199 are accepted (always, smoke runs included). A run refuses to start
with uncommitted changes in pis/ or experiments/, when the grid exceeds --max-configs, or when the results
note or the config already exists. `--label smoke` lifts the dirty-tree and duration checks but may not
write the official note or config.
Outputs: results/step_b/report.md (note), configs/step_b.json (chosen reviser, relative member paths),
results/raw/step_b/tuning<_label>.json (every per-scenario number; gitignored).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pis.privacy import safe_path, scrub  # noqa: E402
from pis.stats import paired_diff  # noqa: E402

DEV_SEEDS = range(1100, 1200)
OFFICIAL_NOTE, OFFICIAL_CONFIG = "results/step_b/report.md", "configs/step_b.json"
STEP_A_CONFIG = "configs/step_a.json"
DEFAULTS = {"seeds": "1140-1199", "grid_mech": "gradient,neighbourhood", "grid_steps_gradient": "2,4",
            "grid_steps_neighbourhood": "1,3", "grid_rev_lam": "1,4", "rev_lr": 0.6, "slo_margin": 1.0,
            "duration": 60.0, "max_configs": 8}
DECIDE_P95_MS, DECIDE_MAX_MS = 50.0, 500.0         # official decision-time limits (normal priority)
BACKGROUND_FACTOR = DEFAULTS_BACKGROUND_FACTOR = 2.0  # tuning runs at background QoS on efficiency cores
GATE_P95_MS, GATE_MAX_MS = BACKGROUND_FACTOR * DECIDE_P95_MS, BACKGROUND_FACTOR * DECIDE_MAX_MS
EXPLOIT_FLAG = 0.5        # a config whose applied revisions lose in reality more often than this is flagged


def set_background_factor(factor: float):
    """Set the background-QoS decision-time gate to `factor` x the official limits."""
    global BACKGROUND_FACTOR, GATE_P95_MS, GATE_MAX_MS
    if not factor >= 1.0:
        raise SystemExit("--background-factor must be >= 1")
    BACKGROUND_FACTOR = factor
    GATE_P95_MS, GATE_MAX_MS = factor * DECIDE_P95_MS, factor * DECIDE_MAX_MS


def load_raw_runs(path: str, cfgs: list[dict], seeds: list[int]) -> list[dict]:
    """The per-run records of an earlier raw file; refuses unless its grid and seeds equal this run's."""
    d = json.loads(Path(path).read_text())
    if [r["cfg"] for r in d["rows"]] != cfgs:
        raise SystemExit("the raw file's grid differs from this run's")
    runs = d["runs"]
    if sorted({r["seed"] for r in runs}) != sorted(seeds):
        raise SystemExit("the raw file's seeds differ from this run's")
    return runs


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def grid(a: dict) -> list[dict]:
    out = []
    for mech in a["grid_mech"].split(","):
        steps = _ints(a[f"grid_steps_{mech}"])
        out += [{"mechanism": mech, "steps": s, "rev_lam": lam}
                for s, lam in itertools.product(steps, _floats(a["grid_rev_lam"]))]
    return out


def check_run(a: dict, dirty: bool, note_exists: bool, config_exists: bool) -> list[str]:
    """Errors that refuse the run. The seed checks hold for every run, smoke included."""
    errors = []
    seeds = parse_seeds(a["seeds"])
    outside = sorted(s for s in set(seeds) if s not in DEV_SEEDS)
    if outside:
        errors.append(f"seeds outside the development range {DEV_SEEDS.start}-{DEV_SEEDS.stop - 1}: "
                      f"{outside[:5]}")
    mechs = a["grid_mech"].split(",")
    if not mechs or any(m not in ("gradient", "neighbourhood") for m in mechs):
        errors.append("--grid-mech takes gradient and/or neighbourhood")
        return errors
    g = grid(a)
    if len(g) > a["max_configs"]:
        errors.append(f"grid has {len(g)} configs > --max-configs {a['max_configs']}")
    if any(c["rev_lam"] <= 0 for c in g):
        errors.append("revision lam must be > 0 (the uncertainty penalty is always active)")
    if any(c["steps"] < 1 for c in g):
        errors.append("revision steps must be >= 1")
    smoke = a.get("label") == "smoke"
    if smoke:
        for key, official in (("out", OFFICIAL_NOTE), ("config_out", OFFICIAL_CONFIG)):
            if Path(a[key]).resolve() == Path(official).resolve():
                errors.append(f"a smoke run must not write {official}")
    else:
        if dirty:
            errors.append("pis/ or experiments/ have uncommitted changes")
        if a["duration"] != DEFAULTS["duration"]:
            errors.append("duration differs from 60 s outside a smoke run")
        if note_exists:
            errors.append("the results note already exists (one tuning run, never overwritten)")
        if config_exists:
            errors.append(f"{OFFICIAL_CONFIG} or --config-out already exists (never overwritten)")
    return errors


def load_step_a(path: str) -> dict:
    """Step A's chosen config (strict JSON with members and params)."""
    def _no_nan(x):
        raise ValueError(f"non-strict JSON constant {x}")
    cfg = json.loads(Path(path).read_text(), parse_constant=_no_nan)
    if not cfg.get("members") or not isinstance(cfg.get("params"), dict):
        raise ValueError("the Step A config needs 'members' and 'params'")
    return cfg


# ------------------------------------------------------------------ worker side
_W: dict = {}


def _init_worker(members: list[str], duration: float):
    import torch
    torch.set_num_threads(1)
    from pis.ensemble import Ensemble
    from pis.reviser import warm_up
    _W.update(ens=Ensemble.from_paths(members), duration=duration)
    warm_up(_W["ens"], DEV_SEEDS.start)


def revision_stats(records: list[dict]) -> dict:
    """Per-scenario revision counts, predicted and real gains, exploitation count, mean consequences."""
    infos = [r["info"] for r in records]
    trig = [i for i in infos if i.get("rev_trigger")]
    rev = [i for i in infos if i.get("revised")]
    probed = [i for i in rev if "rev_real_gain" in i]
    expl = [i for i in probed if i["rev_pred_reward_gain"] > 0 and i["rev_real_gain"] < 0]
    vs_greedy = [i["rev_real_gain_vs_greedy"] for i in probed if "rev_real_gain_vs_greedy" in i]
    cons = {k: float(np.mean([i["consequences"][k] for i in rev])) if rev else math.nan
            for k in ("a_p99_ms", "a_queue_work_ms", "b_rate", "c_work_core_ms")}
    return {"triggers": len(trig), "triggers_untrusted": sum(i["rev_trigger"] == "untrusted" for i in trig),
            "triggers_slo": sum(i["rev_trigger"] == "slo" for i in trig), "revised": len(rev),
            "probed": len(probed), "exploited": len(expl),
            "real_gain_sum": float(sum(i["rev_real_gain"] for i in probed)),
            "vs_greedy": len(vs_greedy), "real_gain_vs_greedy_sum": float(sum(vs_greedy)),
            "pred_reward_gain_sum": float(sum(i["rev_pred_reward_gain"] for i in probed)),
            "rev_ms_max": float(max((i["rev_ms"] for i in trig if "rev_ms" in i), default=math.nan)),
            "consequences_mean": cons}


def run_job(job: dict) -> dict:
    from pis.adapters import make_spec
    from pis.evaluator import Params
    from pis.harness import BaseGreedyController, run_scenario, scenario_metrics
    from pis.loop import LoopController
    from pis.privacy import scrub as _scrub
    from pis.reviser import ReviserLoopController
    seed, kind = job["seed"], job["kind"]
    t0 = time.time()
    try:
        spec = make_spec(seed)
        if kind == "base":
            ctl = BaseGreedyController(spec)
        elif kind == "step_a":
            ctl = LoopController(spec, _W["ens"], Params.from_dict(job["params"]))
        else:
            c = job["cfg"]
            ctl = ReviserLoopController(spec, _W["ens"], Params.from_dict(job["params"]), mechanism=c["mechanism"],
                                        rev_steps=c["steps"], rev_lam=c["rev_lam"], rev_lr=job["rev_lr"],
                                        probe_real=True)
        recs = run_scenario(seed, ctl, _W["duration"])
        out = {**job, "metrics": scenario_metrics(recs, spec), "seconds": time.time() - t0}
        if kind == "reviser":
            out["revision"] = revision_stats(recs)
        return out
    except Exception as e:                       # recorded, never replaced
        return {**job, "error": _scrub(f"{type(e).__name__}: {e}"), "seconds": time.time() - t0}


# ------------------------------------------------------------------ analysis
def compare(ref: dict[int, dict], arm: dict[int, dict]) -> dict:
    """Paired (arm - ref) statistics over the seeds both finished."""
    seeds = sorted(set(ref) & set(arm))
    def d(key):
        return paired_diff(np.array([arm[s][key] for s in seeds]), np.array([ref[s][key] for s in seeds]))
    def mean(key):
        return float(np.mean([arm[s][key] for s in seeds])) if seeds else math.nan
    return {"n": len(seeds), "value": d("value_per_s"), "slo_v2": d("slo_v2_pct"), "slo_v1": d("slo_v1_pct"),
            "late": d("a_late_pct"), "fallback_rate": mean("fallback_rate"), "greedy_rate": mean("greedy_rate"),
            "decide_p95": mean("decide_ms_p95"),
            "decide_p95_worst": float(max((arm[s]["decide_ms_p95"] for s in seeds), default=math.nan)),
            "decide_max": float(max((arm[s]["decide_ms_max"] for s in seeds), default=math.nan)),
            "changes": mean("policy_changes")}


def aggregate_revisions(stats: list[dict]) -> dict:
    """Pooled over scenarios: revision rate, exploitation rate, mean real and predicted gain per revision."""
    tot = {k: sum(s[k] for s in stats) for k in ("triggers", "triggers_untrusted", "triggers_slo", "revised",
                                                  "probed", "exploited", "vs_greedy")}
    probed = tot["probed"]
    rev = [s for s in stats if s["revised"]]
    cons = {k: (float(np.average([s["consequences_mean"][k] for s in rev], weights=[s["revised"] for s in rev]))
                if rev else math.nan) for k in ("a_p99_ms", "a_queue_work_ms", "b_rate", "c_work_core_ms")}
    return {**tot,
            "exploit_rate": tot["exploited"] / probed if probed else math.nan,
            "real_gain_mean": sum(s["real_gain_sum"] for s in stats) / probed if probed else math.nan,
            "real_gain_vs_greedy_mean": (sum(s["real_gain_vs_greedy_sum"] for s in stats) / tot["vs_greedy"]
                                         if tot["vs_greedy"] else math.nan),
            "pred_reward_gain_mean": sum(s["pred_reward_gain_sum"] for s in stats) / probed if probed else math.nan,
            "rev_ms_max": float(max((s["rev_ms_max"] for s in stats if math.isfinite(s["rev_ms_max"])),
                                    default=math.nan)),
            "consequences_mean": cons}


def select_best(rows: list[dict], slo_margin: float, n_seeds: int) -> tuple[int | None, list[bool], bool, bool]:
    """(best feasible row, feasibility of every row, beats base, beats Step A); see the module docstring."""
    def ok(r):
        m = r["vs_base"]
        return (m["n"] == n_seeds and n_seeds >= 2 and m["slo_v2"]["hi"] <= slo_margin
                and math.isfinite(m["value"]["mean"])
                and m["decide_p95_worst"] <= GATE_P95_MS and m["decide_max"] <= GATE_MAX_MS)
    feas = [bool(ok(r)) for r in rows]
    cands = [i for i, f in enumerate(feas) if f]
    if not cands:
        return None, feas, False, False
    best = max(cands, key=lambda i: (rows[i]["vs_base"]["value"]["mean"], -rows[i]["cfg"]["steps"]))
    vb, va = rows[best]["vs_base"]["value"], rows[best]["vs_step_a"]["value"]
    return (best, feas, bool(vb["mean"] > 0 and vb["lo"] > 0),
            bool(rows[best]["vs_step_a"]["n"] == n_seeds and va["mean"] > 0 and va["lo"] > 0))


def mechanism_comparison(rows: list[dict], feas: list[bool], per_seed: dict) -> dict | None:
    """Best config of each mechanism (feasible first, then by value vs base), paired gradient - neighbourhood."""
    best = {}
    for mech in ("gradient", "neighbourhood"):
        idx = [i for i, r in enumerate(rows) if r["cfg"]["mechanism"] == mech]
        if idx:
            best[mech] = max(idx, key=lambda i: (feas[i], _num(rows[i]["vs_base"]["value"]["mean"])))
    if len(best) < 2:
        return None
    g, n = best["gradient"], best["neighbourhood"]
    return {"gradient": g, "neighbourhood": n, "diff": compare(per_seed[n], per_seed[g])}


def exploit_flagged(revisions: dict) -> bool:
    """A config whose applied revisions lost in reality more often than EXPLOIT_FLAG."""
    return _num(revisions["exploit_rate"]) > EXPLOIT_FLAG


def _num(x) -> float:
    return x if isinstance(x, (int, float)) and math.isfinite(x) else -1e18


def _f(x, nd=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else (
        "inf" if isinstance(x, float) and math.isinf(x) else f"{x:.{nd}f}")


def _ci(s: dict, nd=1) -> str:
    return f"{_f(s['mean'], nd)} [{_f(s['lo'], nd)}, {_f(s['hi'], nd)}]"


def _cfg(c: dict) -> str:
    return f"{c['mechanism']} steps {c['steps']} lam {_f(c['rev_lam'], 2)}"


def render(meta: dict, base_summary: dict, step_a_cmp: dict, rows: list[dict], best: int | None, feas: list[bool],
           beats_base: bool, beats_step_a: bool, mech: dict | None, failures: list[dict]) -> str:
    L = [f"# Step B tuning on development seeds{' (' + meta['label'] + ')' if meta.get('label') else ''}", ""]
    L += [f"- {k}: {v}" for k, v in meta.items() if k != "label" and not k.startswith("_")]
    L += ["", "Paired differences are per scenario, mean [95% CI]. Value in value units per second; SLO in "
          "percentage points of windows violating the SLO. Feasible = paired results on every evaluation seed, "
          f"upper bound of the (reviser - base) SLO v2 difference <= +{meta['slo margin (pp)']} pp (SLO v1 "
          "reported only), "
          f"decision time p95 <= {GATE_P95_MS:g} ms in every scenario and max <= {GATE_MAX_MS:g} ms. "
          f"**These are background-QoS limits, {BACKGROUND_FACTOR:g}x the official {DECIDE_P95_MS:g} / "
          f"{DECIDE_MAX_MS:g} ms**: tuning runs at macOS background priority on efficiency cores with several "
          "processes, which inflates wall-clock time; the official limits are checked at normal priority by "
          "experiments/time_decisions.py.", "",
          "## Reference arms", "",
          f"- base scheduler (R6 greedy): {base_summary['n']} scenarios; mean value per second "
          f"{_f(base_summary['value'])}; SLO v2 {_f(base_summary['slo_v2'], 2)} %; SLO v1 "
          f"{_f(base_summary['slo_v1'], 2)} %",
          f"- Step A loop vs base: value {_ci(step_a_cmp['value'])}; SLO v2 {_ci(step_a_cmp['slo_v2'], 2)}; "
          f"fallback {_f(step_a_cmp['fallback_rate'], 3)}; decide p95 {_f(step_a_cmp['decide_p95'])} ms "
          f"(worst {_f(step_a_cmp['decide_p95_worst'])}); n {step_a_cmp['n']}", "",
          "## Grid", "",
          "| mechanism | steps | lam | value vs base | SLO v2 vs base | value vs Step A | SLO v2 vs Step A | "
          "revised / triggers | exploit rate | real gain / revision | real gain vs greedy (untrusted) | "
          "decide p95 ms (worst) | decide max ms | n | feasible |", "|" + "---|" * 15]
    order = sorted(range(len(rows)), key=lambda i: -_num(rows[i]["vs_base"]["value"]["mean"]))
    for i in order:
        r, c, rv = rows[i], rows[i]["cfg"], rows[i]["revisions"]
        mark = " **chosen**" if i == best else ""
        L.append(f"| {c['mechanism']} | {c['steps']} | {_f(c['rev_lam'], 2)} | {_ci(r['vs_base']['value'])} | "
                 f"{_ci(r['vs_base']['slo_v2'], 2)} | {_ci(r['vs_step_a']['value'])} | "
                 f"{_ci(r['vs_step_a']['slo_v2'], 2)} | {rv['revised']} / {rv['triggers']} | "
                 f"{_f(rv['exploit_rate'], 3)} | {_f(rv['real_gain_mean'], 3)} | "
                 f"{_f(rv['real_gain_vs_greedy_mean'], 3)} ({rv['vs_greedy']}) | "
                 f"{_f(r['vs_base']['decide_p95'])} ({_f(r['vs_base']['decide_p95_worst'])}) | "
                 f"{_f(r['vs_base']['decide_max'])} | {r['vs_base']['n']} | {'yes' if feas[i] else 'no'}{mark} |")
    L += ["", "Triggers: Step A's best candidate untrusted (disagreement above tau) or predicted to violate A's "
          "SLO. Revised: the revision was applied (better revision score and, with the trust gate, disagreement "
          "within tau). Exploit rate: share of applied revisions whose predicted reward rose while the real "
          "simulator scored the revised policy below the start policy over the same horizon (same random state); "
          "real gain / revision: mean real score difference (revised - start) per applied revision; real gain vs "
          "greedy: the same against the greedy fallback Step A would have applied, for revisions of untrusted "
          "candidates (count in brackets)."]
    flagged = [i for i, r in enumerate(rows) if exploit_flagged(r["revisions"])]
    L += ["", "## Model exploitation", ""]
    if flagged:
        L += [f"**Exploitation detected** (rate above {EXPLOIT_FLAG:g}) in: "
              + "; ".join(f"{_cfg(rows[i]['cfg'])} ({_f(rows[i]['revisions']['exploit_rate'], 3)})" for i in flagged)
              + ". Remedies: a larger revision penalty (lam), or the other mechanism where it is not flagged."]
    else:
        L.append(f"No config's exploit rate exceeds {EXPLOIT_FLAG:g}.")
    L += ["", "## Gradient vs neighbourhood search", ""]
    if mech is None:
        L.append("Only one mechanism in the grid.")
    else:
        d = mech["diff"]
        L += [f"Best of each mechanism: gradient = {_cfg(rows[mech['gradient']]['cfg'])}; neighbourhood = "
              f"{_cfg(rows[mech['neighbourhood']]['cfg'])}. Paired (gradient - neighbourhood): value "
              f"{_ci(d['value'])}, SLO v2 {_ci(d['slo_v2'], 2)}, n {d['n']}."]
    L += ["", "## Chosen configuration", ""]
    if best is None:
        L.append("No configuration met the SLO non-inferiority and decision-time constraints; none is recommended.")
    else:
        r = rows[best]
        if not beats_base:
            L += [f"**The chosen reviser does not beat the base scheduler**: paired value difference "
                  f"{_ci(r['vs_base']['value'])} (mean not above zero or CI includes zero).", ""]
        if not beats_step_a:
            L += [f"**The chosen reviser does not beat the Step A loop**: paired value difference "
                  f"{_ci(r['vs_step_a']['value'])} (mean not above zero or CI includes zero).", ""]
        if exploit_flagged(r["revisions"]):
            L += [f"**Warning: the chosen config is flagged for model exploitation** (exploit rate "
                  f"{_f(r['revisions']['exploit_rate'], 3)} > {EXPLOIT_FLAG:g}): its applied revisions lost in "
                  "the real simulator more often than not while the model predicted gains.", ""]
        cm = r["revisions"]["consequences_mean"]
        L += [f"{_cfg(r['cfg'])}, learning rate {meta['_rev_lr']}, on top of Step A's Params.", "",
              "Mean predicted consequences of an applied revision (revised - start, over members and horizon): "
              f"A p99 {_f(cm['a_p99_ms'], 2)} ms, A backlog {_f(cm['a_queue_work_ms'], 2)} ms, B rate "
              f"{_f(cm['b_rate'], 3)}, C work {_f(cm['c_work_core_ms'], 1)} core-ms.", "",
              "Selected as the best of the grid on these seeds, so its paired differences are optimistic "
              "(winner's curse); only the held-out evaluation measures it."]
    if failures:
        L += ["", "## Failed runs (recorded, not replaced)", ""]
        L += [f"- {f['kind']} seed {f['seed']}{' config ' + str(f.get('cfg_id')) if 'cfg_id' in f else ''}: "
              f"{f['error']}" for f in failures]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ driver
def _run(pool, jobs, label, t0):
    out = []
    it = map(run_job, jobs) if pool is None else pool.imap_unordered(run_job, jobs)
    for i, r in enumerate(it, 1):
        out.append(r)
        if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
            print(f"[{label}] {i}/{len(jobs)} runs, {time.time() - t0:.0f}s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-a-config", default=STEP_A_CONFIG)
    for k in ("seeds", "grid_mech", "grid_steps_gradient", "grid_steps_neighbourhood", "grid_rev_lam"):
        ap.add_argument("--" + k.replace("_", "-"), default=DEFAULTS[k])
    for k in ("rev_lr", "slo_margin", "duration"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=DEFAULTS[k])
    ap.add_argument("--max-configs", type=int, default=DEFAULTS["max_configs"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--label", default="", help='"smoke" for test runs')
    ap.add_argument("--out", default=OFFICIAL_NOTE)
    ap.add_argument("--config-out", default=OFFICIAL_CONFIG)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--background-factor", type=float, default=DEFAULTS_BACKGROUND_FACTOR,
                    help="decision-time gate = factor x the official limits (measured background/normal ratio)")
    ap.add_argument("--from-raw", default=None,
                    help="re-select from the runs of an earlier raw file instead of simulating (same seeds and grid)")
    a = ap.parse_args(argv)
    args = vars(a)
    set_background_factor(a.background_factor)
    dirty = bool(_git("status", "--porcelain", "--", "pis", "experiments"))
    config_exists = Path(OFFICIAL_CONFIG).exists() or Path(a.config_out).exists()
    errors = check_run(args, dirty, Path(a.out).exists(), config_exists)
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))
    try:
        step_a = load_step_a(a.step_a_config)
    except (OSError, ValueError) as e:
        raise SystemExit(scrub(f"cannot read the Step A config {safe_path(a.step_a_config)}: {e}"))
    members = list(step_a["members"])
    from pis.config import resolve
    missing = [m for m in members if not resolve(m).is_file()]
    if missing:
        raise SystemExit("missing ensemble members: " + ", ".join(safe_path(m) for m in missing))
    from pis.ensemble import _sha256
    from pis.evaluator import Params
    params = Params.from_dict(step_a["params"]).to_dict()       # validates the Step A params
    cfgs = grid(args)
    seeds = parse_seeds(a.seeds)
    t0 = time.time()
    if a.from_raw:
        res = load_raw_runs(a.from_raw, cfgs, seeds)
    else:
        pool = None
        if a.procs > 1:
            from multiprocessing import get_context
            pool = get_context("spawn").Pool(a.procs, initializer=_init_worker, initargs=(members, a.duration))
        else:
            _init_worker(members, a.duration)
        try:
            jobs = ([{"kind": "base", "seed": s} for s in seeds]
                    + [{"kind": "step_a", "seed": s, "params": params} for s in seeds]
                    + [{"kind": "reviser", "seed": s, "cfg_id": i, "cfg": c, "params": params, "rev_lr": a.rev_lr}
                       for i, c in enumerate(cfgs) for s in seeds])
            res = _run(pool, jobs, "base + step A + reviser grid", t0)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    base = {r["seed"]: r["metrics"] for r in res if r["kind"] == "base" and "metrics" in r}
    step_a_arm = {r["seed"]: r["metrics"] for r in res if r["kind"] == "step_a" and "metrics" in r}
    rows, per_cfg = [], {}
    for i, c in enumerate(cfgs):
        done = [r for r in res if r["kind"] == "reviser" and r["cfg_id"] == i and "metrics" in r]
        arm = {r["seed"]: r["metrics"] for r in done}
        per_cfg[i] = arm
        rows.append({"cfg": c, "vs_base": compare(base, arm), "vs_step_a": compare(step_a_arm, arm),
                     "revisions": aggregate_revisions([r["revision"] for r in done])})
    best, feas, beats_base, beats_step_a = select_best(rows, a.slo_margin, len(seeds))
    mech = mechanism_comparison(rows, feas, per_cfg)
    failures = [r for r in res if "error" in r]
    runtime = time.time() - t0
    head = _git("rev-parse", "HEAD") or "unknown"
    meta = {"label": a.label,
            "code": f"git HEAD {head}{' (uncommitted changes)' if dirty else ''}",
            "step A config": f"{safe_path(a.step_a_config)} (Step A beats base: {step_a.get('beats_base')})",
            "members": ", ".join(f"{safe_path(m)} (sha256 {_sha256(resolve(m))[:16]})" for m in members),
            "step A params": json.dumps(params), "evaluation seeds": a.seeds,
            "grid": f"mechanism {a.grid_mech}; gradient steps {a.grid_steps_gradient}; neighbourhood steps "
                    f"{a.grid_steps_neighbourhood}; revision lam {a.grid_rev_lam}",
            "fixed": f"gradient learning rate {a.rev_lr}, scenario {a.duration:g} s",
            "slo margin (pp)": a.slo_margin,
            "runtime": (f"re-selected from {safe_path(a.from_raw)} (no new simulation)" if a.from_raw
                        else f"{runtime:.0f} s with {a.procs} processes"),
            "decision-time gate": f"{BACKGROUND_FACTOR:g}x the official limits", "_rev_lr": a.rev_lr}
    base_summary = {"n": len(base), **{k: float(np.mean([m[v] for m in base.values()])) if base else math.nan
                                       for k, v in (("value", "value_per_s"), ("slo_v2", "slo_v2_pct"),
                                                    ("slo_v1", "slo_v1_pct"))}}
    step_a_cmp = compare(base, step_a_arm)
    text = render(meta, base_summary, step_a_cmp, rows, best, feas, beats_base, beats_step_a, mech, failures)
    meta.pop("_rev_lr")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(scrub(text))
    if best is not None:
        c, rv = cfgs[best], rows[best]["revisions"]
        cfg = {"members": [safe_path(m) for m in members], "step_a_params": params,
               "reviser": {"mechanism": c["mechanism"], "rev_steps": c["steps"], "rev_lam": c["rev_lam"],
                           "rev_lr": a.rev_lr},
               "beats_base": beats_base, "beats_step_a": beats_step_a,
               "exploit_rate": rv["exploit_rate"] if math.isfinite(rv["exploit_rate"]) else None,
               "exploit_flagged": exploit_flagged(rv),
               "step_a_config": safe_path(a.step_a_config), "selected_on_seeds": a.seeds, "code": head,
               "label": a.label}
        cp = Path(a.config_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(scrub(json.dumps(cfg, indent=1, allow_nan=False)) + "\n")
    raw = Path(a.raw or f"results/raw/step_b/tuning{'_' + a.label if a.label else ''}.json")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(scrub(json.dumps({"meta": meta, "rows": rows, "best": best, "base": base, "step_a": step_a_arm,
                                     "mechanisms": mech, "runs": res}, default=str)))
    print(f"chosen config {best} (beats base: {beats_base}, beats Step A: {beats_step_a}) -> "
          f"{safe_path(str(out))}, runtime {runtime:.0f} s")
    return {"best": best, "beats_base": beats_base, "beats_step_a": beats_step_a, "rows": rows, "mechanisms": mech}


if __name__ == "__main__":
    main()
