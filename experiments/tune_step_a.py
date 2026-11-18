"""Step A tuning on development seeds: grid over N, M, H, lam, tau; paired comparison with the base scheduler.

  .venv/bin/python experiments/tune_step_a.py --procs 4

Phases (one process pool):
  1. base scheduler (R6 greedy) on every evaluation seed; calibration runs on separate calibration seeds:
     the base scheduler drives, and at every window with full history the ensemble disagreement of every
     candidate (greedy, max(M) samples, keep-current) is recorded for each (N, H) of the grid. tau for a
     config = the `tau_q`-th percentile of that distribution ("inf" = no trust gate).
  2. the Step A loop for every grid config on every evaluation seed.
  3. at the chosen config: random selection, oracle selection (real simulator), no uncertainty penalty,
     no fallback gate (reported, not used for selection).
Objective: the largest mean paired value per second (loop - base), among configs whose paired upper 95%
bound of (loop - base) SLO v2 window-violation % is at most +`slo_margin` percentage points.

Seeds: only development seeds 1100-1199 are accepted (always, smoke runs included); calibration and
evaluation seeds must not overlap. A run refuses to start with uncommitted changes in pis/ or experiments/,
when the grid exceeds --max-configs, or when the results note or the config already exists. `--label smoke` lifts the
dirty-tree and duration checks but may not write the official note or config.
Outputs: results/step_a/tuning.md (note), configs/step_a.json (chosen Params, relative member paths),
results/raw/step_a/tuning<_label>.json (every per-scenario number; gitignored).
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
OFFICIAL_NOTE, OFFICIAL_CONFIG = "results/step_a/tuning.md", "configs/step_a.json"
MEMBERS = ("worldmodel:models/gru_mix_bounded.pt,models/ens_1.pt,models/ens_2.pt,models/ens_3.pt,"
           "models/ens_4.pt")
DEFAULTS = {"calib_seeds": "1100-1109", "seeds": "1110-1139", "grid_n": "3,5", "grid_m": "4,8",
            "grid_h": "1,3,5", "grid_lam": "0,1", "grid_tau_q": "90,99", "mu": 2.0, "change_cost": 0.0,
            "slo_lam": 10.0, "slo_margin": 1.0, "duration": 60.0, "max_configs": 48}
ABLATIONS = {"random": {"selection": "random"}, "oracle": {"selection": "oracle"},
             "no_uncertainty": {"use_uncertainty": False}, "no_fallback": {"use_fallback": False}}


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


def _tau_qs(text: str) -> list[float]:
    return [math.inf if x.strip() == "inf" else float(x) for x in text.split(",")]


def grid(a: dict) -> list[dict]:
    return [{"n": n, "m": m, "h": h, "lam": lam, "tau_q": q}
            for n, m, h, lam, q in itertools.product(_ints(a["grid_n"]), _ints(a["grid_m"]), _ints(a["grid_h"]),
                                                     _floats(a["grid_lam"]), _tau_qs(a["grid_tau_q"]))]


def check_run(a: dict, dirty: bool, note_exists: bool, config_exists: bool, n_members: int) -> list[str]:
    """Errors that refuse the run. The seed checks hold for every run, smoke included."""
    errors = []
    calib, seeds = parse_seeds(a["calib_seeds"]), parse_seeds(a["seeds"])
    outside = sorted(s for s in set(calib) | set(seeds) if s not in DEV_SEEDS)
    if outside:
        errors.append(f"seeds outside the development range {DEV_SEEDS.start}-{DEV_SEEDS.stop - 1}: "
                      f"{outside[:5]}")
    if set(calib) & set(seeds):
        errors.append("calibration and evaluation seeds overlap")
    g = grid(a)
    if len(g) > a["max_configs"]:
        errors.append(f"grid has {len(g)} configs > --max-configs {a['max_configs']}")
    if max(c["n"] for c in g) > n_members or min(c["n"] for c in g) < 1:
        errors.append(f"grid N must lie in 1..{n_members} (number of members)")
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


# ------------------------------------------------------------------ worker side
_W: dict = {}


def _init_worker(members: list[str], duration: float):
    import torch
    torch.set_num_threads(1)
    from pis.ensemble import Ensemble
    _W.update(full=Ensemble.from_paths(members), duration=duration, subs={})


def _sub(n: int):
    from pis.ensemble import Ensemble
    full = _W["full"]
    if n not in _W["subs"]:
        _W["subs"][n] = Ensemble(full.models[:n], train_std=full.train_std, refs=full.refs[:n],
                                 shas=full.shas[:n])
    return _W["subs"][n]


class _CalibController:
    """Acts as the base scheduler; records the disagreement of every candidate for each (N, H)."""

    def __init__(self, spec, ensemble, ns, hs, m):
        from pis.adapters import BaseScheduler, known_of
        from pis.loop import BASE_CKPT
        self.base, self.known, self.ens = BaseScheduler(BASE_CKPT, spec), known_of(spec), ensemble
        self.ns, self.hs, self.m = ns, hs, m
        self.rng = np.random.default_rng(0)
        self.pool = {f"{n},{h}": [] for n in ns for h in hs}

    def decide(self, tel, current, history):
        from pis.candidates import candidates_from
        t0 = time.perf_counter()
        greedy, samples = self.base.propose(tel, current, self.m, self.rng)
        cands = candidates_from(greedy, samples, current)
        if len(history) >= self.ens.history and cands:
            roll = self.ens.rollout_batch(history, cands, self.known, max(self.hs))
            for n in self.ns:
                for h in self.hs:
                    self.pool[f"{n},{h}"] += self.ens.disagreement(roll[:n, :, :h]).tolist()
        return greedy, {"decide_ms": (time.perf_counter() - t0) * 1000.0}


def run_job(job: dict) -> dict:
    from pis.adapters import make_spec
    from pis.evaluator import Params
    from pis.harness import BaseGreedyController, run_scenario, scenario_metrics
    from pis.loop import LoopController
    from pis.privacy import scrub as _scrub
    seed, kind = job["seed"], job["kind"]
    t0 = time.time()
    try:
        spec = make_spec(seed)
        if kind == "calib":
            ctl = _CalibController(spec, _W["full"], job["ns"], job["hs"], job["m"])
            run_scenario(seed, ctl, _W["duration"])
            return {**job, "pool": ctl.pool, "seconds": time.time() - t0}
        if kind == "base":
            ctl = BaseGreedyController(spec)
        else:
            ctl = LoopController(spec, _sub(job["n"]), Params.from_dict(job["params"]), **job.get("opts", {}))
        recs = run_scenario(seed, ctl, _W["duration"])
        return {**job, "metrics": scenario_metrics(recs, spec), "seconds": time.time() - t0}
    except Exception as e:                       # recorded, never replaced
        return {**job, "error": _scrub(f"{type(e).__name__}: {e}"), "seconds": time.time() - t0}


# ------------------------------------------------------------------ analysis
def calibrate(pools: list[dict], qs: list[float]) -> dict:
    """{"n,h": {q: tau}} from the pooled disagreements of every calibration run."""
    merged: dict[str, list[float]] = {}
    for p in pools:
        for k, v in p.items():
            merged.setdefault(k, []).extend(v)
    out = {}
    for k, v in merged.items():
        arr = np.asarray(v, float)
        arr = arr[np.isfinite(arr)]
        out[k] = {q: (math.inf if math.isinf(q) else (float(np.percentile(arr, q)) if len(arr) else math.nan))
                  for q in qs}
    return out


def compare(base: dict[int, dict], arm: dict[int, dict]) -> dict:
    """Paired (arm - base) statistics over the seeds both finished."""
    seeds = sorted(set(base) & set(arm))
    def d(key):
        return paired_diff(np.array([arm[s][key] for s in seeds]), np.array([base[s][key] for s in seeds]))
    def mean(key):
        return float(np.mean([arm[s][key] for s in seeds])) if seeds else math.nan
    return {"n": len(seeds), "value": d("value_per_s"), "slo_v2": d("slo_v2_pct"), "slo_v1": d("slo_v1_pct"),
            "late": d("a_late_pct"), "fallback_rate": mean("fallback_rate"), "greedy_rate": mean("greedy_rate"),
            "decide_p95": mean("decide_ms_p95"),
            "decide_p95_worst": float(max((arm[s]["decide_ms_p95"] for s in seeds), default=math.nan)),
            "decide_max": float(max((arm[s]["decide_ms_max"] for s in seeds),
                                                                       default=math.nan)),
            "changes": mean("policy_changes")}


DECIDE_P95_MS, DECIDE_MAX_MS = 50.0, 500.0


def _finite_params(p: dict) -> bool:
    """Every value finite; tau may be the explicit "inf" of a no-gate config, never NaN."""
    for k, v in p.items():
        if k == "tau" and v == "inf":
            continue
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return False
    return True


def select_best(rows: list[dict], slo_margin: float, n_seeds: int) -> tuple[int | None, list[bool], bool]:
    """(index of the best feasible row, feasibility of every row, whether the best beats the base).

    Feasible: paired results on every evaluation seed; upper 95% bound of the SLO v2 difference <= slo_margin
    (SLO v1 is reported only); finite params (no NaN tau); worst per-scenario decision p95 <= 50 ms and
    max <= 500 ms. Best: largest mean value difference, ties -> smaller N*M*H. Beats base: that mean > 0 and
    its CI lower bound > 0.
    """
    def ok(r):
        m = r["cmp"]
        return (m["n"] == n_seeds and n_seeds >= 2 and m["slo_v2"]["hi"] <= slo_margin
                and math.isfinite(m["value"]["mean"]) and _finite_params(r["params"])
                and m["decide_p95_worst"] <= DECIDE_P95_MS and m["decide_max"] <= DECIDE_MAX_MS)
    feas = [bool(ok(r)) for r in rows]
    cands = [i for i, f in enumerate(feas) if f]
    if not cands:
        return None, feas, False
    c = rows
    best = max(cands, key=lambda i: (c[i]["cmp"]["value"]["mean"], -c[i]["cfg"]["n"] * c[i]["cfg"]["m"]
                                     * c[i]["cfg"]["h"]))
    v = rows[best]["cmp"]["value"]
    return best, feas, bool(v["mean"] > 0 and v["lo"] > 0)


def _f(x, nd=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else (
        "inf" if isinstance(x, float) and math.isinf(x) else f"{x:.{nd}f}")


def _ci(s: dict, nd=1) -> str:
    return f"{_f(s['mean'], nd)} [{_f(s['lo'], nd)}, {_f(s['hi'], nd)}]"


def render(meta: dict, base_summary: dict, taus: dict, rows: list[dict], best: int | None, feas: list[bool],
           ablations: dict, failures: list[dict], beats_base: bool = False) -> str:
    L = [f"# Step A tuning on development seeds{' (' + meta['label'] + ')' if meta.get('label') else ''}", ""]
    L += [f"- {k}: {v}" for k, v in meta.items() if k != "label" and not k.startswith("_")]
    L += ["", "Paired differences are (loop - base) per scenario, mean [95% CI]. Value in value units per "
          "second; SLO in percentage points of windows violating the SLO. Feasible = upper bound of the "
          f"SLO v2 difference <= +{meta['slo margin (pp)']} pp (SLO v1 reported only), paired results on every "
          f"evaluation seed, finite parameters, decision time p95 <= {DECIDE_P95_MS:g} ms in every scenario and max "
          f"<= {DECIDE_MAX_MS:g} ms.", "",
          "## Base scheduler (R6 greedy)", "",
          f"- scenarios: {base_summary['n']}; mean value per second {_f(base_summary['value'])}; "
          f"SLO v2 {_f(base_summary['slo_v2'], 2)} %; SLO v1 {_f(base_summary['slo_v1'], 2)} %", "",
          "## Calibrated trust thresholds (tau)", "",
          "Percentiles of the ensemble disagreement of every candidate at base-scheduler states "
          "(calibration seeds).", "", "| N | H | " + " | ".join(f"q{_f(q, 0)}" for q in meta["_qs"]) + " |",
          "|---|---|" + "---|" * len(meta["_qs"])]
    for k, v in sorted(taus.items()):
        n, h = k.split(",")
        L.append(f"| {n} | {h} | " + " | ".join(_f(v[q], 4) for q in meta["_qs"]) + " |")
    L += ["", "## Grid", "", "| N | M | H | lam | tau q | tau | value diff | SLO v2 diff | SLO v1 diff | "
          "fallback | greedy applied | decide p95 ms | n | feasible |", "|" + "---|" * 14]
    order = sorted(range(len(rows)), key=lambda i: -(rows[i]["cmp"]["value"]["mean"]
                                                      if math.isfinite(rows[i]["cmp"]["value"]["mean"]) else -1e18))
    for i in order:
        c, m = rows[i]["cfg"], rows[i]["cmp"]
        mark = " **chosen**" if i == best else ""
        L.append(f"| {c['n']} | {c['m']} | {c['h']} | {_f(c['lam'], 2)} | {_f(c['tau_q'], 0)} | "
                 f"{_f(rows[i]['params']['tau'] if not isinstance(rows[i]['params']['tau'], str) else math.inf, 4)} | "
                 f"{_ci(m['value'])} | {_ci(m['slo_v2'], 2)} | {_ci(m['slo_v1'], 2)} | {_f(m['fallback_rate'], 3)} | "
                 f"{_f(m['greedy_rate'], 3)} | {_f(m['decide_p95'], 1)} | {m['n']} | "
                 f"{'yes' if feas[i] else 'no'}{mark} |")
    L += ["", "## Chosen configuration", ""]
    if best is None:
        L.append("No configuration met the SLO non-inferiority constraint; no configuration is recommended.")
    else:
        if not beats_base:
            v = rows[best]["cmp"]["value"]
            L += [f"**No configuration beats the base scheduler**: the best feasible configuration's paired value "
                  f"difference is {_ci(v)}: its mean is not above zero or its 95% CI does not exclude zero. It is "
                  "recorded below (and in the config with `beats_base: false`) for reference only.", ""]
        L.append(f"`{json.dumps(rows[best]['params'])}` with N = {rows[best]['cfg']['n']} members.")
        L += ["", "Selected as the best of the grid on these seeds, so its paired value difference is "
              "optimistic (winner's curse); only the held-out evaluation measures it."]
    if ablations:
        L += ["", "## Ablations at the chosen configuration", "",
              "| arm | value diff vs base | SLO v2 diff vs base | fallback | greedy applied | decide p95 ms | n |",
              "|---|---|---|---|---|---|---|"]
        for name, m in ablations.items():
            L.append(f"| {name} | {_ci(m['value'])} | {_ci(m['slo_v2'], 2)} | {_f(m['fallback_rate'], 3)} | "
                     f"{_f(m['greedy_rate'], 3)} | {_f(m['decide_p95'], 1)} | {m['n']} |")
        L += ["", "random = uniform choice among the same candidates; oracle = each candidate run in a copy of "
              "the real simulator and scored with the same reward terms (upper bound of the candidate set); "
              "no_uncertainty = lam 0 (tau gate kept); no_fallback = no tau gate."]
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
    ap.add_argument("--members", default=MEMBERS, help="comma-separated, member 0 first; prefixes as pis.config")
    for k in ("calib_seeds", "seeds", "grid_n", "grid_m", "grid_h", "grid_lam", "grid_tau_q"):
        ap.add_argument("--" + k.replace("_", "-"), default=DEFAULTS[k])
    for k in ("mu", "change_cost", "slo_lam", "slo_margin", "duration"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=DEFAULTS[k])
    ap.add_argument("--max-configs", type=int, default=DEFAULTS["max_configs"])
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--label", default="", help='"smoke" for test runs')
    ap.add_argument("--out", default=OFFICIAL_NOTE)
    ap.add_argument("--config-out", default=OFFICIAL_CONFIG)
    ap.add_argument("--raw", default=None)
    a = ap.parse_args(argv)
    args = vars(a)
    members = a.members.split(",")
    dirty = bool(_git("status", "--porcelain", "--", "pis", "experiments"))
    config_exists = Path(OFFICIAL_CONFIG).exists() or Path(a.config_out).exists()
    errors = check_run(args, dirty, Path(a.out).exists(), config_exists, len(members))
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))
    from pis.config import resolve
    missing = [m for m in members if not resolve(m).is_file()]
    if missing:
        raise SystemExit("missing ensemble members: " + ", ".join(safe_path(m) for m in missing))
    from pis.evaluator import Params
    from pis.ensemble import _sha256
    cfgs = grid(args)
    calib, seeds = parse_seeds(a.calib_seeds), parse_seeds(a.seeds)
    qs = sorted(set(c["tau_q"] for c in cfgs))
    ns, hs = sorted(set(c["n"] for c in cfgs)), sorted(set(c["h"] for c in cfgs))
    t0 = time.time()
    pool = None
    if a.procs > 1:
        from multiprocessing import get_context
        pool = get_context("spawn").Pool(a.procs, initializer=_init_worker, initargs=(members, a.duration))
    else:
        _init_worker(members, a.duration)
    try:
        jobs = ([{"kind": "calib", "seed": s, "ns": ns, "hs": hs, "m": max(c["m"] for c in cfgs)} for s in calib]
                + [{"kind": "base", "seed": s} for s in seeds])
        res1 = _run(pool, jobs, "calibration + base", t0)
        taus = calibrate([r["pool"] for r in res1 if r["kind"] == "calib" and "pool" in r], qs)
        base = {r["seed"]: r["metrics"] for r in res1 if r["kind"] == "base" and "metrics" in r}
        params = []
        for c in cfgs:
            tau = taus.get(f"{c['n']},{c['h']}", {}).get(c["tau_q"], math.nan)
            params.append(Params(n_candidates_m=c["m"], horizon_h=c["h"], lam=c["lam"], mu=a.mu,
                                 change_cost=a.change_cost, tau=tau, slo_lam=a.slo_lam).to_dict())
        jobs = [{"kind": "loop", "seed": s, "cfg_id": i, "n": c["n"], "params": params[i]}
                for i, c in enumerate(cfgs) for s in seeds]
        res2 = _run(pool, jobs, "grid", t0)
        rows = []
        for i, c in enumerate(cfgs):
            arm = {r["seed"]: r["metrics"] for r in res2 if r["cfg_id"] == i and "metrics" in r}
            rows.append({"cfg": c, "params": params[i], "cmp": compare(base, arm)})
        best, feas, beats_base = select_best(rows, a.slo_margin, len(seeds))
        ablations, res3 = {}, []
        if best is not None and not a.no_ablations:
            jobs = [{"kind": "ablation", "name": name, "seed": s, "n": cfgs[best]["n"], "params": params[best],
                     "opts": opts} for name, opts in ABLATIONS.items() for s in seeds]
            res3 = _run(pool, jobs, "ablations", t0)
            for name in ABLATIONS:
                arm = {r["seed"]: r["metrics"] for r in res3 if r.get("name") == name and "metrics" in r}
                ablations[name] = compare(base, arm)
            ablations = {"loop (chosen)": rows[best]["cmp"], **ablations}
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    failures = [r for r in res1 + res2 + res3 if "error" in r]
    runtime = time.time() - t0
    head = _git("rev-parse", "HEAD") or "unknown"
    meta = {"label": a.label,
            "code": f"git HEAD {head}{' (uncommitted changes)' if dirty else ''}",
            "members": ", ".join(f"{safe_path(m)} (sha256 {_sha256(resolve(m))[:16]})" for m in members),
            "calibration seeds": a.calib_seeds, "evaluation seeds": a.seeds,
            "grid": f"N {a.grid_n}; M {a.grid_m}; H {a.grid_h}; lam {a.grid_lam}; tau percentile {a.grid_tau_q}",
            "fixed": f"mu {a.mu}, change_cost {a.change_cost}, slo_lam {a.slo_lam}, scenario {a.duration:g} s",
            "slo margin (pp)": a.slo_margin,
            "disagreement": "std across members of a.p99_ms, a.queue_work_ms, b.rate, c.work_core_ms (encoded), "
                            "divided by member 0's training input std, mean over keys and horizon",
            "runtime": f"{runtime:.0f} s with {a.procs} processes", "_qs": qs}
    base_summary = {"n": len(base), **{k: float(np.mean([m[v] for m in base.values()])) if base else math.nan
                                       for k, v in (("value", "value_per_s"), ("slo_v2", "slo_v2_pct"),
                                                    ("slo_v1", "slo_v1_pct"))}}
    text = render(meta, base_summary, taus, rows, best, feas, ablations, failures, beats_base)
    meta.pop("_qs")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(scrub(text))
    if best is not None:
        cfg = {"members": [safe_path(m) for m in members[:cfgs[best]["n"]]], "params": params[best],
               "beats_base": beats_base,
               "tau_percentile": cfgs[best]["tau_q"] if math.isfinite(cfgs[best]["tau_q"]) else "inf",
               "selected_on_seeds": a.seeds, "calibration_seeds": a.calib_seeds, "code": head,
               "label": a.label}
        cp = Path(a.config_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(scrub(json.dumps(cfg, indent=1, allow_nan=False)) + "\n")
    raw = Path(a.raw or f"results/raw/step_a/tuning{'_' + a.label if a.label else ''}.json")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(scrub(json.dumps({"meta": meta, "taus": {k: {str(q): t for q, t in v.items()}
                                                             for k, v in taus.items()},
                                     "rows": rows, "best": best, "base": base, "ablations": ablations,
                                     "runs": [{k: v for k, v in r.items() if k != "pool"}
                                              for r in res1 + res2 + res3]}, default=str)))
    print(f"chosen config {best} (beats base: {beats_base}) -> {safe_path(str(out))}, runtime {runtime:.0f} s")
    return {"best": best, "beats_base": beats_base, "rows": rows, "taus": taus, "ablations": ablations}


if __name__ == "__main__":
    main()
