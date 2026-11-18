"""Final paired evaluation: the arms, one run of an arm on one scenario, and the paired report.

Arms (each run on every evaluation scenario, all starting from the same policy and simulator seed):
  base_r6              the base scheduler, RL student R6 acting greedily (the reference)
  base_r3              the earlier RL student R3 acting greedily (reference point)
  loop_a               the Step A loop with its chosen members and Params
  loop_b               the Step B loop (Step A plus revision) with its chosen reviser
  random_selection     Step A's candidates, chosen uniformly at random (lower bound: no ranking)
  oracle_selection     Step A's candidates, chosen by the real simulator (upper bound: candidate headroom)
  loop_no_uncertainty  Step A without the disagreement penalty (lam = 0; the tau gate stays)
  loop_no_fallback     Step A without the tau gate
  adaptive_c9          the non-ML feedback controller, configuration 9 of the sibling's adaptive grid

Arms that rank candidates with the ensemble (loop_a, loop_b, loop_no_uncertainty, loop_no_fallback) are also
probed after every `probe_every`-th model decision, outside the timed part: every candidate is run for H
windows in a copy of the real simulator and compared with the members' forecasts. That gives two run-time
model metrics per scenario:
  gru_error        mean |ensemble-mean forecast - real| over candidates, horizon and the disagreement keys
                   (encoded units divided by member 0's training std, the scale of the disagreement)
  rank_agreement   mean Spearman correlation between the members' predicted reward of the candidates
                   (no uncertainty penalty) and their real reward over the same horizon
  top1_agreement   share of probes where the predicted best candidate is also the real best
"""
from __future__ import annotations

import math
import time

import numpy as np

from pis.privacy import safe_path, scrub
from pis.stats import paired_diff

ARMS = ("base_r6", "base_r3", "loop_a", "loop_b", "random_selection", "oracle_selection",
        "loop_no_uncertainty", "loop_no_fallback", "adaptive_c9")
REFERENCE = "base_r6"
MODEL_ARMS = ("loop_a", "loop_b", "loop_no_uncertainty", "loop_no_fallback")
LOOP_OPTIONS = {"loop_a": {}, "random_selection": {"selection": "random"},
                "oracle_selection": {"selection": "oracle"}, "loop_no_uncertainty": {"use_uncertainty": False},
                "loop_no_fallback": {"use_fallback": False}}
# reported per scenario; (key, label, decimals)
METRICS = (("value_per_s", "value /s", 1), ("slo_v2_pct", "SLO v2 %", 2), ("slo_v1_pct", "SLO v1 %", 2),
           ("a_late_pct", "A late %", 2), ("a_queue_overdue_end", "A overdue backlog at end", 2),
           ("policy_changes", "policy changes", 1))
RUN_METRICS = (("decide_ms_mean", "decide mean ms", 1), ("decide_ms_p95", "decide p95 ms", 1),
               ("decide_ms_max", "decide max ms", 1), ("fallback_rate", "fallback rate", 3),
               ("gru_error", "GRU error", 3), ("rank_agreement", "rank agreement", 3),
               ("top1_agreement", "top-1 agreement", 3))


# ------------------------------------------------------------------ arm definitions
def arm_definitions(protocol: dict, step_a: dict, step_b: dict) -> dict[str, dict]:
    """Every arm of the protocol resolved to what builds its controller (members, Params, options)."""
    out = {}
    for arm, d in protocol["arms"].items():
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        kind = d["kind"]
        if kind == "base":
            out[arm] = {"kind": "base", "checkpoint": d["checkpoint"]}
        elif kind == "adaptive":
            out[arm] = {"kind": "adaptive", **{k: d[k] for k in ("h0", "r_hi", "hold")}}
        elif kind == "loop":
            out[arm] = {"kind": "loop", "members": list(step_a["members"]), "params": dict(step_a["params"]),
                        "options": dict(d.get("options", {}))}
        elif kind == "reviser":
            r = step_b["reviser"]
            out[arm] = {"kind": "reviser", "members": list(step_b["members"]),
                        "params": dict(step_b["step_a_params"]),
                        "reviser": {k: r[k] for k in ("mechanism", "rev_steps", "rev_lam", "rev_lr")}}
        else:
            raise ValueError(f"arm {arm!r}: unknown kind {kind!r}")
    return out


# ------------------------------------------------------------------ run-time model probe
def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks 0..n-1 with ties averaged."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b) -> float:
    """Spearman correlation; nan when either side is constant or there are fewer than 3 values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.nan
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _probed(base):
    """A loop controller class whose model decisions are compared with reality every `probe_every`-th time."""

    class Probed(base):
        def __init__(self, *args, probe_every: int = 0, **kw):
            super().__init__(*args, **kw)
            self.probe_every, self._n_model = int(probe_every), 0

        def decide(self, tel, current, history):
            policy, info = super().decide(tel, current, history)   # timed inside
            if (self.probe_every > 0 and self.selection == "model" and self.last_rollouts is not None
                    and info.get("reason") not in ("short_history", "no_candidates")):
                self._n_model += 1
                if self._n_model % self.probe_every == 0:
                    info["probe"] = self._probe_reality(current)
            return policy, info

        def _probe_reality(self, current: dict) -> dict:
            from pis.adapters import clone_sim
            from pis.config import setup_imports
            from pis.ensemble import DISAGREEMENT_KEYS, N_STATE
            from pis.evaluator import score_candidates
            setup_imports()
            from sim.policy import policy_from_dict
            from worldmodel.features import STATE_NAMES, encode_state
            if self.sim is None:
                raise RuntimeError("the run-time probe needs attach_sim(sim) (run_scenario does this)")
            cands, roll = self.last_candidates, np.asarray(self.last_rollouts, np.float64)
            h = roll.shape[2]
            real = np.empty((1, len(cands), h, N_STATE), dtype=np.float32)
            for i, c in enumerate(cands):
                sim, pol = clone_sim(self.sim), policy_from_dict(c)
                for t in range(h):
                    real[0, i, t] = encode_state(sim.run_window(pol))
            pred, _ = score_candidates(self.spec, roll, cands, current, self.params, lam=0.0)
            true, _ = score_candidates(self.spec, real, cands, current, self.params, lam=0.0)
            idx = [STATE_NAMES.index(k) for k in DISAGREEMENT_KEYS]
            err = np.abs(roll.mean(axis=0)[..., idx] - real[0][..., idx]) / np.asarray(
                self.ensemble.train_std, np.float64)[idx]
            ok = np.isfinite(pred) & np.isfinite(true)
            return {"gru_error": float(np.nanmean(err)),
                    "rank_agreement": spearman(pred[ok], true[ok]),
                    "top1": (bool(np.argmax(np.where(ok, pred, -np.inf)) == np.argmax(np.where(ok, true, -np.inf)))
                             if ok.sum() >= 2 else None)}

    Probed.__name__ = f"Probed{base.__name__}"
    return Probed


def probe_metrics(records: list[dict]) -> dict:
    """Per-scenario run-time model metrics from the probes in `records` (nan when there are none)."""
    probes = [r["info"]["probe"] for r in records if "probe" in r["info"]]
    errs = [p["gru_error"] for p in probes if math.isfinite(p["gru_error"])]
    ranks = [p["rank_agreement"] for p in probes if math.isfinite(p["rank_agreement"])]
    top = [p["top1"] for p in probes if p["top1"] is not None]
    return {"probes": len(probes), "gru_error": float(np.mean(errs)) if errs else math.nan,
            "rank_agreement": float(np.mean(ranks)) if ranks else math.nan, "rank_probes": len(ranks),
            "top1_agreement": float(np.mean(top)) if top else math.nan}


# ------------------------------------------------------------------ controllers
class AdaptiveArm:
    """The sibling's non-ML feedback controller behind the harness's controller protocol."""

    def __init__(self, spec, h0: float, r_hi: float, hold: int):
        from pis.config import setup_imports
        setup_imports()
        from sim.controllers import AdaptiveController
        self.ctl = AdaptiveController(spec, h0=h0, r_hi=r_hi, hold=hold)

    def decide(self, tel, current, history):
        from sim.dataset import policy_dict
        t0 = time.perf_counter()
        p = policy_dict(self.ctl.decide(tel))
        return p, {"decide_ms": (time.perf_counter() - t0) * 1000.0}


_ENSEMBLES: dict[tuple, object] = {}


def ensemble_for(members: list[str]):
    """One loaded ensemble per member list per process."""
    key = tuple(members)
    if key not in _ENSEMBLES:
        from pis.ensemble import Ensemble
        _ENSEMBLES[key] = Ensemble.from_paths(list(members))
    return _ENSEMBLES[key]


def make_controller(defn: dict, spec, probe_every: int = 0):
    from pis.evaluator import Params
    kind = defn["kind"]
    if kind == "base":
        from pis.harness import BaseGreedyController
        return BaseGreedyController(spec, defn["checkpoint"])
    if kind == "adaptive":
        return AdaptiveArm(spec, defn["h0"], defn["r_hi"], defn["hold"])
    ens, params = ensemble_for(defn["members"]), Params.from_dict(defn["params"])
    if kind == "loop":
        from pis.loop import LoopController
        opts = defn.get("options", {})
        pe = probe_every if opts.get("selection", "model") == "model" else 0
        return _probed(LoopController)(spec, ens, params, probe_every=pe, **opts)
    if kind == "reviser":
        from pis.reviser import ReviserLoopController
        r = defn["reviser"]
        return _probed(ReviserLoopController)(spec, ens, params, probe_every=probe_every,
                                              mechanism=r["mechanism"], rev_steps=r["rev_steps"],
                                              rev_lam=r["rev_lam"], rev_lr=r["rev_lr"])
    raise ValueError(f"unknown kind {kind!r}")


def run_arm(arm: str, seed: int, arms: dict[str, dict], duration_s: float = 60.0, probe_every: int = 0) -> dict:
    """Run one arm on one scenario. Returns {"seed", "arm", "failed": False, "metrics", "seconds"}; any
    exception becomes {"seed", "arm", "failed": True, "error": scrubbed message, "seconds"} (never raised)."""
    t0 = time.time()
    try:
        from pis.adapters import make_spec
        from pis.harness import run_scenario, scenario_metrics
        spec = make_spec(seed)
        ctl = make_controller(arms[arm], spec, probe_every)
        recs = run_scenario(seed, ctl, duration_s)
        metrics = {**scenario_metrics(recs, spec), **probe_metrics(recs)}
        revised = [r for r in recs if r["info"].get("revised")]
        if arms[arm]["kind"] == "reviser":
            metrics["revisions"] = len(revised)
        return {"seed": seed, "arm": arm, "failed": False, "metrics": metrics, "seconds": time.time() - t0}
    except Exception as e:                       # recorded, never replaced
        return {"seed": seed, "arm": arm, "failed": True, "error": scrub(f"{type(e).__name__}: {e}"),
                "seconds": time.time() - t0}


# ------------------------------------------------------------------ analysis
def per_arm(raw: list[dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for r in raw:
        if not r.get("failed"):
            out.setdefault(r["arm"], {})[r["seed"]] = r["metrics"]
    return out


def paired(per: dict, a: str, b: str, key: str, conf: float = 0.95) -> dict:
    """Paired (a - b) over the seeds where both arms succeeded."""
    pa, pb = per.get(a, {}), per.get(b, {})
    seeds = sorted(set(pa) & set(pb))
    d = paired_diff(np.array([pa[s][key] for s in seeds], float), np.array([pb[s][key] for s in seeds], float),
                    conf)
    return {**d, "seeds": seeds}


def success(per: dict, arm: str, rule: dict, timing: dict | None) -> dict:
    """The protocol's success rule for one candidate arm against the reference."""
    v = paired(per, arm, rule["reference"], "value_per_s", rule["confidence"])
    s = paired(per, arm, rule["reference"], "slo_v2_pct", rule["confidence"])
    value_ok = bool(v["n"] >= 2 and v["mean"] > 0 and v["lo"] > 0)
    slo_ok = bool(s["n"] >= 2 and s["hi"] <= rule["slo_v2_margin_pp"])
    time_ok = None if timing is None else bool(timing["p95_ok"] and timing["max_ok"])
    return {"value": v, "slo_v2": s, "value_ok": value_ok, "slo_ok": slo_ok, "time_ok": time_ok,
            "met": bool(value_ok and slo_ok and time_ok is True)}


def _f(x, nd=1) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _ci(d: dict, nd=1) -> str:
    star = "*" if d["n"] >= 2 and (d["lo"] > 0 or d["hi"] < 0) else ""
    return f"{d['mean']:+.{nd}f}{star} [{_f(d['lo'], nd)}, {_f(d['hi'], nd)}]" if d["n"] else "n/a"


def _mean(rows: dict[int, dict], key: str) -> float:
    vals = [m[key] for m in rows.values() if key in m and m[key] is not None and math.isfinite(m[key])]
    return float(np.mean(vals)) if vals else math.nan


def report(raw: list[dict], protocol: dict | None = None, meta: dict | None = None,
           timing: dict | None = None, seeds: list[int] | None = None, background: bool = True) -> str:
    """Markdown report of a final-evaluation attempt (not yet scrubbed).

    Paired differences per scenario against the reference arm with Student-t CIs, over the scenarios where
    both arms succeeded; failed runs are listed and marked in the per-scenario table, never replaced.
    `timing` maps "loop_a" / "loop_b" to the normal-priority decision-time verdict (or None if not available);
    `background` says whether this run's own decision times were measured under background QoS.
    """
    protocol = protocol or {}
    arms = list(protocol.get("arms") or dict.fromkeys(a for a in ARMS if any(r["arm"] == a for r in raw)))
    ref = protocol.get("reference_arm", REFERENCE)
    rule = {"reference": ref, "confidence": 0.95, "slo_v2_margin_pp": 1.0, "candidates": ["loop_a", "loop_b"],
            **protocol.get("success_rule", {})}
    timing = timing or {}
    seeds = sorted(seeds if seeds is not None else {r["seed"] for r in raw})
    per = per_arm(raw)
    failed = [r for r in raw if r.get("failed")]
    title = (meta or {}).get("title", "Final paired evaluation")
    L = [f"## {title}", ""]
    L += [f"- {k}: {v}" for k, v in (meta or {}).items() if k != "title"]
    n_total = len(arms) * len(seeds)
    L += ["", "### Runs", "",
          f"{len(raw) - len(failed)} succeeded, {len(failed)} failed of {n_total} ({len(arms)} arms x "
          f"{len(seeds)} scenarios)."]
    missing = [(a, s) for a in arms for s in seeds if not any(r["arm"] == a and r["seed"] == s for r in raw)]
    if failed:
        L += ["", "Failed runs (recorded, not replaced; every comparison uses the scenarios where both arms "
              "succeeded):", ""]
        L += [f"- FAILED {r['arm']} seed {r['seed']}: {r['error']}" for r in sorted(failed, key=lambda r: (
            r["arm"], r["seed"]))]
    if missing:
        L += ["", "Runs without any record (the attempt stopped before they finished): "
              + ", ".join(f"{a} seed {s}" for a, s in missing[:50]) + (" ..." if len(missing) > 50 else "")]

    L += ["", "### Success rule", ""]
    L += [f"Each candidate arm against `{ref}`, paired per scenario: (a) value per second improves, the "
          f"paired {int(rule['confidence'] * 100)}% CI of the difference lies above zero; (b) SLO v2 not worse, "
          f"the upper bound of the paired difference in SLO v2 window-violation % is at most "
          f"+{rule['slo_v2_margin_pp']:g} pp; (c) decision time within the official limits (p95 <= "
          f"{protocol.get('decision_time_ms', {}).get('p95_target', 50):g} ms, max <= "
          f"{protocol.get('decision_time_ms', {}).get('hard_limit', 500):g} ms per scenario), checked at "
          "normal priority by experiments/time_decisions.py, not by this run's wall clock.", "",
          "| arm | value /s vs ref (a) | SLO v2 pp vs ref (b) | decision time (c) | n | success |",
          "|---|---|---|---|---|---|"]
    verdicts = {}
    for arm in rule["candidates"]:
        if arm not in arms:
            continue
        t = timing.get(arm)
        sv = success(per, arm, rule, t)
        verdicts[arm] = sv
        tt = ("not available" if t is None else
              f"{'within' if sv['time_ok'] else '**over**'} (worst p95 {t['worst_p95']:.1f} ms, max "
              f"{t['worst_max']:.1f} ms; {t['source']})")
        L.append(f"| {arm} | {_ci(sv['value'])} {'met' if sv['value_ok'] else 'not met'} | "
                 f"{_ci(sv['slo_v2'], 2)} {'met' if sv['slo_ok'] else 'not met'} | {tt} | {sv['value']['n']} | "
                 f"**{'yes' if sv['met'] else 'no'}** |")
    any_met = [a for a, v in verdicts.items() if v["met"]]
    L += ["", ("**Success: " + ", ".join(any_met) + " met the rule.**") if any_met else
          "**No candidate arm met the success rule.** The result is reported as it is.",
          "", "A CI that includes zero means no detectable difference at this precision, not equivalence. "
          "* marks a CI that excludes zero."]

    L += ["", "### Means over scenarios", "",
          "| arm | n | " + " | ".join(lab for _, lab, _ in METRICS + RUN_METRICS) + " |",
          "|---|---|" + "---|" * (len(METRICS) + len(RUN_METRICS))]
    for arm in arms:
        rows = per.get(arm, {})
        L.append(f"| {arm} | {len(rows)} | " + " | ".join(_f(_mean(rows, k), nd)
                                                          for k, _, nd in METRICS + RUN_METRICS) + " |")
    L += ["", "GRU error, rank agreement and top-1 agreement come from run-time probes of the arms that rank "
          "candidates with the ensemble (every candidate run in a copy of the real simulator after the decision "
          "was timed); n/a for the other arms. Fallback rate: share of decisions where the loop applied the "
          "base scheduler's greedy policy because no candidate was trusted (or history was short)."]

    L += ["", f"### Paired differences vs `{ref}` (arm - {ref})", "",
          "| arm | n | " + " | ".join(lab for _, lab, _ in METRICS) + " |",
          "|---|---|" + "---|" * len(METRICS)]
    for arm in arms:
        if arm == ref:
            continue
        cells = [paired(per, arm, ref, k, rule["confidence"]) for k, _, _ in METRICS]
        L.append(f"| {arm} | {cells[0]['n']} | " + " | ".join(_ci(c, nd) for c, (_, _, nd) in zip(cells, METRICS))
                 + " |")

    secondary = protocol.get("secondary", [])
    if secondary:
        L += ["", "### Secondary comparisons (unadjusted, no decisions)", "",
              "| comparison | what it shows | n | value /s | SLO v2 pp | SLO v1 pp | A late pp |",
              "|---|---|---|---|---|---|---|"]
        for c in secondary:
            a, b = c["a"], c["b"]
            if a not in arms or b not in arms:
                continue
            cells = [paired(per, a, b, k, rule["confidence"]) for k in ("value_per_s", "slo_v2_pct", "slo_v1_pct",
                                                                         "a_late_pct")]
            L.append(f"| {a} - {b} | {c.get('why', '')} | {cells[0]['n']} | {_ci(cells[0])} | "
                     f"{_ci(cells[1], 2)} | {_ci(cells[2], 2)} | {_ci(cells[3], 2)} |")

    v = paired(per, rule["candidates"][0], ref, "value_per_s") if rule["candidates"][0] in arms else None
    if v and v["n"]:
        d = np.array([per[rule['candidates'][0]][s]["value_per_s"] - per[ref][s]["value_per_s"] for s in v["seeds"]])
        order = np.argsort(-np.abs(d))
        L += ["", f"{rule['candidates'][0]} - {ref} value per second: median {np.median(d):+.1f}; better in "
              f"{int((d > 0).sum())}, worse in {int((d < 0).sum())}, equal in {int((d == 0).sum())}; five largest "
              "|difference|: " + ", ".join(f"seed {v['seeds'][i]} {d[i]:+.1f}" for i in order[:5]) + "."]

    where = ("**Measured under background QoS** (launched through scripts/gentle.sh: macOS background priority "
             "on efficiency cores, several processes), which inflates wall-clock time." if background else
             "**Measured at normal priority with several processes in parallel** (a rehearsal, not through "
             "scripts/gentle.sh).")
    L += ["", "### Decision time in this run", "",
          where + " These numbers are reported only; the official limits are checked at normal priority, one "
          "process, by experiments/time_decisions.py (results/step_a/timing.md, results/step_b/timing.md).", "",
          "| arm | mean ms | worst per-scenario p95 ms | max ms | decisions over 500 ms |", "|---|---|---|---|---|"]
    for arm in arms:
        rows = per.get(arm, {})
        if not rows:
            L.append(f"| {arm} | n/a | n/a | n/a | n/a |")
            continue
        L.append(f"| {arm} | {_f(_mean(rows, 'decide_ms_mean'))} | "
                 f"{_f(max(m['decide_ms_p95'] for m in rows.values()))} | "
                 f"{_f(max(m['decide_ms_max'] for m in rows.values()))} | "
                 f"{sum(m.get('decisions_over_500ms', 0) for m in rows.values())} |")

    overdue = sorted({s for a in arms for s, m in per.get(a, {}).items() if m["a_queue_overdue_end"] > 0})
    L += ["", "### Overdue A backlog at the end", ""]
    if not overdue:
        L.append("No scenario ended with an overdue A backlog in any arm.")
    else:
        L += ["| seed | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for s in overdue:
            L.append(f"| {s} | " + " | ".join(str(per[a][s]["a_queue_overdue_end"]) if s in per.get(a, {})
                                               else "FAILED" for a in arms) + " |")

    L += ["", "### Per-scenario value per second / SLO v2 %", "",
          "| seed | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    fail_set = {(r["arm"], r["seed"]) for r in failed}
    for s in seeds:
        cells = []
        for a in arms:
            if s in per.get(a, {}):
                m = per[a][s]
                cells.append(f"{m['value_per_s']:.1f} / {m['slo_v2_pct']:.1f}")
            else:
                cells.append("FAILED" if (a, s) in fail_set else "missing")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def describe_arms(arms: dict[str, dict]) -> dict[str, str]:
    """One line per arm for the report's provenance."""
    out = {}
    for name, d in arms.items():
        if d["kind"] == "base":
            out[name] = f"base scheduler, sched:{d['checkpoint']}, greedy"
        elif d["kind"] == "adaptive":
            out[name] = f"adaptive feedback controller h0 {d['h0']}, r_hi {d['r_hi']}, hold {d['hold']}"
        elif d["kind"] == "loop":
            opts = ", ".join(f"{k}={v}" for k, v in d["options"].items()) or "as tuned"
            out[name] = f"Step A loop ({opts}); members {', '.join(safe_path(m) for m in d['members'])}"
        else:
            r = d["reviser"]
            out[name] = (f"Step B loop, {r['mechanism']} steps {r['rev_steps']} lam {r['rev_lam']} lr "
                         f"{r['rev_lr']}; members {', '.join(safe_path(m) for m in d['members'])}")
    return out
