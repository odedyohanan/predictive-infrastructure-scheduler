"""Shadow mode: five incident predictors run on stored R6 episodes, conformal alarm thresholds, and the
comparison of the learned predictors with the best non-learned one on development seeds.

  .venv/bin/python experiments/b_shadow.py --world-model models/pgru_b1.pt --procs 4
  .venv/bin/python experiments/b_shadow.py --world-model models/rssm_b2.pt --procs 4   # -> shadow_rssm_b2/
  .venv/bin/python experiments/b_shadow.py --label smoke --store data/phase_b/timing_probe \\
      --incident data/phase_b/smoke_g3/lt/incident.json --world-model models/smoke_g4/pgru_smoke.pt \\
      --fit-seeds 3100-3103,3105-3107,3109,3110,3113,3115,3116,3118 --calibration 3104,3108,3111,3112 \\
      --selection 3114,3117,3119 --mc-labels data/phase_b/smoke_g3/mc/mc_labels/labels.parquet \\
      --smoke-dir data/phase_b/smoke_<name>

Predictors: learned = world model (n sampled futures), hazard, analogue; not learned = queueing (its scale
set on the calibration episodes) and change point. Hazard, analogue and the change-point baseline are fitted
on history episodes (by default the world model's training split, seed % 10 not 8 or 9); the world model is
loaded from `--world-model`. The calibration and selection episodes are R6-driven and read from the store;
every predictor gives a risk at every window t from the warm-up on, from the rows up to t only (no
degradation columns). Risks are written per shard of episodes (`pis.shards`: resumable, a changed
configuration refused).

Scoring (`pis.phase_b.alarms`): per predictor a conformal threshold on the calibration episodes (false-alarm
share <= alpha = 0.10, `low` = 0.7 x threshold); on the selection episodes recall of incidents caught with a
lead >= min_lead_windows, median lead, pooled false-alarm rate, alarm events per quiet episode and the Brier
score of the window risk against the Monte Carlo labels, split by cause (a labelled state is "building_up"
when a degradation is active at it). All bootstrap intervals use one shared resample stream (same tag, same
selection episodes): common random numbers across predictors and causes. Question 1: each learned
predictor's building-up recall minus that of the better non-learned one, with a 95 % bootstrap interval over
selection episodes. The predictor with the best building-up recall (ties: lower false-alarm rate) goes to
the alarms config; the note ends with the gate: recall <= 0.2 stops before prevention.

Real runs: the full calibration (3400-3449) and selection (3450-3499) ranges, all five predictors, a clean tree,
configs/phase_b/incident.json (rule, H, min_lead_windows), the Monte Carlo labels and a world model trained on
history seeds only; outputs the raw directory of the world model (results/raw/phase_b/shadow/ for
models/pgru_b1.pt, results/raw/phase_b/shadow_<checkpoint name>/ for any other, so reruns with another world
model never mix with earlier shards), configs/phase_b/alarms.json (which records the world model and that raw
directory for the advisor scripts) and results/phase_b/shadow_dev.md. Smoke runs (`--label smoke`) may use
history episodes as stand-ins, allow a dirty tree and write only into `--smoke-dir`. Long runs go through
scripts/gentle.sh.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
DEFAULT_INCIDENT = "configs/phase_b/incident.json"
DEFAULT_MC = "data/phase_b/mc_labels/labels.parquet"
DEFAULT_WM = runs.SHADOW_WORLD_MODEL           # models/pgru_b1.pt
ALARMS = "configs/phase_b/alarms.json"
NOTE = "results/phase_b/shadow_dev.md"
DEFAULT_OUT = runs.SHADOW_DIR                   # results/raw/phase_b/shadow
ML = ("world_model", "hazard", "analogue")
NON_ML = ("queueing", "change_point")
ALL = ML + NON_ML
SHARD_SIZE = 5
WM_TAG = 17_000_000         # world-model sampling at window t of episode s: WM_TAG + s*1000 + t
GATE_RECALL = 0.2
RISKS_FILE = "risks.parquet"
TIMING_FILE = "timing.json"
FITTED_FILE = "predictors.pkl"
HIDDEN = ("deg_kind", "deg_active", "deg_factor")
SOURCE = "r6"
FULL_EPISODES, FULL_WINDOWS = 100, 600


def raw_dir_for(world_model) -> str:
    """Raw output directory of a real run with this world-model checkpoint."""
    stem = Path(world_model).stem if world_model else Path(DEFAULT_WM).stem
    return DEFAULT_OUT if stem == Path(DEFAULT_WM).stem else f"{DEFAULT_OUT}_{stem}"


def default_fit_seeds() -> list[int]:
    return [s for s in seed_ranges.HISTORY if s % 10 not in (8, 9)]


# ------------------------------------------------------------------ predictors
def make_predictors(store, names, rule, H, fit_seeds, calibration, smoke, overrides) -> dict:
    """Fitted predictors except the world model (loaded in each worker)."""
    from pis.phase_b import features, incidents, predictors
    rule_ = incidents.IncidentRule(rule["k"], rule["interval"], rule["majority"])
    warmup, max_h = int(rule["warmup"]), max(40, int(H))
    ov = overrides or {}
    out = {}
    if "hazard" in names:
        out["hazard"] = predictors.HazardPredictor(rule_, max_h=max_h, warmup=warmup, **ov.get("hazard", {}))
        out["hazard"].fit(store, fit_seeds)
    if "analogue" in names:
        out["analogue"] = predictors.AnalogueSearch(rule_, max_h=max_h, warmup=warmup, **ov.get("analogue", {}))
        out["analogue"].fit(store, fit_seeds)
    if "change_point" in names:
        out["change_point"] = predictors.ChangePoint(**ov.get("change_point", {}))
        out["change_point"].fit(store, fit_seeds)
    if "queueing" in names:
        q = predictors.QueueingModel(rule_, warmup=warmup, **ov.get("queueing", {}))
        if smoke:           # stand-in calibration episodes may be history seeds
            eps = store.episodes().set_index("seed")
            parts = [features.join_known(store.window_rows(s, 0, int(eps.loc[s, "windows"])), eps.reset_index())
                     for s in calibration]
            q.calibrate_episodes(parts, H)
        else:
            q.calibrate(store, calibration, H)
        out["queueing"] = q
    return out


def load_predictors(fitted, names, rule: dict, world_model=None, wm_n: int = 32) -> dict:
    """The predictors in `names` as the shadow run uses them: the fitted ones from its pickle (`fitted`, may
    be None when only the world model is asked for), the world model from its checkpoint with `wm_n`
    futures. Call `for_episode` before each episode."""
    from pis.phase_b import incidents, predictors
    preds = pickle.loads(Path(fitted).read_bytes()) if fitted is not None else {}
    if "world_model" in names:
        from pis.phase_b.world_models import load_world_model
        if world_model is None:
            raise ValueError("the world-model predictor needs a checkpoint")
        rule_ = incidents.IncidentRule(rule["k"], rule["interval"], rule["majority"])
        preds["world_model"] = predictors.WorldModelPredictor(load_world_model(world_model), rule_,
                                                              n=wm_n, warmup=int(rule["warmup"]))
    missing = [n for n in names if n not in preds]
    if missing:
        raise ValueError(f"no fitted predictor {missing} in the shadow run's pickle")
    return {n: preds[n] for n in names}


def for_episode(preds: dict, episode: int) -> None:
    """Sampling seeds of the world-model predictor for an episode: WM_TAG + episode*1000 (+ the window)."""
    if "world_model" in preds:
        preds["world_model"].seed = WM_TAG + int(episode) * 1000


def episode_risks(preds: dict, names, rows, H: int, warmup: int, episode: int, seconds: dict | None = None) -> dict:
    """Risk series of each predictor over one episode's rows (NaN before the warm-up): at window t from the
    rows up to t only, the degradation columns dropped. Adds the time spent per predictor to `seconds`."""
    rows = rows.drop(columns=[c for c in HIDDEN if c in rows.columns])
    for_episode(preds, episode)
    T = len(rows)
    risk = {n: np.full(T, np.nan) for n in names}
    for t in range(warmup, T):
        seen = rows.iloc[:t + 1]
        for n in names:
            t0 = time.perf_counter()
            risk[n][t] = preds[n].risk(seen, H, episode=episode)
            if seconds is not None:
                seconds[n] += time.perf_counter() - t0
    return risk


def run_shard(shard, shard_dir: Path, params: dict) -> None:
    """Risk series of every predictor for the shard's episodes (NaN before the warm-up)."""
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    from pis.phase_b import features
    from pis.phase_b.history import Store
    store = Store(params["store"])
    eps = store.episodes().set_index("seed")
    rule = params["rule"]
    H, warmup = int(rule["H"]), int(rule["warmup"])
    preds = load_predictors(params["fitted"], params["names"], rule, params["world_model"], params["wm_n"])
    names = [n for n in params["names"] if n in preds]
    seconds = {n: 0.0 for n in names}
    frames, windows = [], 0
    for ep in shard.items:
        ep = int(ep)
        T = int(eps.loc[ep, "windows"])
        rows = features.join_known(store.window_rows(ep, 0, T), eps.reset_index())
        risk = episode_risks(preds, names, rows, H, warmup, ep, seconds)
        windows += max(T - warmup, 0)
        frames.append(pd.DataFrame({"episode": np.full(T, ep, dtype=np.int64),
                                    "window": np.arange(T, dtype=np.int64), **risk}))
    pd.concat(frames, ignore_index=True).to_parquet(shard_dir / RISKS_FILE, index=False)
    (shard_dir / TIMING_FILE).write_text(json.dumps({"seconds": seconds, "windows": windows}))


# ------------------------------------------------------------------ code identity
def stored_risks(shadow_dir: Path, episodes, names) -> dict:
    """{episode: {name: risk series}} of a shadow run's finished shards."""
    import pandas as pd
    from pis.shards import collect
    want = {int(e) for e in episodes}
    raw = pd.concat([pd.read_parquet(d / RISKS_FILE) for d in collect(shadow_dir)], ignore_index=True)
    raw = raw[raw.episode.isin(want)]
    return {int(e): {n: g.sort_values("window")[n].to_numpy(float) for n in names} for e, g in raw.groupby("episode")}


def rescore_check(shadow_dir, world_model, episodes: int = 2, windows: int | None = None) -> list[str]:
    """Predictors whose risks, recomputed now from the store on the shadow run's first `episodes` calibration
    episodes (the first `windows` windows; all by default), differ from its stored risks by more than 1e-9
    (NaN where it had NaN). Empty when the current code reproduces the shadow run."""
    from pis.phase_b import features
    from pis.phase_b.history import Store
    shadow_dir = Path(shadow_dir)
    cfg = runs.read_json(shadow_dir / "config.json")
    if not cfg:
        raise ValueError("the shadow run has no config.json")
    names, rule = list(cfg["names"]), cfg["rule"]
    chosen = [int(e) for e in cfg["calibration"][:episodes]]
    store_path = Path(cfg["store"])
    st = Store(store_path if store_path.is_absolute() else ROOT / store_path)
    eps = st.episodes().set_index("seed")
    preds = load_predictors(shadow_dir / FITTED_FILE if any(n != "world_model" for n in names) else None, names,
                            rule, world_model, int(cfg["wm_n"]))
    old = stored_risks(shadow_dir, chosen, names)
    bad = set()
    for e in chosen:
        T = int(eps.loc[e, "windows"]) if windows is None else min(int(windows), int(eps.loc[e, "windows"]))
        rows = features.join_known(st.window_rows(e, 0, T), eps.reset_index())
        now = episode_risks(preds, names, rows, int(rule["H"]), int(rule["warmup"]), e)
        for n in names:
            if e not in old or not np.allclose(now[n], old[e][n][:T], rtol=0, atol=1e-9, equal_nan=True):
                bad.add(n)
    return sorted(bad)


def code_drift(shadow_cfg: dict) -> str | None:
    """A warning when pis/phase_b changed between the shadow run's commit and HEAD (None when not)."""
    import subprocess
    head = shadow_cfg.get("code_head")
    if not head:
        return "the shadow run recorded no code commit"
    p = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", head.split("+")[0], "HEAD", "--", "pis/phase_b"],
                       capture_output=True, text=True)
    return None if p.returncode == 0 else f"pis/phase_b changed since the shadow run's commit {head[:12]}"


def identity_check(shadow_dir, world_model, episodes: int = 2, windows: int | None = None) -> tuple[list, list]:
    """(refusals, warnings) on whether the current code is the code of the shadow run: its risks re-scored
    (`rescore_check`) and the pis/phase_b diff since its commit (`code_drift`)."""
    from pis.privacy import scrub
    refusals, warnings = [], []
    try:
        bad = rescore_check(shadow_dir, world_model, episodes, windows)
        if bad:
            refusals.append("the current code does not reproduce the shadow run's risks of " + ", ".join(bad)
                            + " (calibration episodes re-scored from the store)")
    except Exception as e:                      # any failure to re-score is a refusal, with its reason
        refusals.append(scrub(f"the shadow run's risks could not be re-scored: {type(e).__name__}: {e}"))
    drift = code_drift(runs.read_json(Path(shadow_dir) / "config.json"))
    if drift:
        warnings.append(drift)
    return refusals, warnings


# ------------------------------------------------------------------ scoring
def _recall(res: dict, name: str, cause: str = "building_up") -> float:
    r = res[name]["selection"]["by_cause"][cause]["recall"]
    return 0.0 if r is None or res[name]["threshold"] is None else float(r)


def _fa(res: dict, name: str) -> float:
    f = res[name]["selection"]["false_alarm_rate"]
    return 1.0 if f is None else float(f)


def choose(res: dict) -> str | None:
    """The feasible predictor with the best selection building-up recall; ties -> lower false-alarm rate."""
    ok = [n for n in res if res[n]["threshold"] is not None]
    return min(ok, key=lambda n: (-_recall(res, n), _fa(res, n), n)) if ok else None


def better_non_ml(res: dict) -> str | None:
    """The non-learned predictor with the higher selection building-up recall (an infeasible one has 0)."""
    ok = [n for n in NON_ML if n in res]
    return min(ok, key=lambda n: (-_recall(res, n), _fa(res, n), n)) if ok else None


def gate(best_recall: float | None) -> tuple[str, str]:
    if best_recall is None or best_recall <= GATE_RECALL:
        shown = "none" if best_recall is None else f"{best_recall:.2f}"
        return ("GATE: stop before prevention",
                f"the best building-up recall at lead >= min lead is {shown} (<= {GATE_RECALL}): no predictor "
                "catches building-up incidents early enough at the calibrated false-alarm rate")
    return "GATE: pass", f"the best building-up recall is {best_recall:.2f} (> {GATE_RECALL})"


def _brier(mc, risk_of, active_of) -> dict:
    """Brier of the window risk against MC label risks on labelled states with a risk, split by cause."""
    out = {}
    if mc is None:
        return {c: {"brier": None, "n": 0} for c in ("all", "building_up", "sudden")}
    p, y, cause = [], [], []
    for r in mc.itertuples():
        v = risk_of(int(r.episode), int(r.window))
        if v is None or not np.isfinite(v):
            continue
        p.append(v)
        y.append(float(r.risk))
        cause.append("building_up" if active_of(int(r.episode), int(r.window)) else "sudden")
    p, y, cause = np.array(p), np.array(y), np.array(cause)
    for c in ("all", "building_up", "sudden"):
        m = np.ones(len(p), bool) if c == "all" else cause == c
        out[c] = {"brier": float(((p[m] - y[m]) ** 2).mean()) if m.any() else None, "n": int(m.sum())}
    return out


def evaluate(risks: dict, labels: dict, active: dict, names, calibration, selection, rule, mc) -> dict:
    from pis.phase_b import alarms
    H, lead, tail = int(rule["H"]), int(rule["min_lead_windows"]), int(rule["k"]) - 1
    out = {}
    for n in names:
        thr = alarms.conformal_threshold([(risks[e][n], labels[e]) for e in calibration], alarms.ALPHA,
                                         H=H, tail=tail)
        low = None if thr is None else alarms.LOW_RATIO * thr

        def score(e, n=n, thr=thr, low=low):
            ev = [] if thr is None else alarms.alarm_events(risks[e][n], thr, low)
            return alarms.score_episode(ev, labels[e], H, lead, tail)

        cal = [score(e) for e in calibration]
        sel = [score(e) for e in selection]
        losses = [alarms.episode_loss(s) for s in cal]
        k = len(cal)
        sel_sum = alarms.summarize(sel)
        sel_sum["brier"] = _brier(mc[mc.episode.isin(selection)] if mc is not None else None,
                                  lambda e, w, n=n: float(risks[e][n][w]) if e in risks and w < len(risks[e][n])
                                  else None, lambda e, w: bool(active[e][w]))
        out[n] = {"threshold": thr, "low": low,
                  "calibration": {**alarms.summarize(cal),
                                  "adjusted_risk": None if thr is None
                                  else (k / (k + 1)) * float(np.mean(losses)) + 1 / (k + 1)},
                  "selection": sel_sum,
                  "per_episode": {c: {"caught": [s["caught"][c] for s in sel],
                                      "incidents": [s["incidents"][c] for s in sel]} for c in alarms.CAUSES}}
    return out


def question_one(res: dict) -> dict:
    from pis.phase_b import alarms
    base = better_non_ml(res)
    out = {"non_ml": base, "diffs": {}}
    if base is None:
        return out
    for n in ML:
        if n not in res:
            continue
        out["diffs"][n] = {}
        for c in alarms.CAUSES:
            pa, pb = res[n]["per_episode"][c], res[base]["per_episode"][c]
            ca = pa["caught"] if res[n]["threshold"] is not None else [0] * len(pa["caught"])
            cb = pb["caught"] if res[base]["threshold"] is not None else [0] * len(pb["caught"])
            out["diffs"][n][c] = alarms.bootstrap_recall_diff(ca, cb, pa["incidents"])
    return out


# ------------------------------------------------------------------ note
_f = partial(runs.fmt, default="{:.3f}")


def note_text(cfg: dict, res: dict, q1: dict, best: str | None, gate_line: tuple[str, str], timing: dict) -> str:
    r = cfg["rule"]
    lines = ["# Shadow mode on development seeds", "",
             f"Label: {cfg['label'] or 'real'}. Calibration episodes {len(cfg['calibration'])}, selection episodes "
             f"{len(cfg['selection'])} (R6, from the store); predictors fitted on {len(cfg['fit_seeds'])} history "
             f"episodes. Rule k={r['k']}, interval={r['interval']}, majority={r['majority']}, warm-up "
             f"{r['warmup']}; H={r['H']} windows, min lead {r['min_lead_windows']} windows. World model: "
             f"{cfg['world_model_sha256'] or 'not run'} (n={cfg['wm_n']} futures).",
             "", "Alarm: on at risk >= threshold, off below 0.7 x threshold; conformal risk control of the "
             "per-episode false-alarm share on the calibration episodes, alpha = 0.10. An alarm event is false "
             "when no new incident starts within H after it; events whose H windows run past the episode end "
             "are not judged.", "",
             "The queueing model's scale and the conformal thresholds are both set on the CALIBRATION episodes. "
             "The scale is a monotone map of the queueing statistic, so the alarm-on threshold selects the same "
             "windows whatever the scale; only `low` (0.7 x threshold, in risk units) depends on it.", "",
             "## Thresholds (calibration)", "",
             "| predictor | threshold | low | adjusted risk | false-alarm rate | events |", "|---|---|---|---|---|---|"]
    for n, v in res.items():
        c = v["calibration"]
        thr = "no feasible threshold" if v["threshold"] is None else _f(v["threshold"], "{:.4f}")
        lines.append(f"| {n} | {thr} | {_f(v['low'], '{:.4f}')} | {_f(c['adjusted_risk'])} | "
                     f"{_f(c['false_alarm_rate'])} | {c['events']} |")
    lines += ["", "## Selection episodes", "",
              "| predictor | recall building-up | recall sudden | median lead bu / sudden | false-alarm rate | "
              "alarms per quiet episode all / judged | Brier all / bu / sudden (n states) |",
              "|---|---|---|---|---|---|---|"]
    for n, v in res.items():
        s, b = v["selection"], v["selection"]["by_cause"]
        rec = [f"{_f(b[c]['recall'], '{:.2f}')} ({b[c]['caught']}/{b[c]['incidents']})" for c in b]
        cs = ("all", "building_up", "sudden")
        br = " / ".join(_f(s["brier"][c]["brier"]) for c in cs) + \
            " (n " + "/".join(str(s["brier"][c]["n"]) for c in cs) + ")"
        lines.append(f"| {n} | {rec[0]} | {rec[1]} | {_f(b['building_up']['median_lead'], '{:.1f}')} / "
                     f"{_f(b['sudden']['median_lead'], '{:.1f}')} | {_f(s['false_alarm_rate'])} | "
                     f"{_f(s['alarms_per_quiet_episode'], '{:.2f}')} / "
                     f"{_f(s.get('judged_alarms_per_quiet_episode'), '{:.2f}')} | {br} |")
    lines += ["", "Brier: Monte Carlo labelled selection states with a finite risk of that predictor (n states "
              "per predictor: all / building-up / sudden). Quiet episode = no counted incident. Alarms per quiet "
              "episode: \"all\" includes the censored events near the episode's end (neither true nor false), "
              "\"judged\" counts only the true and false ones.",
              "", "## Question 1: learned vs not learned", "",
              f"Better non-learned predictor (selection building-up recall): {q1.get('non_ml') or '-'}. "
              "Difference of recall (lead >= min lead), 95 % bootstrap interval over selection episodes "
              "(2000 resamples):", "", "| predictor | building-up | sudden |", "|---|---|---|"]
    for n, d in q1.get("diffs", {}).items():
        cells = [f"{_f(d[c]['diff'], '{:+.2f}')} [{_f(d[c]['lo'], '{:+.2f}')}, {_f(d[c]['hi'], '{:+.2f}')}]"
                 for c in ("building_up", "sudden")]
        lines.append(f"| {n} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Timing", "",
              f"{timing['ms_per_window']:.1f} ms per window for all predictors together ("
              + ", ".join(f"{n} {ms:.1f}" for n, ms in timing["per_predictor_ms"].items()) + "); "
              f"{FULL_EPISODES} episodes x {FULL_WINDOWS} windows projected {timing['projected_h_one_proc']:.1f} h "
              "on one process.",
              "", "## Decision", "",
              f"Alarm predictor: {best or 'none'}"
              + (f" (threshold {res[best]['threshold']:.4f}, low {res[best]['low']:.4f})." if best else "."),
              "", gate_line[0], "", f"Reason: {gate_line[1]}."]
    if cfg.get("warnings"):
        lines += ["", "Warnings: " + "; ".join(cfg["warnings"]) + "."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ run
def _check_config(out: Path, cfg: dict) -> None:
    p = out / "config.json"
    if p.exists() and json.loads(p.read_text()) != json.loads(json.dumps(cfg)):
        raise ValueError("the shadow output directory was produced with a different configuration; "
                         "use a new output directory")


def shadow(store, rule: dict, names, fit_seeds, calibration, selection, world_model, wm_n: int, mc_path,
           out, note, alarms_path, procs: int, label: str = "", overrides: dict | None = None) -> dict:
    """Risk series (sharded), thresholds, selection statistics, note and alarms config."""
    import pandas as pd
    from pis.phase_b import alarms, incidents
    from pis.phase_b.history import Store
    from pis.shards import collect, make_shards, run_shards
    out, store_path = Path(out), Path(store)
    names = [n for n in ALL if n in names]
    calibration, selection = sorted(int(s) for s in calibration), sorted(int(s) for s in selection)
    st = Store(store_path)
    eps = st.episodes().set_index("seed")
    episodes = calibration + selection
    missing = sorted(set(episodes) - set(eps.index))
    if missing:
        raise ValueError(f"no stored episode for seeds {missing[:10]}")
    other = sorted(s for s in episodes if eps.loc[s, "source"] != SOURCE)
    if other:
        raise ValueError(f"episodes {other[:10]} are not R6-driven")
    warnings = []
    wm_sha = None
    if "world_model" in names:
        from pis.phase_b.world_models import load_world_model
        if world_model is None or not Path(world_model).exists():
            raise ValueError("the world-model predictor needs an existing --world-model checkpoint")
        m = load_world_model(world_model)
        wm_sha = m.meta["sha256"]
        seen = set(m.meta.get("train_seeds", [])) | set(m.meta.get("val_seeds", []))
        if label != "smoke" and (not seen or any(s not in seed_ranges.HISTORY for s in seen)):
            raise ValueError("the world model must be trained on recorded history seeds only")
        if seen & set(episodes):
            msg = f"the world model was trained on evaluated episodes {sorted(seen & set(episodes))[:5]}"
            if label != "smoke":
                raise ValueError(msg)
            warnings.append(msg)
    if mc_path is not None and Path(mc_path).exists():
        try:
            runs.check_mc_config(Path(mc_path), rule)
        except ValueError as e:
            if label != "smoke":
                raise
            warnings.append(f"Monte Carlo labels not checked against the rule ({e})")
    cfg = {"label": label, "store": str(store_path), "rule": rule, "names": names, "fit_seeds": list(fit_seeds),
           "calibration": calibration, "selection": selection, "world_model_sha256": wm_sha, "wm_n": wm_n,
           "overrides": overrides or {}, "shard_size": SHARD_SIZE, "wm_tag": WM_TAG}
    # the code the risks come from: the commit of the run that started this directory
    cfg["code_head"] = runs.read_json(out / "config.json").get("code_head") or runs.git_head()
    out.mkdir(parents=True, exist_ok=True)
    _check_config(out, cfg)
    if not (out / "config.json").exists():
        # the configuration is pinned before the fitted predictors exist: a crash in between can never leave
        # a pickle that a later run with another configuration would reuse
        tmp = out / "config.json.tmp"
        tmp.write_text(json.dumps(json.loads(json.dumps(cfg)), indent=2, sort_keys=True) + "\n")
        tmp.replace(out / "config.json")
    fitted = out / FITTED_FILE
    if not fitted.exists():
        H = int(rule["H"])
        preds = make_predictors(st, names, rule, H, fit_seeds, calibration, label == "smoke", overrides)
        tmp = out / (FITTED_FILE + ".tmp")
        tmp.write_bytes(pickle.dumps(preds))
        tmp.replace(fitted)
    params = {"store": str(store_path), "fitted": str(fitted), "rule": rule, "names": names,
              "world_model": None if world_model is None else str(world_model), "wm_n": wm_n}
    shards = make_shards(episodes, SHARD_SIZE)
    t0 = time.time()
    res = run_shards(out, shards, partial(run_shard, params=params), cfg, procs)
    print(f"shards done {len(res['done'])}, skipped {len(res['skipped'])}, failed {res['failed']} "
          f"in {time.time() - t0:.0f} s", flush=True)
    dirs = collect(out)
    if len(dirs) != len(shards):
        raise RuntimeError("unfinished shards: rerun the same command")
    raw = pd.concat([pd.read_parquet(d / RISKS_FILE) for d in dirs], ignore_index=True)
    risks = {int(e): {n: g.sort_values("window")[n].to_numpy(float) for n in names}
             for e, g in raw.groupby("episode")}
    secs = {n: 0.0 for n in names}
    wins = 0
    for d in dirs:
        tj = json.loads((d / TIMING_FILE).read_text())
        wins += tj["windows"]
        for n in names:
            secs[n] += tj["seconds"].get(n, 0.0)
    per = {n: 1000 * secs[n] / wins for n in names} if wins else {n: float("nan") for n in names}
    ms = float(sum(per.values()))
    timing = {"ms_per_window": ms, "per_predictor_ms": per,
              "projected_h_one_proc": FULL_EPISODES * FULL_WINDOWS * ms / 1000 / 3600}
    rule_ = incidents.IncidentRule(rule["k"], rule["interval"], rule["majority"])
    labels, active = {}, {}
    for e in episodes:
        w = st.window_rows(e, 0, int(eps.loc[e, "windows"]))
        active[e] = w.deg_active.to_numpy(bool)
        labels[e] = incidents.label_series(w.slo_v2_violation.to_numpy(bool), active[e], rule_, int(rule["warmup"]))
    mc = pd.read_parquet(mc_path) if mc_path is not None and Path(mc_path).exists() else None
    if mc is None:
        warnings.append("no Monte Carlo labels: Brier scores not computed")
    results = evaluate(risks, labels, active, names, calibration, selection, rule, mc)
    q1 = question_one(results)
    best = choose(results)
    g = gate(_recall(results, best) if best else None)
    runs.write_json(Path(alarms_path), {
        "predictor": best, "threshold": results[best]["threshold"] if best else None,
        "low": results[best]["low"] if best else None, "H": int(rule["H"]),
        "min_lead_windows": int(rule["min_lead_windows"]), "alpha": alarms.ALPHA,
        "world_model": None if world_model is None else runs.repo_path(world_model, label == "smoke"),
        "shadow_dir": runs.repo_path(out, label == "smoke")})
    runs.write_note(Path(note), note_text({**cfg, "warnings": warnings}, results, q1, best, g, timing))
    return {"predictors": results, "q1": q1, "best": best, "gate": g, "shards": res, "ms_per_window": ms,
            "timing": timing}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictors", default="all")
    ap.add_argument("--calibration", default="3400-3449")
    ap.add_argument("--selection", default="3450-3499")
    ap.add_argument("--fit-seeds", default=None, help="history seeds (default: seed %% 10 not 8 or 9)")
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--incident", default=DEFAULT_INCIDENT)
    ap.add_argument("--world-model", default=DEFAULT_WM)
    ap.add_argument("--wm-n", type=int, default=32)
    ap.add_argument("--mc-labels", default=DEFAULT_MC)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--label", default="")
    ap.add_argument("--smoke-dir", default=None)
    a = ap.parse_args(argv)
    if a.label not in runs.LABELS:
        raise SystemExit(f"refusing to run: unknown label {a.label!r}")
    smoke = a.label == "smoke"
    names = list(ALL) if a.predictors == "all" else [n.strip() for n in a.predictors.split(",")]
    try:
        if set(names) - set(ALL) or not names:
            raise ValueError(f"unknown predictors {sorted(set(names) - set(ALL))}; expected some of {list(ALL)}")
        cal, sel = seed_ranges.parse(a.calibration), seed_ranges.parse(a.selection)
        fit = seed_ranges.parse(a.fit_seeds) if a.fit_seeds else default_fit_seeds()
        extra = ("HISTORY",) if smoke else ()
        runs.check_seeds(cal, ("CALIBRATION",) + extra)
        runs.check_seeds(sel, ("SELECTION",) + extra)
        seed_ranges.check(fit, "HISTORY")
        for x in (cal, sel, fit):
            if len(set(x)) != len(x):
                raise ValueError("repeated seeds")
        if set(cal) & set(sel) or set(fit) & (set(cal) | set(sel)):
            raise ValueError("fit, calibration and selection seeds overlap")
        sdir = runs.smoke_dir(a.smoke_dir) if smoke else None
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    if not smoke:
        if a.smoke_dir is not None:
            raise SystemExit("refusing to run: --smoke-dir needs --label smoke")
        if cal != list(seed_ranges.CALIBRATION) or sel != list(seed_ranges.SELECTION):
            raise SystemExit("refusing to run: a real run uses the whole calibration and selection ranges")
        if names != list(ALL) or a.wm_n != 32:
            raise SystemExit("refusing to run: a real run compares all five predictors with n = 32 futures")
        if Path(a.incident).resolve() != (ROOT / DEFAULT_INCIDENT).resolve():
            raise SystemExit(f"refusing to run: a real run reads the frozen {DEFAULT_INCIDENT}")
        if not Path(a.mc_labels).exists():
            raise SystemExit("refusing to run: a real run needs the Monte Carlo labels")
        if runs.tree_dirty(ignore=(ALARMS, NOTE)):
            raise SystemExit("refusing to run: the working tree has uncommitted changes (commit, or --label smoke)")
    inc = runs.read_json(Path(a.incident))
    need = ["k", "interval", "majority", "warmup", "H", "min_lead_windows"]
    if any(k not in inc for k in need):
        raise SystemExit(f"refusing to run: {a.incident} lacks {[k for k in need if k not in inc]}")
    rule = {k: inc[k] for k in need}
    out = (sdir / "shadow") if smoke else ROOT / raw_dir_for(a.world_model)
    note = (sdir / "shadow_dev.md") if smoke else ROOT / NOTE
    alarms_path = (sdir / "alarms.json") if smoke else ROOT / ALARMS
    try:
        res = shadow(a.store, rule, names, fit, cal, sel, a.world_model if "world_model" in names else None,
                     a.wm_n, a.mc_labels, out, note, alarms_path, a.procs, a.label)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    t = res["timing"]
    print(f"best {res['best']}; {res['gate'][0]}; {t['ms_per_window']:.1f} ms per window (all predictors); "
          f"{FULL_EPISODES} x {FULL_WINDOWS} windows projected {t['projected_h_one_proc']:.1f} h on one process",
          flush=True)
    return res


if __name__ == "__main__":
    main()
