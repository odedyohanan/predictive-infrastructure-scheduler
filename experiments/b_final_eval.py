"""Final paired evaluation of the incident advisor on the held-out seeds 3000-3099, by the frozen protocol in
configs/phase_b/final_eval.json.

Official run (once; a crash or a bug: fix it, rerun as --attempt 2, report every attempt, never tune on the
results; an interrupted attempt continues with --resume after its "(started)" section is committed):
  scripts/gentle.sh b_final_eval .venv/bin/python experiments/b_final_eval.py --procs 4
Freezing the protocol (writes the file once from the committed configs and notes; commit it on its own;
prints the oracle arm's projected cost from the tuning run's mode shares):
  .venv/bin/python experiments/b_final_eval.py --write-protocol
Rehearsal on development seeds (stand-in configs; never touches the official outputs):
  .venv/bin/python experiments/b_final_eval.py --write-protocol --protocol <smoke>/final_eval.json \\
      --incident ... --alarms ... --advisor ... --world-model ... --shadow-dir ... --timing-note ... --shadow-note ...
  .venv/bin/python experiments/b_final_eval.py --protocol <smoke>/final_eval.json --seeds 3450-3452 \\
      --duration 30 --label rehearsal --out <smoke>/final.md --raw-dir <smoke>/final_raw --procs 1

Every seed runs every primary arm of the protocol on `scenario_for(seed)` (same scenario, same start policy)
inside one shard of SHARD_SIZE seeds: r6 (the reference), advisor (model selection with the tuned params),
loop_a (the Step A loop), advisor_random and always_on (the advisor's selections; "always" has the mode on in
every window from the warm-up on). The primary result is written to the raw directory before the diagnostic
advisor_oracle arm (real-simulator branches, the protocol's `oracle` branches and horizon) runs in its own
shards; the oracle never enters a success rule. Held-out episodes never enter the
history store. Per arm and seed the raw output keeps time in incident, counted incidents (by cause: onset while
a degradation is active = building-up, otherwise sudden), value per second, SLO v2 %, decision time, the
intervention rate (share of windows in mode), the fallback share, non-finite risks, and the per-window
violation, degradation and mode series; the r6 arm also keeps its predictor rows and the risk series of the
shadow run's predictors (fitted pickle and world model, as the shadow run used them).

Question 2 (advisor vs R6): chronic = R6's episode spends more than half its windows in incident; success when
the paired 95 % CI upper bound of (advisor - R6) time in incident over the non-chronic episodes is below zero,
the paired 95 % lower bound of (advisor - R6) value per second over all episodes is at least -1 % of R6's mean
value per second, and the committed timing note (experiments/b_time_advisor.py: unlabelled, same params,
alarm and world model, committed at HEAD) has p95 <= 50 ms and max <= 500 ms. Chronic episodes are reported
separately; decision times measured in this run (background priority) are reported only. Question 1: the
protocol's primary learned predictor (best building-up recall on the shadow run's selection episodes) has a
building-up recall (lead >= min lead) above the better non-learned one's on the held-out R6 episodes, 95 %
bootstrap interval over episodes (tag 14000, 2000 resamples) above zero; the other learned predictors are
secondary (reported, no success claim); alarm thresholds
are the conformal ones of the shadow run's calibration episodes, never refitted here. The shadow-only
variant (the shadow run's gate stopped before prevention) runs r6 alone and does not attempt question 2.

Guards. A run on any held-out seed uses exactly the protocol's seeds and duration, no label, background
priority, a clean tree, the protocol at configs/phase_b/final_eval.json with exactly one commit that touches
only it, the frozen files it names (tracked ones committed at HEAD, all matching their sha256), and the
official outputs. Attempts are numbered 1..3, each needs the previous attempt's raw directory and the same
frozen files; an attempt with a report section or a raw directory is refused unless it is resumed (same code
since it started). Other runs use SELECTION seeds only, label "rehearsal" or "smoke", and outputs outside
results/phase_b/final.md and results/raw/phase_b/final. Before any simulation the current code must reproduce
the shadow run's risks of two calibration episodes re-scored from the store (an official run refuses
otherwise, naming the predictors; a pis/phase_b diff since the shadow run's commit is a warning). Outputs: one "## Attempt N" section in
results/phase_b/final.md, appended as a "(started)" stub before any simulation and replaced at the end; raw
shards under results/raw/phase_b/final/attempt_N/ (git-ignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pis.phase_b import runs  # noqa: E402
from pis.phase_b import seeds as seed_ranges  # noqa: E402
from pis.privacy import safe_path, scrub  # noqa: E402

PROTOCOL = "configs/phase_b/final_eval.json"
OFFICIAL_REPORT, OFFICIAL_RAW = "results/phase_b/final.md", "results/raw/phase_b/final"
INPUTS = {"incident": "configs/phase_b/incident.json", "alarms": "configs/phase_b/alarms.json",
          "advisor": "configs/phase_b/advisor.json", "world_model": "models/pgru_b1.pt",
          "shadow_dir": "results/raw/phase_b/shadow", "timing_note": "results/phase_b/advisor_timing.md",
          "shadow_note": "results/phase_b/shadow_dev.md", "step_a": "configs/step_a.json",
          "phase_a_protocol": "configs/final_eval.json"}
OFFICIAL_FILES = {**{k: v for k, v in INPUTS.items() if k != "shadow_dir"},
                  "fitted": INPUTS["shadow_dir"] + "/predictors.pkl",
                  "shadow_config": INPUTS["shadow_dir"] + "/config.json", "shadow_risks": INPUTS["shadow_dir"]}
TRACKED = ("incident", "alarms", "advisor", "timing_note", "shadow_note", "step_a", "phase_a_protocol")
ARMS = ("r6", "advisor", "loop_a", "advisor_random", "advisor_oracle", "always_on")
SHADOW_ARMS = ("r6",)
ORACLE_ARM = "advisor_oracle"
PRIMARY_ARMS = tuple(a for a in ARMS if a != ORACLE_ARM)
ORACLE_BRANCHES = 4
SELECTION_OF = {"advisor": "model", "advisor_random": "random", "advisor_oracle": "oracle", "always_on": "always"}
REFERENCE = "r6"
GATE_STOP, GATE_PASS = "GATE: stop before prevention", "GATE: pass"
DURATION_S = 300.0
SHARD_SIZE = 5
MAX_ATTEMPTS = 3
REHEARSAL_LABELS = ("rehearsal", "smoke")
VALUE_TOL = 0.01
DECIDE_P95_MS, DECIDE_MAX_MS = 50.0, 500.0
CONF = 0.95
ALARM_KEYS = ("predictor", "threshold", "low", "H")
RULE_KEYS = ("k", "interval", "majority", "warmup")
CAUSES = ("building_up", "sudden")
EPISODES_FILE, ROWS_FILE, RISKS_FILE = "episodes.json", "rows.parquet", "risks.parquet"
SHARDS, ORACLE_SHARDS = "shards", "oracle_shards"
PRIMARY_FILE = "primary_result.json"
DEFAULT_TUNE_DIR = "results/raw/phase_b/advisor_tune"


# ------------------------------------------------------------------ files and git
def _abs(path, root: Path = ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shadow_risks_sha256(shadow_dir: Path) -> str:
    """One digest over the finished shards' risk files of a shadow run (shard id and bytes, by id)."""
    from pis.shards import collect
    h = hashlib.sha256()
    dirs = [d for d in collect(shadow_dir) if (d / "risks.parquet").is_file()]
    if not dirs:
        raise ValueError(f"no finished shadow shards with risks in {safe_path(str(shadow_dir))}")
    for d in dirs:
        h.update(d.name.encode() + b"\0")
        h.update(_sha256(d / "risks.parquet").encode())
    return h.hexdigest()


def file_sha(key: str, path: str, root: Path = ROOT) -> str | None:
    p = _abs(path, root)
    if key == "shadow_risks":
        try:
            return shadow_risks_sha256(p)
        except ValueError:
            return None
    return _sha256(p) if p.is_file() else None


def _git(*args) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def _repo_rel(path) -> str | None:
    try:
        return str(_abs(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return None


def committed_clean(path) -> bool:
    """Tracked at HEAD and identical to HEAD in the working tree."""
    rel = _repo_rel(path)
    if rel is None or _git("cat-file", "-e", f"HEAD:{rel}")[0] != 0:
        return False
    return _git("diff", "--quiet", "HEAD", "--", rel)[0] == 0


def freeze_state() -> dict:
    """The commits that touched the protocol file (newest first) and the files of the oldest one."""
    commits = [c for c in _git("log", "--format=%H", "--", PROTOCOL)[1].splitlines() if c]
    files = _git("show", "--name-only", "--format=", commits[-1])[1].split() if commits else []
    return {"commits": commits, "files": files}


def code_changed(run_config: dict | None) -> bool:
    """pis/ or experiments/ differ between the commit an attempt started from and HEAD."""
    head = (run_config or {}).get("head")
    if not head:
        return True
    return _git("diff", "--quiet", head, "HEAD", "--", "pis", "experiments")[0] != 0


def load_json(path) -> dict:
    def _no_nan(x):
        raise ValueError(f"non-strict JSON constant {x}")
    return json.loads(_abs(path).read_text(), parse_constant=_no_nan)


# ------------------------------------------------------------------ protocol
def _gate(note: Path) -> str:
    text = note.read_text() if note.is_file() else ""
    if GATE_STOP in text:
        return "shadow_only"
    if GATE_PASS in text:
        return "full"
    raise ValueError(f"{safe_path(str(note))} has no GATE line (missing or not a shadow note)")


def make_protocol(paths: dict | None = None, root: Path = ROOT, oracle_branches: int = ORACLE_BRANCHES,
                  oracle_horizon: int | None = None) -> dict:
    """The frozen protocol (written once to configs/phase_b/final_eval.json and committed on its own). `paths`
    replaces the official input files (rehearsals only). The oracle arm branches `oracle_branches` times over
    `oracle_horizon` windows (default: the advisor's h). The primary learned predictor of question 1 is read
    from the shadow run's results (`primary_predictor`). Raises ValueError when an input is missing or
    unusable."""
    p = {**INPUTS, **{k: str(v) for k, v in (paths or {}).items() if v is not None}}
    variant = _gate(_abs(p["shadow_note"], root))
    inc = load_json(_abs(p["incident"], root))
    need = list(RULE_KEYS) + ["H", "min_lead_windows"]
    if any(k not in inc for k in need):
        raise ValueError(f"the incident config lacks {[k for k in need if k not in inc]}")
    if float(inc.get("chronic", 0.5)) != 0.5:
        raise ValueError("the chronic share of the incident config is not 0.5 (incidents.is_chronic)")
    shadow_dir = _abs(p["shadow_dir"], root)
    shadow_cfg = load_json(shadow_dir / "config.json") if (shadow_dir / "config.json").is_file() else None
    if not shadow_cfg or not shadow_cfg.get("names"):
        raise ValueError("the shadow run's config.json is missing or names no predictor")
    entries = {"incident": p["incident"], "alarms": p["alarms"], "world_model": p["world_model"],
               "shadow_note": p["shadow_note"], "step_a": p["step_a"], "phase_a_protocol": p["phase_a_protocol"],
               "fitted": str(Path(p["shadow_dir"]) / "predictors.pkl"),
               "shadow_config": str(Path(p["shadow_dir"]) / "config.json"), "shadow_risks": p["shadow_dir"]}
    if variant == "full":
        entries.update(advisor=p["advisor"], timing_note=p["timing_note"])
    files = {}
    for key, path in entries.items():
        sha = file_sha(key, path, root)
        if sha is None:
            raise ValueError(f"{key} file {safe_path(path)} is missing")
        files[key] = {"path": path, "sha256": sha}
    oracle = None
    if variant == "full":
        adv = load_json(_abs(p["advisor"], root))
        alarms = load_json(_abs(p["alarms"], root))
        if not adv.get("params"):
            raise ValueError("the advisor config has no chosen params")
        horizon = int(adv["params"]["h"] if oracle_horizon is None else oracle_horizon)
        if int(oracle_branches) < 1 or horizon < 1:
            raise ValueError("the oracle needs at least one branch and one window")
        oracle = {"branches": int(oracle_branches), "horizon": horizon,
                  "role": "diagnostic upper bound only: runs in its own shards after every primary arm has "
                          "finished, never part of a success rule"}
        if alarms.get("predictor") is None or alarms.get("threshold") is None:
            raise ValueError("the alarms config has no predictor with a threshold")
    else:
        files.update(advisor=None, timing_note=None)
    names = list(shadow_cfg["names"])
    shadow = runs.load_script("b_shadow")
    ml = [n for n in shadow.ML if n in names]
    if not ml:
        raise ValueError("the shadow run scored no learned predictor")
    primary = primary_predictor({**shadow_cfg, "_dir": p["shadow_dir"]}, names, root)
    if primary not in ml:
        raise ValueError(f"the primary predictor {primary!r} is not a learned one")
    arms_doc = {
        "r6": "reference: the R6 RL student, greedy (models/rl_r6/final.pt of the scheduler repo)",
        "advisor": "candidate: R6 until the alarm, then the chance-constrained model selection with the "
                   "advisor config's params",
        "loop_a": "Phase A Step A loop (configs/step_a.json), as the history's loop_a source",
        "advisor_random": "ablation: uniform choice among the advisor's candidates in the mode",
        "advisor_oracle": "ablation, upper bound: candidates judged by branches of the real degraded simulator",
        "always_on": "ablation: the advisor's mode on in every window from the warm-up on, without the alarm",
    }
    arms = {a: {"role": arms_doc[a], **({"selection": SELECTION_OF[a]} if a in SELECTION_OF else {})}
            for a in (ARMS if variant == "full" else SHADOW_ARMS)}
    q2 = "not attempted" if variant == "shadow_only" else {
        "candidate": "advisor", "reference": REFERENCE, "confidence": CONF,
        "time_in_incident": "paired 95% CI upper bound of (advisor - r6) time_in_incident_s over the episodes "
                            "that are not chronic for r6 is below zero",
        "value": "paired 95% lower bound of (advisor - r6) value_per_s over all episodes >= -value_tol x |r6's "
                 "mean value_per_s| over the same episodes (absolute value of the mean)",
        "value_tol": VALUE_TOL,
        "decision_time": "worst per-episode p95 <= decide_p95_ms and max <= decide_max_ms at normal priority, "
                         "from the timing note (unlabelled, the advisor config's params, the alarms config, the "
                         "world model's sha256, committed at HEAD); decision time measured in this run is "
                         "reported only",
        "decide_p95_ms": DECIDE_P95_MS, "decide_max_ms": DECIDE_MAX_MS,
        "chronic": "r6's episode has more than half its windows in incident; excluded from the time criterion, "
                   "reported separately",
        "pairs": "per seed, over the seeds where both arms ran; Student-t CI (pis.stats.paired_diff)"}
    from pis.phase_b import alarms as alarm_mod
    return {
        "name": "incident advisor: final paired evaluation",
        "variant": variant,
        "seeds": {"first": seed_ranges.HELD_OUT.start, "last": seed_ranges.HELD_OUT.stop - 1},
        "duration_s": DURATION_S,
        "scenario": "pis.phase_b.scenarios.scenario_for(seed): spec and degradation fixed by the seed, not "
                    "announced to any controller; 500 ms windows; every arm starts from the same policy",
        "reference_arm": REFERENCE,
        "arms": arms,
        "oracle": oracle,
        "shard_size": SHARD_SIZE,
        "files": files,
        "predictors": {"names": names, "ml": [n for n in shadow.ML if n in names],
                       "non_ml": [n for n in shadow.NON_ML if n in names], "wm_n": int(shadow_cfg.get("wm_n", 32)),
                       "thresholds": "conformal on the shadow run's calibration episodes (alpha "
                                     f"{alarm_mod.ALPHA}, low = {alarm_mod.LOW_RATIO} x threshold), recomputed "
                                     "from its risks, never refitted on held-out episodes"},
        "metrics": {
            "per_arm_and_seed": ["time_in_incident_s", "incidents", "incidents and time by cause", "value_per_s",
                                 "slo_v2_pct", "decide_ms_p95", "decide_ms_max", "mode_share (intervention rate)",
                                 "fallback_share", "risk_nan"],
            "incident_rule": {k: inc[k] for k in need},
            "cause": "building_up = the incident's onset falls while a degradation is active; sudden otherwise",
            "false_alarm_cost": "paired (arm - r6) value_per_s over non-chronic episodes without a counted r6 "
                                "incident",
            "predictors_on_held_out": "recall by cause at lead >= min_lead_windows, median lead, false-alarm "
                                      "rate, alarms per quiet episode, Brier of the window risk against the "
                                      "realised outcome (a new counted onset within H)"},
        "success_rule": {
            "question_2": q2,
            "question_1": {"names": names, "ml": ml,
                           "non_ml": [n for n in shadow.NON_ML if n in names], "cause": "building_up",
                           "min_lead_windows": int(inc["min_lead_windows"]),
                           "primary": primary, "secondary": [n for n in ml if n != primary],
                           "primary_choice": "the learned predictor with the best building-up recall on the "
                                             "shadow run's SELECTION episodes (ties: lower false-alarm rate), "
                                             "fixed before any held-out run",
                           "rule": "the primary learned predictor's held-out building-up recall minus that of the "
                                   "better non-learned predictor; success when the bootstrap 95% interval over "
                                   "episodes lies above zero. The other learned predictors are secondary: their "
                                   "intervals are reported, no success is claimed for them",
                           "bootstrap": {"reps": alarm_mod.BOOT_REPS, "tag": alarm_mod.BOOT_TAG, "conf": CONF}},
            "not_decidable": "question 2 is not decidable (never a success) when any r6 or advisor run of the "
                             "attempt failed; question 1 is not decidable when any r6 run or its predictor "
                             "scoring failed (those episodes are excluded and counted)",
            "otherwise": "any other result is reported as it is; an interval including zero is no detectable "
                         "difference"},
        "failure_handling": "a run that raises is recorded with a scrubbed message and listed, never dropped or "
                            "replaced; pairs use the seeds where both arms ran. A crash or a bug: fix, rerun as "
                            "the next attempt, report every attempt; no tuning on these results.",
        "outputs": {"report": OFFICIAL_REPORT, "raw": OFFICIAL_RAW + "/attempt_<N>/"},
    }


def choose_primary(res: dict, ml) -> str | None:
    """The learned predictor with the best selection building-up recall (ties: lower false-alarm rate, then
    name), from `b_shadow.evaluate` results."""
    shadow = runs.load_script("b_shadow")
    ok = [n for n in ml if n in res]
    return min(ok, key=lambda n: (-shadow._recall(res, n), shadow._fa(res, n), n)) if ok else None


def primary_predictor(shadow_cfg: dict, names, root: Path = ROOT) -> str | None:
    """Question 1's primary predictor from the shadow run: its risks and the store's labels of its calibration
    and selection episodes, scored as the shadow run scores them."""
    shadow = runs.load_script("b_shadow")
    cal = [int(e) for e in shadow_cfg["calibration"]]
    sel = [int(e) for e in shadow_cfg["selection"]]
    risks, labels, active = shadow_inputs(shadow_cfg, names, cal + sel, root)
    res = shadow.evaluate(risks, labels, active, [n for n in shadow.ALL if n in names], cal, sel,
                          shadow_cfg["rule"], None)
    return choose_primary(res, [n for n in shadow.ML if n in names])


def oracle_projection(tune_dir, advisor_cfg: dict, inc: dict, branches: int, horizon: int, n_seeds: int = 100,
                      windows: int = 600) -> dict | None:
    """Rough wall time of the oracle arm from a tuning run: windows in mode (the chosen config's mode share per
    episode) x candidates (upper bound: greedy + m samples + every named intervention) x branches x
    (horizon + k - 1) simulated windows, at R6's simulated seconds per window in that run. None without the
    tuning results."""
    from pis.adapters import initial_policy
    from pis.probe import named_interventions
    from pis.shards import collect
    d = _abs(tune_dir)
    eps = [e for x in collect(d) if (x / "episodes.json").is_file()
           for e in json.loads((x / "episodes.json").read_text())]
    chosen = advisor_cfg.get("chosen")
    adv = [e for e in eps if e.get("arm") == chosen]
    ref = [e for e in eps if e.get("arm") == REFERENCE and e.get("windows")]
    if not adv or not ref:
        return None
    share = np.array([float(e["mode_share"]) for e in adv])
    sec_per_window = float(np.mean([e["seconds"] / e["windows"] for e in ref]))
    cands = 1 + int(advisor_cfg["params"]["m"]) + len(named_interventions(initial_policy()))
    per_decision = cands * int(branches) * (int(horizon) + int(inc["k"]) - 1)
    per_episode_s = share * windows * per_decision * sec_per_window
    total_h = n_seeds * float(per_episode_s.mean()) / 3600
    shards = math.ceil(n_seeds / SHARD_SIZE)
    return {"episodes": len(adv), "mode_share_mean": float(share.mean()), "mode_share_max": float(share.max()),
            "candidates_max": cands, "sim_windows_per_decision": per_decision,
            "sim_s_per_window": sec_per_window, "one_proc_h": total_h,
            "four_procs_h": total_h * math.ceil(shards / 4) / shards}


def validate_protocol(p: dict) -> list[str]:
    need = ("variant", "seeds", "duration_s", "reference_arm", "arms", "files", "predictors", "success_rule")
    errors = [f"protocol lacks {k!r}" for k in need if k not in p]
    if errors:
        return errors
    want = ARMS if p["variant"] == "full" else SHADOW_ARMS if p["variant"] == "shadow_only" else None
    if want is None:
        return [f"unknown protocol variant {p['variant']!r}"]
    if tuple(p["arms"]) != want:
        errors.append(f"protocol arms {list(p['arms'])} differ from {list(want)}")
    if p["reference_arm"] != REFERENCE:
        errors.append("the reference arm must be r6")
    if p["variant"] == "full":
        o = p.get("oracle") or {}
        if not (isinstance(o.get("branches"), int) and o["branches"] >= 1 and isinstance(o.get("horizon"), int)
                and o["horizon"] >= 1):
            errors.append("the full protocol needs oracle branches and horizon >= 1")
    q1 = p["success_rule"].get("question_1") or {}
    if q1.get("primary") not in (q1.get("ml") or []):
        errors.append("question 1 needs one primary learned predictor")
    if protocol_seeds(p) != list(seed_ranges.HELD_OUT):
        errors.append("the protocol's seeds are not the held-out range 3000-3099")
    if p["duration_s"] != DURATION_S:
        errors.append(f"the protocol's duration is not {DURATION_S:g} s")
    q2 = p["success_rule"].get("question_2")
    if (p["variant"] == "shadow_only") != (q2 == "not attempted"):
        errors.append("question 2 is attempted exactly when the variant is full")
    if "question_1" not in p["success_rule"]:
        errors.append("protocol lacks the question 1 rule")
    return errors


def protocol_seeds(p: dict) -> list[int]:
    return list(range(p["seeds"]["first"], p["seeds"]["last"] + 1))


# ------------------------------------------------------------------ guards
def _inside(path, root) -> bool:
    p, r = _abs(path).resolve(), _abs(root).resolve()
    return p == r or r in p.parents


def _section(text: str, attempt: int) -> str | None:
    """None, "started" or "finished" for the attempt's section in the report."""
    m = re.search(rf"^## Attempt {attempt}\b(.*)$", text, flags=re.M)
    if not m:
        return None
    return "started" if m.group(1).rstrip().endswith("(started)") else "finished"


def check_run(a: dict, protocol: dict, state: dict) -> list[str]:
    """Errors that refuse the run. `state`: dirty, committed {path: bool}, sha_mismatch [file keys], report_text,
    out_exists, raw_exists (this attempt), prev_raw_exists, freeze {commits, files}, run_config (this attempt's,
    or None), code_changed."""
    errors = []
    seeds = list(a["seeds_list"])
    if not seeds:
        return ["no seeds"]
    n = a["attempt"]
    if any(s in seed_ranges.HELD_OUT for s in seeds):
        if sorted(set(seeds)) != protocol_seeds(protocol) or len(seeds) != len(set(seeds)):
            errors.append("a run on the held-out seeds uses exactly the protocol's seeds "
                          f"{protocol['seeds']['first']}-{protocol['seeds']['last']}")
        if a["label"]:
            errors.append("a run on the held-out seeds takes no label")
        if a.get("priority", "background") != "background":
            errors.append("a run on the held-out seeds is launched through scripts/gentle.sh "
                          "(--priority background)")
        if a["duration"] != protocol["duration_s"]:
            errors.append(f"duration differs from the protocol's {protocol['duration_s']:g} s")
        if state["dirty"]:
            errors.append("the working tree is not clean")
        for path, ok in state["committed"].items():
            if not ok:
                errors.append(f"{safe_path(path)} is not committed at HEAD (or differs from it)")
        if _abs(a["protocol"]).resolve() != _abs(PROTOCOL).resolve():
            errors.append(f"a run on the held-out seeds reads {PROTOCOL} (--protocol)")
        for key, v in protocol["files"].items():
            if v is not None and Path(v["path"]) != Path(OFFICIAL_FILES[key]):
                errors.append(f"the protocol's {key} is not the frozen {OFFICIAL_FILES[key]}")
        freeze = state.get("freeze", {"commits": [], "files": []})
        if len(freeze["commits"]) != 1 or freeze["files"] != [PROTOCOL]:
            errors.append(f"{PROTOCOL} must have exactly one commit, touching only that file (found "
                          f"{len(freeze['commits'])} commit(s), files {freeze['files']})")
        if not (_abs(a["out"]).resolve() == _abs(OFFICIAL_REPORT).resolve()
                and _abs(a["raw_dir"]).resolve() == _abs(OFFICIAL_RAW).resolve()):
            errors.append(f"a run on the held-out seeds writes {OFFICIAL_REPORT} and {OFFICIAL_RAW}")
        if not 1 <= n <= MAX_ATTEMPTS:
            errors.append(f"attempt must be 1..{MAX_ATTEMPTS}")
        elif n > 1 and not state["prev_raw_exists"]:
            errors.append(f"attempt {n} needs the previous attempt's raw directory (attempt {n - 1})")
    else:
        try:
            seed_ranges.check(seeds, "SELECTION")
        except ValueError as e:
            errors.append(f"a development run uses SELECTION seeds only: {e}")
        if a["label"] not in REHEARSAL_LABELS:
            errors.append(f"a development-seed run needs --label {' or '.join(REHEARSAL_LABELS)}")
        if _inside(a["out"], OFFICIAL_REPORT) or _inside(a["raw_dir"], OFFICIAL_RAW):
            errors.append(f"a rehearsal must not write into {OFFICIAL_REPORT} or {OFFICIAL_RAW}")
        if state["out_exists"] and not a.get("resume"):
            errors.append("the report file already exists (never overwritten)")
    for key in state.get("sha_mismatch", []):
        errors.append(f"{key} is missing or differs from the protocol's sha256")
    section = _section(state["report_text"], n)
    if a.get("resume"):
        if not state["raw_exists"]:
            errors.append(f"nothing to resume: attempt {n} has no raw directory")
        elif not state.get("run_config"):
            errors.append(f"attempt {n}'s run_config.json is missing")
        if section != "started":
            errors.append(f"attempt {n} has no unfinished (started) section to resume")
        if state.get("code_changed"):
            errors.append("pis/ or experiments/ changed since the attempt started: run the next attempt instead")
    else:
        if section is not None:
            errors.append(f"attempt {n} already has a section in the report")
        if state["raw_exists"]:
            errors.append(f"the raw directory of attempt {n} already exists (never overwritten)")
    return errors


def consistency(protocol: dict, inc: dict, alarms: dict, advisor: dict | None, shadow_cfg: dict) -> list[str]:
    """Disagreements between the frozen inputs (refuse an official run, warnings in a rehearsal)."""
    out = []
    wm_sha = protocol["files"]["world_model"]["sha256"]
    if shadow_cfg.get("label"):
        out.append(f"the shadow run is labelled {shadow_cfg['label']!r}")
    if shadow_cfg.get("world_model_sha256") != wm_sha:
        out.append("the shadow run scored another world model")
    if {k: shadow_cfg.get("rule", {}).get(k) for k in RULE_KEYS + ("H", "min_lead_windows")} != \
            {k: inc.get(k) for k in RULE_KEYS + ("H", "min_lead_windows")}:
        out.append("the shadow run used another incident rule")
    if alarms.get("H") != inc.get("H"):
        out.append("the alarms config's H differs from the incident config's")
    if protocol["variant"] == "full":
        adv = advisor or {}
        if adv.get("label"):
            out.append(f"the advisor config is labelled {adv['label']!r}")
        if adv.get("alarms") != {k: alarms.get(k) for k in ALARM_KEYS}:
            out.append("the advisor config was tuned with another alarm")
        if adv.get("rule") != {k: inc.get(k) for k in RULE_KEYS}:
            out.append("the advisor config was tuned with another incident rule")
        if adv.get("world_model_sha256") != wm_sha:
            out.append("the advisor config was tuned with another world model")
        if adv.get("fitted_sha256") != protocol["files"]["fitted"]["sha256"]:
            out.append("the advisor config was tuned with other fitted predictors")
        if adv.get("wm_n") != shadow_cfg.get("wm_n"):
            out.append("the advisor config's wm_n differs from the shadow run's")
    return out


# ------------------------------------------------------------------ timing note
def read_timing(path) -> dict | None:
    """The fields of an advisor timing note (experiments/b_time_advisor.py), or None without a verdict."""
    p = _abs(path)
    if not p.is_file():
        return None
    text = p.read_text()
    m95 = re.search(r"^- worst per-episode p95 ([\d.]+) ms: (within|\*\*over\*\*)", text, flags=re.M)
    mmax = re.search(r"^- worst max ([\d.]+) ms: (within|\*\*over\*\*)", text, flags=re.M)
    if not (m95 and mmax):
        return None
    title = re.search(r"^# Advisor decision time at normal priority(?: \((\w+)\))?\s*$", text, flags=re.M)
    cfg = re.search(r"^- config: .*; params (\{.*\}); selection (\w+)\s*$", text, flags=re.M)
    alarm = re.search(r"^- alarm: (.*?)\s*$", text, flags=re.M)
    sha = re.search(r"^- world model sha256 (\S+)", text, flags=re.M)
    code = re.search(r"^- code: (.*?)\s*$", text, flags=re.M)
    return {"worst_p95": float(m95.group(1)), "worst_max": float(mmax.group(1)),
            "p95_ok": m95.group(2) == "within", "max_ok": mmax.group(2) == "within",
            "title": bool(title), "label": (title.group(1) or "") if title else "",
            "params": cfg.group(1) if cfg else None, "selection": cfg.group(2) if cfg else None,
            "alarm": alarm.group(1) if alarm else None, "sha256": sha.group(1) if sha else None,
            "code": code.group(1) if code else ""}


def _alarm_line(alarms: dict) -> str:
    return (f"predictor {alarms.get('predictor')}, threshold {alarms.get('threshold')}, low "
            f"{alarms.get('low')}, H {alarms.get('H')}")


def timing_for(path, advisor_cfg: dict, alarms: dict, wm_sha: str, official: bool, committed: bool) -> dict:
    """The timing verdict for the decision-time criterion, or {"unavailable": reason}."""
    t = read_timing(path)
    if t is None:
        return {"unavailable": f"{safe_path(str(path))} missing or without a verdict"}
    reasons = []
    if official and not committed:
        reasons.append("not committed at HEAD")
    if not t["title"]:
        reasons.append("not an advisor timing note")
    if t["label"]:
        reasons.append(f"labelled {t['label']!r}")
    if "(uncommitted changes)" in t["code"]:
        reasons.append("measured with uncommitted changes")
    try:
        params = json.loads(t["params"]) if t["params"] else None
    except ValueError:
        params = None
    if params != advisor_cfg.get("params"):
        reasons.append("params differ from the advisor config")
    if t["selection"] != "model":
        reasons.append(f"selection {t['selection']}, not model")
    if t["alarm"] != _alarm_line(alarms):
        reasons.append("alarm differs from the alarms config")
    if t["sha256"] != wm_sha:
        reasons.append("world model sha256 differs from the protocol's")
    return {"unavailable": f"{safe_path(str(path))}: " + "; ".join(reasons)} if reasons else t


# ------------------------------------------------------------------ calibration inputs
def calibration_inputs(shadow_cfg: dict, names) -> tuple[dict, dict, dict]:
    """Risk series of the shadow run's calibration episodes (its raw shards) and their incident labels and
    degradation flags (from the store it read): (risks {episode: {name: array}}, labels, active)."""
    return shadow_inputs(shadow_cfg, names, [int(e) for e in shadow_cfg["calibration"]])


def shadow_inputs(shadow_cfg: dict, names, episodes, root: Path = ROOT) -> tuple[dict, dict, dict]:
    """`calibration_inputs` for any of the shadow run's episodes."""
    import pandas as pd
    from pis.phase_b import incidents
    from pis.phase_b.history import Store
    from pis.shards import collect
    shadow_dir = _abs(shadow_cfg["_dir"], root)
    cal = [int(e) for e in episodes]
    raw = pd.concat([pd.read_parquet(d / "risks.parquet") for d in collect(shadow_dir)], ignore_index=True)
    raw = raw[raw.episode.isin(cal)]
    risks = {int(e): {n: g.sort_values("window")[n].to_numpy(float) for n in names} for e, g in raw.groupby("episode")}
    missing = sorted(set(cal) - set(risks))
    if missing:
        raise ValueError(f"the shadow run has no risks for episodes {missing[:5]}")
    rule = shadow_cfg["rule"]
    rule_ = incidents.IncidentRule(int(rule["k"]), int(rule["interval"]), float(rule["majority"]))
    st = Store(_abs(shadow_cfg["store"], root))
    eps = st.episodes().set_index("seed")
    labels, active = {}, {}
    for e in cal:
        w = st.window_rows(e, 0, int(eps.loc[e, "windows"]))
        active[e] = w.deg_active.to_numpy(bool)
        labels[e] = incidents.label_series(w.slo_v2_violation.to_numpy(bool), active[e], rule_, int(rule["warmup"]))
    return risks, labels, active


def calibration_threshold(risks: dict, labels: dict, name: str, calibration, rule: dict) -> float | None:
    """The conformal alarm threshold of one predictor, as the shadow run computes it."""
    from pis.phase_b import alarms
    return alarms.conformal_threshold([(risks[e][name], labels[e]) for e in calibration], alarms.ALPHA,
                                      H=int(rule["H"]), tail=int(rule["k"]) - 1)


# ------------------------------------------------------------------ worker
def _series(flags) -> str:
    return "".join("1" if f else "0" for f in flags)


def _num(x) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


def episode_record(records: list[dict], spec, inc: dict, seed: int, arm: str, seconds: float) -> dict:
    from pis.harness import scenario_metrics
    from pis.phase_b import incidents
    tune = runs.load_script("b_tune_advisor")
    m = tune.episode_metrics(records, spec, inc)
    lab = incidents.label(records, tune.rule_of(inc), int(inc["warmup"]))
    by_cause = {c: {"incidents": sum(i.cause == c for i in lab.incidents),
                    "time_s": float(sum(i.end - i.onset for i in lab.incidents if i.cause == c)) * incidents.WINDOW_S}
                for c in CAUSES}
    info = [r["info"] for r in records]
    return {"seed": int(seed), "arm": arm, "failed": None, "seconds": float(seconds), **m,
            "decide_ms_p95": _num(m["decide_ms_p95"]), "decide_ms_max": _num(m["decide_ms_max"]),
            "slo_v2_pct": scenario_metrics(records, spec)["slo_v2_pct"], "by_cause": by_cause,
            "risk_nan": int(sum(bool(i.get("risk_nan")) for i in info)),
            "reasons": dict(Counter(str(i.get("reason")) for i in info if i.get("mode"))),
            "viol": _series(incidents.violations(records)),
            "active": _series(r["tel"].get("phase_b", {}).get("active", False) for r in records),
            "mode": _series(bool(i.get("mode", False)) for i in info),
            "decide_ms": [None if _num(i.get("decide_ms")) is None else round(float(i["decide_ms"]), 3)
                          for i in info]}


def run_shard(shard, shard_dir: Path, params: dict) -> None:
    """Every arm on each seed of the shard; the r6 arm also gives the predictors' risk series."""
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    from pis.config import setup_imports
    from pis.phase_b import features, policies
    from pis.phase_b.advisor import AdvisorParams
    from pis.phase_b.episode import run_episode
    setup_imports()
    from sched.randomize import known_params
    tune, shadow = runs.load_script("b_tune_advisor"), runs.load_script("b_shadow")
    inc, alarms, dur, rule = params["incident"], params["alarms"], float(params["duration_s"]), params["rule"]
    arms = params["arms"]
    stack = None
    if any(a in SELECTION_OF for a in arms):
        stack = tune.load_stack(alarms, inc, params["world_model"], params["fitted"], params["wm_n"])
        adv_params = AdvisorParams(**params["advisor_params"])
    names = params["names"]
    preds = shadow.load_predictors(params["fitted"] if any(n != "world_model" for n in names) else None, names,
                                   rule, params["world_model"], params["wm_n"]) if REFERENCE in arms else {}
    out, rows_all, risk_frames = [], [], []
    for seed in (int(s) for s in shard.items):
        sc = tune.scenario(seed, dur)
        for arm in arms:
            t0 = time.perf_counter()
            try:
                if arm in SELECTION_OF:
                    ctl = tune.make_advisor(stack, sc.spec, seed, alarms, inc, adv_params, SELECTION_OF[arm], dur,
                                            oracle_branches=params.get("oracle_branches"),
                                            oracle_horizon=params.get("oracle_horizon"))
                else:
                    ctl = policies.make(arm, sc.spec, seed)
                recs = run_episode(sc, ctl, arm)
                rec = episode_record(recs, sc.spec, inc, seed, arm, time.perf_counter() - t0)
            except Exception as e:                      # recorded and listed, never replaced
                tb = traceback.format_exc(limit=3)
                rec = {"seed": seed, "arm": arm, "failed": scrub(f"{type(e).__name__}: {e}"),
                       "traceback": scrub(tb), "seconds": time.perf_counter() - t0}
            if arm == REFERENCE and not rec["failed"]:
                # scoring has its own failure record: a predictor that raises never fails the R6 run
                t1 = time.perf_counter()
                rec["scoring_failed"] = None
                try:
                    rows = features.rows_from_records(recs, known_params(sc.spec), episode=seed)
                    risk = shadow.episode_risks(preds, names, rows, int(rule["H"]), int(rule["warmup"]), seed)
                    rows_all.append(rows)
                    risk_frames.append(pd.DataFrame({"episode": np.full(len(rows), seed, dtype=np.int64),
                                                     "window": np.arange(len(rows), dtype=np.int64), **risk}))
                except Exception as e:
                    rec["scoring_failed"] = scrub(f"{type(e).__name__}: {e}")
                    rec["scoring_traceback"] = scrub(traceback.format_exc(limit=3))
                rec["scoring_s"] = time.perf_counter() - t1
            out.append(rec)
    (shard_dir / EPISODES_FILE).write_text(json.dumps(out, indent=1, allow_nan=False))
    if rows_all:
        pd.concat(rows_all, ignore_index=True).to_parquet(shard_dir / ROWS_FILE, index=False)
        pd.concat(risk_frames, ignore_index=True).to_parquet(shard_dir / RISKS_FILE, index=False)


# ------------------------------------------------------------------ question 2
def _mean(xs) -> float:
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def _diff(a: list, b: list) -> dict:
    from pis.stats import paired_diff
    return paired_diff(np.asarray(a, float), np.asarray(b, float), CONF)


def question_two(episodes: list[dict], arms, rule, timing: dict | None) -> dict:
    """Paired comparisons of every arm with r6 and the success verdict of the advisor (None when the rule is
    "not attempted")."""
    ok = {(e["seed"], e["arm"]): e for e in episodes if not e.get("failed")}
    failed = sorted((e["seed"], e["arm"], e["failed"]) for e in episodes if e.get("failed"))
    ref_seeds = sorted(s for (s, a) in ok if a == REFERENCE)
    chronic = {s for s in ref_seeds if ok[(s, REFERENCE)]["chronic"]}
    calm_ref = [s for s in ref_seeds if s not in chronic]
    ref = {"episodes": len(ref_seeds), "chronic": len(chronic),
           "value_per_s": _mean(ok[(s, REFERENCE)]["value_per_s"] for s in ref_seeds),
           "time_non_chronic": _mean(ok[(s, REFERENCE)]["time_in_incident_s"] for s in calm_ref),
           "incidents": int(sum(ok[(s, REFERENCE)]["incidents"] for s in ref_seeds)),
           "slo_v2_pct": _mean(ok[(s, REFERENCE)]["slo_v2_pct"] for s in ref_seeds)}
    out = {}
    for arm in arms:
        if arm == REFERENCE:
            continue
        pairs = [s for s in ref_seeds if (s, arm) in ok]
        calm = [s for s in pairs if s not in chronic]
        chron = [s for s in pairs if s in chronic]
        quiet = [s for s in calm if ok[(s, REFERENCE)]["incidents"] == 0]

        def col(key, seeds, who, sub=None):
            return [ok[(s, who)][key] if sub is None else ok[(s, who)][key][sub[0]][sub[1]] for s in seeds]

        def diff(key, seeds, sub=None):
            return _diff(col(key, seeds, arm, sub), col(key, seeds, REFERENCE, sub))
        rows = [ok[(s, arm)] for s in pairs]
        fb = [r["fallback_share"] for r in rows if r.get("fallback_share") is not None]
        out[arm] = {
            "pairs": len(pairs), "non_chronic": len(calm), "chronic": len(chron), "quiet": len(quiet),
            "time_arm": _mean(col("time_in_incident_s", calm, arm)),
            "time_r6": _mean(col("time_in_incident_s", calm, REFERENCE)),
            "time_diff": diff("time_in_incident_s", calm), "time_diff_chronic": diff("time_in_incident_s", chron),
            "value_diff": diff("value_per_s", pairs), "r6_value": _mean(col("value_per_s", pairs, REFERENCE)),
            "slo_diff": diff("slo_v2_pct", pairs),
            "incidents_arm": int(sum(col("incidents", pairs, arm))),
            "incidents_r6": int(sum(col("incidents", pairs, REFERENCE))),
            "by_cause": {c: {"incidents_arm": int(sum(col("by_cause", calm, arm, (c, "incidents")))),
                             "incidents_r6": int(sum(col("by_cause", calm, REFERENCE, (c, "incidents")))),
                             "time_diff": diff("by_cause", calm, (c, "time_s"))} for c in CAUSES},
            "quiet_value_diff": diff("value_per_s", quiet),
            "mode_share": _mean(r.get("mode_share") for r in rows), "fallback_share": _mean(fb),
            "risk_nan": int(sum(r.get("risk_nan", 0) for r in rows)),
            "decide_ms_p95": max((r["decide_ms_p95"] for r in rows if r.get("decide_ms_p95") is not None),
                                 default=None),
            "decide_ms_max": max((r["decide_ms_max"] for r in rows if r.get("decide_ms_max") is not None),
                                 default=None)}
    success = None
    if isinstance(rule, dict) and rule.get("candidate") in out:
        a = out[rule["candidate"]]
        hi, lo = a["time_diff"]["hi"], a["value_diff"]["lo"]
        bound = -float(rule["value_tol"]) * abs(a["r6_value"])
        time_ok = bool(math.isfinite(hi) and hi < 0)
        value_ok = bool(math.isfinite(lo) and math.isfinite(bound) and lo >= bound)
        if timing is None or "unavailable" in timing:
            timing_ok, ttext = False, "not available: " + (timing or {}).get("unavailable", "no timing note")
        else:
            timing_ok = bool(timing["p95_ok"] and timing["max_ok"] and timing["worst_p95"] <= rule["decide_p95_ms"]
                             and timing["worst_max"] <= rule["decide_max_ms"])
            ttext = (f"worst p95 {timing['worst_p95']:.1f} ms (limit {rule['decide_p95_ms']:g}), worst max "
                     f"{timing['worst_max']:.1f} ms (limit {rule['decide_max_ms']:g})")
        blocking = [f for f in failed if f[1] in (REFERENCE, rule["candidate"])]
        decidable = not blocking
        success = {"time_ok": time_ok, "value_ok": value_ok, "timing_ok": timing_ok, "timing": ttext,
                   "value_bound": bound, "decidable": decidable, "blocking": blocking,
                   "met": decidable and time_ok and value_ok and timing_ok}
    scoring_failed = sorted((e["seed"], e["scoring_failed"]) for e in episodes
                            if e["arm"] == REFERENCE and not e.get("failed") and e.get("scoring_failed"))
    return {"reference": ref, "arms": out, "success": success, "failed": failed, "scoring_failed": scoring_failed}


# ------------------------------------------------------------------ question 1
def outcome_brier(risks: dict, labels: dict, active: dict, names, episodes, rule: dict) -> dict:
    """Brier of the window risk against the realised outcome (a new counted onset in (t, t + H]) on the windows
    with a finite risk whose outcome can be judged, split by cause (a degradation active at t)."""
    from pis.phase_b import incidents
    H, tail = int(rule["H"]), int(rule["k"]) - 1
    out = {}
    for n in names:
        p, y, c = [], [], []
        for e in episodes:
            r, lab = np.asarray(risks[e][n], float), labels[e]
            last = len(lab.in_incident) - 1
            for t in range(min(len(r), last + 1)):
                if not np.isfinite(r[t]) or t + H + tail > last:
                    continue
                p.append(r[t])
                y.append(float(incidents.onset_within(lab, t, H)))
                c.append(bool(active[e][t]))
        p, y, c = np.array(p), np.array(y), np.array(c, bool)
        out[n] = {}
        for k in ("all",) + CAUSES:
            m = np.ones(len(p), bool) if k == "all" else (c if k == "building_up" else ~c)
            out[n][k] = {"brier": float(((p[m] - y[m]) ** 2).mean()) if m.any() else None, "n": int(m.sum())}
    return out


def question_one(risks: dict, labels: dict, active: dict, names, calibration, held_out, rule: dict,
                 blocking=(), primary: str | None = None) -> dict:
    """The shadow run's scoring with thresholds from the calibration episodes, applied to the held-out ones.
    `blocking`: failed R6 runs or scorings of the attempt (seed, what, message); any makes Q1 not decidable.
    `primary`: the protocol's primary learned predictor, the only one the success verdict is about."""
    shared = set(map(int, calibration)) & set(map(int, held_out))
    if shared:
        raise ValueError(f"held-out and calibration episodes overlap: {sorted(shared)[:5]}")
    shadow = runs.load_script("b_shadow")
    names = [n for n in shadow.ALL if n in names]
    res = shadow.evaluate(risks, labels, active, names, list(calibration), list(held_out), rule, None)
    q1 = shadow.question_one(res)
    verdict = {n: (None if d["building_up"]["lo"] is None else bool(d["building_up"]["lo"] > 0))
               for n, d in q1["diffs"].items()}
    blocking = list(blocking)
    return {"predictors": res, "q1": q1, "verdict": verdict, "decidable": not blocking, "blocking": blocking,
            "above_zero": sorted(n for n, v in verdict.items() if v), "primary": primary,
            "primary_above": None if primary is None or blocking else verdict.get(primary),
            "outcome_brier": outcome_brier(risks, labels, active, names, list(held_out), rule)}


# ------------------------------------------------------------------ report
def _f(x, fmt="{:.2f}") -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def _ci(d: dict, fmt="{:+.2f}") -> str:
    return f"{_f(d['mean'], fmt)} [{_f(d['lo'], fmt)}, {_f(d['hi'], fmt)}] (n {d['n']})"


def report_text(protocol: dict, meta: dict, q2: dict, q1: dict, timing: dict | None, warnings: list[str]) -> str:
    L = [f"## {meta['title']}", ""] + [f"- {k}: {v}" for k, v in meta.items() if k != "title"]
    L += ["", "### Question 2: advisor vs R6, time in incident", ""]
    ref = q2["reference"]
    rule = protocol["success_rule"]["question_2"]
    L.append(f"R6: {ref['episodes']} episodes, {ref['chronic']} chronic (more than half the windows in incident); "
             f"mean value per second {_f(ref['value_per_s'])}, SLO v2 {_f(ref['slo_v2_pct'], '{:.1f}')} %, mean "
             f"time in incident over the non-chronic episodes {_f(ref['time_non_chronic'])} s, "
             f"{ref['incidents']} counted incidents.")
    if rule == "not attempted":
        L += ["", "Question 2: not attempted (the shadow-mode gate stopped the phase before prevention: "
              f"{safe_path(protocol['files']['shadow_note']['path'])})."]
    else:
        L += ["", "Paired differences (arm - R6), 95 % Student-t intervals; time in incident over the non-chronic "
              "episodes, value per second and SLO v2 over all paired episodes. Decision times measured in this "
              "run ran at background priority and are reported only.", "",
              "| arm | time in incident arm / R6 (s) | time diff | value/s diff | SLO v2 diff (pp) | incidents arm / R6 "
              "| intervention rate | fallback share | non-finite risks | decide p95 / max ms |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for arm, a in q2["arms"].items():
            L.append(f"| {arm} | {_f(a['time_arm'])} / {_f(a['time_r6'])} | {_ci(a['time_diff'])} | "
                     f"{_ci(a['value_diff'])} | {_ci(a['slo_diff'])} | {a['incidents_arm']} / {a['incidents_r6']} | "
                     f"{_f(a['mode_share'])} | {_f(a['fallback_share'])} | {a['risk_nan']} | "
                     f"{_f(a['decide_ms_p95'], '{:.1f}')} / {_f(a['decide_ms_max'], '{:.1f}')} |")
        orc = q2.get("oracle")
        if orc:
            L += ["", f"Diagnostic oracle arm ({ORACLE_ARM}: real-simulator branches, own shards after every "
                  f"primary arm; never part of the success rule; {orc['status']}):", "",
                  "| arm | time in incident arm / R6 (s) | time diff | value/s diff | intervention rate | "
                  "fallback share |", "|---|---|---|---|---|---|"]
            for arm, a in orc["arms"].items():
                L.append(f"| {arm} | {_f(a['time_arm'])} / {_f(a['time_r6'])} | {_ci(a['time_diff'])} | "
                         f"{_ci(a['value_diff'])} | {_f(a['mode_share'])} | {_f(a['fallback_share'])} |")
        L += ["", "Chronic episodes (excluded above, reported separately): time in incident diff (arm - R6).", "",
              "| arm | time diff |", "|---|---|"]
        L += [f"| {arm} | {_ci(a['time_diff_chronic'])} |" for arm, a in q2["arms"].items()]
        L += ["", "By cause, non-chronic episodes (building-up: onset while a degradation is active; sudden: "
              "otherwise): counted incidents and time in those incidents.", "",
              "| arm | building-up incidents arm / R6 | building-up time diff | sudden incidents arm / R6 | "
              "sudden time diff |", "|---|---|---|---|---|"]
        for arm, a in q2["arms"].items():
            b, s = a["by_cause"]["building_up"], a["by_cause"]["sudden"]
            L.append(f"| {arm} | {b['incidents_arm']} / {b['incidents_r6']} | {_ci(b['time_diff'])} | "
                     f"{s['incidents_arm']} / {s['incidents_r6']} | {_ci(s['time_diff'])} |")
        L += ["", "False-alarm cost: value per second diff (arm - R6) on the non-chronic episodes without a "
              "counted R6 incident.", "", "| arm | value/s diff |", "|---|---|"]
        L += [f"| {arm} | {_ci(a['quiet_value_diff'])} |" for arm, a in q2["arms"].items()]
        s = q2["success"]
        L += ["", f"Success rule ({rule['candidate']} vs {rule['reference']}):", ""]
        if s is None:
            L.append(f"- no paired result for {rule['candidate']}: question 2 not met")
        else:
            a = q2["arms"][rule["candidate"]]
            L += [f"- (a) time in incident: upper bound {_f(a['time_diff']['hi'], '{:+.2f}')} s < 0: "
                  f"{'met' if s['time_ok'] else 'not met'}",
                  f"- (b) value per second: lower bound {_f(a['value_diff']['lo'], '{:+.3f}')} >= "
                  f"{_f(s['value_bound'], '{:+.3f}')} (-{rule['value_tol']:.0%} of |R6's mean|): "
                  f"{'met' if s['value_ok'] else 'not met'}",
                  f"- (c) decision time at normal priority: {s['timing']}: {'met' if s['timing_ok'] else 'not met'}",
                  ""]
            if not s["decidable"]:
                L += [f"{len(s['blocking'])} {rule['reference']} or {rule['candidate']} run(s) failed in this "
                      "attempt: by the protocol the verdict is not decidable (never a success).", "",
                      "**Question 2: not decidable**"]
            else:
                L.append(f"**Question 2: {'SUCCESS' if s['met'] else 'not met'}**")
    L += ["", "### Question 1: learned vs not learned predictors on the held-out R6 episodes", "",
          "Thresholds: conformal, from the shadow run's calibration episodes (never refitted here). Recall at lead "
          ">= min lead; Brier of the window risk against the realised outcome (new counted onset within H).", "",
          "| predictor | threshold | recall building-up | recall sudden | median lead bu | false-alarm rate | "
          "alarms per quiet episode | Brier all / bu / sudden (n) |", "|---|---|---|---|---|---|---|---|"]
    for n, v in q1["predictors"].items():
        sel, b = v["selection"], v["selection"]["by_cause"]
        br = q1["outcome_brier"][n]
        rec = [f"{_f(b[c]['recall'])} ({b[c]['caught']}/{b[c]['incidents']})" for c in CAUSES]
        L.append(f"| {n} | {_f(v['threshold'], '{:.4f}') if v['threshold'] is not None else 'none feasible'} | "
                 f"{rec[0]} | {rec[1]} | {_f(b['building_up']['median_lead'], '{:.1f}')} | "
                 f"{_f(sel['false_alarm_rate'], '{:.3f}')} | {_f(sel['alarms_per_quiet_episode'])} | "
                 + " / ".join(_f(br[c]['brier'], '{:.3f}') for c in ('all',) + CAUSES)
                 + f" ({'/'.join(str(br[c]['n']) for c in ('all',) + CAUSES)}) |")
    L += ["", f"Better non-learned predictor (held-out building-up recall): {q1['q1'].get('non_ml') or '-'}. "
          "Recall difference, 95 % bootstrap interval over episodes (2000 resamples, shared stream):", "",
          "| predictor | role | building-up | sudden | above zero |", "|---|---|---|---|---|"]
    primary = q1.get("primary")
    for n, d in q1["q1"].get("diffs", {}).items():
        cells = [f"{_f(d[c]['diff'], '{:+.2f}')} [{_f(d[c]['lo'], '{:+.2f}')}, {_f(d[c]['hi'], '{:+.2f}')}]"
                 for c in CAUSES]
        v = q1["verdict"].get(n)
        role = "primary" if n == primary else "secondary (no success claim)"
        L.append(f"| {n} | {role} | {cells[0]} | {cells[1]} | {'-' if v is None else 'yes' if v else 'no'} |")
    L += ["", f"The protocol fixed one primary learned predictor before the run ({primary or '-'}: best building-up "
          "recall on the shadow run's selection episodes); the success rule is judged on it alone. The other "
          "learned predictors are secondary and reported without a success claim.", ""]
    if not q1.get("decidable", True):
        L += [f"{len(q1['blocking'])} R6 run(s) or their predictor scoring failed in this attempt; those episodes "
              "are excluded above and by the protocol the verdict is not decidable.", "",
              "**Question 1: not decidable**"]
    elif primary is None:
        L.append("**Question 1: no primary predictor in the protocol: not decidable**")
    elif q1.get("primary_above"):
        L.append(f"**Question 1: SUCCESS ({primary} above the better non-learned predictor)**")
    else:
        L.append(f"**Question 1: not met ({primary} not above the better non-learned predictor)**")
    scoring = q2.get("scoring_failed", [])
    if scoring:
        L += ["", "### Failed predictor scoring (R6 run kept, episode excluded from question 1)", ""]
        L += [f"- seed {s}: {msg}" for s, msg in scoring]
    if q2["failed"]:
        L += ["", "### Failed runs", ""] + [f"- seed {s}, {arm}: {msg}" for s, arm, msg in q2["failed"]]
    if warnings:
        L += ["", "### Warnings", ""] + [f"- {w}" for w in warnings]
    return "\n".join(L) + "\n"


def replace_section(text: str, attempt: int, new: str) -> str:
    """`text` with the section of `attempt` (from its heading to the next attempt) replaced, or `new` appended."""
    m = re.search(rf"^## Attempt {attempt}\b.*$", text, flags=re.M)
    if not m:
        return text + new
    nxt = re.search(r"^## Attempt \d+\b", text[m.end():], flags=re.M)
    end = m.end() + nxt.start() if nxt else len(text)
    return text[:m.start()] + new + text[end:]


def projection(episodes: list[dict], arms, n_seeds: int, duration_s: float, full_s: float, procs: int) -> dict:
    """Projected wall time of a run with `n_seeds` seeds of `full_s` from this run's per-arm seconds (linear in
    the episode length; scoring included)."""
    per_arm = {a: _mean(e["seconds"] for e in episodes if e["arm"] == a and not e.get("failed")) for a in arms}
    scoring = _mean(e.get("scoring_s") for e in episodes if e["arm"] == REFERENCE and not e.get("failed"))
    per_seed = sum(v for v in per_arm.values() if math.isfinite(v)) + (scoring if math.isfinite(scoring) else 0)
    one = n_seeds * per_seed * full_s / duration_s / 3600
    shards = math.ceil(n_seeds / SHARD_SIZE)
    return {"per_arm_s": per_arm, "scoring_s": scoring, "one_proc_h": one,
            "procs": procs, "procs_h": one * math.ceil(shards / max(procs, 1)) / shards,
            "four_procs_h": one * math.ceil(shards / 4) / shards}


# ------------------------------------------------------------------ driver
PATH_ARGS = ("incident", "alarms", "advisor", "world_model", "shadow_dir", "timing_note", "shadow_note")


def _write_protocol(a) -> None:
    target = _abs(a.protocol)
    if target.exists():
        raise SystemExit(f"{safe_path(a.protocol)} already exists (frozen, never rewritten)")
    overrides = {k: getattr(a, k) for k in PATH_ARGS if getattr(a, k) is not None}
    if target.resolve() == _abs(PROTOCOL).resolve() and overrides:
        raise SystemExit("refusing to write the protocol: the frozen protocol reads the official files")
    try:
        p = make_protocol(overrides, oracle_branches=a.oracle_branches, oracle_horizon=a.oracle_horizon)
    except (OSError, ValueError) as e:
        raise SystemExit(scrub(f"refusing to write the protocol: {e}"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(p, indent=1, allow_nan=False) + "\n")
    print(f"wrote {safe_path(a.protocol)} ({p['variant']}; question 1 primary predictor "
          f"{p['success_rule']['question_1']['primary']})")
    if p["oracle"]:
        proj = oracle_projection(a.tune_dir, load_json(p["files"]["advisor"]["path"]),
                                 load_json(p["files"]["incident"]["path"]), p["oracle"]["branches"],
                                 p["oracle"]["horizon"], len(protocol_seeds(p)), int(p["duration_s"] / 0.5))
        if proj is None:
            print(f"oracle projection unavailable: no tuning episodes of the chosen config in {safe_path(a.tune_dir)}")
        else:
            print(f"oracle arm projection ({p['oracle']['branches']} branches x {p['oracle']['horizon']} windows): "
                  f"mode share mean {proj['mode_share_mean']:.3f} (max {proj['mode_share_max']:.3f}) over "
                  f"{proj['episodes']} tuning episodes, <= {proj['sim_windows_per_decision']} simulated windows per "
                  f"decision -> about {proj['one_proc_h']:.1f} h on one process, {proj['four_procs_h']:.1f} h with 4")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", default=PROTOCOL)
    ap.add_argument("--write-protocol", action="store_true", help="write the protocol file once and stop")
    for k in PATH_ARGS:
        ap.add_argument(f"--{k.replace('_', '-')}", default=None, help="--write-protocol only (rehearsals)")
    ap.add_argument("--oracle-branches", type=int, default=ORACLE_BRANCHES, help="--write-protocol only")
    ap.add_argument("--oracle-horizon", type=int, default=None, help="--write-protocol only (default: h)")
    ap.add_argument("--tune-dir", default=DEFAULT_TUNE_DIR, help="--write-protocol only: tuning raw results "
                                                                  "for the oracle projection")
    ap.add_argument("--seeds", default=None, help="default: the protocol's seeds")
    ap.add_argument("--duration", type=float, default=None, help="default: the protocol's episode length")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted attempt")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--label", default="", help='"rehearsal" or "smoke" for development-seed runs')
    ap.add_argument("--priority", choices=("background", "normal"), default="background")
    ap.add_argument("--out", default=OFFICIAL_REPORT)
    ap.add_argument("--raw-dir", default=OFFICIAL_RAW)
    a = ap.parse_args(argv)
    if a.write_protocol:
        return _write_protocol(a)
    if any(getattr(a, k) is not None for k in PATH_ARGS):
        raise SystemExit("refusing to run: input paths come from the protocol (they are --write-protocol options)")
    try:
        protocol = load_json(a.protocol)
    except (OSError, ValueError) as e:
        raise SystemExit(scrub(f"refusing to run: cannot read the protocol {safe_path(a.protocol)}: {e}"))
    errors = validate_protocol(protocol)
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))
    try:
        seeds = seed_ranges.parse(a.seeds) if a.seeds else protocol_seeds(protocol)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    duration = float(protocol["duration_s"] if a.duration is None else a.duration)
    official = any(s in seed_ranges.HELD_OUT for s in seeds)
    raw_root = _abs(a.raw_dir)
    raw_attempt, out = raw_root / f"attempt_{a.attempt}", _abs(a.out)
    files = protocol["files"]
    try:
        run_config = json.loads((raw_attempt / "run_config.json").read_text())
    except (OSError, ValueError):
        run_config = None
    tracked = [a.protocol] + [files[k]["path"] for k in TRACKED if files.get(k)]
    state = {"dirty": runs.tree_dirty(), "committed": {p: committed_clean(p) for p in tracked},
             "sha_mismatch": [k for k, v in files.items() if v and file_sha(k, v["path"]) != v["sha256"]],
             "report_text": out.read_text() if out.is_file() else "", "out_exists": out.exists(),
             "raw_exists": raw_attempt.exists(), "prev_raw_exists": (raw_root / f"attempt_{a.attempt - 1}").is_dir(),
             "freeze": freeze_state(), "run_config": run_config, "code_changed": code_changed(run_config)}
    args = {"seeds_list": seeds, "label": a.label, "duration": duration, "out": a.out, "raw_dir": a.raw_dir,
            "attempt": a.attempt, "resume": a.resume, "priority": a.priority, "protocol": a.protocol}
    errors = check_run(args, protocol, state)
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))

    full = protocol["variant"] == "full"
    inc, alarms = load_json(files["incident"]["path"]), load_json(files["alarms"]["path"])
    advisor = load_json(files["advisor"]["path"]) if full else None
    shadow_cfg = {**load_json(files["shadow_config"]["path"]), "_dir": files["shadow_risks"]["path"]}
    rule = {k: inc[k] for k in RULE_KEYS + ("H", "min_lead_windows")}
    names = list(protocol["predictors"]["names"])
    wm_sha = files["world_model"]["sha256"]
    problems = consistency(protocol, inc, alarms, advisor, shadow_cfg)
    timing = None
    if full:
        timing = timing_for(files["timing_note"]["path"], advisor, alarms, wm_sha, official,
                            committed_clean(files["timing_note"]["path"]))
        if "unavailable" in timing:
            problems.append(f"decision time unavailable: {timing['unavailable']}")
    try:
        cal_risks, cal_labels, cal_active = calibration_inputs(shadow_cfg, names)
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(scrub(f"refusing to run: cannot read the shadow run's calibration risks: {e}"))
    calibration = sorted(int(e) for e in shadow_cfg["calibration"])
    if alarms.get("predictor") in names:
        thr = calibration_threshold(cal_risks, cal_labels, alarms["predictor"], calibration, rule)
        if thr is None or alarms.get("threshold") is None or abs(thr - float(alarms["threshold"])) > 1e-12:
            problems.append(f"the calibration threshold of {alarms['predictor']} ({thr}) differs from the alarms "
                            f"config's ({alarms.get('threshold')})")
    refusals, drift = runs.load_script("b_shadow").identity_check(_abs(files["shadow_risks"]["path"]),
                                                                  _abs(files["world_model"]["path"]))
    problems += refusals
    for w in drift:
        print(f"warning: {w}", flush=True)
    from pis.config import sibling_paths
    pa = load_json(files["phase_a_protocol"]["path"])["arms"]["base_r6"]
    ckpt = sibling_paths()["sched"] / pa["checkpoint"]
    if not ckpt.is_file() or _sha256(ckpt) != pa["sha256"]:
        raise SystemExit("refusing to run: the R6 checkpoint differs from the Phase A protocol's")
    if official and problems:
        raise SystemExit(scrub("refusing to run: " + "; ".join(problems)))
    problems += drift                           # warnings only
    pinned = {"protocol_sha256": _sha256(_abs(a.protocol)), "files": {k: v["sha256"] for k, v in files.items() if v}}
    prev = raw_root / f"attempt_{a.attempt - 1}"
    if official and a.attempt > 1 and not a.resume:
        try:
            prev_pinned = json.loads((prev / "run_config.json").read_text()).get("pinned")
        except (OSError, ValueError):
            prev_pinned = None
        if prev_pinned != pinned:
            raise SystemExit(f"refusing to run: attempt {a.attempt} must use the frozen files of attempt "
                             f"{a.attempt - 1} (its run_config.json differs or is missing)")

    head = _git("rev-parse", "HEAD")[1] or "unknown"
    arms = list(protocol["arms"])
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    if a.resume:
        config = run_config
        config.setdefault("resumed_utc", []).append(now)
    else:
        import torch
        contiguous = seeds == list(range(seeds[0], seeds[-1] + 1))
        seed_text = (str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}-{seeds[-1]}") if contiguous \
            else ",".join(map(str, seeds))
        freeze = state["freeze"]["commits"][-1] if state["freeze"]["commits"] and \
            _abs(a.protocol).resolve() == _abs(PROTOCOL).resolve() else None
        config = {"attempt": a.attempt, "label": a.label, "official": official, "head": head,
                  "code": f"git HEAD {head}{' (uncommitted changes)' if state['dirty'] else ''}",
                  "protocol": {"path": safe_path(a.protocol), "sha256": pinned["protocol_sha256"],
                               "freeze_commit": freeze, "variant": protocol["variant"]},
                  "pinned": pinned, "arms": arms, "n_seeds": len(seeds),
                  "seeds": seed_text, "duration_s": duration, "procs": a.procs, "priority": a.priority,
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "torch": torch.__version__},
                  "start_utc": now}
    title = f"Attempt {a.attempt}{' (' + a.label + ')' if a.label else ''}"
    meta = {"title": title, "code": config["code"],
            "protocol": f"{config['protocol']['path']} (sha256 {config['protocol']['sha256'][:16]}, variant "
                        f"{protocol['variant']}" + (f", frozen in commit {config['protocol']['freeze_commit']}"
                                                    if config["protocol"].get("freeze_commit") else "") + ")",
            "frozen files": ", ".join(f"{k} {safe_path(v['path'])} ({v['sha256'][:16]})"
                                      for k, v in files.items() if v),
            "seeds": f"{config['seeds']} ({len(seeds)} episodes, {duration:g} s each)",
            "arms": ", ".join(arms),
            "raw": f"{OFFICIAL_RAW}/attempt_{a.attempt}/" if official else safe_path(str(raw_attempt))}
    if a.attempt > 1 and prev.is_dir():
        meta["previous attempt"] = f"attempt {a.attempt - 1}: {summarise_attempt(prev)}"
    if not a.resume:
        raw_attempt.mkdir(parents=True, exist_ok=False)          # never overwrite an earlier attempt
        (raw_attempt / "run_config.json").write_text(scrub(json.dumps(config, indent=1)) + "\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        head_text = "" if state["report_text"] else (
            "# Incident advisor: final paired evaluation\n\nProtocol: configs/phase_b/final_eval.json, frozen in its "
            "own commit before the run. One section per attempt, written when the attempt starts and completed "
            "when it finishes; never edited afterwards.\n\n")
        stub = (f"## {title} (started)\n\n" + "".join(f"- {k}: {v}\n" for k, v in meta.items() if k != "title")
                + f"- started (UTC): {config['start_utc']}\n\nThis attempt has not finished: it is running, or it "
                "stopped (continue it with --resume after committing this section); its finished shards are in the "
                "raw directory.\n\n")
        with out.open("a") as f:
            f.write(scrub(head_text + stub))
    else:
        (raw_attempt / "run_config.json").write_text(scrub(json.dumps(config, indent=1)) + "\n")

    from pis.shards import collect, make_shards, run_shards
    oracle = protocol.get("oracle") or {}
    primary_arms = [x for x in arms if x != ORACLE_ARM]
    oracle_arms = [x for x in arms if x == ORACLE_ARM]
    params = {"incident": inc, "alarms": alarms, "duration_s": duration, "rule": rule, "names": names,
              "world_model": str(_abs(files["world_model"]["path"])), "fitted": str(_abs(files["fitted"]["path"])),
              "wm_n": int(protocol["predictors"]["wm_n"]),
              "advisor_params": advisor["params"] if full else None,
              "oracle_branches": oracle.get("branches"), "oracle_horizon": oracle.get("horizon")}
    shards = make_shards(seeds, SHARD_SIZE)

    def run_set(sub: str, which: list[str]):
        cfg = {**pinned, "seeds": seeds, "duration_s": duration, "arms": which, "label": a.label,
               "shard_size": SHARD_SIZE, "names": names, "oracle": oracle if ORACLE_ARM in which else None}
        t_set = time.time()
        res = run_shards(raw_attempt / sub, shards, partial(run_shard, params={**params, "arms": which}), cfg, a.procs)
        config["runtime_s"] = round(config.get("runtime_s", 0) + time.time() - t_set)
        print(f"{sub}: done {len(res['done'])}, skipped {len(res['skipped'])}, failed {res['failed']} in "
              f"{time.time() - t_set:.0f} s", flush=True)
        return collect(raw_attempt / sub)

    print(f"{title}: {len(seeds)} seeds x {len(primary_arms)} primary arms in {len(shards)} shards"
          + (f", then {ORACLE_ARM} in its own shards" if oracle_arms else "") + f", {a.procs} processes", flush=True)
    dirs = run_set(SHARDS, primary_arms)
    if len(dirs) != len(shards):
        (raw_attempt / "run_config.json").write_text(scrub(json.dumps(config, indent=1)) + "\n")
        print(f"unfinished shards: commit the report's started section, then rerun with --attempt {a.attempt} "
              "--resume", flush=True)
        return {"unfinished": True}

    import pandas as pd
    from pis.phase_b import incidents
    episodes = [e for d in dirs for e in json.loads((d / EPISODES_FILE).read_text())]
    # the primary result is final before the diagnostic oracle arm starts
    primary_q2 = question_two(episodes, primary_arms, protocol["success_rule"]["question_2"], timing)
    (raw_attempt / PRIMARY_FILE).write_text(scrub(json.dumps(
        {"success": primary_q2["success"], "failed": primary_q2["failed"],
         "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, indent=1, default=str)) + "\n")
    if primary_q2["success"] is not None:
        s_ = primary_q2["success"]
        print("primary question 2: " + ("not decidable" if not s_["decidable"] else "SUCCESS" if s_["met"]
                                        else "not met"), flush=True)
    oracle_status = None
    if oracle_arms:
        odirs = run_set(ORACLE_SHARDS, oracle_arms)
        oracle_status = "complete" if len(odirs) == len(shards) else \
            f"incomplete: {len(odirs)} of {len(shards)} shards finished"
        episodes += [e for d in odirs for e in json.loads((d / EPISODES_FILE).read_text())]
    held_risk_files = [d / RISKS_FILE for d in dirs if (d / RISKS_FILE).is_file()]
    risks, labels, active = dict(cal_risks), dict(cal_labels), dict(cal_active)
    rule_ = incidents.IncidentRule(int(rule["k"]), int(rule["interval"]), float(rule["majority"]))
    scored = []
    shared = set(map(int, calibration)) & {int(e["seed"]) for e in episodes}
    if shared:
        raise SystemExit(f"held-out and calibration episodes overlap: {sorted(shared)[:5]}")
    if held_risk_files:
        raw = pd.concat([pd.read_parquet(p) for p in held_risk_files], ignore_index=True)
        ref = {e["seed"]: e for e in episodes if e["arm"] == REFERENCE and not e.get("failed")}
        for e, g in raw.groupby("episode"):
            e = int(e)
            risks[e] = {n: g.sort_values("window")[n].to_numpy(float) for n in names}
            active[e] = np.array([c == "1" for c in ref[e]["active"]], bool)
            labels[e] = incidents.label_series(np.array([c == "1" for c in ref[e]["viol"]], bool), active[e], rule_,
                                               int(rule["warmup"]))
            scored.append(e)
    q2 = primary_q2
    if oracle_arms:
        q2["oracle"] = {**question_two(episodes, [REFERENCE] + oracle_arms, "not attempted", None),
                        "status": oracle_status}
    blocking = [(s, arm, msg) for s, arm, msg in q2["failed"] if arm == REFERENCE] + \
        [(s, "scoring", msg) for s, msg in q2["scoring_failed"]]
    q1 = question_one(risks, labels, active, names, calibration, sorted(scored), rule, blocking,
                      protocol["success_rule"]["question_1"].get("primary"))
    proj = projection(episodes, arms, len(protocol_seeds(protocol)), duration, float(protocol["duration_s"]),
                      a.procs)
    config.update(end_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                  runs_ok=sum(not e.get("failed") for e in episodes), runs_failed=len(q2["failed"]))
    (raw_attempt / "run_config.json").write_text(scrub(json.dumps(config, indent=1)) + "\n")
    meta["runtime"] = (f"{config['runtime_s']} s with {a.procs} process(es) ({a.priority} priority); seconds per "
                       "episode by arm (mean): " + ", ".join(f"{k} {_f(v, '{:.1f}')}" for k, v in
                                                            proj["per_arm_s"].items())
                       + f"; predictor scoring {_f(proj['scoring_s'], '{:.1f}')} s per R6 episode")
    meta["projection"] = (f"Projected official run ({len(protocol_seeds(protocol))} seeds x "
                          f"{protocol['duration_s']:g} s, all arms): {proj['one_proc_h']:.1f} h on one process, "
                          f"{proj['procs_h']:.1f} h with {a.procs} process(es), {proj['four_procs_h']:.1f} h with 4; linear in the episode length, so a rough "
                          "estimate from short episodes")
    meta["finished (UTC)"] = config["end_utc"]
    if oracle_status and oracle_status != "complete":
        problems.append(f"the diagnostic {ORACLE_ARM} arm is {oracle_status}")
    text = report_text(protocol, meta, q2, q1, timing, problems)
    out.write_text(replace_section(out.read_text(), a.attempt, scrub(text + "\n")))
    print(f"done: {config['runs_ok']} runs ok, {config['runs_failed']} failed -> {safe_path(a.out)}", flush=True)
    return {"episodes": episodes, "q2": q2, "q1": q1, "text": text, "config": config, "projection": proj}


def summarise_attempt(raw_attempt: Path) -> str:
    """One line on an earlier attempt from its raw directory: runs recorded, failed, completed or not."""
    from pis.shards import collect
    recs = [e for d in collect(raw_attempt / SHARDS) for e in json.loads((d / EPISODES_FILE).read_text())]
    try:
        cfg = json.loads((raw_attempt / "run_config.json").read_text())
    except (OSError, ValueError):
        cfg = {}
    expected = cfg.get("n_seeds", 0) * len(cfg.get("arms", [])) or "?"
    failed = sum(bool(r.get("failed")) for r in recs)
    return (f"{len(recs)} of {expected} runs recorded ({len(recs) - failed} ok, {failed} failed), "
            f"{'completed' if 'end_utc' in cfg else 'did not complete'}")


if __name__ == "__main__":
    main()
