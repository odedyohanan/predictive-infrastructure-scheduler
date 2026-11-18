"""Final paired evaluation on the held-out seeds 2000-2099, by the frozen protocol in configs/final_eval.json.

Official run (once; a crash or a bug: fix it, rerun as --attempt 2, report both, never tune on the results):
  scripts/gentle.sh final_eval .venv/bin/python experiments/final_eval.py --procs 4
Rehearsal on development seeds (never touches the official outputs):
  .venv/bin/python experiments/final_eval.py --seeds 1150-1154 --label rehearsal --duration 10 \\
      --step-a-config <scratch>/step_a.json --step-b-config <scratch>/step_b.json \\
      --out <scratch>/report.md --raw-dir <scratch>/raw --procs 2

The protocol file fixes the seeds, the scenario length, the arms (`pis.runner.ARMS`), the metrics, the success
rule (value per second vs base_r6 with a paired 95% CI above zero; SLO v2 non-inferior within +1 pp on the
paired upper 95% bound; decision time within p95 50 ms / max 500 ms at normal priority, read from the timing
notes of experiments/time_decisions.py) and the failure handling. `make_protocol()` is the generator of that
file; the run reads the file, never the generator.

Guards. Seeds 1000-1099 are never accepted. A run on any seed in 2000-2099 must use exactly the protocol's
seeds and scenario length, no label, the official outputs, a clean working tree, configs/final_eval.json with
exactly one commit that touches only it, the step configs the protocol names, all three committed at HEAD,
unlabelled step configs (Step B tuned on top of this Step A), and ensemble members whose sha256 prefixes match
the committed tuning notes (results/step_a/tuning.md, results/step_b/report.md). Attempts are numbered 1..3,
each needs the previous attempt's raw directory, and an attempt that already has a report section or a raw
directory is refused. A timing note counts for part (c) of the success rule only if it is unlabelled, records
the evaluated params, reviser and member sha256 prefixes, and (official run) is committed at HEAD; otherwise
part (c) is "not available" with the reason. Any other run must use
development seeds 1100-1199 only, label "rehearsal" or "smoke", and outputs outside results/final and
results/raw/final; it never overwrites a report or a raw attempt either.

Outputs: one "## Attempt N" section in results/final/report.md (tracked), appended as a "(started)" stub with
the provenance before any run (so a crash leaves it, and the dirty tree forces committing it before the next
attempt) and replaced by the full section at the end; attempt N > 1 summarises attempt N-1 from its raw
records. Per-scenario records
streamed to results/raw/final/attempt<N>/runs.jsonl plus run_config.json (gitignored). Recorded: git HEAD,
sha256 of the protocol, of the step configs and of every model checkpoint. Failed runs are recorded with
scrubbed messages and listed in the report, never dropped or replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pis.privacy import safe_path, scrub  # noqa: E402
from pis.runner import ARMS, LOOP_OPTIONS, arm_definitions, describe_arms, report, run_arm  # noqa: E402

PROTOCOL = "configs/final_eval.json"
STEP_CONFIGS = {"step_a": "configs/step_a.json", "step_b": "configs/step_b.json"}
TIMING_NOTES = {"loop_a": "results/step_a/timing.md", "loop_b": "results/step_b/timing.md"}
TUNING_NOTES = {"step_a": "results/step_a/tuning.md", "step_b": "results/step_b/report.md"}
OFFICIAL_REPORT, OFFICIAL_RAW = "results/final/report.md", "results/raw/final"
FINAL_SEEDS, DEV_SEEDS, USED_SEEDS = range(2000, 2100), range(1100, 1200), range(1000, 1100)
MAX_ATTEMPTS = 3
REHEARSAL_LABELS = ("rehearsal", "smoke")
SLOW_FIRST = ("oracle_selection", "loop_b", "loop_no_fallback", "loop_no_uncertainty", "loop_a",
              "random_selection", "base_r6", "base_r3", "adaptive_c9")


def make_protocol() -> dict:
    """The frozen protocol (written once to configs/final_eval.json and committed on its own)."""
    loop = {"kind": "loop", "config": STEP_CONFIGS["step_a"]}
    return {
        "name": "final paired evaluation",
        "seeds": {"first": FINAL_SEEDS.start, "last": FINAL_SEEDS.stop - 1},
        "duration_s": 60.0,
        "scenario": "spec = random_spec(default_rng(900000 + seed)); Simulator(spec, seed, duration_s); one "
                    "decision per 500 ms window; every arm starts from the same isolated policy",
        "reference_arm": "base_r6",
        "arms": {
            "base_r6": {"kind": "base", "checkpoint": "models/rl_r6/final.pt",
                        "sha256": "ca02c87cf70d40a735b698cb5561c20b2816864349de33bf04d6824c0c2247fd",
                        "role": "reference: the base scheduler as delivered, greedy"},
            "base_r3": {"kind": "base", "checkpoint": "models/rl_r3/final.pt",
                        "sha256": "e0cb957b8dc293defbdc5cbe043f02f8b60069c267b239d07a3cb28b3892ed21",
                        "role": "earlier RL scheduler, greedy (reference point)"},
            "loop_a": {**loop, "options": LOOP_OPTIONS["loop_a"],
                       "role": "candidate: Step A loop, members and Params from the Step A config"},
            "loop_b": {"kind": "reviser", "config": STEP_CONFIGS["step_b"],
                       "role": "candidate: Step B loop, members, Step A Params and reviser from the Step B config"},
            "random_selection": {**loop, "options": LOOP_OPTIONS["random_selection"],
                                 "role": "ablation, lower bound: uniform choice among the same candidates"},
            "oracle_selection": {**loop, "options": LOOP_OPTIONS["oracle_selection"],
                                 "role": "ablation, upper bound: candidates ranked by the real simulator"},
            "loop_no_uncertainty": {**loop, "options": LOOP_OPTIONS["loop_no_uncertainty"],
                                    "role": "ablation: Step A without the disagreement penalty"},
            "loop_no_fallback": {**loop, "options": LOOP_OPTIONS["loop_no_fallback"],
                                 "role": "ablation: Step A without the trust gate and fallback"},
            "adaptive_c9": {"kind": "adaptive", "h0": 2.0, "r_hi": 0.6, "hold": 6,
                            "role": "non-ML feedback baseline, configuration 9 of the sibling's adaptive grid"},
        },
        "metrics": {
            "primary": "value_per_s: (B batches x value + C core-ms x value) per second, as the RL reward",
            "per_scenario": ["value_per_s", "slo_v2_pct", "slo_v1_pct", "a_late_pct", "a_queue_overdue_end",
                             "policy_changes", "decide_ms_mean", "decide_ms_p95", "decide_ms_max",
                             "fallback_rate", "gru_error", "rank_agreement", "top1_agreement"],
            "slo": "window-violation % of A's p99 SLO; v2 (sees queued requests) is primary, v1 reported",
            "run_time_model_probe": "arms that rank with the ensemble: every probe_every-th model decision, every "
                                    "candidate run for H windows in a copy of the real simulator after the decision "
                                    "is timed; GRU error = mean |ensemble-mean forecast - real| on the "
                                    "disagreement keys in training-std units; rank agreement = Spearman of "
                                    "predicted vs real reward over candidates",
            "probe_every": 5,
        },
        "success_rule": {
            "candidates": ["loop_a", "loop_b"],
            "reference": "base_r6",
            "confidence": 0.95,
            "value": "paired 95% CI of (candidate - base_r6) value_per_s lies above zero",
            "slo_v2_margin_pp": 1.0,
            "slo": "paired upper 95% bound of (candidate - base_r6) slo_v2_pct <= +1.0 percentage point",
            "decision_time": "worst per-scenario p95 <= 50 ms and max <= 500 ms at normal priority, from "
                             "experiments/time_decisions.py (results/step_a/timing.md, results/step_b/timing.md); "
                             "decision time measured in this run is under background QoS and reported only",
            "pairs": "per scenario, over the scenarios where both arms succeeded; Student-t CI",
            "otherwise": "any other result is reported as it is; a CI including zero is no detectable difference",
        },
        "decision_time_ms": {"p95_target": 50.0, "hard_limit": 500.0},
        "secondary": [
            {"a": "loop_a", "b": "random_selection", "why": "value of the model's ranking"},
            {"a": "oracle_selection", "b": "loop_a", "why": "headroom left in the candidate set"},
            {"a": "loop_b", "b": "loop_a", "why": "value of revision"},
            {"a": "loop_a", "b": "loop_no_uncertainty", "why": "value of the uncertainty penalty"},
            {"a": "loop_a", "b": "loop_no_fallback", "why": "value of the trust gate and fallback"},
            {"a": "base_r6", "b": "base_r3", "why": "R6 vs R3 reference"},
            {"a": "loop_a", "b": "adaptive_c9", "why": "Step A vs non-ML feedback"},
            {"a": "loop_b", "b": "adaptive_c9", "why": "Step B vs non-ML feedback"},
        ],
        "failure_handling": "a run that raises is recorded with a scrubbed message and listed, never dropped or "
                            "replaced; comparisons use the scenarios where both arms succeeded. A crash or a bug: "
                            "fix, rerun as the next attempt, report every attempt; no tuning on these results.",
        "outputs": {"report": OFFICIAL_REPORT, "raw": OFFICIAL_RAW + "/attempt<N>/"},
    }


def _git(*args) -> tuple[int, str]:
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def committed_clean(path: str) -> bool:
    """Tracked at HEAD and identical to HEAD in the working tree."""
    ok = _git("cat-file", "-e", f"HEAD:{path}")[0] == 0
    return ok and _git("diff", "--quiet", "HEAD", "--", path)[0] == 0


def freeze_state() -> dict:
    """The commits that touched the protocol file (newest first) and the files of the oldest one."""
    commits = [c for c in _git("log", "--format=%H", "--", PROTOCOL)[1].splitlines() if c]
    files = _git("show", "--name-only", "--format=", commits[-1])[1].split() if commits else []
    return {"commits": commits, "files": files}


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _inside(path: str, root: str) -> bool:
    p, r = Path(path).resolve(), Path(root).resolve()
    return p == r or r in p.parents


def load_json(path: str) -> dict:
    def _no_nan(x):
        raise ValueError(f"non-strict JSON constant {x}")
    return json.loads(Path(path).read_text(), parse_constant=_no_nan)


def validate_protocol(p: dict) -> list[str]:
    errors = []
    for k in ("seeds", "duration_s", "reference_arm", "arms", "success_rule", "metrics"):
        if k not in p:
            errors.append(f"protocol lacks {k!r}")
    if errors:
        return errors
    if tuple(p["arms"]) != ARMS:
        errors.append(f"protocol arms {list(p['arms'])} differ from {list(ARMS)}")
    if p["reference_arm"] != "base_r6" or p["success_rule"].get("reference") != "base_r6":
        errors.append("the reference arm must be base_r6")
    return errors


def protocol_seeds(p: dict) -> list[int]:
    return list(range(p["seeds"]["first"], p["seeds"]["last"] + 1))


def check_run(a: dict, protocol: dict, state: dict) -> list[str]:
    """Errors that refuse the run. `state`: dirty (whole tree), committed {path: bool}, report_text,
    raw_exists (this attempt), prev_raw_exists (previous attempt), out_exists, freeze {commits, files} (the
    commits touching the protocol and the files of the first one)."""
    errors = []
    seeds = a["seeds_list"]
    if not seeds:
        return ["no seeds"]
    used = sorted(s for s in set(seeds) if s in USED_SEEDS)
    if used:
        errors.append(f"seeds {USED_SEEDS.start}-{USED_SEEDS.stop - 1} are never used here: {used[:5]}")
    final = any(s in FINAL_SEEDS for s in seeds)
    if final:
        if sorted(set(seeds)) != protocol_seeds(protocol) or len(seeds) != len(set(seeds)):
            errors.append("a run on the held-out seeds uses exactly the protocol's seeds "
                          f"{protocol['seeds']['first']}-{protocol['seeds']['last']}")
        if a["label"]:
            errors.append("a run on the held-out seeds takes no label")
        if a.get("priority", "background") != "background":
            errors.append("a run on the held-out seeds is launched through scripts/gentle.sh (--priority background)")
        if a["duration"] != protocol["duration_s"]:
            errors.append(f"duration differs from the protocol's {protocol['duration_s']:g} s")
        if state["dirty"]:
            errors.append("the working tree is not clean")
        for path, ok in state["committed"].items():
            if not ok:
                errors.append(f"{path} is not committed at HEAD (or differs from it)")
        expected = {"protocol": PROTOCOL, "step_a_config": protocol["arms"]["loop_a"].get("config"),
                    "step_b_config": protocol["arms"]["loop_b"].get("config")}
        for key, want in expected.items():
            got = a.get(key, want)
            if not want or Path(got).resolve() != Path(want).resolve():
                errors.append(f"a run on the held-out seeds reads {want} (--{key.replace('_', '-')})")
        freeze = state.get("freeze", {"commits": [], "files": []})
        if len(freeze["commits"]) != 1 or freeze["files"] != [PROTOCOL]:
            errors.append(f"{PROTOCOL} must have exactly one commit, touching only that file (found "
                          f"{len(freeze['commits'])} commit(s), files {freeze['files']})")
        if not _inside(a["out"], OFFICIAL_REPORT) or not _inside(a["raw_dir"], OFFICIAL_RAW):
            errors.append(f"a run on the held-out seeds writes {OFFICIAL_REPORT} and {OFFICIAL_RAW}")
        if not 1 <= a["attempt"] <= MAX_ATTEMPTS:
            errors.append(f"attempt must be 1..{MAX_ATTEMPTS}")
        elif a["attempt"] > 1 and not state["prev_raw_exists"]:
            errors.append(f"attempt {a['attempt']} needs attempt {a['attempt'] - 1}'s raw directory")
    else:
        outside = sorted(s for s in set(seeds) if s not in DEV_SEEDS)
        if outside:
            errors.append(f"seeds outside the development range {DEV_SEEDS.start}-{DEV_SEEDS.stop - 1}: "
                          f"{outside[:5]}")
        if a["label"] not in REHEARSAL_LABELS:
            errors.append(f"a development-seed run needs --label {' or '.join(REHEARSAL_LABELS)}")
        if _inside(a["out"], "results/final") or _inside(a["raw_dir"], OFFICIAL_RAW):
            errors.append("a rehearsal must not write into results/final or results/raw/final")
        if state["out_exists"]:
            errors.append("the report file already exists (never overwritten)")
    if re.search(rf"^## Attempt {a['attempt']}\b", state["report_text"], flags=re.M):
        errors.append(f"attempt {a['attempt']} already has a section in the report")
    if state["raw_exists"]:
        errors.append(f"the raw directory of attempt {a['attempt']} already exists (never overwritten)")
    return errors


def read_timing(path: str) -> dict | None:
    """Verdict of a normal-priority timing note of experiments/time_decisions.py, or None if absent."""
    p = Path(path)
    if not p.is_file():
        return None
    text = p.read_text()
    m95 = re.search(r"^- worst per-scenario p95 ([\d.]+) ms: (within|\*\*over\*\*)", text, flags=re.M)
    mmax = re.search(r"^- worst max ([\d.]+) ms: (within|\*\*over\*\*)", text, flags=re.M)
    if not (m95 and mmax):
        return None
    label = re.search(r"^# .*\((\w+)\)\s*$", text, flags=re.M)
    return {"worst_p95": float(m95.group(1)), "worst_max": float(mmax.group(1)),
            "p95_ok": m95.group(2) == "within", "max_ok": mmax.group(2) == "within",
            "label": label.group(1) if label else "", "source": path + (f" ({label.group(1)})" if label else "")}


_MEMBER = re.compile(r"([^\s,;]+) \(sha256 ([0-9a-f]{16})\)")


def recorded_members(text: str) -> list[tuple[str, str]] | None:
    """(path, sha256[:16]) of each member on a note's "- members:" line (tuning, Step B and timing notes)."""
    m = re.search(r"^- members: (.*)$", text, flags=re.M)
    return _MEMBER.findall(m.group(1)) if m else None


def member_mismatch(text: str, members: list[str], shas: dict[str, str], exact: bool) -> str | None:
    """Why the note's recorded members do not match `members` (sha256 by member), or None when they do.
    `exact`: the same members in the same order; otherwise every member must be among the recorded ones."""
    rec = recorded_members(text)
    if not rec:
        return "no member sha256 recorded"
    if exact and [p for p, _ in rec] != [safe_path(m) for m in members]:
        return "recorded members differ from the config's"
    got = dict(rec)
    for m in members:
        k = safe_path(m)
        if k not in got:
            return f"member {k} not recorded"
        if got[k] != shas[m][:16]:
            return f"member {k} sha256 {shas[m][:16]} differs from the recorded {got[k]}"
    return None


def recorded_params(text: str) -> tuple[dict | None, dict | None]:
    """(controller params, reviser) recorded on a timing note's "- controller params:" line."""
    m = re.search(r"^- controller params: (.*)$", text, flags=re.M)
    if not m:
        return None, None
    params, _, reviser = m.group(1).partition("; reviser ")
    try:
        return json.loads(params), (json.loads(reviser) if reviser else None)
    except ValueError:
        return None, None


def _same_params(a: dict | None, b: dict) -> bool:
    from pis.evaluator import Params
    try:
        return a is not None and Params.from_dict(a).to_dict() == Params.from_dict(b).to_dict()
    except (TypeError, ValueError, KeyError):
        return False


def timing_for(path: str, kind: str, cfg: dict, shas: dict[str, str], official: bool, committed: bool) -> dict:
    """The timing verdict for part (c) of the success rule, or {"unavailable": reason}: the note must exist and
    be unlabelled, record the config's params (and reviser) and member sha256 prefixes, and in an official run
    be committed at HEAD."""
    t = read_timing(path)
    if t is None:
        return {"unavailable": f"{path} missing or without a verdict"}
    text = Path(path).read_text()
    reasons = []
    if official and not committed:
        reasons.append("not committed at HEAD")
    if t["label"]:
        reasons.append(f"labelled {t['label']!r}")
    if not re.search(rf"^# Decision time, Step {kind.upper()}\b", text, flags=re.M):
        reasons.append(f"not a Step {kind.upper()} timing note")
    params, reviser = recorded_params(text)
    if not _same_params(params, cfg["params"] if kind == "a" else cfg["step_a_params"]):
        reasons.append("params differ from the step config")
    if kind == "b" and reviser != cfg["reviser"]:
        reasons.append("reviser differs from the step config")
    mm = member_mismatch(text, list(cfg["members"]), shas, exact=True)
    if mm:
        reasons.append(mm)
    return {"unavailable": f"{path}: " + "; ".join(reasons)} if reasons else t


def summarise_attempt(raw_attempt: Path) -> str:
    """One line on an earlier attempt from its raw directory: records, ok / failed, completed or not."""
    recs = []
    runs = raw_attempt / "runs.jsonl"
    if runs.is_file():
        recs = [json.loads(x) for x in runs.read_text().splitlines() if x.strip()]
    cfg = {}
    try:
        cfg = json.loads((raw_attempt / "run_config.json").read_text())
    except (OSError, ValueError):
        pass
    expected = cfg.get("n_seeds", 0) * len(cfg.get("arms", {})) or "?"
    failed = sum(bool(r.get("failed")) for r in recs)
    return (f"{len(recs)} of {expected} runs recorded ({len(recs) - failed} ok, {failed} failed), "
            f"{'completed' if 'end_utc' in cfg else 'did not complete'}")


def replace_section(text: str, attempt: int, new: str) -> str:
    """`text` with the section of `attempt` (from its "## Attempt N" heading to the next attempt) replaced."""
    m = re.search(rf"^## Attempt {attempt}\b.*$", text, flags=re.M)
    if not m:
        return text + new
    nxt = re.search(r"^## Attempt \d+\b", text[m.end():], flags=re.M)
    end = m.end() + nxt.start() if nxt else len(text)
    return text[:m.start()] + new + text[end:]


# ------------------------------------------------------------------ worker side
_W: dict = {}


def _init_worker(arms: dict, duration: float, probe_every: int):
    import torch
    torch.set_num_threads(1)
    from pis.runner import ensemble_for
    _W.update(arms=arms, duration=duration, probe_every=probe_every)
    members = {tuple(d["members"]) for d in arms.values() if "members" in d}
    for m in members:
        ensemble_for(list(m))
    if any(d["kind"] == "reviser" for d in arms.values()):
        from pis.reviser import warm_up
        for d in arms.values():
            if d["kind"] == "reviser":
                warm_up(ensemble_for(d["members"]), DEV_SEEDS.start)


def _job(job: tuple[str, int]) -> dict:
    arm, seed = job
    return run_arm(arm, seed, _W["arms"], _W["duration"], _W["probe_every"])


# ------------------------------------------------------------------ driver
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=PROTOCOL)
    ap.add_argument("--step-a-config", default=STEP_CONFIGS["step_a"])
    ap.add_argument("--step-b-config", default=STEP_CONFIGS["step_b"])
    ap.add_argument("--seeds", default=None, help="default: the protocol's seeds")
    ap.add_argument("--duration", type=float, default=None, help="default: the protocol's scenario length")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--label", default="", help='"rehearsal" or "smoke" for development-seed runs')
    ap.add_argument("--priority", choices=("background", "normal"), default="background",
                    help="how this run's processes are scheduled (scripts/gentle.sh = background); labels the "
                         "decision times measured in the run")
    ap.add_argument("--out", default=OFFICIAL_REPORT)
    ap.add_argument("--raw-dir", default=OFFICIAL_RAW)
    a = ap.parse_args(argv)
    try:
        protocol = load_json(a.protocol)
    except (OSError, ValueError) as e:
        raise SystemExit(scrub(f"cannot read the protocol {safe_path(a.protocol)}: {e}"))
    errors = validate_protocol(protocol)
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))
    seeds = parse_seeds(a.seeds) if a.seeds else protocol_seeds(protocol)
    duration = protocol["duration_s"] if a.duration is None else a.duration
    raw_attempt = Path(a.raw_dir) / f"attempt{a.attempt}"
    out = Path(a.out)
    args = {"seeds_list": seeds, "label": a.label, "duration": duration, "out": a.out, "raw_dir": a.raw_dir,
            "attempt": a.attempt, "priority": a.priority, "protocol": a.protocol,
            "step_a_config": a.step_a_config, "step_b_config": a.step_b_config}
    state = {"dirty": bool(_git("status", "--porcelain")[1]),
             "committed": {p: committed_clean(p) for p in (a.protocol, a.step_a_config, a.step_b_config)},
             "report_text": out.read_text() if out.exists() else "", "out_exists": out.exists(),
             "raw_exists": raw_attempt.exists(),
             "prev_raw_exists": (Path(a.raw_dir) / f"attempt{a.attempt - 1}").is_dir(),
             "freeze": freeze_state()}
    errors = check_run(args, protocol, state)
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))
    official = any(s in FINAL_SEEDS for s in seeds)

    steps = {}
    for key, path in (("step_a", a.step_a_config), ("step_b", a.step_b_config)):
        try:
            steps[key] = load_json(path)
        except (OSError, ValueError) as e:
            raise SystemExit(scrub(f"cannot read the {key} config {safe_path(path)}: {e}"))
    sa, sb = steps["step_a"], steps["step_b"]
    if not sa.get("members") or not isinstance(sa.get("params"), dict):
        raise SystemExit("the Step A config needs 'members' and 'params'")
    if not sb.get("members") or not isinstance(sb.get("step_a_params"), dict) or not sb.get("reviser"):
        raise SystemExit("the Step B config needs 'members', 'step_a_params' and 'reviser'")
    if official:
        problems = [f"{k} config has label {v['label']!r}" for k, v in steps.items() if v.get("label")]
        if sb["step_a_params"] != sa["params"] or sb["members"] != sa["members"]:
            problems.append("the Step B config was not tuned on top of this Step A config")
        if problems:
            raise SystemExit("refusing to run: " + "; ".join(problems))

    from pis.config import resolve, setup_imports, sibling_paths
    from pis.evaluator import Params
    setup_imports()
    from sched.controllers import ADAPTIVE_GRID
    ad = protocol["arms"]["adaptive_c9"]
    if {k: ad[k] for k in ("h0", "r_hi", "hold")} != ADAPTIVE_GRID[9]:
        raise SystemExit("adaptive_c9 in the protocol differs from the sibling's adaptive grid entry 9")
    Params.from_dict(sa["params"]), Params.from_dict(sb["step_a_params"])     # validates both
    arms = arm_definitions(protocol, sa, sb)
    members = sorted({m for d in arms.values() for m in d.get("members", [])})
    missing = [m for m in members if not resolve(m).is_file()]
    if missing:
        raise SystemExit("missing ensemble members: " + ", ".join(safe_path(m) for m in missing))
    shas = {m: _sha256(resolve(m)) for m in members}
    models = {f"member {safe_path(m)}": shas[m] for m in members}
    if official:
        problems = []
        for key, exact in (("step_a", False), ("step_b", True)):
            note = TUNING_NOTES[key]
            if not committed_clean(note):
                problems.append(f"{note} is not committed at HEAD")
                continue
            mm = member_mismatch(Path(note).read_text(), list(steps[key]["members"]), shas, exact)
            if mm:
                problems.append(f"{note}: {mm}")
        if problems:
            raise SystemExit("refusing to run: " + "; ".join(problems))
    for name in ("base_r6", "base_r3"):
        d = protocol["arms"][name]
        sha = _sha256(sibling_paths()["sched"] / d["checkpoint"])
        if sha != d["sha256"]:
            raise SystemExit(f"{name} checkpoint differs from the protocol (sha256 {sha[:16]})")
        models[f"{name} sched:{d['checkpoint']}"] = sha

    head = _git("rev-parse", "HEAD")[1] or "unknown"
    probe_every = int(protocol["metrics"].get("probe_every", 0))
    import numpy
    import torch
    config = {"attempt": a.attempt, "label": a.label, "official": official,
              "code": f"git HEAD {head}{' (uncommitted changes)' if state['dirty'] else ''}",
              "protocol": {"path": safe_path(a.protocol), "sha256": _sha256(Path(a.protocol)),
                           "freeze_commit": (state["freeze"]["commits"][-1] if state["freeze"]["commits"] and
                                             Path(a.protocol).resolve() == Path(PROTOCOL).resolve() else None)},
              "step_configs": {k: {"path": safe_path(p), "sha256": _sha256(Path(p)), "code": steps[k].get("code"),
                                   "selected_on_seeds": steps[k].get("selected_on_seeds"),
                                   "label": steps[k].get("label")}
                               for k, p in (("step_a", a.step_a_config), ("step_b", a.step_b_config))},
              "models_sha256": models, "arms": describe_arms(arms), "seeds": f"{seeds[0]}-{seeds[-1]}"
              if seeds == list(range(seeds[0], seeds[-1] + 1)) else ",".join(map(str, seeds)),
              "n_seeds": len(seeds), "duration_s": duration, "probe_every": probe_every, "procs": a.procs,
              "priority": a.priority,
              "versions": {"python": platform.python_version(), "numpy": numpy.__version__,
                           "torch": torch.__version__},
              "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    title = f"Attempt {a.attempt}{' (' + a.label + ')' if a.label else ''}"
    freeze = config["protocol"]["freeze_commit"]
    meta = {"title": title, "code": config["code"],
            "protocol": f"{config['protocol']['path']} (sha256 {config['protocol']['sha256'][:16]}"
                        + (f", frozen in commit {freeze}" if freeze else "") + ")",
            "step configs": "; ".join(f"{k} {v['path']} (sha256 {v['sha256'][:16]}, selected on seeds "
                                      f"{v['selected_on_seeds']}, label {v['label']!r})"
                                      for k, v in config["step_configs"].items()),
            "models": ", ".join(f"{k} ({v[:16]})" for k, v in models.items()),
            "arms": "; ".join(f"{k}: {v}" for k, v in config["arms"].items()),
            "seeds": f"{config['seeds']} ({len(seeds)} scenarios, {duration:g} s each)",
            "run-time probe": f"every {probe_every}th model decision",
            "raw": safe_path(str(raw_attempt)) if not official else f"{OFFICIAL_RAW}/attempt{a.attempt}/"}
    prev = Path(a.raw_dir) / f"attempt{a.attempt - 1}"
    if a.attempt > 1 and prev.is_dir():
        meta["previous attempt"] = f"attempt {a.attempt - 1}: {summarise_attempt(prev)}"
    raw_attempt.mkdir(parents=True, exist_ok=False)          # never overwrite an earlier attempt
    (raw_attempt / "run_config.json").write_text(scrub(json.dumps(config, indent=1)) + "\n")
    # the attempt's section exists from the start: a crash leaves it in the report (and the tree dirty)
    out.parent.mkdir(parents=True, exist_ok=True)
    head_text = "" if state["report_text"] else (
        "# Final paired evaluation\n\nProtocol: configs/final_eval.json, frozen in its own commit before the run. "
        "One section per attempt, written when the attempt starts and completed when it finishes; never edited "
        "afterwards.\n\n")
    stub = (f"## {title} (started)\n\n" + "".join(f"- {k}: {v}\n" for k, v in meta.items() if k != "title")
            + f"- started (UTC): {config['start_utc']}\n\nThis attempt has not finished: it is running, or it "
            "stopped; its finished runs are in the raw records.\n\n")
    with out.open("a") as f:
        f.write(scrub(head_text + stub))
    order = {arm: i for i, arm in enumerate(SLOW_FIRST)}
    jobs = sorted(((arm, s) for arm in arms for s in seeds), key=lambda j: (order[j[0]], j[1]))
    print(f"attempt {a.attempt}{' (' + a.label + ')' if a.label else ''}: {len(jobs)} runs "
          f"({len(arms)} arms x {len(seeds)} scenarios), {a.procs} processes", flush=True)
    t0 = time.time()
    raw: list[dict] = []
    pool = None
    if a.procs > 1:
        from multiprocessing import get_context
        pool = get_context("spawn").Pool(a.procs, initializer=_init_worker, initargs=(arms, duration, probe_every))
        it = pool.imap_unordered(_job, jobs)
    else:
        _init_worker(arms, duration, probe_every)
        it = map(_job, jobs)
    try:
        with (raw_attempt / "runs.jsonl").open("a") as f:
            for i, r in enumerate(it, 1):
                raw.append(r)
                f.write(scrub(json.dumps(r)) + "\n")
                f.flush()
                if i % max(1, len(jobs) // 50) == 0 or i == len(jobs) or r["failed"]:
                    fail = f" ({sum(x['failed'] for x in raw)} failed)" if any(x["failed"] for x in raw) else ""
                    print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f} s{fail}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    runtime = time.time() - t0
    config.update(end_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), runtime_s=round(runtime),
                  runs_ok=sum(not r["failed"] for r in raw), runs_failed=sum(r["failed"] for r in raw))
    (raw_attempt / "run_config.json").write_text(scrub(json.dumps(config, indent=1)) + "\n")

    timing = {arm: timing_for(path, kind, steps[key], shas, official, committed_clean(path))
              for arm, path, kind, key in (("loop_a", TIMING_NOTES["loop_a"], "a", "step_a"),
                                           ("loop_b", TIMING_NOTES["loop_b"], "b", "step_b"))}
    per_arm_s = {arm: [r["seconds"] for r in raw if r["arm"] == arm] for arm in arms}
    meta["runtime"] = (f"{runtime:.0f} s with {a.procs} processes ({a.priority} priority), started "
                       f"{config['start_utc']} UTC; seconds per run by arm (mean): "
                       + ", ".join(f"{k} {sum(v) / len(v):.1f}" for k, v in per_arm_s.items() if v))
    text = report(raw, protocol, meta, timing, seeds, background=a.priority == "background")
    out.write_text(replace_section(out.read_text(), a.attempt, scrub(text + "\n")))
    print(f"done: {config['runs_ok']} ok, {config['runs_failed']} failed, {runtime:.0f} s -> {safe_path(a.out)}")
    return {"raw": raw, "text": text, "config": config}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write-protocol":
        p = Path(PROTOCOL)
        if p.exists():
            raise SystemExit(f"{PROTOCOL} already exists (frozen, never rewritten)")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(make_protocol(), indent=1) + "\n")
        print(f"wrote {PROTOCOL}")
    else:
        main()
