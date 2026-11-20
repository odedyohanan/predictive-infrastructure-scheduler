"""Train a sampled world model on history episodes and calibrate it on selection episodes.

  .venv/bin/python experiments/b_train_world_model.py --out models/pgru_b1.pt
  .venv/bin/python experiments/b_train_world_model.py --arch rssm        # -> models/rssm_b2.pt
  .venv/bin/python experiments/b_train_world_model.py --label smoke --seeds 3100-3119 \\
      --store data/phase_b/timing_probe --calib-seeds 3119 --horizon 6 --epochs 3 --hidden 16 \\
      --incident data/phase_b/smoke_g3/incident.json \\
      --smoke-dir data/phase_b/smoke_<name> --models-dir models/smoke_<name>

`--arch`: "pgru" the probabilistic GRU (`pis.phase_b.wm_prob`, the default), "rssm" the recurrent
state-space model (`pis.phase_b.wm_rssm`: deterministic GRU state + Gaussian latent, KL with free bits;
`--latent`, `--free-bits`; `--layers` is then the depth of its prior, posterior and decoder networks).
The default checkpoint is models/pgru_b1.pt or models/rssm_b2.pt.

History seeds split by seed % 10: 8 = validation (early stopping), 9 = test (never used here), the rest
train. Calibration (`pis.phase_b.wm_prob.calibration`) runs on the selection seeds with the frozen incident
rule and H from configs/phase_b/incident.json (refused when missing) and the Monte Carlo labels when they
exist (otherwise the Brier fields stay empty and the ladder verdict is undecided).

Real runs need a clean tree and write the checkpoint (`--out`), `<out>.recipe.json`, `<out>.history.json`
(both with the checkpoint's sha256) and results/phase_b/wm_<name>.md. Smoke runs (`--label smoke`) allow a
dirty tree, history validation/test seeds for calibration and a smaller everything, and write only under
`--smoke-dir` (data/phase_b/smoke*) and `--models-dir` (models/smoke*). Long runs go through
scripts/gentle.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pis.phase_b import runs  # noqa: E402
from pis.phase_b import seeds as seed_ranges  # noqa: E402
from pis.privacy import safe_path  # noqa: E402

DEFAULT_STORE = "data/phase_b/history"
DEFAULT_INCIDENT = "configs/phase_b/incident.json"
DEFAULT_MC = "data/phase_b/mc_labels/labels.parquet"
DEFAULT_OUT = {"pgru": "models/pgru_b1.pt", "rssm": "models/rssm_b2.pt"}
ARCH_NAMES = {"pgru": "Probabilistic GRU", "rssm": "Recurrent state-space (RSSM)"}
DEFAULT_LAYERS = {"pgru": 2, "rssm": 1}
CALIB_SEEDS = "3450-3499"
RULE_KEYS = ("k", "interval", "majority", "warmup", "H")


def split(seeds: list[int]) -> tuple[list[int], list[int], list[int]]:
    """(train, validation, test) by seed % 10."""
    return ([s for s in seeds if s % 10 not in (8, 9)], [s for s in seeds if s % 10 == 8],
            [s for s in seeds if s % 10 == 9])


def under(path: str, prefix: str) -> Path:
    """`path` resolved, required to lie under ROOT/<prefix>* (links such as models/ or data/phase_b/ allowed)."""
    p = Path(path)
    p = (ROOT / p if not p.is_absolute() else p).resolve()
    head, _, stem = prefix.rpartition("/")
    base = (ROOT / head).resolve()
    try:
        rel = p.relative_to(base)
    except ValueError:
        raise ValueError(f"{path} must lie under {prefix}*") from None
    if not rel.parts or not rel.parts[0].startswith(stem):
        raise ValueError(f"{path} must lie under {prefix}*")
    return p


def read_incident(path: Path, horizon: int | None) -> dict:
    if not Path(path).exists():
        raise ValueError(f"the incident config {path.name} does not exist yet (fix the incident rule first)")
    cfg = runs.read_json(path)
    need = [k for k in RULE_KEYS if k not in cfg and not (k == "H" and horizon)]
    if need:
        raise ValueError(f"the incident config lacks {need}")
    return {**cfg, "H": int(horizon or cfg["H"])}


check_mc_config = runs.check_mc_config


def arch_text(rec: dict) -> str:
    """The recipe line of the note (and, for the RSSM, the architecture before it)."""
    if rec.get("arch", "pgru") != "rssm":
        return (f"Recipe: context {rec['context']} (+ room {rec['room']}), burn-in {rec['burn_in']}, hidden "
                f"{rec['hidden']} x {rec['layers']} layers, lr {rec['lr']}, batch {rec['batch']}, patience "
                f"{rec['patience']}, seed {rec['seed']}.")
    return (f"Architecture: recurrent state-space model (`pis.phase_b.wm_rssm`): deterministic GRU state of "
            f"{rec['hidden']}, diagonal Gaussian latent of {rec['latent']} with prior p(z | h) and posterior "
            f"q(z | h, change), decoder to a Gaussian over the change of the 43 state numbers (standardised, "
            f"p50/p99 masked without completions); prior, posterior and decoder networks with {rec['layers']} "
            f"hidden layer(s). Loss: masked Gaussian NLL + KL(q || p) floored at {rec['free_bits']} nats (free "
            f"bits), per target number; early stopping on the validation negative ELBO. Rollouts: history "
            f"filtered through the posterior, futures sampled through the prior, states clipped to the training "
            f"range.\n\nRecipe: context {rec['context']} (+ room {rec['room']}), burn-in {rec['burn_in']}, "
            f"hidden {rec['hidden']}, latent {rec['latent']}, layers {rec['layers']}, free bits "
            f"{rec['free_bits']}, lr {rec['lr']}, batch {rec['batch']}, patience {rec['patience']}, seed "
            f"{rec['seed']}.")


def note_text(name: str, rec: dict, hist: list[dict], cal: dict, verdict) -> str:
    from pis.phase_b.wm_prob import NONZERO_OUTPUTS, OUTPUTS
    h = str(cal["horizon"])
    arch = rec.get("arch", "pgru")
    loss = "loss" if arch == "rssm" else "NLL"
    lines = [f"# {ARCH_NAMES[arch]} world model {name}", "",
             f"Checkpoint sha256 `{rec['sha256']}`; {rec['epochs_run']} epochs, best validation {loss} "
             f"{rec['best_val_loss']:.4f}; trained on {len(rec['train_seeds'])} history episodes "
             f"({rec['n_train_steps']} steps), validation {len(rec['val_seeds'])}; {rec['seconds']} s.", "",
             arch_text(rec), "",
             f"| epoch | train {loss} | val {loss} |", "|---|---|---|"]
    step = max(len(hist) // 10, 1)
    lines += [f"| {e['epoch']} | {e['train_loss']:.4f} | {e['val_loss']:.4f} |" for e in hist[::step]]
    lines += ["", f"## Calibration on {len(cal['seeds'])} episodes ({cal['starts']} starts, {cal['n']} samples, "
              f"H = {h})", "",
              f"Coverage of the central {cal['level']:.0%} interval and CRPS (encoded units). Overdue and late "
              "counts are 0 in most windows, where an interval containing 0 covers for free: they are shown on "
              "all windows and on the windows with a non-zero truth (the ladder judges the latter); p99 only on "
              "windows with completions. n = windows judged at H.", "",
              "| output | windows | " + " | ".join(f"cov h={x}" for x in cal["horizons"]) + " | "
              + " | ".join(f"CRPS h={x}" for x in cal["horizons"]) + " | n |",
              "|---|---|" + "---|" * (2 * len(cal["horizons"]) + 1)]
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    for o in OUTPUTS:
        groups = [("all" if o != "a.p99_ms" else "with completions", cal["coverage"], cal["crps"], cal["counts"])]
        if o in NONZERO_OUTPUTS:
            groups.append(("non-zero", cal["coverage_nonzero"], cal["crps_nonzero"], cal["counts_nonzero"]))
        for label, cv, cr, cn in groups:
            lines.append(f"| {o} | {label} | " + " | ".join(fmt(cv[o][str(x)]) for x in cal["horizons"]) + " | "
                         + " | ".join(fmt(cr[o][str(x)]) for x in cal["horizons"]) + f" | {cn[o][h]} |")
    if cal.get("brier") is not None:
        lines += ["", f"Brier of the rollout incident risk vs Monte Carlo labels: {cal['brier']:.4f} "
                  f"(base rate {cal['base_rate']:.3f}: {cal['brier_base']:.4f}); skill {fmt(cal['brier_skill'])} "
                  f"over {cal['n_labels']} labelled states."]
    else:
        lines += ["", "Brier: no Monte Carlo labels available."]
    r = cal.get("rank")
    if r:
        per = ", ".join(f"{k} {fmt(v)}" for k, v in sorted(r["per_intervention"].items()))
        lines += ["", f"Intervention rank agreement (Spearman of predicted vs branched effect on mean "
                  f"a.queue_overdue at h={r['horizon']}, {r['reseeds']} reseeds): {fmt(r['spearman'])} over "
                  f"{r['effects']} effects on {r['states']} states ({per})."]
    lines += ["", "Ladder criterion (coverage at H within [0.70, 0.90] for the three outputs, non-zero windows for "
              "overdue and late, and Brier skill > 0): " + {True: "PASS", False: "FAIL", None: "undecided (no Brier skill)"}[verdict] + "."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="3100-3399")
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--arch", default="pgru", choices=sorted(DEFAULT_OUT))
    ap.add_argument("--out", default=None, help="default: models/pgru_b1.pt or models/rssm_b2.pt by --arch")
    ap.add_argument("--incident", default=DEFAULT_INCIDENT)
    ap.add_argument("--mc-labels", default=DEFAULT_MC)
    ap.add_argument("--calib-seeds", default=CALIB_SEEDS)
    ap.add_argument("--label", default="")
    ap.add_argument("--smoke-dir", default=None)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--horizon", type=int, default=None, help="smoke only (real runs use H)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=None, help="default: 2 (pgru), 1 (rssm)")
    ap.add_argument("--latent", type=int, default=16, help="rssm only")
    ap.add_argument("--free-bits", type=float, default=1.0, help="rssm only (nats)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--context", type=int, default=32)
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--room", type=int, default=32, help="extra sequence length for rollouts (steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--rank-states", type=int, default=50)
    ap.add_argument("--rank-reseeds", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=None, help="smoke only: stop training after this")
    a = ap.parse_args(argv)
    if a.label not in runs.LABELS:
        raise SystemExit(f"refusing to run: unknown label {a.label!r}")
    smoke = a.label == "smoke"
    a.out = a.out or DEFAULT_OUT[a.arch]
    a.layers = DEFAULT_LAYERS[a.arch] if a.layers is None else a.layers
    try:
        seeds = seed_ranges.parse(a.seeds)
        seed_ranges.check(seeds, "HISTORY")
        calib = seed_ranges.parse(a.calib_seeds)
        if smoke:
            runs.check_seeds(calib, ("SELECTION", "HISTORY"))
            bad = [s for s in calib if s in seed_ranges.HISTORY and s % 10 != 9]
            if bad:
                raise ValueError(f"smoke calibration on history seeds uses test seeds only (seed % 10 == 9): {bad}")
            sdir = runs.smoke_dir(a.smoke_dir)
            mdir = under(a.models_dir or "", "models/smoke")
            out = mdir / Path(a.out).name
        else:
            seed_ranges.check(calib, "SELECTION")
            if a.smoke_dir or a.models_dir or a.horizon or a.max_seconds:
                raise ValueError("--smoke-dir, --models-dir, --horizon and --max-seconds need --label smoke")
            out = under(a.out, "models/")
            if out.parent.name.startswith("smoke"):
                raise ValueError("real runs do not write into smoke model folders")
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    train_s, val_s, _ = split(seeds)
    if not smoke and runs.tree_dirty():
        raise SystemExit("refusing to run: the working tree has uncommitted changes (commit, or --label smoke)")
    try:
        inc = read_incident(ROOT / a.incident if not Path(a.incident).is_absolute() else Path(a.incident),
                            a.horizon if smoke else None)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    mc_path = ROOT / a.mc_labels if not Path(a.mc_labels).is_absolute() else Path(a.mc_labels)
    if mc_path.exists():
        try:
            check_mc_config(mc_path, inc)
        except ValueError as e:
            raise SystemExit(f"refusing to run: {e}")
    if not train_s or not val_s:
        raise SystemExit("refusing to run: need training and validation (seed % 10 == 8) history seeds")

    import pandas as pd
    from pis.phase_b import incidents, wm_prob, world_models
    from pis.phase_b.history import Store
    store = Store(ROOT / a.store if not Path(a.store).is_absolute() else a.store)
    t0 = time.time()

    def log(h):
        if isinstance(h, str):
            print(h, flush=True)
        elif h["epoch"] % 5 == 0:
            print(f"epoch {h['epoch']:4d}  train {h['train_loss']:.4f}  val {h['val_loss']:.4f}  "
                  f"{h['seconds']:.0f}s", flush=True)

    recipe = dict(context=a.context, burn_in=a.burn_in, room=a.room, hidden=a.hidden, layers=a.layers, lr=a.lr,
                  batch=a.batch, epochs=a.epochs, patience=a.patience, seed=a.seed)
    if a.arch == "rssm":
        recipe.update(latent=a.latent, free_bits=a.free_bits)
    fit = world_models.trainer(a.arch)
    try:
        model, hist = fit(store, train_s, val_s, log=log, max_seconds=a.max_seconds, **recipe)
    except ValueError as e:
        raise SystemExit(f"refusing to run: {e}")
    train_seconds = round(time.time() - t0)
    model.meta.update({"arch": a.arch, "recipe": recipe, "store": safe_path(str(a.store)), "label": a.label})
    out.parent.mkdir(parents=True, exist_ok=True)
    sha = model.save(out)
    rec = {**recipe, "train_seeds": train_s, "val_seeds": val_s, "n_train_steps": model.meta["n_train_steps"],
           "best_val_loss": model.meta["best_val_loss"], "epochs_run": len(hist), "sha256": sha,
           "seconds": train_seconds, "store": safe_path(str(a.store)), "label": a.label,
           "out": out.name, "arch": a.arch, "trainer": f"{fit.__module__}.fit"}
    runs.write_json(out.with_suffix(".recipe.json"), rec)
    runs.write_json(out.with_suffix(".history.json"), {"sha256": sha, "epochs": hist})
    print(f"trained {a.arch} {len(hist)} epochs in {train_seconds} s, best val loss {rec['best_val_loss']:.4f}",
          flush=True)

    rule = incidents.IncidentRule(int(inc["k"]), int(inc["interval"]), float(inc["majority"]))
    mc = pd.read_parquet(mc_path) if mc_path.exists() else None
    cal = wm_prob.calibration(model, store, calib, inc["H"], rule=rule, mc=mc, n=a.samples, every=a.every,
                              warmup=int(inc["warmup"]), rank_states=a.rank_states, rank_reseeds=a.rank_reseeds,
                              seed=a.seed, log=log)
    verdict = wm_prob.ladder_pass(cal)
    name = out.stem
    note = (sdir / f"wm_{name}.md") if smoke else ROOT / f"results/phase_b/wm_{name}.md"
    raw = (sdir / f"wm_{name}.json") if smoke else ROOT / f"results/raw/phase_b/wm_{name}.json"
    runs.write_note(note, note_text(name, rec, hist, cal, verdict))
    runs.write_json(raw, {k: v for k, v in cal.items() if k != "rank"}
                    | {"rank": None if cal["rank"] is None else
                       {k: v for k, v in cal["rank"].items() if k != "rows"}, "ladder_pass": verdict})
    judged = {o: (cal["coverage_nonzero"] if o in wm_prob.NONZERO_OUTPUTS else cal["coverage"])[o][str(inc["H"])]
              for o in wm_prob.OUTPUTS}
    print(f"calibration: coverage at H (non-zero for counts) {json.dumps(judged)}"
          f"; brier skill {cal['brier_skill']}; rank {cal['rank'] and cal['rank']['spearman']}; "
          f"ladder {verdict}", flush=True)
    return {"recipe": rec, "calibration": cal, "ladder_pass": verdict}


if __name__ == "__main__":
    main()
