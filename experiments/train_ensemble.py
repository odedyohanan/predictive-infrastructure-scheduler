"""Train extra ensemble members with the recipe of the gate-passing world model, one seed each.

Member 0 is the gate-passing model itself (`worldmodel:models/gru_mix_bounded.pt`, read-only). Its recipe,
taken from its checkpoint meta and the sibling repo's `experiments/train.py` + `experiments/add_bounds.py`:
GRU, history 8, hidden 128, Adam lr 1e-3, batch 512, up to 300 epochs, early stopping patience 20, trained
by `worldmodel.train.train_model` on the train split (val split for early stopping) of
`adaptive_v1` + `teacher_v2` + `dagger_r1`, transitions built by `worldmodel.history.build_history_arrays`,
then state bounds = min/max of every encoded state over the train split (`worldmodel.bounds.state_bounds`).
Member n (n >= 1) uses the same recipe with seed n (member 0 used seed 0) and is saved as
`<out-dir>/ens_<n>.pt`, with `ens_<n>.recipe.json` and `ens_<n>.history.json` next to it.

  .venv/bin/python experiments/train_ensemble.py --members 1-4 --procs 2

A run refuses to start when pis/ or experiments/ have uncommitted changes or a target file exists (unless
--force); `--label smoke` lifts the dirty-tree check but must not write into models/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pis.config import resolve, setup_imports  # noqa: E402
from pis.privacy import safe_path, scrub  # noqa: E402

setup_imports()

DATA = "worldmodel:data/adaptive_v1,old_data:teacher_v2,old_data:dagger_r1"
RECIPE = {"arch": "gru", "history": 8, "hidden": 128, "layers": 2, "lr": 1e-3, "batch": 512,
          "epochs": 300, "patience": 20}
OFFICIAL_DIR = "models"


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def parse_members(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    if not out or min(out) < 1:
        raise ValueError("members are numbered from 1 (member 0 is the gate-passing model)")
    return sorted(set(out))


def check_run(args: dict, dirty: bool, existing: list[str]) -> list[str]:
    errors = []
    smoke = args.get("label") == "smoke"
    if dirty and not smoke:
        errors.append("pis/ or experiments/ have uncommitted changes")
    if smoke and Path(args["out_dir"]).resolve() == Path(OFFICIAL_DIR).resolve():
        errors.append("a smoke run must not write into models/")
    if existing and not args.get("force"):
        errors.append("target exists: " + ", ".join(existing))
    return errors


def _arrays(dirs: list[str], split: str, history: int):
    from worldmodel.data import load_records
    from worldmodel.history import build_history_arrays
    from worldmodel.model import WorldModel
    recs = load_records(dirs, split)
    H, _, Y, M, _ = build_history_arrays(recs, history)
    shaper = WorldModel(None, None, None, {"arch": RECIPE["arch"], "history": history})
    return (shaper.prepare(H), Y, M), recs


def train_member(job: dict) -> dict:
    """Train one member; returns its recipe record (no absolute paths)."""
    import torch
    from worldmodel.bounds import state_bounds
    from worldmodel.features import INPUT_NAMES, STATE_NAMES
    from worldmodel.train import train_model
    torch.set_num_threads(1)
    n, seed, r = job["member"], job["seed"], job["recipe"]
    t0 = time.time()
    dirs = [str(resolve(x)) for x in job["data"]]
    tr, tr_recs = _arrays(dirs, "train", r["history"])
    va, _ = _arrays(dirs, "val", r["history"])

    def log(h):
        if h["epoch"] % 10 == 0:
            print(f"[member {n}] epoch {h['epoch']:4d}  train {h['train_loss']:.4f}  val {h['val_loss']:.4f}",
                  flush=True)

    model, hist = train_model(tr, va, epochs=r["epochs"], lr=r["lr"], batch=r["batch"],
                              patience=r["patience"], seed=seed, hidden=r["hidden"], layers=r["layers"],
                              log=log, arch=r["arch"], history=r["history"])
    lo, hi = state_bounds(tr_recs)
    out = Path(job["out_dir"]) / f"ens_{n}.pt"
    rec = {**r, "seed": seed, "member": n, "data": list(job["data"]), "out": safe_path(str(out)),
           "trainer": "worldmodel.train.train_model", "bounds": "worldmodel.bounds.state_bounds(train split)",
           "code": job["code"], "label": job["label"]}
    model.meta.update({"args": rec, "state_names": STATE_NAMES, "input_names": INPUT_NAMES,
                       "n_train": len(tr[0]), "n_val": len(va[0]), "state_min": lo.tolist(),
                       "state_max": hi.tolist(), "bounds_from": list(job["data"])})
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    rec.update(best_val_loss=model.meta["best_val_loss"], epochs_run=len(hist), n_train=len(tr[0]),
               n_val=len(va[0]), sha256=sha, seconds=round(time.time() - t0))
    out.with_suffix(".history.json").write_text(json.dumps(hist))
    out.with_suffix(".recipe.json").write_text(scrub(json.dumps(rec, indent=1)))
    print(f"[member {n}] best val loss {rec['best_val_loss']:.4f} after {len(hist)} epochs, "
          f"{rec['seconds']} s -> {rec['out']} (sha256 {sha})", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="1-4", help="e.g. 1-4 or 1,3 (member n trains with seed n)")
    ap.add_argument("--out-dir", default=OFFICIAL_DIR)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--epochs", type=int, default=RECIPE["epochs"], help="smoke only; the recipe is 300")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--label", default="", help='"smoke" for test runs')
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    members = parse_members(a.members)
    existing = [safe_path(str(Path(a.out_dir) / f"ens_{n}.pt")) for n in members
                if (Path(a.out_dir) / f"ens_{n}.pt").exists()]
    dirty = bool(_git("status", "--porcelain", "--", "pis", "experiments"))
    errors = check_run(vars(a), dirty, existing)
    if errors:
        raise SystemExit("refusing to run: " + "; ".join(errors))
    if a.epochs != RECIPE["epochs"] and a.label != "smoke":
        raise SystemExit("refusing to run: --epochs differs from the recipe outside a smoke run")
    data = a.data.split(",")
    for r in data:
        if not resolve(r).is_dir():
            raise SystemExit(f"data set {r} not found")
    code = f"git HEAD {_git('rev-parse', 'HEAD') or 'unknown'}{' (uncommitted changes)' if dirty else ''}"
    jobs = [{"member": n, "seed": n, "data": data, "out_dir": a.out_dir, "code": code, "label": a.label,
             "recipe": {**RECIPE, "epochs": a.epochs}} for n in members]
    t0 = time.time()
    if a.procs <= 1 or len(jobs) == 1:
        recs = [train_member(j) for j in jobs]
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(a.procs, len(jobs))) as pool:
            recs = pool.map(train_member, jobs)
    print(f"trained {len(recs)} members in {time.time() - t0:.0f} s")
    return recs


if __name__ == "__main__":
    main()
