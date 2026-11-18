"""Train a world model with the sibling's recipe (`worldmodel.train.train_model`) and add state bounds.

Same recipe as the sibling's `gru_mix` (GRU, history 8, Adam lr 1e-3, batch 512, patience 20, hidden 128,
seed 0), on the same three data sets plus our counterfactual data. Context records of the counterfactual
data only provide history (their rows are dropped after the history arrays are built). The checkpoint is saved already bounded (as `add_bounds.py`
would: min/max of every encoded state over the train split), with a recipe file next to it.

  .venv/bin/python experiments/train_world_model.py --out models/gru_cf_v1.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pis.config import resolve, setup_imports  # noqa: E402
from pis.counterfactual import history_arrays_without_context, read_records  # noqa: E402
from pis.privacy import scrub  # noqa: E402

setup_imports()

from worldmodel.bounds import state_bounds  # noqa: E402
from worldmodel.features import INPUT_NAMES, STATE_NAMES  # noqa: E402
from worldmodel.model import WorldModel  # noqa: E402
from worldmodel.train import train_model  # noqa: E402

DEFAULT_DATA = ("worldmodel:data/adaptive_v1,old_data:teacher_v2,old_data:dagger_r1,"
                "local:data/counterfactual_v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="comma-separated; prefixes worldmodel:, old_data:, local: (see pis.config.resolve)")
    ap.add_argument("--out", default="models/gru_cf_v1.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["mlp", "gru"], default="gru")
    ap.add_argument("--history", type=int, default=8)
    a = ap.parse_args()
    refs = a.data.split(",")
    dirs = [str(resolve(r)) for r in refs]
    for r, d in zip(refs, dirs):
        if not Path(d).is_dir():
            raise SystemExit(f"data set {r} not found")
    t0 = time.time()
    shaper = WorldModel(None, None, None, {"arch": a.arch, "history": a.history})

    def arrays(split):
        recs = read_records(dirs, split)
        H, S, Y, M, _ = history_arrays_without_context(recs, a.history)
        return (shaper.prepare(H), Y, M), recs

    (tr, tr_recs), (va, _) = arrays("train"), arrays("val")
    print(f"{a.arch} history {a.history}: train {len(tr[0])} / val {len(va[0])} transitions "
          f"({time.time() - t0:.0f}s to load)", flush=True)

    def log(h):
        if h["epoch"] % 10 == 0:
            print(f"epoch {h['epoch']:4d}  train {h['train_loss']:.4f}  val {h['val_loss']:.4f}", flush=True)

    model, hist = train_model(tr, va, epochs=a.epochs, lr=a.lr, batch=a.batch, patience=a.patience,
                              seed=a.seed, hidden=a.hidden, layers=a.layers, log=log,
                              arch=a.arch, history=a.history)
    lo, hi = state_bounds(tr_recs)
    recipe = {**vars(a), "data": refs, "trainer": "worldmodel.train.train_model",
              "context_rows_dropped": True}
    model.meta.update({"args": recipe, "state_names": STATE_NAMES, "input_names": INPUT_NAMES,
                       "n_train": len(tr[0]), "n_val": len(va[0]),
                       "state_min": lo.tolist(), "state_max": hi.tolist(), "bounds_from": refs})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".history.json").write_text(json.dumps(hist))
    out.with_suffix(".recipe.json").write_text(scrub(json.dumps(
        {**recipe, "best_val_loss": model.meta["best_val_loss"], "epochs_run": len(hist),
         "sha256": sha, "seconds": round(time.time() - t0)}, indent=1)))
    print(f"best val loss {model.meta['best_val_loss']:.4f} after {len(hist)} epochs -> {a.out} "
          f"(sha256 {sha})")


if __name__ == "__main__":
    main()
