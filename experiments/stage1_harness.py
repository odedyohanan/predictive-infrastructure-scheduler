"""Stage 1: R6 greedy runs on dev seeds; check the world model's forecasts against the simulator."""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pis.adapters import make_spec  # noqa: E402
from pis.config import sibling_paths  # noqa: E402
from pis.forecaster import DEFAULT_MODEL, load_world_model  # noqa: E402
from pis.harness import (BASE_CKPT, BaseGreedyController, forecast_vs_fact,  # noqa: E402
                         merge_results, render_report, run_scenario, write_report)

DEV_SEEDS = range(1100, 1110)
OUT = Path("results/stage1/report.md")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = sibling_paths()
    model = load_world_model()
    per_seed, decide_ms = [], []
    t0 = time.perf_counter()
    for seed in DEV_SEEDS:
        spec = make_spec(seed)
        records = run_scenario(seed, BaseGreedyController(spec))
        decide_ms += [r["info"]["decide_ms"] for r in records]
        per_seed.append(forecast_vs_fact(records, model, spec))
        print(f"seed {seed}: {len(records)} windows")
    meta = {
        "dev seeds": f"{DEV_SEEDS.start}-{DEV_SEEDS.stop - 1}",
        "scenario": "60 s, base scheduler R6 greedy",
        "base checkpoint": f"sched:{BASE_CKPT} (sha256 {sha256(paths['sched'] / BASE_CKPT)[:16]})",
        "world model": f"worldmodel:{DEFAULT_MODEL} (sha256 {sha256(paths['worldmodel'] / DEFAULT_MODEL)[:16]})",
        "R6 decide time": f"{np.mean(decide_ms):.2f} ms mean",
    }
    write_report(render_report(merge_results(per_seed), meta), OUT)
    print(f"wrote {OUT} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
