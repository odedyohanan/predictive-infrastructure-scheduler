"""The episode history: Parquet files written per episode, read through DuckDB views.

Files. `write_episode` writes two files into a shard directory: `windows-<seed>.parquet` (one row per
500 ms window) and `episodes-<seed>.parquet` (one row per episode). `Store(root)` reads every finished
shard under `root` (those with a `done` marker, see `pis.shards`) and never writes.

Window columns (dots in source names become underscores; the mappings below are the reference):
  episode, window, t        episode id (= seed), window index from 0, simulated time at window end (s)
  STATE_COLUMNS             the 39 world-model state fields of the window's telemetry, in the model's
                            ENCODED space (`worldmodel.features.encode_state`: log1p on a.p50_ms,
                            a.p99_ms, a.queue_len, a.queue_work_ms; everything else in its own units)
  RAW_COLUMNS               raw counts of the window: a_queue_overdue (queued A requests older than the
                            SLO at window end), a_late_new, a_completed_ontime, a_arrivals
  POLICY_COLUMNS            the policy that actually ran during this window (after masking dead CPUs and
                            GPUs), encoded by `worldmodel.features.encode_policy`; this row's state is
                            its outcome
  value                     useful work done in the window: B batches x value per batch + C core-ms x
                            value per core-ms (per window, not per second)
  slo_v2_violation          `sim.slo.window_violation(tel, slo, "v2")`, as `incidents.violations`
  deg_kind, deg_active,     the degradation's kind, whether it was active at the window's start, its
  deg_factor                share of full strength
  alive_cpus                number of CPUs alive in the window
  source                    the policy source that drove the episode

`slo_v2_violation` and `deg_active` feed `incidents.label_series(viol, active, rule)` directly.

Episode columns include KNOWN_COLUMNS (`known_<name>`): the parameters known in advance, at their
start-of-episode values. World-model training pairs (`Store.training_sequences`) take state_t, the policy
of row t+1 (decided after seeing window t) and these known values as input, and state_{t+1} as target.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pis.config import setup_imports, sibling_paths
from pis.phase_b import incidents
from pis.phase_b import seeds as seed_ranges
from pis.shards import collect

setup_imports()

from sched.randomize import known_params  # noqa: E402
from worldmodel.features import KNOWN_NAMES, POLICY_NAMES, STATE_NAMES, encode_policy, encode_state  # noqa: E402

GENERATOR_VERSION = 2
ROOT = Path(__file__).resolve().parents[2]


def _sql_name(name: str) -> str:
    return name.replace(".", "_")


STATE_COLUMNS = {n: _sql_name(n) for n in STATE_NAMES}
RAW_COUNTS = ("queue_overdue", "late_new", "completed_ontime", "arrivals")
RAW_COLUMNS = {f"a.{k}": f"a_{k}" for k in RAW_COUNTS}
POLICY_COLUMNS = {n: f"policy_{n}" for n in POLICY_NAMES}
WINDOW_COLUMNS = (["episode", "window", "t"] + list(STATE_COLUMNS.values()) + list(RAW_COLUMNS.values())
                  + list(POLICY_COLUMNS.values())
                  + ["value", "slo_v2_violation", "deg_kind", "deg_active", "deg_factor", "alive_cpus", "source"])
# episode columns: the parameters every controller and model knows in advance (start-of-episode values)
KNOWN_COLUMNS = {n: f"known_{n}" for n in KNOWN_NAMES}
# world-model inputs per step (state, next policy, known; the order of worldmodel.features.INPUT_NAMES)
INPUT_COLUMNS = list(STATE_COLUMNS.values()) + list(POLICY_COLUMNS.values()) + list(KNOWN_COLUMNS.values())
TARGET_COLUMNS = list(STATE_COLUMNS.values())


# ------------------------------------------------------------------ writing
def _git_head(path: Path) -> str:
    """HEAD of a repo, with "+dirty" when a tracked file differs from it (untracked files do not count)."""
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


@lru_cache(maxsize=1)
def commits() -> str:
    """Commits of this repo and the sibling repos the episode ran on, as JSON."""
    sib = sibling_paths()
    return json.dumps({"pis": _git_head(ROOT), "sched": _git_head(sib["sched"]),
                       "worldmodel": _git_head(sib["worldmodel"])}, sort_keys=True)


def window_frame(scenario, records: list[dict], source: str) -> pd.DataFrame:
    spec = scenario.spec
    tels = [r["tel"] for r in records]
    states = np.stack([encode_state(t) for t in tels]).astype(np.float32)
    pols = np.stack([encode_policy(r["applied"]) for r in records]).astype(np.float32)
    cols: dict[str, object] = {
        "episode": np.full(len(records), scenario.seed, dtype=np.int64),
        "window": np.arange(len(records), dtype=np.int64),
        "t": np.array([t["t"] for t in tels], dtype=np.float64),
    }
    for i, c in enumerate(STATE_COLUMNS.values()):
        cols[c] = states[:, i]
    for k in RAW_COUNTS:
        cols[f"a_{k}"] = np.array([t["a"][k] for t in tels], dtype=np.int64)
    for i, c in enumerate(POLICY_COLUMNS.values()):
        cols[c] = pols[:, i]
    cols["value"] = np.array([spec.b.value_per_batch * t["b"]["batches"]
                              + spec.c.value_per_core_ms * t["c"]["work_core_ms"] for t in tels], dtype=np.float64)
    cols["slo_v2_violation"] = incidents.violations(records, spec.a.slo_p99_ms)
    cols["deg_kind"] = [t["phase_b"]["kind"] for t in tels]
    cols["deg_active"] = np.array([t["phase_b"]["active"] for t in tels], dtype=bool)
    cols["deg_factor"] = np.array([t["phase_b"]["factor"] for t in tels], dtype=np.float64)
    cols["alive_cpus"] = np.array([len(t["phase_b"]["alive_cpus"]) for t in tels], dtype=np.int64)
    cols["source"] = [source] * len(records)
    return pd.DataFrame(cols, columns=WINDOW_COLUMNS)


def write_episode(out_dir, scenario, records: list[dict], source: str, wall_s: float | None = None) -> None:
    """Write one episode's window rows and its episode row into `out_dir`."""
    if not records:
        raise ValueError("an episode without windows")
    out_dir = Path(out_dir)
    d = scenario.degradation
    window_frame(scenario, records, source).to_parquet(out_dir / f"windows-{scenario.seed}.parquet", index=False)
    ep = {"seed": scenario.seed, "kind": d.kind,
          "onset_s": d.onset_s if d.kind != "none" else None,
          "ramp_s": d.ramp_s if d.kind != "none" else None,
          "magnitude": d.magnitude if d.kind != "none" else None,
          "source": source, "duration_s": float(scenario.duration_s), "windows": len(records),
          "spec": json.dumps(dataclasses.asdict(scenario.spec), sort_keys=True),
          **{c: float(known_params(scenario.spec)[n]) for n, c in KNOWN_COLUMNS.items()},
          "generator_version": GENERATOR_VERSION, "commits": commits(),
          "wall_s": float(wall_s) if wall_s is not None else None}
    pd.DataFrame([ep]).astype({"onset_s": "float64", "ramp_s": "float64", "magnitude": "float64",
                               "wall_s": "float64"}).to_parquet(
        out_dir / f"episodes-{scenario.seed}.parquet", index=False)


# ------------------------------------------------------------------ reading
class Store:
    """Read-only DuckDB views `windows` and `episodes` over every finished shard under `root`.

    A store holding any held-out seed is refused when it is opened, so no query (training, inference or
    analysis) can ever return held-out rows. Training queries accept history seeds only and every requested
    seed must have a finished episode.
    """

    def __init__(self, root):
        import duckdb
        dirs = collect(Path(root))
        wins = sorted(str(p) for d in dirs for p in d.glob("windows-*.parquet"))
        eps = sorted(str(p) for d in dirs for p in d.glob("episodes-*.parquet"))
        if not wins or not eps:
            raise ValueError("no finished shards with episodes under the store root")
        self._db = duckdb.connect(":memory:")
        self._db.execute(f"CREATE VIEW windows AS SELECT * FROM read_parquet({_list(wins)})")
        self._db.execute(f"CREATE VIEW episodes AS SELECT * FROM read_parquet({_list(eps)})")
        self._columns = [r[0] for r in self._db.execute("DESCRIBE windows").fetchall()]
        self._seeds = {int(r[0]) for r in self._db.execute("SELECT seed FROM episodes").fetchall()}
        held = sorted(s for s in self._seeds if s in seed_ranges.HELD_OUT)
        if held:
            raise ValueError(f"the store holds held-out seeds {held[:5]}; held-out episodes are never stored")

    def _df(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self._db.execute(sql, params or []).df()

    def _cols(self, columns) -> str:
        bad = [c for c in columns if c not in self._columns]
        if bad:
            raise ValueError(f"unknown window columns {bad}")
        return ", ".join(f'"{c}"' for c in columns)

    # -------------------------------------------------- training (history seeds only)
    def training_windows(self, seeds) -> pd.DataFrame:
        """Training: every window row of the given history seeds, by episode and window."""
        seeds = self._training_seeds(seeds)
        return self._df('SELECT * FROM windows WHERE episode IN (SELECT unnest(?)) ORDER BY episode, "window"',
                        [seeds])

    def training_sequences(self, seeds, context: int,
                           horizon: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Training: world-model-ready sliding sequences inside each episode of the given history seeds.

        For a sequence ending at window t:
          X [N, context, 75]  step j (window w = t-context+1+j) = state of w + the policy applied in window
                              w+1 (row w+1's policy_* columns: the decision taken after seeing w) + the known
                              parameters of the episode; columns INPUT_COLUMNS, the order of
                              `worldmodel.features.encode_input`
          Y [N, horizon, 39]  states of windows t+1..t+horizon (TARGET_COLUMNS: state only, no policy)
        So X[i, -1] holds state_t and the policy that produced state_{t+1} = Y[i, 0]. The policies of
        later horizon steps are row t+2.. policy_* columns (`window_rows`). The index frame holds (episode,
        window = t) per sequence. Sequences never cross episodes.
        """
        if context < 1 or horizon < 1:
            raise ValueError("context and horizon must be >= 1")
        seeds = self._training_seeds(seeds)
        states, pols = list(STATE_COLUMNS.values()), list(POLICY_COLUMNS.values())
        df = self._df(f'SELECT episode, "window", {self._cols(states + pols)} FROM windows '
                      'WHERE episode IN (SELECT unnest(?)) ORDER BY episode, "window"', [seeds])
        known = self._df(f"SELECT seed, {', '.join(KNOWN_COLUMNS.values())} FROM episodes "
                         "WHERE seed IN (SELECT unnest(?))", [seeds]).set_index("seed")
        xs, ys, idx = [], [], []
        for ep, part in df.groupby("episode", sort=True):
            s = part[states].to_numpy(dtype=np.float32)
            p = part[pols].to_numpy(dtype=np.float32)
            k = known.loc[ep].to_numpy(dtype=np.float32)
            n = len(s) - context - horizon + 1
            if n <= 0:
                continue
            steps = np.concatenate([s[:-1], p[1:], np.broadcast_to(k, (len(s) - 1, len(k)))], axis=1)
            for i in range(n):
                t = i + context - 1
                xs.append(steps[i:t + 1])
                ys.append(s[t + 1:t + 1 + horizon])
                idx.append((int(ep), int(part.window.iloc[t])))
        x = np.stack(xs) if xs else np.zeros((0, context, len(INPUT_COLUMNS)), np.float32)
        y = np.stack(ys) if ys else np.zeros((0, horizon, len(TARGET_COLUMNS)), np.float32)
        return x, y, pd.DataFrame(idx, columns=["episode", "window"])

    def _training_seeds(self, seeds) -> list[int]:
        """Calibration, selection and held-out seeds never feed training; every seed must be stored."""
        seeds = [int(s) for s in seeds]
        seed_ranges.check(seeds, "HISTORY")
        missing = sorted(set(seeds) - self._seeds)
        if missing:
            raise ValueError(f"no finished episode for history seeds {missing[:10]}"
                             f"{' and more' if len(missing) > 10 else ''}")
        return seeds

    # -------------------------------------------------- inference
    def recent_windows(self, episode: int, upto: int, n: int) -> pd.DataFrame:
        """Inference: the last `n` window rows of an episode ending at window `upto` (never after it)."""
        return self._df('SELECT * FROM windows WHERE episode = ? AND "window" <= ? AND "window" > ? '
                        'ORDER BY "window"',
                        [int(episode), int(upto), int(upto) - int(n)])

    # -------------------------------------------------- analysis
    def episodes(self, where: str | None = None) -> pd.DataFrame:
        """Analysis: episode rows ordered by seed, optionally filtered by an SQL condition."""
        cond = f" WHERE {where}" if where else ""
        return self._df(f"SELECT * FROM episodes{cond} ORDER BY seed")

    def window_rows(self, episode: int, start: int, end: int) -> pd.DataFrame:
        """Analysis: an episode's window rows with start <= window < end."""
        return self._df('SELECT * FROM windows WHERE episode = ? AND "window" >= ? AND "window" < ? '
                        'ORDER BY "window"',
                        [int(episode), int(start), int(end)])

    def state_matrix(self, seeds, columns: list[str]) -> np.ndarray:
        """Analysis: the given window columns of the given episodes as a matrix [rows, columns]."""
        sql = (f"SELECT {self._cols(columns)} FROM windows WHERE episode IN (SELECT unnest(?)) "
               'ORDER BY episode, "window"')
        return self._df(sql, [[int(s) for s in seeds]]).to_numpy(dtype=np.float64)


def _list(paths: list[str]) -> str:
    return "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"
