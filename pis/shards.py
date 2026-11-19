"""Resumable sharded runs: each shard writes its own directory and a `done` marker last.

Rerunning skips finished shards and redoes the rest from scratch; a changed configuration is
refused instead of being mixed with shards produced under the old one.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pis.privacy import scrub

log = logging.getLogger(__name__)
DONE = "done"
CONFIG_FILE = "config.json"
ERROR_FILE = "error.txt"


@dataclass
class Shard:
    id: str
    items: list


def make_shards(items, size: int) -> list[Shard]:
    if size < 1:
        raise ValueError("shard size must be >= 1")
    items = list(items)
    return [Shard(f"s{i // size:04d}", items[i:i + size]) for i in range(0, len(items), size)]


def _check_config(out_dir: Path, config: dict) -> None:
    path = out_dir / CONFIG_FILE
    normal = json.loads(json.dumps(config))
    if path.exists():
        old = json.loads(path.read_text())
        if old != normal:
            raise ValueError(f"{scrub(str(out_dir))} was produced with a different configuration; "
                             "use a new output directory")
    else:
        path.write_text(json.dumps(normal, indent=2, sort_keys=True) + "\n")


def _run_one(job: tuple[Shard, Path, Callable[[Shard, Path], None]]) -> tuple[str, bool, str]:
    shard, out_dir, fn = job
    d = out_dir / shard.id
    if d.exists():                      # leftovers of an unfinished attempt
        shutil.rmtree(d)
    d.mkdir(parents=True)
    try:
        fn(shard, d)
    except Exception:
        msg = scrub(traceback.format_exc())
        (d / ERROR_FILE).write_text(msg)
        return shard.id, False, msg
    (d / DONE).write_text("")
    return shard.id, True, ""


def run_shards(out_dir: Path, shards: list[Shard], fn: Callable[[Shard, Path], None], config: dict,
               procs: int) -> dict:
    """Run `fn(shard, out_dir/<id>/)` for every unfinished shard.

    `fn` must be a top-level function (workers are spawned). Returns {"done", "skipped", "failed"}
    as lists of shard ids. A failing shard is logged and left without `done`.
    """
    out_dir = Path(out_dir)
    ids = [s.id for s in shards]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate shard ids")
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_config(out_dir, config)
    skipped = [s.id for s in shards if (out_dir / s.id / DONE).exists()]
    todo = [(s, out_dir, fn) for s in shards if s.id not in skipped]
    if procs > 1 and len(todo) > 1:
        with mp.get_context("spawn").Pool(min(procs, len(todo))) as pool:
            results = list(pool.imap_unordered(_run_one, todo))
    else:
        results = [_run_one(j) for j in todo]
    done, failed = [], []
    for sid, ok, msg in results:
        if ok:
            done.append(sid)
        else:
            failed.append(sid)
            log.error("shard %s failed:\n%s", sid, msg)
    return {"done": done, "skipped": skipped, "failed": failed}


def collect(out_dir: Path) -> list[Path]:
    """Finished shard directories, sorted by id."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    return sorted(p for p in out_dir.iterdir() if p.is_dir() and (p / DONE).exists())
