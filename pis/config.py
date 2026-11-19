"""Locations of the sibling repositories and import setup."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pis.privacy import scrub

_DEFAULTS = {
    "sched": ("SCHED_REPO", "../priority-scheduler"),
    "worldmodel": ("WORLDMODEL_REPO", "../scheduler-simulation-model"),
    "old_data": ("OLD_DATA", "../priority_scheduler_model/data"),
}


def sibling_paths() -> dict[str, Path]:
    """Paths of the sibling repos; environment first, relative default second."""
    return {k: Path(os.environ.get(env, default)) for k, (env, default) in _DEFAULTS.items()}


def setup_imports() -> None:
    """Put the two sibling repo roots on sys.path (idempotent)."""
    paths = sibling_paths()
    for key in ("sched", "worldmodel"):
        root = paths[key]
        if not root.is_dir():
            raise FileNotFoundError(
                f"sibling repo '{key}' not found at '{scrub(str(root))}' (set {_DEFAULTS[key][0]})"
            )
        resolved = str(root.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def resolve(ref: str) -> Path:
    """'worldmodel:x', 'sched:x', 'old_data:x' -> path inside that sibling; 'local:x' or 'x' -> x here."""
    key, sep, rest = ref.partition(":")
    if sep and key in _DEFAULTS:
        return sibling_paths()[key] / rest
    return Path(rest if sep and key == "local" else ref)
