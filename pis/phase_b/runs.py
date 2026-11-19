"""Guards shared by the incident-advisor experiment scripts: clean tree, seed ranges, smoke output places."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from pis.phase_b import seeds as seed_ranges
from pis.privacy import scrub

ROOT = Path(__file__).resolve().parents[2]
SMOKE_PREFIX = "data/phase_b/smoke"     # smoke outputs live in directories starting with this
LABELS = ("", "smoke")


def tree_dirty(ignore=()) -> bool:
    """Uncommitted changes (untracked files included), except the paths in `ignore` (repo-relative)."""
    p = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return True
    skip = {str(Path(i)) for i in ignore}
    return any(line[3:].strip().strip('"') not in skip for line in p.stdout.splitlines() if line.strip())


def git_head() -> str:
    """The commit of the working tree (git rev-parse HEAD), "unknown" outside a repository."""
    p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() or "unknown"


def check_seeds(seeds, allowed: tuple[str, ...]) -> None:
    """Every seed must lie in one of the named ranges (and none may be refused)."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("no seeds given")
    for s in seeds:
        for name in allowed:
            if s in seed_ranges.RANGES[name]:
                seed_ranges.check([s], name)
                break
        else:
            seed_ranges.check([s], allowed[0])      # raises with the reason


def smoke_dir(path: str | None) -> Path:
    """The directory a smoke run writes everything into: required, and under data/phase_b/smoke*."""
    if not path:
        raise ValueError("a smoke run needs --smoke-dir (a directory under data/phase_b/smoke*)")
    p = Path(path).resolve()
    try:
        rel = p.relative_to(ROOT.resolve())
    except ValueError:
        rel = None
    if rel is None:
        # data/phase_b may be a link: compare against its resolved form too
        base = (ROOT / "data" / "phase_b").resolve()
        try:
            rel = Path("data/phase_b") / p.relative_to(base)
        except ValueError:
            raise ValueError("--smoke-dir must lie under data/phase_b/smoke*") from None
    if not str(rel).startswith(SMOKE_PREFIX):
        raise ValueError("--smoke-dir must lie under data/phase_b/smoke*")
    return p


def load_script(name: str):
    """Import one of the experiment scripts by file path (the sibling repos have an `experiments` package
    too), to share its code with another script."""
    import importlib.util
    import sys
    key = f"pis_exp_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def write_note(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scrub(text))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def fmt(x, spec: str | None = None, *, default: str = "{:.2f}") -> str:
    """A number for a note with `spec` (else `default`): "-" for None, NaN and infinities. Scripts bind their
    own default with functools.partial."""
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (spec or default).format(x)


def check_mc_config(labels: Path, inc: dict) -> None:
    """The Monte Carlo labels must come from a run with the same rule (k, interval, majority, warm-up) and H;
    their run's config.json sits next to the labels file."""
    cfg_path = Path(labels).parent / "config.json"
    if not cfg_path.exists():
        raise ValueError("the Monte Carlo labels have no run config.json next to them")
    cfg = read_json(cfg_path)
    want = {"k": int(inc["k"]), "interval": int(inc["interval"]), "majority": float(inc["majority"]),
            "warmup": int(inc["warmup"]), "horizon": int(inc["H"])}
    diff = [k for k, v in want.items() if k not in cfg or cfg[k] != v]
    if diff:
        raise ValueError(f"the Monte Carlo labels were made with a different {', '.join(diff)} "
                         f"({ {k: cfg.get(k) for k in diff} } vs { {k: want[k] for k in diff} })")


PRIO_DARWIN_PROCESS = 4         # <sys/resource.h>; getpriority gives 1 for a process in background QoS


def background_priority() -> tuple[bool, str]:
    """Whether this process runs at background priority (scripts/gentle.sh: macOS background QoS via
    `taskpolicy -b`, or a positive nice value elsewhere), and how that was read."""
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    try:
        darwin_bg = os.getpriority(PRIO_DARWIN_PROCESS, 0) == 1
    except OSError:
        darwin_bg = False
    if darwin_bg:
        return True, "macOS background QoS"
    return (nice > 0), f"nice {nice}"
