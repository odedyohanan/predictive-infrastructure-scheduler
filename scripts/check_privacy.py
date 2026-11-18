"""Scan tracked files for machine-specific or secret strings. Exit 1 on a hit."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Patterns are built from fragments so this file does not match itself.
PATTERNS = [
    ("home path (mac)", re.compile("/" + "Users" + "/[A-Za-z0-9._-]+")),
    ("home path (linux)", re.compile("/" + "home" + "/[A-Za-z0-9._-]+")),
    ("home path (windows)", re.compile("[A-Za-z]:" + r"\\" + "Users" + r"\\")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")),
    ("github token", re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{8,}")),
]
ALLOWED_EMAILS = {"noreply@" + "anthropic.com"}


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout.decode()
    return [root / p for p in out.split("\0") if p]


def scan(root: Path) -> list[str]:
    denylist = root / ".privacy-denylist.local"
    extra = []
    if denylist.is_file():
        extra = [s.strip() for s in denylist.read_text().splitlines() if s.strip()]
    hits = []
    for path in tracked_files(root):
        if not path.is_file():
            continue
        try:
            lines = path.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(root)
        for n, line in enumerate(lines, 1):
            for name, pat in PATTERNS:
                for m in pat.finditer(line):
                    if name == "email" and m.group(0) in ALLOWED_EMAILS:
                        continue
                    hits.append(f"{rel}:{n}: {name}")
            for s in extra:
                if s in line:
                    hits.append(f"{rel}:{n}: denylist entry")
    return hits


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(".")
    hits = scan(root.resolve())
    for h in hits:
        print(h)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
