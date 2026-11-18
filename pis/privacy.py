"""Scrub machine-specific strings (paths, emails) from text."""
from __future__ import annotations

import os
import re
from pathlib import Path

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Home-directory-style paths (macOS, Linux, Windows); prefixes built from fragments.
_HOME_RE = re.compile(
    "|".join(
        [
            "/" + "Users" + r"/[^\s/\"']+(?:/[^\s\"']*)?",
            "/" + "home" + r"/[^\s/\"']+(?:/[^\s\"']*)?",
            "[A-Za-z]:" + r"\\" + "Users" + r"\\[^\s\\\"']+(?:\\[^\s\"']*)?",
        ]
    )
)


def scrub(text: str) -> str:
    """Replace home dir, cwd, emails and home-style paths with placeholders."""
    for literal in sorted({str(Path.home()), os.getcwd()}, key=len, reverse=True):
        if literal and literal != "/":
            text = text.replace(literal, "<path>")
    text = _HOME_RE.sub("<path>", text)
    return _EMAIL.sub("<email>", text)
