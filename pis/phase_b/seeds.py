"""Seed ranges of the incident-advisor experiments and the guards every script applies."""
from __future__ import annotations

from typing import Iterable

HELD_OUT = range(3000, 3100)      # final evaluation only, used once
HISTORY = range(3100, 3400)       # generation and training; seed % 10: 8 = validation, 9 = test
CALIBRATION = range(3400, 3450)   # incident rule, conformal thresholds, chronic rule, lead time
SELECTION = range(3450, 3500)     # predictor choice, advisor tuning, development shadow results

RANGES = {"HELD_OUT": HELD_OUT, "HISTORY": HISTORY, "CALIBRATION": CALIBRATION, "SELECTION": SELECTION}
# Seeds consumed by earlier experiments (development and the first held-out evaluation).
REFUSED = (range(1000, 1200), range(2000, 2100))


def parse(text: str) -> list[int]:
    """"3100-3109,3120" -> [3100, ..., 3109, 3120]."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty item in seed list {text!r}")
        lo, sep, hi = part.partition("-")
        try:
            a = int(lo)
            b = int(hi) if sep else a
        except ValueError:
            raise ValueError(f"bad seed item {part!r}") from None
        if b < a:
            raise ValueError(f"descending seed range {part!r}")
        out.extend(range(a, b + 1))
    return out


def check(seeds: Iterable[int], allowed: str) -> None:
    """Raise ValueError unless every seed lies in the named range and none is refused."""
    if allowed not in RANGES:
        raise ValueError(f"unknown seed range {allowed!r}; expected one of {sorted(RANGES)}")
    seeds = list(seeds)
    if not seeds:
        raise ValueError("no seeds given")
    refused = [s for s in seeds if any(s in r for r in REFUSED)]
    if refused:
        raise ValueError(f"seeds {refused[:5]} belong to earlier experiments and are never used here")
    outside = [s for s in seeds if s not in RANGES[allowed]]
    if outside:
        r = RANGES[allowed]
        raise ValueError(f"seeds {outside[:5]} are outside {allowed} ({r.start}-{r.stop - 1})")
