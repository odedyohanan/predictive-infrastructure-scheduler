"""Paired comparisons with Student-t confidence intervals (no scipy dependency)."""
from __future__ import annotations

import math

import numpy as np


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Inverse CDF of Student's t by bisection (accurate to ~1e-10)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def paired_diff(a: np.ndarray, b: np.ndarray, conf: float = 0.95) -> dict:
    """Mean of a - b over pairs with a two-sided Student-t confidence interval."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-d arrays of the same length (one value per pair)")
    d = a - b
    n = len(d)
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "sd": float("nan"), "lo": float("nan"), "hi": float("nan"), "conf": conf}
    sd = float(d.std(ddof=1))
    half = t_quantile(0.5 + conf / 2.0, n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half, "conf": conf}
