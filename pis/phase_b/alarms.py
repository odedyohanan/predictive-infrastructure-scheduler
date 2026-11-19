"""Alarm events from a per-window risk series, their scoring against incident labels, and the conformal
alarm threshold.

Hysteresis: the alarm turns on when the risk reaches `threshold` and off when it falls below `low`; an alarm
event is the first window of each on-stretch. A window without a risk (NaN, e.g. the warm-up) is off.

An alarm event at window e is true when a new counted incident starts in (e, e + H] (`incidents.onset_within`,
the target of the Monte Carlo labels: an incident already in force at e does not count), false otherwise.
Events too close to the episode's end for that to be judged (e + H + tail > last window, `tail` = k - 1 for
the run rule) are censored: neither true nor false. An incident is caught with lead l when the first alarm
event in [onset - H, onset - 1] is at onset - l; it counts for recall when l >= the minimum lead. Incidents
with onsets in the warm-up are not counted (`incidents.label_series` drops them).

Conformal risk control over episodes: the loss of episode i at threshold lambda is its share of false alarm
events (0 without events); lambda* is the smallest grid threshold such that every grid threshold at or above
it keeps (n / (n + 1)) * mean_i L_i + 1 / (n + 1) <= alpha (the scan runs from the top and stops at the
first failure, because the loss is not monotone in lambda). `low` = LOW_RATIO * lambda*.
"""
from __future__ import annotations

import numpy as np

from pis.phase_b import incidents

ALPHA = 0.10
LOW_RATIO = 0.7
MAX_GRID = 2000
CAUSES = ("building_up", "sudden")
BOOT_TAG = 14_000
BOOT_REPS = 2000


# ------------------------------------------------------------------ events
def alarm_events(risk: np.ndarray, threshold: float, low: float) -> list[int]:
    """First windows of the alarm's on-stretches (on at risk >= threshold, off at risk < low or NaN)."""
    if low > threshold:
        raise ValueError(f"low {low} above threshold {threshold}")
    r = np.asarray(risk, dtype=float)
    on_at = np.flatnonzero(r >= threshold)              # NaN compares False: never turns on
    off_at = np.flatnonzero(~(r >= low))                # NaN counts as off
    out: list[int] = []
    i = 0
    while True:
        j = np.searchsorted(on_at, i)
        if j == len(on_at):
            return out
        e = int(on_at[j])
        out.append(e)
        k = np.searchsorted(off_at, e, side="right")
        if k == len(off_at):
            return out
        i = int(off_at[k])


# ------------------------------------------------------------------ scoring
def score_episode(events, labels: incidents.Labels, H: int, min_lead: int, tail: int = 0) -> dict:
    """True, false and censored alarm events; per cause the counted incidents, the caught ones (lead >=
    `min_lead`), the leads of the caught ones (`leads`) and of every incident with an event in its lead window
    (`early`)."""
    events = sorted(int(e) for e in events)
    last = len(labels.in_incident) - 1
    true = false = censored = 0
    for e in events:
        if e + H + tail > last:
            censored += 1
        elif incidents.onset_within(labels, e, H):
            true += 1
        else:
            false += 1
    ev = np.array(events, dtype=np.int64)
    out = {"events": len(events), "true": true, "false": false, "censored": censored,
           "incidents": {c: 0 for c in CAUSES}, "caught": {c: 0 for c in CAUSES},
           "leads": {c: [] for c in CAUSES}, "early": {c: [] for c in CAUSES}}
    for inc in labels.incidents:
        c = inc.cause
        out["incidents"][c] += 1
        j = np.searchsorted(ev, inc.onset - H)
        if j < len(ev) and ev[j] <= inc.onset - 1:
            lead = int(inc.onset - ev[j])
            out["early"][c].append(lead)
            if lead >= min_lead:
                out["caught"][c] += 1
                out["leads"][c].append(lead)
    return out


def episode_loss(score: dict) -> float:
    """Share of false alarm events among the judged ones; 0 without any."""
    n = score["true"] + score["false"]
    return score["false"] / n if n else 0.0


def summarize(scores: list[dict]) -> dict:
    """Pooled statistics over episodes. Quiet episode = no counted incident. Undefined ratios are None."""
    tot = {k: sum(s[k] for s in scores) for k in ("events", "true", "false", "censored")}
    judged = tot["true"] + tot["false"]
    quiet = [s for s in scores if not sum(s["incidents"].values())]
    by = {}
    for c in CAUSES:
        n = sum(s["incidents"][c] for s in scores)
        caught = sum(s["caught"][c] for s in scores)
        leads = [x for s in scores for x in s["leads"][c]]
        by[c] = {"incidents": n, "caught": caught, "recall": caught / n if n else None,
                 "median_lead": float(np.median(leads)) if leads else None}
    return {**tot, "episodes": len(scores), "false_alarm_rate": tot["false"] / judged if judged else None,
            "quiet_episodes": len(quiet),
            "alarms_per_quiet_episode": sum(s["events"] for s in quiet) / len(quiet) if quiet else None,
            "by_cause": by}


# ------------------------------------------------------------------ conformal threshold
def threshold_grid(risks, max_grid: int = MAX_GRID) -> np.ndarray:
    """Sorted unique risk values (NaN dropped), thinned to at most `max_grid` quantiles."""
    v = np.concatenate([np.asarray(r, dtype=float).ravel() for r in risks]) if len(risks) else np.zeros(0)
    v = np.unique(v[np.isfinite(v)])
    if len(v) > max_grid:
        v = np.unique(np.quantile(v, np.linspace(0.0, 1.0, max_grid), method="nearest"))
    return v


def conformal_threshold(per_episode_scores, alpha: float = ALPHA, *, H: int, tail: int = 0,
                        low_ratio: float = LOW_RATIO, max_grid: int = MAX_GRID) -> float | None:
    """lambda* over calibration episodes given as (risk series, incident labels) pairs; None when even the
    highest grid threshold fails."""
    eps = list(per_episode_scores)
    n = len(eps)
    if n == 0:
        raise ValueError("no calibration episodes")
    best = None
    for lam in threshold_grid([r for r, _ in eps], max_grid)[::-1]:
        lam = float(lam)
        losses = [episode_loss(score_episode(alarm_events(r, lam, low_ratio * lam), lab, H, 1, tail))
                  for r, lab in eps]
        if (n / (n + 1)) * float(np.mean(losses)) + 1.0 / (n + 1) > alpha:
            break
        best = lam
    return best


# ------------------------------------------------------------------ bootstrap
def bootstrap_recall_diff(caught_a, caught_b, incidents_n, reps: int = BOOT_REPS, tag: int = BOOT_TAG,
                          conf: float = 0.95) -> dict:
    """Recall of A minus recall of B over the incidents of resampled episodes (per-episode counts), with a
    percentile interval. Resamples without any incident are skipped."""
    a, b, n = (np.asarray(x, dtype=float) for x in (caught_a, caught_b, incidents_n))
    if not n.sum():
        return {"diff": None, "lo": None, "hi": None, "reps": 0}
    rng = np.random.default_rng(tag)
    idx = rng.integers(0, len(n), size=(reps, len(n)))
    tot = n[idx].sum(axis=1)
    ok = tot > 0
    d = (a[idx].sum(axis=1)[ok] - b[idx].sum(axis=1)[ok]) / tot[ok]
    q = (1 - conf) / 2
    return {"diff": float((a.sum() - b.sum()) / n.sum()), "lo": float(np.quantile(d, q)),
            "hi": float(np.quantile(d, 1 - q)), "reps": int(ok.sum())}
