"""Incident predictors: each turns the window rows seen so far into the risk that a new incident starts
within the next H windows.

Every predictor has `name`, `risk(window_rows, H, episode=None) -> float` and, when it learns from data,
`fit(store, seeds)`, which accepts history seeds only (the private `_fit_frames` behind it takes frames
directly and has no seed guard: tests only). `window_rows` are rows shaped like the store's
window rows (`pis.phase_b.history`; live records through `features.rows_from_records`) of one episode, up
to and including the current window, oldest first, with the episode's known_* columns
(`features.join_known`). `episode` names the episode the rows belong to when they carry no `episode`
column (the analogue search needs it to leave the episode's own history out). Predictors read only the state, raw-count, policy, violation and known columns,
never the degradation columns.

  WorldModelPredictor  share of sampled world-model futures (the policy in force held) with a new onset
  HazardPredictor      discrete-time logistic hazard on features of the last 8 windows
  AnalogueSearch       share of the k nearest history situations followed by an onset within H
  QueueingModel        fluid projection of A's backlog against its effective capacity (no learning)
  ChangePoint          CUSUM and Bayesian online change-point detection on A's p99 and overdue backlog

Labels for learning: a window t of a history episode is followed by an onset within H when the first
counted incident onset after t (`incidents.label_series` on the whole episode with the rule) lies in
(t, t + H], as `incidents.onset_within` judges it.
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import pandas as pd

from pis.phase_b import features, incidents
from pis.phase_b import seeds as seed_ranges

WINDOW_S = incidents.WINDOW_S
SLO_S = incidents.SLO_MS / 1000.0
L3_DOMAINS = ((0, 1, 2, 3), (4, 5, 6, 7))
N_CPUS = 8
NO_ONSET = 1 << 30


class Predictor(Protocol):
    name: str

    def risk(self, window_rows: pd.DataFrame, H: int, episode: int | None = None) -> float:
        ...


# ------------------------------------------------------------------ data
def _history(seeds) -> list[int]:
    seeds = [int(s) for s in seeds]
    seed_ranges.check(seeds, "HISTORY")
    return seeds


def history_frames(store, seeds) -> pd.DataFrame:
    """Training: window rows of history episodes with their known_* columns (history seeds only)."""
    seeds = _history(seeds)
    return features.join_known(store.training_windows(seeds), store.episodes())


def onset_distance(viol, active, rule: incidents.IncidentRule, warmup: int = incidents.WARMUP) -> np.ndarray:
    """Per window t: windows until the first counted onset after t (NO_ONSET when none follows)."""
    lab = incidents.label_series(np.asarray(viol, bool), np.asarray(active, bool), rule, warmup)
    onsets = np.array([i.onset for i in lab.incidents], dtype=np.int64)
    t = np.arange(len(viol))
    if not len(onsets):
        return np.full(len(viol), NO_ONSET, dtype=np.int64)
    j = np.searchsorted(onsets, t, side="right")
    return np.where(j < len(onsets), onsets[np.minimum(j, len(onsets) - 1)] - t, NO_ONSET).astype(np.int64)


def _episodes(frame: pd.DataFrame):
    for ep, part in frame.groupby("episode", sort=True):
        yield int(ep), part.sort_values("window")


# ------------------------------------------------------------------ A-side signals
def a_capacity(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Cores A can get under the policy in force (a CPU shared with B or C counts by share weight) and
    the neighbours' busy share in A's L3 domains (the interference driver), per row."""
    def col(w, k):
        return rows[f"policy_elig_{w}_{k}"].to_numpy(float)
    sh = [rows[f"policy_share_{w}"].to_numpy(float) for w in "ABC"]
    cores = np.zeros(len(rows))
    dom = np.zeros((len(rows), N_CPUS), dtype=bool)
    for k in range(N_CPUS):
        a, b, c = col(0, k), col(1, k), col(2, k)
        denom = sh[0] * a + sh[1] * b + sh[2] * c
        cores += np.where((a > 0) & (denom > 0), sh[0] / np.maximum(denom, 1e-9), 0.0)
    for d in L3_DOMAINS:
        any_a = np.any([col(0, k) > 0 for k in d], axis=0)
        for k in d:
            dom[:, k] = any_a
    others = np.stack([rows[f"usage_1_{k}"].to_numpy(float) + rows[f"usage_2_{k}"].to_numpy(float)
                       for k in range(N_CPUS)], axis=1)
    busy = np.where(dom.any(axis=1), (others * dom).sum(axis=1) / np.maximum(dom.sum(axis=1), 1), 0.0)
    return cores, np.clip(busy, 0.0, 1.0)


def a_service_rate(rows: pd.DataFrame) -> np.ndarray:
    """Requests per second A can serve: cores / (service time x (1 + interference x busy)). Known
    parameters are the episode's start values (drift is not announced). Thermal throttling slows the GPUs
    only, so it does not enter A's capacity."""
    cores, busy = a_capacity(rows)
    svc = rows["known_a_service_ms"].to_numpy(float)
    inter = rows["known_a_interference"].to_numpy(float)
    return cores * 1000.0 / (svc * (1.0 + inter * busy))


SIGNALS = ("overdue", "late", "late_frac", "p99", "qlen", "qwork", "util", "rate")


def a_signals(rows: pd.DataFrame) -> np.ndarray:
    """[T, len(SIGNALS)]: log overdue backlog, log late requests, late share of resolved requests, encoded
    p99, encoded queue length and work, utilisation (arrival rate / service rate), arrival rate (k req/s)."""
    late = rows["a_late_new"].to_numpy(float)
    ontime = rows["a_completed_ontime"].to_numpy(float)
    resolved = late + ontime
    rate = rows["a_rate"].to_numpy(float)
    util = rate / np.maximum(a_service_rate(rows), 1e-6)
    return np.stack([np.log1p(rows["a_queue_overdue"].to_numpy(float)), np.log1p(late),
                     np.where(resolved > 0, late / np.maximum(resolved, 1), 0.0),
                     rows["a_p99_ms"].to_numpy(float), rows["a_queue_len"].to_numpy(float),
                     rows["a_queue_work_ms"].to_numpy(float), np.minimum(util, 5.0), rate / 1000.0], axis=1)


def rolling(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal last value, mean and least-squares slope (per window) over the last `w` rows of x [T, m];
    rows before the start repeat the first row."""
    x = np.asarray(x, float)
    pad = np.concatenate([np.repeat(x[:1], w - 1, axis=0), x], axis=0)
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(pad, axis=0)], axis=0)
    mean = (c[w:] - c[:-w]) / w
    u = np.arange(w, dtype=float) - (w - 1) / 2
    slope = np.zeros_like(x)
    for j in range(w):                       # sum over the window of u_j * x_{t - w + 1 + j}
        slope += u[j] * pad[j:j + len(x)]
    slope /= (u * u).sum()
    return x, mean, slope


def _violation_features(viol: np.ndarray, w: int) -> np.ndarray:
    """Causal current violation-run length (log1p) and violated share of the last `w` windows."""
    v = np.asarray(viol, float)
    run = np.zeros(len(v))
    for i in range(len(v)):
        run[i] = (run[i - 1] + 1) * v[i] if i else v[i]
    c = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(1, len(v) + 1)
    share = (c[i] - c[np.maximum(i - w, 0)]) / w
    return np.stack([np.log1p(run), share], axis=1)


# ------------------------------------------------------------------ logistic fitting
def _fit_logistic_hazard(x: np.ndarray, d: np.ndarray, at_risk_max: np.ndarray, max_h: int, l2: float,
                         iters: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """logit h_i(t) = w . x_t + b_i for steps i = 1..max_h, fitted by full-batch Adam on the discrete-time
    likelihood. Window t is at risk at step i while d_t >= i and i <= at_risk_max_t (censoring); the event
    is d_t == i."""
    import torch
    torch.manual_seed(seed)
    steps = np.arange(1, max_h + 1)
    at_risk = (d[:, None] >= steps[None]) & (steps[None] <= at_risk_max[:, None])
    event = (d[:, None] == steps[None]) & at_risk
    X = torch.from_numpy(x.astype(np.float32))
    R = torch.from_numpy(at_risk.astype(np.float32))
    E = torch.from_numpy(event.astype(np.float32))
    rate = max(float(event.sum()) / max(float(at_risk.sum()), 1.0), 1e-6)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.full((max_h,), math.log(rate / (1 - rate)), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    n = R.sum().clamp(min=1.0)
    for _ in range(iters):
        logit = (X @ w)[:, None] + b[None]
        nll = (torch.nn.functional.binary_cross_entropy_with_logits(logit, E, reduction="none") * R).sum() / n
        loss = nll + l2 * (w * w).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy().astype(np.float64), b.detach().numpy().astype(np.float64)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ------------------------------------------------------------------ world model
class WorldModelPredictor:
    """Share of `n` sampled futures of the probabilistic world model, with the policy in force (the newest
    row's) held, in which a new counted incident starts within H (`wm_prob.rollout_risk`)."""
    name = "world_model"

    def __init__(self, model, rule: incidents.IncidentRule, n: int = 32, warmup: int = incidents.WARMUP,
                 seed: int = 0):
        self.model, self.rule, self.n, self.warmup, self.seed = model, rule, n, warmup, seed

    def fit(self, store, seeds) -> None:
        """The world model is trained by its own script; only the seed guard applies here."""
        _history(seeds)

    def risk(self, window_rows: pd.DataFrame, H: int, episode: int | None = None) -> float:
        from pis.phase_b.wm_prob import rollout_risk
        win = features.encode_rows(window_rows.iloc[-self.model.context:])
        if win.known is None:
            raise ValueError("window rows need the episode's known_* columns")
        t = int(window_rows["window"].iloc[-1]) if "window" in window_rows else len(window_rows) - 1
        return rollout_risk(self.model, win, window_rows, self.rule, H, self.n, self.warmup,
                            seed=self.seed + t)


# ------------------------------------------------------------------ hazard
class HazardPredictor:
    """Discrete-time logistic hazard: h_i = P(onset at t + i | none in (t, t + i)) for i = 1..max_h, one
    shared weight vector on the window features and one intercept per step (the baseline hazard).

    Features of window t (all from windows <= t): last value, mean and slope over the last 8 windows of the
    A-side SIGNALS (levels, slopes, the early signal = overdue slope, utilisation), the current violation
    run and the violated share of the last 8 windows; standardised on the training windows. The violation
    run is counted over every row given (not cut at 8), so `hazards` computes the features on the whole
    `window_rows`, exactly as training computes them on the whole episode.
    """
    name = "hazard"

    def __init__(self, rule: incidents.IncidentRule, max_h: int = 40, window: int = 8,
                 warmup: int = incidents.WARMUP, l2: float = 1e-3, iters: int = 400):
        self.rule, self.max_h, self.window, self.warmup, self.l2, self.iters = rule, max_h, window, warmup, l2, iters
        self.w = self.b = self.mu = self.sd = None

    def features(self, rows: pd.DataFrame) -> np.ndarray:
        last, mean, slope = rolling(a_signals(rows), self.window)
        vf = _violation_features(rows["slo_v2_violation"].to_numpy(bool), self.window)
        return np.concatenate([last, mean, slope, vf], axis=1)

    def window_features(self, window_rows: pd.DataFrame) -> np.ndarray:
        """Features of the newest row, as `hazards` uses them."""
        return self.features(window_rows)[-1]

    def fit(self, store, seeds) -> None:
        self._fit_frames(history_frames(store, seeds))

    def _fit_frames(self, frame: pd.DataFrame) -> None:
        xs, ds, caps = [], [], []
        for _, part in _episodes(frame):
            T = len(part)
            x = self.features(part)
            d = onset_distance(part.slo_v2_violation, part.deg_active, self.rule, self.warmup)
            t = np.arange(T)
            keep = t >= self.warmup
            xs.append(x[keep])
            ds.append(d[keep])
            caps.append(((T - self.rule.k) - t)[keep])        # later onsets cannot be seen in the episode
        x, d, cap = np.concatenate(xs), np.concatenate(ds), np.concatenate(caps)
        self.mu, self.sd = x.mean(axis=0), np.where(x.std(axis=0) < 1e-6, 1.0, x.std(axis=0))
        self.w, self.b = _fit_logistic_hazard((x - self.mu) / self.sd, d, cap, self.max_h, self.l2, self.iters)

    def hazards(self, window_rows: pd.DataFrame, H: int, episode: int | None = None) -> np.ndarray:
        if self.w is None:
            raise ValueError("fit the hazard model first")
        if H > self.max_h:
            raise ValueError(f"H = {H} exceeds the fitted {self.max_h} steps")
        x = self.window_features(window_rows)
        return _sigmoid(float(((x - self.mu) / self.sd) @ self.w) + self.b[:H])

    def risk(self, window_rows: pd.DataFrame, H: int, episode: int | None = None) -> float:
        return float(1.0 - np.prod(1.0 - self.hazards(window_rows, H)))


# ------------------------------------------------------------------ analogues
class AnalogueSearch:
    """k nearest history situations by the standardised summary of the last 16 windows (mean, slope and
    last value of the A-side SIGNALS); risk = share of them followed by an onset within H. Neighbours from
    the queried episode (the `episode` argument, else the rows' `episode` column; one of them is required)
    are excluded. Reference windows lie after the warm-up and early enough that an
    onset up to `max_h` windows later is visible in their episode."""
    name = "analogue"

    def __init__(self, rule: incidents.IncidentRule, k: int = 50, window: int = 16, max_h: int = 40,
                 warmup: int = incidents.WARMUP):
        self.rule, self.k, self.window, self.max_h, self.warmup = rule, k, window, max_h, warmup
        self.ref = self.dist = self.ep = self.mu = self.sd = None

    def embed(self, rows: pd.DataFrame) -> np.ndarray:
        last, mean, slope = rolling(a_signals(rows), self.window)
        return np.concatenate([mean, slope, last], axis=1)

    def fit(self, store, seeds) -> None:
        self._fit_frames(history_frames(store, seeds))

    def _fit_frames(self, frame: pd.DataFrame) -> None:
        es, ds, eps = [], [], []
        for ep, part in _episodes(frame):
            T = len(part)
            e = self.embed(part)
            d = onset_distance(part.slo_v2_violation, part.deg_active, self.rule, self.warmup)
            t = np.arange(T)
            keep = (t >= max(self.warmup, self.window - 1)) & (t <= T - self.rule.k - self.max_h)
            es.append(e[keep])
            ds.append(d[keep])
            eps.append(np.full(int(keep.sum()), ep))
        e = np.concatenate(es)
        self.mu, self.sd = e.mean(axis=0), np.where(e.std(axis=0) < 1e-6, 1.0, e.std(axis=0))
        self.ref = ((e - self.mu) / self.sd).astype(np.float32)
        self.ref_sq = (self.ref.astype(np.float64) ** 2).sum(axis=1)
        self.dist, self.ep = np.concatenate(ds), np.concatenate(eps)

    def neighbours(self, window_rows: pd.DataFrame, episode: int | None = None) -> np.ndarray:
        if self.ref is None:
            raise ValueError("fit the analogue search first")
        if episode is None:
            if "episode" not in window_rows:
                raise ValueError("the analogue search needs the rows' episode (argument or column) to leave "
                                 "the episode's own history out")
            episode = int(window_rows["episode"].iloc[-1])
        q = ((self.embed(window_rows.iloc[-self.window:])[-1] - self.mu) / self.sd).astype(np.float32)
        d2 = self.ref_sq - 2.0 * (self.ref @ q) + float(q @ q)
        d2 = np.where(self.ep == int(episode), np.inf, d2)
        k = min(self.k, int(np.isfinite(d2).sum()))
        if k == 0:
            raise ValueError("no reference windows outside the queried episode")
        return np.argpartition(d2, k - 1)[:k]

    def risk(self, window_rows: pd.DataFrame, H: int, episode: int | None = None) -> float:
        if H > self.max_h:
            raise ValueError(f"H = {H} exceeds the reference horizon {self.max_h}")
        return float((self.dist[self.neighbours(window_rows, episode)] <= H).mean())


# ------------------------------------------------------------------ queueing
class QueueingModel:
    """Fluid projection of A's queue. Service rate mu = A's cores under the policy / (service time x (1 +
    interference x neighbours' busy share)) with the known start-of-episode parameters; arrival rate =
    current rate plus its trend (slope over the last 8 windows), held for H windows; backlog Q_h = max(0,
    Q_{h-1} + (lambda_h - mu) x 0.5 s) from the current queue length. Projected overdue backlog = what
    cannot be served within the SLO (Q_h - mu x SLO), at least the current overdue count; its peak z is
    compared with the incident threshold (1 % of a window's arrivals, at least one request):
    risk = logistic((log1p z - log1p threshold) / scale). No learning: `scale` is set by `calibrate` on
    calibration seeds."""
    name = "queueing"

    def __init__(self, rule: incidents.IncidentRule, scale: float = 0.5, window: int = 8,
                 warmup: int = incidents.WARMUP):
        self.rule, self.scale, self.window, self.warmup = rule, scale, window, warmup

    def statistic(self, rows: pd.DataFrame, H: int) -> np.ndarray:
        """Per row: (log1p z - log1p threshold), using rows up to it only."""
        lam = rows["a_rate"].to_numpy(float)
        _, _, slope = rolling(lam[:, None], self.window)
        slope = slope[:, 0]
        mu = a_service_rate(rows)
        q = np.expm1(rows["a_queue_len"].to_numpy(float))
        over0 = rows["a_queue_overdue"].to_numpy(float)
        peak = over0.copy()
        for h in range(1, H + 1):
            lam_h = np.maximum(lam + slope * h, 0.0)
            q = np.maximum(q + (lam_h - mu) * WINDOW_S, 0.0)
            peak = np.maximum(peak, q - mu * SLO_S)
        thr = np.maximum(0.01 * lam * WINDOW_S, 1.0)
        return np.log1p(np.maximum(peak, 0.0)) - np.log1p(thr)

    def risk(self, window_rows: pd.DataFrame, H: int, episode: int | None = None) -> float:
        z = self.statistic(window_rows.iloc[-self.window:], H)[-1]
        return float(_sigmoid(z / self.scale))

    def calibrate(self, store, seeds, H: int, grid=None) -> float:
        """Set `scale` by maximum likelihood of "onset within H" over the windows of calibration episodes."""
        seeds = [int(s) for s in seeds]
        seed_ranges.check(seeds, "CALIBRATION")
        eps = store.episodes()
        zs, ys = [], []
        for s in seeds:
            n = int(eps.loc[eps.seed == s, "windows"].iloc[0])
            part = features.join_known(store.window_rows(s, 0, n), eps)
            z, y = self._labelled(part, H)
            zs.append(z)
            ys.append(y)
        return self.calibrate_arrays(np.concatenate(zs), np.concatenate(ys), grid)

    def _labelled(self, part: pd.DataFrame, H: int) -> tuple[np.ndarray, np.ndarray]:
        T = len(part)
        d = onset_distance(part.slo_v2_violation, part.deg_active, self.rule, self.warmup)
        t = np.arange(T)
        keep = (t >= self.warmup) & (t + H <= T - self.rule.k)
        return self.statistic(part, H)[keep], (d <= H)[keep]

    def calibrate_arrays(self, z: np.ndarray, y: np.ndarray, grid=None) -> float:
        grid = np.geomspace(0.05, 20.0, 60) if grid is None else np.asarray(grid, float)
        y = np.asarray(y, float)

        def nll(s):
            p = np.clip(_sigmoid(z / s), 1e-9, 1 - 1e-9)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        self.scale = float(grid[int(np.argmin([nll(s) for s in grid]))])
        return self.scale


# ------------------------------------------------------------------ change points
def bocpd_short_mass(z: np.ndarray, hazard: float = 1 / 200, short: int = 10, rmax: int = 300,
                     noise_var: float = 1.0, prior_var: float = 1.0) -> np.ndarray:
    """Bayesian online change-point detection on a standardised series (Gaussian, known noise variance,
    unknown mean with prior N(0, prior_var), constant hazard). Returns per step the posterior mass of a
    change within the last `short` observations (the run since the series' start excluded). The run-length
    distribution is truncated at `rmax` (the longest runs pooled)."""
    z = np.asarray(z, float)
    p = np.array([1.0])
    mu = np.array([0.0])
    prec = np.array([1.0 / prior_var])
    out = np.zeros(len(z))
    for t, x in enumerate(z):
        var = 1.0 / prec + noise_var
        pred = np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
        grow = p * pred * (1 - hazard)
        cp = float((p * pred).sum() * hazard)
        p = np.concatenate([[cp], grow])
        mu = np.concatenate([[0.0], (prec * mu + x / noise_var) / (prec + 1.0 / noise_var)])
        prec = np.concatenate([[1.0 / prior_var], prec + 1.0 / noise_var])
        if len(p) > rmax + 1:
            p[rmax] += p[rmax + 1:].sum()
            p, mu, prec = p[:rmax + 1], mu[:rmax + 1], prec[:rmax + 1]
        s = p.sum()
        p = p / s if s > 0 else np.r_[1.0, np.zeros(len(p) - 1)]
        r = np.arange(len(p))                        # observations in the current run
        out[t] = float(p[(r <= short) & (r < t + 1)].sum())
    return out


class ChangePoint:
    """Change detection on A's encoded p99 and log overdue backlog, standardised by their mean and spread
    on ordinary (no degradation) history episodes. Per series: one-sided upward CUSUM (drift `drift`,
    normalised by the alarm level `h` and capped at 1) and BOCPD's short-run mass, counted only when the
    last 5 values sit above the baseline (a change upward). risk = the largest of these over both series;
    it describes the present, so H is not used. Only the last `rmax` rows are read."""
    name = "change_point"
    COLUMNS = ("a_p99_ms", "a_queue_overdue")

    def __init__(self, drift: float = 0.5, h: float = 5.0, hazard: float = 1 / 200, short: int = 10,
                 rmax: int = 300, min_sd: float = 0.25):
        self.drift, self.h, self.hazard, self.short, self.rmax, self.min_sd = drift, h, hazard, short, rmax, min_sd
        self.mu = self.sd = None

    def series(self, rows: pd.DataFrame) -> np.ndarray:
        return np.stack([rows["a_p99_ms"].to_numpy(float), np.log1p(rows["a_queue_overdue"].to_numpy(float))],
                        axis=1)

    def fit(self, store, seeds) -> None:
        self._fit_frames(history_frames(store, seeds), store.episodes())

    def _fit_frames(self, frame: pd.DataFrame, episodes: pd.DataFrame | None = None) -> None:
        if episodes is not None:
            quiet = set(episodes.loc[episodes.kind == "none", "seed"].astype(int))
            frame = frame[frame.episode.isin(quiet)]
        if frame.empty:
            raise ValueError("no ordinary episodes to set the baseline")
        x = self.series(frame)
        self.mu, self.sd = x.mean(axis=0), np.maximum(x.std(axis=0), self.min_sd)

    def scores(self, window_rows: pd.DataFrame) -> dict:
        if self.mu is None:
            raise ValueError("fit the change-point baseline first")
        z = (self.series(window_rows.iloc[-self.rmax:]) - self.mu) / self.sd
        out = {}
        for j, c in enumerate(self.COLUMNS):
            s = 0.0
            for v in z[:, j]:
                s = max(0.0, s + v - self.drift)
            up = float(np.clip(z[-5:, j].mean() / 2.0, 0.0, 1.0))
            out[c] = {"cusum": min(s / self.h, 1.0),
                      "bocpd": float(bocpd_short_mass(z[:, j], self.hazard, self.short, self.rmax)[-1]) * up}
        return out

    def risk(self, window_rows: pd.DataFrame, H: int, episode: int | None = None) -> float:
        return float(max(v for sc in self.scores(window_rows).values() for v in sc.values()))
