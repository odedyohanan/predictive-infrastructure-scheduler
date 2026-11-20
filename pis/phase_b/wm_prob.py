"""Probabilistic GRU world model: a Gaussian over the next window's state change, sampled into futures.

`ProbGRU` reads a sequence of 79-number inputs (`pis.phase_b.features`) and outputs, at every position,
the mean and log-variance of the change of the 43 state numbers (standardised). Training minimises the
Gaussian negative log-likelihood, masked like the deterministic model (p50/p99 targets are skipped when no
request completed). Each episode is cut into sequences of `context + room` steps that overlap by `burn_in`
steps (a random shift per epoch); a sequence's first `burn_in` positions only build up the recurrent state
and carry no loss, except in the sequence that starts the episode, where every position counts. So the
network has learnt to predict after anything from 1 to `context + room` windows.

`ProbWorldModel.sample_rollouts` draws futures by ancestral sampling: the last `context` windows of real
history warm the recurrent state, then each step draws a change, clips the new state to the range seen in
training, and feeds it back one step at a time (for horizons beyond `room` the last `context` inputs are
run again from a fresh state). `rollout_risk` turns sampled futures into incident risk with the
incident rule, exactly as the Monte Carlo labels judge real branches. `calibration` measures predictive
intervals, CRPS, the Brier score of the rollout risk against Monte Carlo labels, and the rank agreement
of predicted and real (branched) effects of the named interventions.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pis.config import setup_imports
from pis.phase_b import features, incidents
from pis.phase_b import seeds as seed_ranges

setup_imports()

from worldmodel.model import Normalizer  # noqa: E402
from worldmodel.features import encode_policy  # noqa: E402

CONTEXT = 32
BURN_IN = 8
ROOM = 32           # training sequences are CONTEXT + ROOM long: rollouts of up to ROOM steps stay in range
LOGVAR_MIN, LOGVAR_MAX = -12.0, 6.0
OUTPUTS = ("a.queue_overdue", "a.late_new", "a.p99_ms")     # judged by the interval coverage
# Counts that are 0 in most windows: an interval containing 0 covers those for free, so their coverage is
# also reported (and judged) on the windows where the truth is non-zero.
NONZERO_OUTPUTS = ("a.queue_overdue", "a.late_new")
LEVEL = 0.8                                                  # central predictive interval
RANK_TAG = 16_000_000        # reseed j of every intervention branch at any state of episode s: RANK_TAG + s*1000 + j
                             # (shared by the episode's states and by all policies of a state)
RANK_HORIZON = 5
INTERVENTION_OUTCOME = "a.queue_overdue"


# ------------------------------------------------------------------ network
class ProbGRU(nn.Module):
    def __init__(self, in_dim: int = features.N_INPUT, out_dim: int = features.N_STATE, hidden: int = 128,
                 layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2 * out_dim)
        self.in_dim, self.out_dim, self.hidden, self.layers = in_dim, out_dim, hidden, layers

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """x [B, L, in] (standardised), optional recurrent state -> mean, log-variance [B, L, out] of the
        standardised change, and the recurrent state after the last position."""
        out, h = self.gru(x, h)
        mean, logvar = self.head(out).chunk(2, dim=-1)
        return mean, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX), h

    def params(self) -> dict:
        return dict(in_dim=self.in_dim, out_dim=self.out_dim, hidden=self.hidden, layers=self.layers)


def gaussian_nll(mean, logvar, y, mask) -> torch.Tensor:
    """Mean over unmasked numbers of 0.5 * (log var + (y - mean)^2 / var); the 0.5 log 2 pi constant is left out."""
    nll = 0.5 * (logvar + (y - mean) ** 2 * torch.exp(-logvar))
    return (nll * mask).sum() / mask.sum().clamp(min=1.0)


# ------------------------------------------------------------------ the model
class WorldModelBase:
    """Network + normalisers + meta (context, training bounds, recipe; sha256 once saved or loaded), and what
    every sampled world model shares: policy handling, batched candidates, checkpoint files. A subclass sets
    `KIND` (recorded in the checkpoint), `NET` (the network class) and implements `_sample`."""
    KIND = ""
    NET = None

    def __init__(self, net, x_norm: Normalizer, y_norm: Normalizer, meta: dict | None = None):
        self.net, self.x_norm, self.y_norm, self.meta = net, x_norm, y_norm, dict(meta or {})
        self.net.eval()

    @property
    def context(self) -> int:
        return int(self.meta.get("context", CONTEXT))

    @property
    def room(self) -> int:
        return int(self.meta.get("room", 0))

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if "state_min" not in self.meta:
            return None
        return (np.asarray(self.meta["state_min"], np.float32), np.asarray(self.meta["state_max"], np.float32))

    def _policy_steps(self, policy, horizon: int) -> np.ndarray:
        if isinstance(policy, dict):
            policy = encode_policy(policy)
        p = np.asarray(policy, dtype=np.float32)
        if p.ndim == 1:
            return np.broadcast_to(p, (horizon, p.shape[0]))
        if len(p) < horizon:
            raise ValueError(f"{len(p)} policies for a horizon of {horizon}")
        return p[:horizon]

    def sample_rollouts(self, history: features.Window, policy, known, horizon: int, n: int,
                        seed: int | None = None) -> np.ndarray:
        """`n` sampled futures [n, horizon, 43] (encoded) after the newest window of `history`.

        `policy`: one policy (dict or encoded 33-vector) held for the whole horizon, or one encoded policy
        per step [horizon, 33]; the first runs in the window right after the history. `known`: dict or
        3-vector, or None to use `history.known`. States are clipped to the training range (and at 0).
        """
        pols = self._policy_steps(policy, horizon)
        return self._sample(history, np.broadcast_to(pols, (n,) + pols.shape), known, horizon, seed)

    def sample_rollouts_batch(self, history: features.Window, policies, known, horizon: int, n: int,
                              seed: int | None = None) -> np.ndarray:
        """`n` sampled futures for each of several candidate policies in one pass: [K, n, horizon, 43].

        Each element of `policies` is what `sample_rollouts` accepts as `policy`. All K x n futures share
        one random stream (row-major: candidate, then future); a single candidate gets exactly the futures
        `sample_rollouts` draws with the same seed.
        """
        pols = np.stack([self._policy_steps(p, horizon) for p in policies])
        rows = np.repeat(pols, n, axis=0)
        out = self._sample(history, rows, known, horizon, seed)
        return out.reshape(len(pols), n, horizon, out.shape[-1])

    def _sample(self, history: features.Window, pols: np.ndarray, known, horizon: int,
                seed: int | None) -> np.ndarray:
        raise NotImplementedError

    def _clip_bounds(self, n_state: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Lower (at least 0) and upper (None when unknown) bounds that sampled states are clipped to."""
        lo, hi = self.bounds if self.bounds is not None else (np.zeros(n_state, np.float32), None)
        return np.maximum(lo, 0.0), hi

    def save(self, path) -> str:
        """Write the checkpoint; returns (and records in `meta`) its sha256."""
        path = Path(path)
        meta = {k: v for k, v in self.meta.items() if k != "sha256"}
        torch.save({"kind": self.KIND, "state_dict": self.net.state_dict(), "arch": self.net.params(),
                    "x_norm": (self.x_norm.mean, self.x_norm.std), "y_norm": (self.y_norm.mean, self.y_norm.std),
                    "meta": meta}, path)
        self.meta["sha256"] = sha256(path)
        return self.meta["sha256"]

    @classmethod
    def from_checkpoint(cls, ck: dict, path) -> "WorldModelBase":
        """The model of a loaded checkpoint dict `ck` read from `path` (its kind must be this class's)."""
        kind = checkpoint_kind(ck)
        if kind != cls.KIND:
            raise ValueError(f"the checkpoint holds a {kind!r} world model, not {cls.KIND!r}")
        net = cls.NET(**ck["arch"])
        net.load_state_dict(ck["state_dict"])
        m = cls(net, Normalizer(*ck["x_norm"]), Normalizer(*ck["y_norm"]), ck.get("meta") or {})
        m.meta["sha256"] = sha256(path)
        return m

    @classmethod
    def load(cls, path):
        return cls.from_checkpoint(torch.load(path, weights_only=False), path)


def checkpoint_kind(ck: dict) -> str:
    """The world-model kind a checkpoint records; checkpoints written before kinds existed hold a "pgru"."""
    return str(ck.get("kind") or "pgru")


class ProbWorldModel(WorldModelBase):
    """The probabilistic GRU: a Gaussian over the next change straight from the recurrent state."""
    KIND = "pgru"
    NET = ProbGRU

    @torch.inference_mode()
    def _run(self, xs: np.ndarray, h=None):
        """Standardised inputs [N, L, in] (and a recurrent state) -> mean, log-variance at the last
        position, new recurrent state."""
        mean, logvar, h = self.net(torch.from_numpy(xs), h)
        return mean[:, -1].numpy(), logvar[:, -1].numpy(), h

    def _last(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standardised mean and log-variance at the last position of raw inputs [N, L, in]."""
        mean, logvar, _ = self._run(self.x_norm.transform(x))
        return mean, logvar

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of the next change, in encoded units, for input windows [N, L, in]."""
        mean, logvar = self._last(np.asarray(x, dtype=np.float32))
        std = self.y_norm.std.astype(np.float64)
        return (mean * std + self.y_norm.mean), np.exp(logvar) * std ** 2

    def _sample(self, history: features.Window, pols: np.ndarray, known, horizon: int,
                seed: int | None) -> np.ndarray:
        """Ancestral sampling of one future per row of `pols` [R, horizon, 33]."""
        k = features.known_vector(known if known is not None else history.known)
        n = pols.shape[0]
        C, longest = self.context, self.context + self.room
        S, P = history.states[-C:], history.policies[-C:]
        past = np.concatenate([S[:-1], P[1:], np.broadcast_to(k, (len(S) - 1, len(k)))], axis=1)
        seq = [np.broadcast_to(self.x_norm.transform(past), (n,) + past.shape)] if len(past) else []
        s = np.broadcast_to(S[-1], (n, S.shape[1])).astype(np.float32)
        lo, hi = self._clip_bounds(S.shape[1])
        rng = np.random.default_rng(seed)
        std, mu = self.y_norm.std, self.y_norm.mean
        out = np.empty((n, horizon, S.shape[1]), dtype=np.float32)
        kk = np.broadcast_to(k, (n, len(k)))
        state, used = None, 0
        for h in range(horizon):
            x = np.concatenate([s, pols[:, h], kk], axis=1)
            seq.append(self.x_norm.transform(x)[:, None])
            if state is None or used >= longest:        # (re)start from the last `context` inputs
                win = np.concatenate(seq, axis=1)[:, -C:]
                seq = [win]
                mean, logvar, state = self._run(np.ascontiguousarray(win))
                used = win.shape[1]
            else:
                mean, logvar, state = self._run(seq[-1], state)
                used += 1
            delta = (mean + np.exp(0.5 * logvar) * rng.standard_normal(mean.shape)) * std + mu
            s = np.maximum(s + delta.astype(np.float32), lo)
            if hi is not None:
                s = np.minimum(s, hi)
            out[:, h] = s
        return out


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ training
def _y_normalizer(ys: np.ndarray, ms: np.ndarray) -> Normalizer:
    """Per-number mean and std over the unmasked targets only."""
    w = ms.sum(axis=0)
    mean = (ys * ms).sum(axis=0) / np.maximum(w, 1.0)
    var = (((ys - mean) ** 2) * ms).sum(axis=0) / np.maximum(w, 1.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return Normalizer(mean.astype(np.float32), std.astype(np.float32))


def _windows(lengths: list[int], offsets: np.ndarray, length: int, burn_in: int, shift: int):
    """Sequence starts (global row index) and whether each starts its episode."""
    stride = max(length - burn_in, 1)
    starts, first = [], []
    for L, off in zip(lengths, offsets):
        if L < length:
            continue
        s = list(range(shift % stride if shift else 0, L - length + 1, stride))
        if not s or s[0] != 0:
            s = [0] + s
        starts += [off + i for i in s]
        first += [i == 0 for i in s]
    return np.array(starts, dtype=np.int64), np.array(first, dtype=bool)


def fit_loop(make_net, seq_loss, train_data: list, val_data: list, context: int, burn_in: int, room: int,
             lr: float, batch: int, epochs: int, patience: int, seed: int, log=None,
             max_seconds: float | None = None):
    """The training loop shared by the sampled world models.

    Fits normalisers on the training data (Y on unmasked targets only), builds the network with
    `make_net(in_dim, out_dim)` and minimises `seq_loss(net, x, y, m, keep, training)` over sequences of
    `context + room` steps (`keep` [B, L, 1] is 0 on the burn-in positions of sequences that do not start
    their episode) with Adam, early stopping on the validation loss. Returns the network with the
    best-validation weights, the X and Y normalisers, the best validation loss and the per-epoch history
    {epoch, train_loss, val_loss, seconds}."""
    if context <= burn_in or room < 0:
        raise ValueError("context must exceed burn_in and room must be >= 0")
    length = context + room
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def stack(data):
        X = np.concatenate([d[0] for d in data]).astype(np.float32)
        Y = np.concatenate([d[1] for d in data]).astype(np.float32)
        M = np.concatenate([d[2] for d in data]).astype(np.float32)
        lengths = [len(d[0]) for d in data]
        return X, Y, M, lengths, np.cumsum([0] + lengths[:-1])

    X, Y, M, L, off = stack(train_data)
    Xv, Yv, Mv, Lv, offv = stack(val_data)
    xn = Normalizer.fit(X)
    yn = _y_normalizer(Y, M)
    Xt, Yt, Mt = (torch.from_numpy(a) for a in (xn.transform(X), yn.transform(Y), M))
    Xvt, Yvt, Mvt = (torch.from_numpy(a) for a in (xn.transform(Xv), yn.transform(Yv), Mv))
    net = make_net(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    pos = torch.arange(length)

    def loss_of(Xa, Ya, Ma, starts, first, training):
        idx = torch.from_numpy(starts)[:, None] + pos
        keep = ((pos >= burn_in)[None, :] | torch.from_numpy(first)[:, None]).float()[..., None]
        return seq_loss(net, Xa[idx], Ya[idx], Ma[idx], keep, training)

    vs, vf = _windows(Lv, offv, length, burn_in, 0)
    if not len(vs) or not len(_windows(L, off, length, burn_in, 0)[0]):
        raise ValueError(f"no training and validation episodes with at least {length} steps")
    best, best_state, bad, hist, t0 = math.inf, None, 0, [], time.perf_counter()
    for ep in range(epochs):
        starts, first = _windows(L, off, length, burn_in, int(rng.integers(1 << 30)))
        perm = rng.permutation(len(starts))
        net.train()
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = loss_of(Xt, Yt, Mt, starts[b], first[b], True)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        net.eval()
        with torch.no_grad():
            vl = sum(loss_of(Xvt, Yvt, Mvt, vs[i:i + 1024], vf[i:i + 1024], False).item() * len(vs[i:i + 1024])
                     for i in range(0, len(vs), 1024)) / len(vs)
        hist.append({"epoch": ep, "train_loss": tot / cnt, "val_loss": vl,
                     "seconds": round(time.perf_counter() - t0, 2)})
        if log:
            log(hist[-1])
        if vl < best - 1e-6:
            best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            break
    net.load_state_dict(best_state)
    return net, xn, yn, best, hist


def _pgru_loss(net, x, y, m, keep, training):
    mean, logvar, _ = net(x)
    return gaussian_nll(mean, logvar, y, m * keep)


def train(train_data: list, val_data: list, context: int = CONTEXT, burn_in: int = BURN_IN, room: int = ROOM,
          hidden: int = 128,
          layers: int = 2, lr: float = 1e-3, batch: int = 256, epochs: int = 300, patience: int = 20,
          seed: int = 0, log=None, max_seconds: float | None = None) -> tuple[ProbWorldModel, list[dict]]:
    """Fit on per-episode (X [L, in], Y [L, out], M [L, out]) arrays; returns the best-validation model and
    the per-epoch history {epoch, train_loss, val_loss, seconds}."""
    net, xn, yn, best, hist = fit_loop(lambda i, o: ProbGRU(i, o, hidden, layers), _pgru_loss, train_data,
                                       val_data, context, burn_in, room, lr, batch, epochs, patience, seed, log,
                                       max_seconds)
    meta = {"context": context, "room": room, "burn_in": burn_in, "best_val_loss": best, "epochs_run": len(hist),
            "hidden": hidden, "layers": layers, "lr": lr, "batch": batch, "seed": seed}
    return ProbWorldModel(net, xn, yn, meta), hist


def fit_history(train_fn, store, train_seeds, val_seeds, **recipe):
    """`train_fn(train_data, val_data, **recipe)` on history episodes, the training bounds and seeds added to
    the model's meta; calibration, selection and held-out seeds are refused, as is overlap."""
    train_seeds, val_seeds = [int(s) for s in train_seeds], [int(s) for s in val_seeds]
    seed_ranges.check(train_seeds, "HISTORY")
    seed_ranges.check(val_seeds, "HISTORY")
    if set(train_seeds) & set(val_seeds):
        raise ValueError("training and validation seeds overlap")
    tr, va = features.episode_data(store, train_seeds), features.episode_data(store, val_seeds)
    S = np.concatenate([w.states for w in tr])
    model, hist = train_fn([features.steps(w) for w in tr], [features.steps(w) for w in va], **recipe)
    model.meta.update({"state_min": np.maximum(S.min(axis=0), 0.0).tolist(), "state_max": S.max(axis=0).tolist(),
                       "train_seeds": train_seeds, "val_seeds": val_seeds, "state_names": features.STATE_B,
                       "input_names": features.INPUT_B,
                       "n_train_steps": int(sum(len(w.states) - 1 for w in tr))})
    return model, hist


def fit(store, train_seeds, val_seeds, **recipe) -> tuple[ProbWorldModel, list[dict]]:
    """Train on history episodes; calibration, selection and held-out seeds are refused, as is overlap."""
    return fit_history(train, store, train_seeds, val_seeds, **recipe)


# ------------------------------------------------------------------ incident risk from futures
def rollout_risk(model, window: features.Window, rows: pd.DataFrame, rule: incidents.IncidentRule, horizon: int,
                 n: int, warmup: int = incidents.WARMUP, policy=None, known=None, seed: int | None = None) -> float:
    """Share of `n` sampled futures in which a new counted incident starts in (t, t + horizon].

    `rows` are the observed rows up to the current window t (columns `slo_v2_violation`, optionally
    `window`); `window` their encoding. The policy in force (the newest row's) is held
    unless `policy` is given. Futures run `horizon + k - 1` windows (a run starting at t + horizon needs k
    windows) and their windows are never "degradation active"; onsets are judged on the observed series
    joined with each future, as the Monte Carlo labels judge real branches (`incidents.onset_share`, the
    advisor's judgement too).
    """
    from pis.phase_b.mc_labels import branch_windows
    pol = window.policies[-1] if policy is None else policy
    m = branch_windows(horizon, rule)
    roll = model.sample_rollouts(window, pol, known, m, n, seed=seed)
    pv = features.predicted_violation(features.decode(roll))
    v0 = rows["slo_v2_violation"].to_numpy(dtype=bool)
    w0 = int(rows["window"].iloc[0]) if "window" in rows and len(rows) else 0
    return incidents.onset_share(v0, pv[:n], rule, horizon, max(warmup - w0, 0))


# ------------------------------------------------------------------ calibration
def interval_coverage(samples: np.ndarray, truth: np.ndarray, level: float = LEVEL) -> float:
    """Share of truths inside the central `level` interval of their samples; samples [n, m], truth [m]."""
    q = (1 - level) / 2
    lo, hi = np.quantile(samples, q, axis=0), np.quantile(samples, 1 - q, axis=0)
    return float(((truth >= lo) & (truth <= hi)).mean())


def crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Mean sample CRPS: E|X - y| - 0.5 E|X - X'| (sorted-sample form); samples [n, m], truth [m]."""
    s = np.sort(np.asarray(samples, float), axis=0)
    n = s.shape[0]
    a = np.abs(s - truth[None]).mean(axis=0)
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    b = (w * s).sum(axis=0) / (n * n)
    return float((a - b).mean())


def _calibration_seeds(seeds) -> list[int]:
    seeds = [int(s) for s in seeds]
    for s in seeds:
        if s in seed_ranges.HELD_OUT:
            raise ValueError(f"seed {s} is held out")
        if s in seed_ranges.HISTORY and s % 10 not in (8, 9):
            raise ValueError(f"seed {s} is a training seed; calibration uses unseen episodes")
        if not any(s in r for r in (seed_ranges.HISTORY, seed_ranges.CALIBRATION, seed_ranges.SELECTION)):
            raise ValueError(f"seed {s} is outside the experiment's seed ranges")
    return seeds


def _episode_rows(store, seed: int, eps: pd.DataFrame) -> pd.DataFrame:
    n = int(eps.loc[seed, "windows"])
    return features.join_known(store.window_rows(seed, 0, n), eps.reset_index())


def predictive_scores(model, store, seeds, horizon: int, n: int = 64, every: int = 20,
                      warmup: int = incidents.WARMUP, seed: int = 0) -> dict:
    """Coverage of the central 80 % interval and CRPS at horizons 1, 5 and H for OUTPUTS, with the real
    future policies fed in. p99 is judged only where the true window completed requests. For
    NONZERO_OUTPUTS coverage and CRPS are also given over the windows whose true value is non-zero
    (`coverage_nonzero`, `crps_nonzero`); `counts`, `counts_nonzero` hold the numbers of windows."""
    eps = store.episodes().set_index("seed")
    hs = sorted({1, min(5, horizon), horizon})
    idx = [features.IDX[o] for o in OUTPUTS]
    smp = {(o, h): [] for o in OUTPUTS for h in hs}
    tru = {(o, h): [] for o in OUTPUTS for h in hs}
    starts = 0
    for s in seeds:
        w = features.encode_rows(_episode_rows(store, s, eps))
        T = len(w.states)
        for t in range(max(model.context - 1, warmup), T - horizon, every):
            hist = features.Window(w.states[:t + 1], w.policies[:t + 1], w.known)
            r = model.sample_rollouts(hist, w.policies[t + 1:t + 1 + horizon], None, horizon, n,
                                      seed=seed + 7919 * s + t)
            starts += 1
            for o, i in zip(OUTPUTS, idx):
                for h in hs:
                    y = w.states[t + h, i]
                    if o == "a.p99_ms" and y <= 0:
                        continue
                    smp[(o, h)].append(r[:, h - 1, i])
                    tru[(o, h)].append(y)
    cov, cr, cnt = ({o: {} for o in OUTPUTS} for _ in range(3))
    cov_nz, cr_nz, cnt_nz = ({o: {} for o in NONZERO_OUTPUTS} for _ in range(3))
    for (o, h), xs in smp.items():
        S = np.stack(xs, axis=1) if xs else np.zeros((n, 0))
        Y = np.array(tru[(o, h)], dtype=float)
        key = str(h)
        cnt[o][key] = len(Y)
        cov[o][key], cr[o][key] = (interval_coverage(S, Y), crps(S, Y)) if len(Y) else (None, None)
        if o in NONZERO_OUTPUTS:
            nz = Y > 0
            cnt_nz[o][key] = int(nz.sum())
            cov_nz[o][key], cr_nz[o][key] = ((interval_coverage(S[:, nz], Y[nz]), crps(S[:, nz], Y[nz]))
                                            if nz.any() else (None, None))
    return {"coverage": cov, "crps": cr, "counts": cnt, "coverage_nonzero": cov_nz, "crps_nonzero": cr_nz,
            "counts_nonzero": cnt_nz, "starts": starts, "horizons": hs}


def brier(model, store, mc: pd.DataFrame, seeds, rule, horizon: int, n: int = 64,
          warmup: int = incidents.WARMUP, seed: int = 0) -> dict:
    """Brier score of the rollout risk against Monte Carlo risk labels, and its skill over the labels'
    base rate (a constant forecast of their mean)."""
    eps = store.episodes().set_index("seed")
    lab = mc[mc.episode.isin(list(seeds))].sort_values(["episode", "window"])
    if lab.empty:
        return {"brier": None, "brier_base": None, "brier_skill": None, "n_labels": 0}
    pred = []
    for ep, part in lab.groupby("episode", sort=True):
        rows = _episode_rows(store, int(ep), eps)
        w = features.encode_rows(rows)
        for t in part.window:
            t = int(t)
            hist = features.Window(w.states[:t + 1], w.policies[:t + 1], w.known)
            pred.append(rollout_risk(model, hist, rows.iloc[:t + 1], rule, horizon, n, warmup,
                                     seed=seed + 104729 * int(ep) + t))
    y, p = lab.risk.to_numpy(float), np.array(pred)
    b = float(((p - y) ** 2).mean())
    b0 = float(((y.mean() - y) ** 2).mean())
    return {"brier": b, "brier_base": b0, "brier_skill": (1 - b / b0) if b0 > 0 else None,
            "n_labels": len(y), "base_rate": float(y.mean()), "mean_pred": float(p.mean())}


def _rank_states(seeds, n_states: int, horizon: int, start: int, tag: int = RANK_TAG) -> dict[int, list[int]]:
    """`n_states` (episode, window) pairs among the replayable (R6) seeds, uniform, deterministic."""
    from pis.phase_b.mc_labels import SOURCE
    from pis.phase_b.policies import source_for
    from pis.phase_b.scenarios import scenario_for
    ok = [s for s in seeds if source_for(s) == SOURCE]
    if not ok:
        raise ValueError("no R6-driven seeds to branch")
    cands = []
    for s in ok:
        sc = scenario_for(s)
        T = int(round(sc.duration_s / sc.spec.policy_period_s))
        cands += [(s, t) for t in range(start, T - 1 - horizon)]
    rng = np.random.default_rng(tag)
    pick = rng.choice(len(cands), size=min(n_states, len(cands)), replace=False)
    out: dict[int, list[int]] = {}
    for i in sorted(pick):
        out.setdefault(cands[i][0], []).append(cands[i][1])
    return out


def intervention_agreement(model, seeds, n_states: int = 50, reseeds: int = 8, horizon: int = RANK_HORIZON,
                           n: int = 64, warmup: int = incidents.WARMUP, seed: int = 0) -> dict:
    """Spearman agreement of predicted and real effects (after - before) of the named interventions on
    mean `a.queue_overdue` at `horizon`, over replayed states of R6-driven seeds.

    Real: each policy held for `horizon` windows in `reseeds` branches of the degraded simulator (the same
    reseeds for every policy of a state). Predicted: the mean over `n` sampled rollouts of the policy
    (masked for the CPUs and GPUs lost at that time, as the simulator masks it).
    """
    from pis.candidates import policy_key
    from pis.phase_b.mc_labels import continue_branch, policy_in_force, replay_visit
    from pis.phase_b.scenarios import mask_policy
    from pis.probe import named_interventions
    from pis.runner import spearman
    from sched.randomize import known_params
    j = features.IDX[INTERVENTION_OUTCOME]
    plan = _rank_states(seeds, n_states, horizon, max(model.context, warmup))
    effects: list[dict] = []
    t0 = time.perf_counter()

    def visit(st):
        sc = st.dsim.scenario
        win = features.encode_records(st.records, known_params(sc.spec))
        d, tn = sc.degradation, round(st.dsim.sim.t, 3)
        true_c, pred_c = {}, {}

        def outcome(p):
            key = policy_key(p)
            if key not in true_c:
                true_c[key] = float(np.mean([
                    continue_branch(st, horizon, RANK_TAG + st.seed * 1000 + r, hold=p)[-1]["tel"]["a"][
                        "queue_overdue"] for r in range(reseeds)]))
                masked = mask_policy(p, d.dead_cpus(tn), d.dead_gpus(tn))
                roll = model.sample_rollouts(win, encode_policy(masked), None, horizon, n,
                                             seed=seed + st.seed * 1000 + st.t)
                pred_c[key] = float(features.decode(roll[:, -1])[:, j].mean())
            return true_c[key], pred_c[key]

        for name, (before, after) in named_interventions(policy_in_force(st)).items():
            (tb, pb), (ta, pa) = outcome(before), outcome(after)
            effects.append({"episode": st.seed, "window": st.t, "intervention": name,
                            "true": ta - tb, "pred": pa - pb})

    for s, ts in plan.items():
        replay_visit(s, ts, visit)
    df = pd.DataFrame(effects)
    per = {name: _finite(spearman(g.pred, g.true)) for name, g in df.groupby("intervention")} if len(df) else {}
    return {"spearman": _finite(spearman(df.pred, df.true)) if len(df) else None, "per_intervention": per,
            "states": int(sum(len(v) for v in plan.values())), "effects": len(df), "reseeds": reseeds,
            "horizon": horizon, "seconds": round(time.perf_counter() - t0, 1),
            "rows": df.to_dict("records")}


def _finite(x: float) -> float | None:
    """None for an undefined correlation (constant or too few values)."""
    return None if x is None or not math.isfinite(x) else float(x)


def calibration(model, store, seeds, horizon: int, rule: incidents.IncidentRule | None = None,
                mc: pd.DataFrame | None = None, n: int = 64, every: int = 20, warmup: int = incidents.WARMUP,
                rank_states: int = 50, rank_reseeds: int = 8, seed: int = 0, log=None) -> dict:
    """Predictive coverage and CRPS on the seeds' stored episodes; Brier against `mc` (columns episode,
    window, risk, n; None -> Brier fields None); intervention rank agreement on `rank_states` replayed
    states (0 -> skipped). Seeds: calibration, selection, or history validation/test episodes."""
    seeds = _calibration_seeds(seeds)
    t0 = time.perf_counter()
    out = {"seeds": seeds, "horizon": horizon, "n": n, "level": LEVEL}
    out.update(predictive_scores(model, store, seeds, horizon, n, every, warmup, seed))
    out["seconds_predictive"] = round(time.perf_counter() - t0, 1)
    if log:
        log(f"predictive scores on {out['starts']} starts in {out['seconds_predictive']} s")
    if mc is not None:
        if rule is None:
            raise ValueError("the Brier score needs the incident rule")
        t1 = time.perf_counter()
        out.update(brier(model, store, mc, seeds, rule, horizon, n, warmup, seed))
        out["seconds_brier"] = round(time.perf_counter() - t1, 1)
    else:
        out.update({"brier": None, "brier_base": None, "brier_skill": None, "n_labels": 0})
    out["rank"] = intervention_agreement(model, seeds, rank_states, rank_reseeds, RANK_HORIZON, n, warmup,
                                         seed) if rank_states > 0 else None
    if log and out["rank"]:
        log(f"intervention agreement on {out['rank']['states']} states in {out['rank']['seconds']} s")
    return out


def ladder_pass(cal: dict, lo: float = 0.70, hi: float = 0.90) -> bool | None:
    """Rung passes: 80 % interval coverage at H within [lo, hi] for the three outputs (on the windows with a
    non-zero truth for NONZERO_OUTPUTS, on windows with completions for p99) and Brier skill > 0. None when
    the coverage passes but the Brier skill is unavailable."""
    h = str(cal["horizon"])

    def judged(o):
        return (cal["coverage_nonzero"] if o in NONZERO_OUTPUTS else cal["coverage"])[o].get(h)
    cov_ok = all(judged(o) is not None and lo <= judged(o) <= hi for o in OUTPUTS)
    if cal.get("brier_skill") is None:
        return None if cov_ok else False
    return bool(cov_ok and cal["brier_skill"] > 0)
