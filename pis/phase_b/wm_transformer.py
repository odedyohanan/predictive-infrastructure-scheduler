"""Causal transformer world model: a Gaussian over the next window's state change from the last `context` windows.

At position j the network reads the same 79-number input as the probabilistic GRU (`pis.phase_b.features`:
state j, the policy of window j+1, the known parameters), standardised, projected to `d_model` numbers and
added to a learned embedding of the position in the window (0 .. context - 1). A stack of pre-norm blocks
(causal multi-head self-attention, then a ReLU feed-forward layer, each with a residual connection) lets
every position see itself and the positions before it, never after; a linear head gives the mean and
log-variance of the standardised change of the 43 state numbers, clamped like the GRU's.

Training goes through `wm_prob.fit_loop` with the masking, normalisation and early stopping of the GRU model:
sequences of exactly `context` steps (no room: the attention window is the sequence) that overlap by
`burn_in` steps, the first `burn_in` positions carrying no loss except in the sequence that starts the
episode. So the network has learnt to predict after anything from 1 to `context` windows, the positions
counted from the first window it sees.

Sampling (`TransformerWorldModel.sample_rollouts`) is autoregressive over a sliding window: the last
`context` inputs (real history, then sampled states) are run through the causal stack, a change is drawn from
the Gaussian at the last position, the new state is clipped to the training range and appended, and the
window slides by one once it is full. A sliding window changes what every position sees, so nothing is
cached between steps; instead the history inputs still in the window, the same for every future, run once
per step (`CausalTransformer.last_shared`) and the n sampled continuations run as one batch, and the last
block computes the last position only. All noise comes from one numpy stream per call, so futures are a
function of the seed.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pis.phase_b import features
from pis.phase_b.wm_prob import (BURN_IN, CONTEXT, LOGVAR_MAX, LOGVAR_MIN, WorldModelBase, fit_history, fit_loop,
                                 gaussian_nll)

D_MODEL = 64
HEADS = 4
FF = 128
LAYERS = 4


# ------------------------------------------------------------------ network
class _Block(nn.Module):
    """Pre-norm transformer block: causal self-attention and a ReLU feed-forward layer, both residual."""

    def __init__(self, d_model: int, heads: int, ff: int, dropout: float):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be a multiple of heads")
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(), nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)

    def _qkv(self, h: torch.Tensor):
        B, L, D = h.shape
        return self.qkv(self.norm1(h)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)

    def _rest(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        h = h + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return h + self.drop(self.ff(self.norm2(h)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q, k, v = self._qkv(h)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self._rest(h, a)

    def forward_shared(self, hp: torch.Tensor, hs: torch.Tensor,
                       last: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Inference on a prefix hp [1, P, D] shared by every row and a suffix hs [n, Q, D] that follows it:
        the same as `forward` on the prefix repeated n times joined with the suffix, the prefix run once.
        `last`: only the suffix's last position is returned ([n, 1, D]) and the prefix output is skipped (None)."""
        n, Q = hs.shape[:2]
        P = hp.shape[1]
        qs, ks, vs = self._qkv(hs)
        hp_out = None
        if P:
            qp, kp, vp = self._qkv(hp)
            if not last:
                hp_out = self._rest(hp, F.scaled_dot_product_attention(qp, kp, vp, is_causal=True))
            ks = torch.cat([kp.expand(n, -1, -1, -1), ks], dim=2)
            vs = torch.cat([vp.expand(n, -1, -1, -1), vs], dim=2)
        if last:                                         # the last position sees everything
            return hp_out, self._rest(hs[:, -1:], F.scaled_dot_product_attention(qs[:, :, -1:], ks, vs))
        if not P:
            return hp, self._rest(hs, F.scaled_dot_product_attention(qs, ks, vs, is_causal=True))
        mask = torch.arange(P + Q)[None, :] <= (P + torch.arange(Q))[:, None]      # suffix i sees 0 .. P + i
        return hp_out, self._rest(hs, F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask))


class CausalTransformer(nn.Module):
    def __init__(self, in_dim: int = features.N_INPUT, out_dim: int = features.N_STATE, d_model: int = D_MODEL,
                 heads: int = HEADS, ff: int = FF, layers: int = LAYERS, context: int = CONTEXT,
                 dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(context, d_model) * 0.02)
        self.blocks = nn.ModuleList(_Block(d_model, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2 * out_dim)
        self.in_dim, self.out_dim, self.d_model, self.heads, self.ff, self.layers, self.context, self.dropout = (
            in_dim, out_dim, d_model, heads, ff, layers, context, dropout)

    def params(self) -> dict:
        return dict(in_dim=self.in_dim, out_dim=self.out_dim, d_model=self.d_model, heads=self.heads, ff=self.ff,
                    layers=self.layers, context=self.context, dropout=self.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, L, in] (standardised, L <= context, position 0 the first window seen) -> mean, log-variance
        [B, L, out] of the standardised change; position j depends on positions 0..j only."""
        L = x.shape[1]
        if L > self.context:
            raise ValueError(f"{L} positions exceed the context of {self.context}")
        h = self.inp(x) + self.pos[:L]
        for b in self.blocks:
            h = b(h)
        return self._out(h)

    def _out(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.head(self.norm(h)).chunk(2, dim=-1)
        return mean, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def last_shared(self, prefix: torch.Tensor, suffix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log-variance [n, out] at the last position of the sequences made of a prefix [P, in] shared
        by every row and a per-row suffix [n, Q, in] (standardised, P + Q <= context, Q >= 1); the prefix is
        run once instead of n times."""
        P, Q = prefix.shape[0], suffix.shape[1]
        if P + Q > self.context:
            raise ValueError(f"{P + Q} positions exceed the context of {self.context}")
        hp = self.inp(prefix[None]) + self.pos[:P]
        hs = self.inp(suffix) + self.pos[P:P + Q]
        for i, b in enumerate(self.blocks):
            hp, hs = b.forward_shared(hp, hs, last=i == len(self.blocks) - 1)
        return self._out(hs[:, -1])


# ------------------------------------------------------------------ the model
class TransformerWorldModel(WorldModelBase):
    """The causal transformer: a Gaussian over the next change from the last `context` windows."""
    KIND = "transformer"
    NET = CausalTransformer

    @property
    def context(self) -> int:
        return min(int(self.meta.get("context", CONTEXT)), self.net.context)

    def _last(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standardised mean and log-variance at the last position of standardised inputs [N, L, in]."""
        mean, logvar = self.net(torch.from_numpy(np.ascontiguousarray(xs, dtype=np.float32)))
        return mean[:, -1].numpy(), logvar[:, -1].numpy()

    @torch.inference_mode()
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of the next change, in encoded units, for input windows [N, L, in] (only the last
        `context` positions are seen)."""
        x = np.asarray(x, dtype=np.float32)[:, -self.context:]
        mean, logvar = self._last(self.x_norm.transform(x))
        std = self.y_norm.std.astype(np.float64)
        return mean * std + self.y_norm.mean, np.exp(logvar) * std ** 2

    @torch.inference_mode()
    def _sample(self, history: features.Window, pols: np.ndarray, known, horizon: int,
                seed: int | None) -> np.ndarray:
        """Ancestral sampling of one future per row of `pols` [R, horizon, 33] over a sliding window."""
        k = features.known_vector(known if known is not None else history.known)
        n = pols.shape[0]
        C = self.context
        S, P = history.states[-C:], history.policies[-C:]
        past = np.concatenate([S[:-1], P[1:], np.broadcast_to(k, (len(S) - 1, len(k)))], axis=1)
        # the window is the history inputs still in it (the same for every row, run once) and the sampled ones
        past = torch.from_numpy(np.ascontiguousarray(self.x_norm.transform(past), dtype=np.float32))
        new = torch.empty((n, min(horizon, C), self.net.in_dim))
        q = 0                                                # sampled inputs in the window
        s = np.broadcast_to(S[-1], (n, S.shape[1])).astype(np.float32)
        lo, hi = self._clip_bounds(S.shape[1])
        rng = np.random.default_rng(seed)
        std, mu = self.y_norm.std, self.y_norm.mean
        out = np.empty((n, horizon, S.shape[1]), dtype=np.float32)
        kk = np.broadcast_to(k, (n, len(k)))
        for h in range(horizon):
            if q == new.shape[1]:                            # slide: drop the oldest sampled input
                new[:, :-1] = new[:, 1:].clone()
                q -= 1
            new[:, q] = torch.from_numpy(self.x_norm.transform(np.concatenate([s, pols[:, h], kk], axis=1)))
            q += 1
            prefix = past[max(len(past) - (C - q), 0):]
            mean, logvar = self.net.last_shared(prefix, new[:, :q])
            mean, logvar = mean.numpy(), logvar.numpy()
            delta = (mean + np.exp(0.5 * logvar) * rng.standard_normal(mean.shape)) * std + mu
            s = np.maximum(s + delta.astype(np.float32), lo)
            if hi is not None:
                s = np.minimum(s, hi)
            out[:, h] = s
        return out


# ------------------------------------------------------------------ training
def _loss(net, x, y, m, keep, training):
    mean, logvar = net(x)
    return gaussian_nll(mean, logvar, y, m * keep)


def train(train_data: list, val_data: list, context: int = CONTEXT, burn_in: int = BURN_IN, room: int = 0,
          d_model: int = D_MODEL, heads: int = HEADS, ff: int = FF, layers: int = LAYERS, dropout: float = 0.0,
          lr: float = 1e-3, batch: int = 256, epochs: int = 300, patience: int = 20, seed: int = 0, log=None,
          max_seconds: float | None = None) -> tuple[TransformerWorldModel, list[dict]]:
    """Fit on per-episode (X [L, in], Y [L, out], M [L, out]) arrays; returns the best-validation model and
    the per-epoch history {epoch, train_loss, val_loss, seconds}. The attention window is the training
    sequence, so `room` must be 0 (rollouts slide the window instead)."""
    if room != 0:
        raise ValueError("the transformer trains on sequences of exactly `context` steps: room must be 0")
    if not 0.0 <= dropout < 1.0 or not math.isfinite(dropout):
        raise ValueError("dropout must lie in [0, 1)")
    net, xn, yn, best, hist = fit_loop(lambda i, o: CausalTransformer(i, o, d_model, heads, ff, layers, context,
                                                                      dropout),
                                       _loss, train_data, val_data, context, burn_in, 0, lr, batch, epochs,
                                       patience, seed, log, max_seconds)
    meta = {"kind": TransformerWorldModel.KIND, "context": context, "room": 0, "burn_in": burn_in,
            "best_val_loss": best, "epochs_run": len(hist), "d_model": d_model, "heads": heads, "ff": ff,
            "layers": layers, "dropout": dropout, "lr": lr, "batch": batch, "seed": seed}
    return TransformerWorldModel(net, xn, yn, meta), hist


def fit(store, train_seeds, val_seeds, **recipe) -> tuple[TransformerWorldModel, list[dict]]:
    """Train on history episodes; calibration, selection and held-out seeds are refused, as is overlap."""
    return fit_history(train, store, train_seeds, val_seeds, **recipe)
