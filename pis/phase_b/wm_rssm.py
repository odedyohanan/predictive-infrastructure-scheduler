"""Recurrent state-space world model (RSSM): a deterministic GRU state plus a Gaussian latent per step.

At position j the network reads the same 79-number input as the probabilistic GRU (`pis.phase_b.features`:
state j, the policy of window j+1, the known parameters) and the previous latent z_{j-1}:

    h_j = GRU(h_{j-1}, embed(x_j, z_{j-1}))
    prior      p(z_j | h_j)              diagonal Gaussian
    posterior  q(z_j | h_j, y_j)         diagonal Gaussian; y_j = standardised change state j+1 - state j
                                         (masked numbers zeroed, the mask given alongside)
    decoder    p(y_j | h_j, z_j)         Gaussian mean and log-variance of the change

Training maximises the evidence lower bound with the masking, normalisation, sequence windowing, burn-in
and early stopping of `wm_prob` (`wm_prob.fit_loop`): the masked Gaussian NLL of the change under the
decoder (z from the posterior, reparameterised) plus KL(q || p) summed over latent numbers. The KL is
averaged over the kept positions and floored at `free_bits` nats ("free bits": below the floor it costs a
constant, so the posterior may carry that much information about the change without penalty); the
validation loss, which drives early stopping, is the plain negative ELBO without the floor. Both are per
target number (the KL divided by the 43 state numbers), on the scale of the GRU's NLL.

Sampling (`RSSMWorldModel.sample_rollouts`): the last `context` windows of history are filtered through
the posterior (their changes are known), the first future step's latent comes from the prior, and each step
then draws a latent from the prior and a change from the decoder, clips the new state to the training
range and feeds it back. Beyond `room` steps the last `context` positions are filtered again from a fresh
state, the sampled changes playing the observed ones, as the GRU model restarts. All noise comes from one
numpy stream per call, so futures are a function of the seed.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pis.phase_b import features
from pis.phase_b.wm_prob import (BURN_IN, CONTEXT, LOGVAR_MAX, LOGVAR_MIN, ROOM, WorldModelBase, fit_history,
                                 fit_loop, gaussian_nll)

LATENT = 16
FREE_BITS = 1.0
MIN_STD = 0.1


def _mlp(i: int, hidden: int, o: int, layers: int) -> nn.Sequential:
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ELU()]
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, o))


def _mask(states: np.ndarray) -> np.ndarray:
    """Target mask of the changes into `states` (all ones for a state vector other than the 43 numbers)."""
    if states.shape[-1] != features.N_STATE:
        return np.ones(states.shape, dtype=np.float32)
    return features.target_mask(states)


# ------------------------------------------------------------------ network
class RSSM(nn.Module):
    def __init__(self, in_dim: int = features.N_INPUT, out_dim: int = features.N_STATE, hidden: int = 128,
                 latent: int = LATENT, layers: int = 1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim + latent, hidden), nn.ELU())
        self.cell = nn.GRUCell(hidden, hidden)
        self.prior_net = _mlp(hidden, hidden, 2 * latent, layers)
        self.post_net = _mlp(hidden + 2 * out_dim, hidden, 2 * latent, layers)
        self.dec = _mlp(hidden + latent, hidden, 2 * out_dim, layers)
        self.in_dim, self.out_dim, self.hidden, self.latent, self.layers = in_dim, out_dim, hidden, latent, layers

    def params(self) -> dict:
        return dict(in_dim=self.in_dim, out_dim=self.out_dim, hidden=self.hidden, latent=self.latent,
                    layers=self.layers)

    def initial(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(batch, self.hidden), torch.zeros(batch, self.latent)

    def step(self, h: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Deterministic state after input x (standardised) given the previous state and latent."""
        return self.cell(self.embed(torch.cat([x, z], dim=-1)), h)

    @staticmethod
    def _gauss(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, s = raw.chunk(2, dim=-1)
        return mu, F.softplus(s) + MIN_STD

    def prior(self, h):
        return self._gauss(self.prior_net(h))

    def posterior(self, h, y, m):
        return self._gauss(self.post_net(torch.cat([h, y * m, m], dim=-1)))

    def decode(self, h, z) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.dec(torch.cat([h, z], dim=-1)).chunk(2, dim=-1)
        return mean, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def forward(self, x, y, m, generator: torch.Generator | None = None):
        """Teacher-forced pass over x, y, m [B, L, .] (standardised), latents from the posterior ->
        decoder mean, log-variance [B, L, out] and KL(q || p) [B, L] (summed over latent numbers)."""
        B, L = x.shape[:2]
        h, z = self.initial(B)
        means, logvars, kls = [], [], []
        for t in range(L):
            h = self.step(h, z, x[:, t])
            pm, ps = self.prior(h)
            qm, qs = self.posterior(h, y[:, t], m[:, t])
            z = qm + qs * torch.randn(qm.shape, generator=generator)
            mean, logvar = self.decode(h, z)
            means.append(mean)
            logvars.append(logvar)
            kls.append((torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5).sum(-1))
        return torch.stack(means, 1), torch.stack(logvars, 1), torch.stack(kls, 1)


def elbo_terms(net: RSSM, x, y, m, keep, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked Gaussian NLL per target number and the KL per kept position."""
    mean, logvar, kl = net(x, y, m, generator)
    nll = gaussian_nll(mean, logvar, y, m * keep)
    k = keep[..., 0]
    return nll, (kl * k).sum() / k.sum().clamp(min=1.0)


def objective(nll, kl, free_bits: float, out_dim: int) -> torch.Tensor:
    """Training loss per target number: NLL + max(KL, free_bits) / out_dim."""
    return nll + torch.clamp(kl, min=free_bits) / out_dim


# ------------------------------------------------------------------ the model
class RSSMWorldModel(WorldModelBase):
    KIND = "rssm"
    NET = RSSM

    def _filter(self, xs: list, sts: list, rng: np.random.Generator):
        """Deterministic state and latent after the positions `xs` (standardised inputs [n, in]); the change
        after every position but the last is known from the input states `sts` ([n, 43], raw) and its latent
        drawn from the posterior; the last position's latent is drawn from the prior. `rng` None -> means."""
        net = self.net
        h, z = net.initial(xs[0].shape[0])
        for j, x in enumerate(xs):
            h = net.step(h, z, torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
            if j < len(xs) - 1:
                nxt = sts[j + 1]
                y = self.y_norm.transform(nxt - sts[j]).astype(np.float32)
                mu, sd = net.posterior(h, torch.from_numpy(y), torch.from_numpy(_mask(nxt)))
            else:
                mu, sd = net.prior(h)
            z = mu if rng is None else mu + sd * torch.from_numpy(rng.standard_normal(mu.shape).astype(np.float32))
        return h, z

    @torch.inference_mode()
    def predict(self, x: np.ndarray, samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of the next change, in encoded units, for input windows [N, L, in]: the earlier
        positions filtered with posterior means, the variance that of the prior mixture (latent spread plus
        decoder variance, `samples` latent draws of a fixed stream)."""
        x = np.asarray(x, dtype=np.float32)
        xs = [self.x_norm.transform(x[:, j]) for j in range(x.shape[1])]
        sts = [x[:, j, :self.net.out_dim] for j in range(x.shape[1])]
        h, _ = self._filter(xs, sts, None)
        pm, ps = self.net.prior(h)
        rng = np.random.default_rng(0)
        eps = torch.from_numpy(rng.standard_normal((samples,) + tuple(pm.shape)).astype(np.float32))
        z = pm[None] + ps[None] * eps
        mean, logvar = self.net.decode(h[None].expand(samples, *h.shape), z)
        mu = mean.mean(0).double().numpy()
        var = (logvar.exp().mean(0) + mean.var(0, unbiased=False)).double().numpy()
        std = self.y_norm.std.astype(np.float64)
        return mu * std + self.y_norm.mean, var * std ** 2

    @torch.inference_mode()
    def _sample(self, history: features.Window, pols: np.ndarray, known, horizon: int,
                seed: int | None) -> np.ndarray:
        """Ancestral sampling of one future per row of `pols` [R, horizon, 33]."""
        k = features.known_vector(known if known is not None else history.known)
        n = pols.shape[0]
        C, longest = self.context, self.context + self.room
        S, P = history.states[-C:], history.policies[-C:]
        kk = np.broadcast_to(k, (n, len(k)))
        xs, sts = [], []                    # per position: standardised input [n, 79], input state [n, 43]
        for j in range(len(S) - 1):
            st = np.broadcast_to(S[j], (n, S.shape[1])).astype(np.float32)
            xs.append(self.x_norm.transform(np.concatenate([st, np.broadcast_to(P[j + 1], (n, P.shape[1])), kk],
                                                           axis=1)))
            sts.append(st)
        s = np.broadcast_to(S[-1], (n, S.shape[1])).astype(np.float32)
        lo, hi = self._clip_bounds(S.shape[1])
        rng = np.random.default_rng(seed)
        std, mu = self.y_norm.std, self.y_norm.mean
        out = np.empty((n, horizon, S.shape[1]), dtype=np.float32)
        h = z = None
        used = 0
        for step in range(horizon):
            x = self.x_norm.transform(np.concatenate([s, pols[:, step], kk], axis=1))
            xs.append(x)
            sts.append(s)
            if h is None or used >= longest:          # (re)start: filter the last `context` positions
                xs, sts = xs[-C:], sts[-C:]
                h, z = self._filter(xs, sts, rng)
                used = len(xs)
            else:
                h = self.net.step(h, z, torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
                pm, ps = self.net.prior(h)
                z = pm + ps * torch.from_numpy(rng.standard_normal(pm.shape).astype(np.float32))
                used += 1
            mean, logvar = self.net.decode(h, z)
            mean, logvar = mean.numpy(), logvar.numpy()
            delta = (mean + np.exp(0.5 * logvar) * rng.standard_normal(mean.shape)) * std + mu
            s = np.maximum(s + delta.astype(np.float32), lo)
            if hi is not None:
                s = np.minimum(s, hi)
            out[:, step] = s
        return out


# ------------------------------------------------------------------ training
def train(train_data: list, val_data: list, context: int = CONTEXT, burn_in: int = BURN_IN, room: int = ROOM,
          hidden: int = 128, latent: int = LATENT, layers: int = 1, free_bits: float = FREE_BITS,
          lr: float = 1e-3, batch: int = 256, epochs: int = 300, patience: int = 20, seed: int = 0, log=None,
          max_seconds: float | None = None) -> tuple[RSSMWorldModel, list[dict]]:
    """Fit on per-episode (X [L, in], Y [L, out], M [L, out]) arrays; returns the best-validation model and
    the per-epoch history {epoch, train_loss, val_loss, seconds}."""

    def loss(net, x, y, m, keep, training):
        # validation draws its posterior noise from a fixed stream: the same weights give the same loss
        gen = None if training else torch.Generator().manual_seed(seed)
        nll, kl = elbo_terms(net, x, y, m, keep, gen)
        return objective(nll, kl, free_bits, net.out_dim) if training else nll + kl / net.out_dim

    net, xn, yn, best, hist = fit_loop(lambda i, o: RSSM(i, o, hidden, latent, layers), loss, train_data,
                                       val_data, context, burn_in, room, lr, batch, epochs, patience, seed, log,
                                       max_seconds)
    meta = {"kind": RSSMWorldModel.KIND, "context": context, "room": room, "burn_in": burn_in,
            "best_val_loss": best, "epochs_run": len(hist), "hidden": hidden, "latent": latent, "layers": layers,
            "free_bits": free_bits, "lr": lr, "batch": batch, "seed": seed}
    return RSSMWorldModel(net, xn, yn, meta), hist


def fit(store, train_seeds, val_seeds, **recipe) -> tuple[RSSMWorldModel, list[dict]]:
    """Train on history episodes; calibration, selection and held-out seeds are refused, as is overlap."""
    return fit_history(train, store, train_seeds, val_seeds, **recipe)
