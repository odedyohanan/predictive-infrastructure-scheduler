"""The sampled world models by kind: loading a checkpoint whatever its kind, and the trainer of each kind.

A checkpoint records its kind ("pgru": `wm_prob.ProbWorldModel`, "rssm": `wm_rssm.RSSMWorldModel`,
"transformer": `wm_transformer.TransformerWorldModel`); checkpoints written before kinds were recorded are
probabilistic GRUs. Every kind offers the same interface (`context`, `room`, `bounds`, `meta`, `predict`,
`sample_rollouts`, `sample_rollouts_batch`, `save`), so predictors, calibration and the advisor take any of
them.
"""
from __future__ import annotations

import importlib

import torch

ARCHS = ("pgru", "rssm", "transformer")
_CLASSES = {"pgru": ("pis.phase_b.wm_prob", "ProbWorldModel"), "rssm": ("pis.phase_b.wm_rssm", "RSSMWorldModel"),
            "transformer": ("pis.phase_b.wm_transformer", "TransformerWorldModel")}


def _module(arch: str):
    if arch not in _CLASSES:
        raise ValueError(f"unknown world-model kind {arch!r}; expected one of {list(ARCHS)}")
    return importlib.import_module(_CLASSES[arch][0])


def model_class(arch: str):
    return getattr(_module(arch), _CLASSES[arch][1])


def kind_of(path) -> str:
    from pis.phase_b.wm_prob import checkpoint_kind
    return checkpoint_kind(torch.load(path, weights_only=False))


def load_world_model(path):
    """The world model saved at `path`, of the class its recorded kind names."""
    from pis.phase_b.wm_prob import checkpoint_kind
    ck = torch.load(path, weights_only=False)
    return model_class(checkpoint_kind(ck)).from_checkpoint(ck, path)


def trainer(arch: str):
    """`fit(store, train_seeds, val_seeds, **recipe)` of the kind."""
    return _module(arch).fit
