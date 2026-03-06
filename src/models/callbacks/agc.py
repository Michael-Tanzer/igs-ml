"""Adaptive Gradient Clipping (AGC) Lightning callback.

NFNet models are Normalizer-Free and designed specifically for SGD+AGC
(not Adam). Without AGC, Adam's moment estimates cause activation variance
to grow unboundedly with weight-standardized (ScaledStdConv) layers,
leading to NaN loss after a few hundred steps.

AGC clips each parameter's gradient so that:
    ||grad|| / ||weight|| <= clip_factor

This is the same clip used in the original NFNet paper and timm's training
scripts. It replaces (or supplements) the global norm clip in the trainer.

Reference: Brock et al. 2021, "High-Performance Large-Scale Image Recognition
Without Normalization" (https://arxiv.org/abs/2102.06171)

Usage in config::

    callbacks:
      agc:
        _target_: src.models.callbacks.agc.AGCCallback
        clip_factor: 0.01    # standard NFNet value
        eps: 1e-3
"""

import lightning as L
import torch
from timm.utils import adaptive_clip_grad


class AGCCallback(L.Callback):
    """Apply Adaptive Gradient Clipping before each optimizer step.

    Clips per-parameter gradient norms relative to parameter norms.
    Skips bias and 1-D parameters (BN/LN scale+bias) which should not
    be clipped — consistent with the original AGC implementation.

    Args:
        clip_factor: Max allowed ratio ||grad|| / ||weight||. NFNet paper
            uses 0.01. Larger values = less aggressive clipping.
        eps: Minimum parameter norm to avoid dividing by zero.
    """

    def __init__(self, clip_factor: float = 0.01, eps: float = 1e-3):
        super().__init__()
        self.clip_factor = clip_factor
        self.eps = eps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Collect parameters to clip: skip biases and 1-D params (norm layers)
        params = [
            p for p in pl_module.parameters()
            if p.grad is not None and p.ndim > 1
        ]
        adaptive_clip_grad(params, clip_factor=self.clip_factor, eps=self.eps)
