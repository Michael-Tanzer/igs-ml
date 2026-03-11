"""Callback that calibrates the optimal logit threshold after each epoch.

Accumulates raw (unshifted) logits during training or validation, then finds
the threshold that optimises a configurable criterion and writes it into the
model's ``threshold_logit`` buffer.  Because the model always shifts its output
by this buffer, all downstream metrics (step-level and epoch-level)
automatically operate at the optimal decision boundary.
"""

from typing import List

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class ThresholdCalibrator(Callback):
    """Finds the optimal logit threshold from training or validation data each epoch.

    After each epoch, concatenates accumulated raw logits and targets,
    runs a grid search over every unique logit value, and writes the threshold
    that optimises ``criterion`` into the model's ``threshold_logit`` buffer
    via ``set_threshold()``.

    The callback reads ``batch.raw_logits`` (unshifted logits stored by
    ``DataObjectModelWrapper``) so that threshold search always operates in
    raw logit space, regardless of any shift already applied to ``batch.output``.

    Args:
        criterion: A torchmetrics ``Metric`` (or any callable taking
            ``(preds, targets) -> scalar``) used to score each candidate
            threshold.  Since predictions are pre-binarised to ``{0, 1}``
            before being passed, set ``threshold=0.5`` on binary metrics
            to make their internal thresholding a no-op.
        mode: ``"max"`` to find the threshold that maximises the criterion,
            ``"min"`` to minimise it.  Same convention as
            ``EarlyStopping`` / ``ModelCheckpoint``.
        calibrate_on: ``"train"`` to calibrate on training data (default),
            ``"val"`` to calibrate on validation data.  Validation-based
            calibration is more robust when the model overfits, since the
            threshold is tuned to a distribution closer to the test set.
    """

    def __init__(self, criterion, mode: str = "max", calibrate_on: str = "train"):
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        if calibrate_on not in ("train", "val"):
            raise ValueError(f"calibrate_on must be 'train' or 'val', got '{calibrate_on}'")
        self.criterion = criterion
        self.mode = mode
        self.calibrate_on = calibrate_on
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def _is_better(self, score: float, best: float) -> bool:
        if self.mode == "max":
            return score > best
        return score < best

    def _accumulate(self, batch):
        """Accumulate raw logits and targets from a batch."""
        raw_logits = getattr(batch, "raw_logits", None)
        if raw_logits is None or raw_logits.numel() == 0:
            return
        self._preds.append(raw_logits.detach().cpu().float())
        self._targets.append(batch.target.detach().cpu().long())

    def _calibrate(self, trainer: Trainer, pl_module: LightningModule):
        """Compute optimal threshold and write it into the model buffer."""
        if not self._preds:
            return

        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)

        unique_vals, _ = torch.sort(torch.unique(preds))
        if len(unique_vals) == 1:
            threshold, best_score = unique_vals[0].item(), 0.0
        else:
            best_score = float("inf") if self.mode == "min" else float("-inf")
            threshold = unique_vals[0].item()
            for t in unique_vals:
                binary = (preds > t).int()
                score = self.criterion(binary, targets)
                self.criterion.reset()
                score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                if score_val != score_val:  # skip NaN
                    continue
                if self._is_better(score_val, best_score):
                    best_score = score_val
                    threshold = t.item()

        pl_module.net.set_threshold(threshold)
        pl_module.log(f"{self.calibrate_on}/opt_threshold_logit", threshold, prog_bar=False)
        pl_module.log(f"{self.calibrate_on}/opt_criterion_score", best_score, prog_bar=False)

        self._preds.clear()
        self._targets.clear()

    # -- Train hooks --

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        if self.calibrate_on == "train":
            self._accumulate(batch)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.calibrate_on == "train":
            self._calibrate(trainer, pl_module)

    # -- Validation hooks --

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if self.calibrate_on == "val":
            self._accumulate(batch)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.calibrate_on == "val":
            self._calibrate(trainer, pl_module)
