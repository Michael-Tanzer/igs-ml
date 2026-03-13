"""Callback that jointly optimises classifier threshold C and count threshold T.

Finds the (C, T) pair that minimises the Limit of Detection (LoD) subject to
a target patient-level specificity constraint.  C is written into the model's
``threshold_logit`` buffer (same as ``ThresholdCalibrator``), and T is written
into the model's ``count_threshold`` buffer.

**Do not use together with ThresholdCalibrator** — both write to
``threshold_logit``.
"""

from collections import defaultdict
from typing import List

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class DualThresholdCalibrator(Callback):
    """Joint (C, T) optimisation minimising LoD at a target specificity.

    Algorithm
    ---------
    For each unique raw logit value C:
        1. Binarise all patches at C.
        2. Count positive-predicted patches per patient.
        3. For negative patients, find the smallest count threshold T such that
           the fraction of negative patients with positive count < T is >= K
           (the target specificity).
        4. For positive patients, compute patient-level sensitivity using T
           (a patient is "detected" if its positive patch count >= T).
        5. Compute LoD = (3.3 * sigma_F + T) / mu_S.
    Select the (C, T) pair that minimises LoD.

    Args:
        target_specificity: Desired patient-level specificity (K). Default 0.95.
        calibrate_on: ``"train"`` or ``"val"``.  Default ``"val"``.
    """

    def __init__(
        self,
        target_specificity: float = 0.95,
        calibrate_on: str = "val",
        ema_alpha: float = 0.3,
    ):
        super().__init__()
        if calibrate_on not in ("train", "val"):
            raise ValueError(f"calibrate_on must be 'train' or 'val', got '{calibrate_on}'")
        if not 0 < target_specificity < 1:
            raise ValueError(f"target_specificity must be in (0, 1), got {target_specificity}")
        self.target_specificity = target_specificity
        self.calibrate_on = calibrate_on
        self.ema_alpha = ema_alpha
        self._ema_c: float | None = None
        self._ema_t: float | None = None
        self._best_lod: float = float("inf")
        self._best_c: float | None = None
        self._best_t: int | None = None

        self._raw_logits: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._patient_ids: List[List[str]] = []

    # ── Accumulation ──────────────────────────────────────────────────────

    def _accumulate(self, batch):
        raw_logits = getattr(batch, "raw_logits", None)
        if raw_logits is None or raw_logits.numel() == 0:
            return
        if batch.malaria is None:
            return
        self._raw_logits.append(raw_logits.detach().cpu().float())
        self._targets.append(batch.target.detach().cpu().float())
        pids = batch.malaria.patient_id
        if not isinstance(pids, (list, tuple)):
            pids = [pids]
        self._patient_ids.append(list(pids))

    # ── Calibration ───────────────────────────────────────────────────────

    def _calibrate(self, trainer: Trainer, pl_module: LightningModule):
        if not self._raw_logits:
            return

        raw_logits = torch.cat(self._raw_logits)
        targets = torch.cat(self._targets)
        all_pids: List[str] = []
        for pid_batch in self._patient_ids:
            all_pids.extend(pid_batch)

        # Group by PID
        pid_to_indices: dict[str, List[int]] = defaultdict(list)
        for i, pid in enumerate(all_pids):
            pid_to_indices[str(pid)].append(i)

        # Classify patients as positive/negative by ground truth
        positive_pids: List[str] = []
        negative_pids: List[str] = []
        for pid, indices in pid_to_indices.items():
            idx = torch.tensor(indices, dtype=torch.long)
            if targets[idx].sum() > 0:
                positive_pids.append(pid)
            else:
                negative_pids.append(pid)

        if not positive_pids or not negative_pids:
            self._clear()
            return

        # Exhaustive search over every unique logit value (finds true optimum)
        candidates, _ = torch.sort(torch.unique(raw_logits))

        best_lod = float("inf")
        best_c = candidates[0].item()
        best_t = 1
        best_spec = 0.0
        best_sens = 0.0

        # Pre-compute index tensors
        neg_idx = {pid: torch.tensor(pid_to_indices[pid], dtype=torch.long) for pid in negative_pids}
        pos_idx_map = {pid: torch.tensor(pid_to_indices[pid], dtype=torch.long) for pid in positive_pids}

        for c in candidates:
            binary = (raw_logits > c).float()

            # Count positive-predicted patches per patient
            neg_counts: List[int] = []
            for pid in negative_pids:
                neg_counts.append(int((binary[neg_idx[pid]] == 1).sum().item()))

            # Find T: smallest integer such that
            # (fraction of neg patients with count < T) >= target_specificity
            sorted_counts = sorted(neg_counts)
            n_neg = len(sorted_counts)
            t = 1
            for candidate_t in range(1, max(sorted_counts) + 2 if sorted_counts else 2):
                n_below = sum(1 for c_val in sorted_counts if c_val < candidate_t)
                if n_below / n_neg >= self.target_specificity:
                    t = candidate_t
                    break
            else:
                # Could not achieve target specificity at any T
                t = max(sorted_counts) + 1 if sorted_counts else 1

            achieved_spec = sum(1 for c_val in neg_counts if c_val < t) / n_neg

            # Patient-level sensitivity: fraction of positive patients with count >= T
            sensitivities: List[float] = []
            n_detected = 0
            for pid in positive_pids:
                p_binary = binary[pos_idx_map[pid]]
                pos_count = int((p_binary == 1).sum().item())
                if pos_count >= t:
                    n_detected += 1
                    p_targets = targets[pos_idx_map[pid]]
                    tp = ((p_binary == 1) & (p_targets == 1)).sum().item()
                    fn = ((p_binary == 0) & (p_targets == 1)).sum().item()
                    denom = tp + fn
                    sensitivities.append(tp / denom if denom > 0 else 0.0)

            if not sensitivities:
                continue

            mu_s = sum(sensitivities) / len(sensitivities)
            sigma_f = (
                torch.tensor(neg_counts, dtype=torch.float)
                .std(correction=1).item()
                if len(neg_counts) > 1 else 0.0
            )
            patient_sens = n_detected / len(positive_pids)

            lod = (3.3 * sigma_f + t) / max(mu_s, 1e-6)

            if lod < best_lod:
                best_lod = lod
                best_c = c.item()
                best_t = t
                best_spec = achieved_spec
                best_sens = patient_sens

        # EMA smoothing to reduce epoch-to-epoch volatility
        alpha = self.ema_alpha
        if self._ema_c is None:
            self._ema_c = best_c
            self._ema_t = float(best_t)
        else:
            self._ema_c = alpha * best_c + (1 - alpha) * self._ema_c
            self._ema_t = alpha * float(best_t) + (1 - alpha) * self._ema_t

        smoothed_c = self._ema_c
        smoothed_t = max(1, round(self._ema_t))

        # Track best LoD across all epochs
        if best_lod < self._best_lod:
            self._best_lod = best_lod
            self._best_c = best_c
            self._best_t = best_t

        # Write thresholds (use smoothed values for stability)
        pl_module.net.set_threshold(smoothed_c)
        if hasattr(pl_module.net, "set_count_threshold"):
            pl_module.net.set_count_threshold(float(smoothed_t))

        # Update count_threshold on epoch metrics
        for metric in pl_module.epoch_metrics:
            if hasattr(metric, "count_threshold"):
                metric.count_threshold = smoothed_t

        # Log
        stage = self.calibrate_on
        pl_module.log(f"{stage}/opt_C", smoothed_c, prog_bar=False)
        pl_module.log(f"{stage}/opt_T", float(smoothed_t), prog_bar=False)
        pl_module.log(f"{stage}/opt_LoD", best_lod, prog_bar=False)
        pl_module.log(f"{stage}/opt_C_raw", best_c, prog_bar=False)
        pl_module.log(f"{stage}/opt_specificity", best_spec, prog_bar=False)
        pl_module.log(f"{stage}/opt_sensitivity", best_sens, prog_bar=False)
        pl_module.log(f"{stage}/best_LoD", self._best_lod, prog_bar=False)

        self._clear()

    def _clear(self):
        self._raw_logits.clear()
        self._targets.clear()
        self._patient_ids.clear()

    # ── Train hooks ───────────────────────────────────────────────────────

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.calibrate_on == "train":
            self._accumulate(batch)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.calibrate_on == "train":
            self._calibrate(trainer, pl_module)

    # ── Validation hooks ──────────────────────────────────────────────────

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
    ):
        if self.calibrate_on == "val":
            self._accumulate(batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.calibrate_on == "val":
            self._calibrate(trainer, pl_module)
