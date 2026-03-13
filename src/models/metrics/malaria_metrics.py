"""Patient-level metrics for malaria patch classification.

These metrics require accumulating predictions across a full epoch (val or test)
before computing, since patient-level statistics require seeing all objects
from each patient together.

Metrics computed:
    mu_S, sigma_S    -- mean/std of per-positive-patient object sensitivity
    mu_F, sigma_F    -- mean/std of per-negative-patient false positive count
    mu_F_rate, sigma_F_rate -- mean/std of per-negative-patient false positive rate
    lod              -- Limit of Detection in examined-volume units
    lod_puL          -- Empirical LoD in p/uL (95% detection threshold)
    lod_puL_calibrated -- Volume-calibrated LoD in p/uL
    patient_sens     -- fraction of positive patients correctly diagnosed
    patient_spec     -- fraction of negative patients with zero FPs
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.data_objects import DataObject


def _parse_parasitemia(raw) -> Optional[float]:
    """Parse parasitemia value to float. Returns None if unavailable.

    The SQL query COALESCEs missing values to -1. Values may arrive as
    str, int, or float depending on collation path.
    """
    try:
        val = float(raw)
        if val < 0:
            return None
        return val
    except (ValueError, TypeError):
        return None


class MalariaPatientMetrics(torch.nn.Module):
    """Epoch-level patient metrics for malaria binary classification.

    Accumulates per-batch predictions and groups them by patient_id (blood sample id) at
    epoch end to compute clinically meaningful metrics.

    A patient is considered **positive** if any of their objects has
    ``target == 1`` (y_binary_strict). All other patients are **negative**.

    When ``raw_logits`` are available, this class performs its own internal
    (C, T) threshold search to find the operating point that minimises LoD
    at a target specificity — making patient metrics self-consistent within
    each epoch and independent of the ``DualThresholdCalibrator`` callback
    (which runs after metrics and would otherwise cause a one-epoch lag).

    When ``raw_logits`` are NOT available, falls back to binarising at
    ``> 0.0`` on the shifted model outputs (legacy behaviour).

    This class uses ``add_to_modules = False`` so that ``BaseLitModule``
    does NOT register the internal torchmetric sub-modules -- accumulation
    is managed manually here, not via torchmetrics state.
    """

    name = "malaria_patient"
    add_to_modules = False

    def __init__(self, target_specificity: float = 0.95):
        super().__init__()
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._raw_logits: List[torch.Tensor] = []
        self._pids: List[List[str]] = []
        self._parasitemias: List[List[str]] = []
        self.count_threshold: int = 1
        self.target_specificity = target_specificity

    def reset(self):
        """Clear accumulated state (call at the start of each epoch)."""
        self._preds.clear()
        self._targets.clear()
        self._raw_logits.clear()
        self._pids.clear()
        self._parasitemias.clear()

    def update(self, batch: DataObject, is_train: bool = False):
        """Accumulate one batch of predictions.

        Args:
            batch: Computed DataObject with .output (logits), .target (binary),
                   and .malaria.patient_id (list of int per sample).
            is_train: If True, skip accumulation (patient metrics are eval-only).
        """
        if is_train:
            return
        if batch.malaria is None:
            return

        patient_ids = batch.malaria.patient_id
        if not isinstance(patient_ids, (list, tuple)):
            patient_ids = [patient_ids]

        self._preds.append(batch.output.detach().cpu().float())
        self._targets.append(batch.target.detach().cpu().float())
        self._pids.append(list(patient_ids))

        raw_logits = getattr(batch, "raw_logits", None)
        if raw_logits is not None and raw_logits.numel() > 0:
            self._raw_logits.append(raw_logits.detach().cpu().float())

        parasitemias = batch.malaria.parasitemia
        if not isinstance(parasitemias, (list, tuple)):
            parasitemias = [parasitemias]
        self._parasitemias.append(list(parasitemias))

    def _find_optimal_ct(
        self,
        all_raw_logits: torch.Tensor,
        all_targets: torch.Tensor,
        pid_to_indices: Dict[str, List[int]],
        positive_pids: List[str],
        negative_pids: List[str],
    ) -> Tuple[float, int]:
        """Find (C, T) that minimises LoD at target specificity.

        Sweeps all unique logit values for maximum precision.

        Returns:
            (best_C, best_T) — optimal logit threshold and count threshold.
        """
        candidates, _ = torch.sort(torch.unique(all_raw_logits))

        # Pre-compute index tensors for each patient (avoid repeated tensor creation)
        neg_idx = {pid: torch.tensor(pid_to_indices[pid], dtype=torch.long) for pid in negative_pids}
        pos_idx = {pid: torch.tensor(pid_to_indices[pid], dtype=torch.long) for pid in positive_pids}
        pos_targets = {pid: all_targets[pos_idx[pid]] for pid in positive_pids}

        best_lod = float("inf")
        best_c = candidates[0].item()
        best_t = 1

        for c in candidates:
            binary = (all_raw_logits > c).float()

            # Count FPs per negative patient
            neg_counts = [int((binary[neg_idx[pid]] == 1).sum().item()) for pid in negative_pids]

            # Find smallest T achieving target specificity
            sorted_counts = sorted(neg_counts)
            n_neg = len(sorted_counts)
            t = 1
            for candidate_t in range(1, (max(sorted_counts) + 2) if sorted_counts else 2):
                n_below = sum(1 for cv in sorted_counts if cv < candidate_t)
                if n_below / n_neg >= self.target_specificity:
                    t = candidate_t
                    break
            else:
                t = (max(sorted_counts) + 1) if sorted_counts else 1

            # Patient-level sensitivity using T
            sensitivities = []
            n_detected = 0
            for pid in positive_pids:
                p_binary = binary[pos_idx[pid]]
                pos_count = int((p_binary == 1).sum().item())
                if pos_count >= t:
                    n_detected += 1
                    p_targets = pos_targets[pid]
                    tp = ((p_binary == 1) & (p_targets == 1)).sum().item()
                    fn = ((p_binary == 0) & (p_targets == 1)).sum().item()
                    denom = tp + fn
                    sensitivities.append(tp / denom if denom > 0 else 0.0)

            if not sensitivities:
                continue

            mu_s = sum(sensitivities) / len(sensitivities)
            sigma_f = (
                torch.tensor(neg_counts, dtype=torch.float).std(correction=1).item()
                if len(neg_counts) > 1 else 0.0
            )
            lod = (3.3 * sigma_f + t) / max(mu_s, 1e-6)

            if lod < best_lod:
                best_lod = lod
                best_c = c.item()
                best_t = t

        return best_c, best_t

    def _compute_froc(
        self,
        all_raw_logits: torch.Tensor,
        all_targets: torch.Tensor,
        pid_to_indices: Dict[str, List[int]],
        positive_pids: List[str],
        negative_pids: List[str],
        n_thresholds: int = 200,
    ) -> List[Tuple[float, float]]:
        """Compute FROC curve: sensitivity vs avg FP per negative patient.

        Returns list of (avg_fp, mean_sensitivity) tuples sorted by avg_fp.
        """
        thresholds = torch.linspace(
            all_raw_logits.min().item(),
            all_raw_logits.max().item(),
            n_thresholds,
        )
        froc_points: List[Tuple[float, float]] = []

        for t in thresholds:
            binary = (all_raw_logits > t).float()

            # Avg FP per negative patient
            if negative_pids:
                fp_sum = 0.0
                for pid in negative_pids:
                    idx = torch.tensor(pid_to_indices[pid], dtype=torch.long)
                    fp_sum += (binary[idx] == 1).sum().item()
                avg_fp = fp_sum / len(negative_pids)
            else:
                avg_fp = 0.0

            # Mean sensitivity across positive patients
            if positive_pids:
                sens_sum = 0.0
                for pid in positive_pids:
                    idx = torch.tensor(pid_to_indices[pid], dtype=torch.long)
                    p_preds = binary[idx]
                    p_targets = all_targets[idx]
                    tp = ((p_preds == 1) & (p_targets == 1)).sum().item()
                    fn = ((p_preds == 0) & (p_targets == 1)).sum().item()
                    denom = tp + fn
                    sens_sum += tp / denom if denom > 0 else 0.0
                mean_sens = sens_sum / len(positive_pids)
            else:
                mean_sens = 0.0

            froc_points.append((avg_fp, mean_sens))

        # Sort by avg_fp ascending
        froc_points.sort(key=lambda x: x[0])
        return froc_points

    @staticmethod
    def _make_froc_figure(froc_points: List[Tuple[float, float]]) -> np.ndarray:
        """Render FROC curve to a numpy RGB array."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        fps = [p[0] for p in froc_points]
        sens = [p[1] for p in froc_points]
        ax.plot(fps, sens, "b-", linewidth=2)
        ax.set_xlabel("Avg FP per negative patient")
        ax.set_ylabel("Mean sensitivity")
        ax.set_title("FROC Curve")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf

    def compute(self) -> Dict[str, Any]:
        """Compute patient-level metrics from accumulated batches.

        Returns:
            Dictionary of metric name -> scalar float value.
        """
        if not self._preds:
            return {}

        # Flatten across batches
        all_preds = torch.cat(self._preds)     # (N,)
        all_targets = torch.cat(self._targets)  # (N,)
        all_pids: List[str] = []
        for pid_batch in self._pids:
            all_pids.extend(pid_batch)
        all_parasitemias: List[str] = []
        for p_batch in self._parasitemias:
            all_parasitemias.extend(p_batch)

        # Group by PID
        pid_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, pid in enumerate(all_pids):
            pid_to_indices[str(pid)].append(i)

        # Per-patient parasitemia (all patches from one patient share the same value)
        pid_to_parasitemia: Dict[str, Optional[float]] = {}
        for pid, indices in pid_to_indices.items():
            pid_to_parasitemia[pid] = _parse_parasitemia(all_parasitemias[indices[0]])

        # Classify patients by ground truth
        positive_pids: List[str] = []
        negative_pids: List[str] = []
        for pid, indices in pid_to_indices.items():
            idx = torch.tensor(indices, dtype=torch.long)
            if all_targets[idx].sum() > 0:
                positive_pids.append(pid)
            else:
                negative_pids.append(pid)

        # ── Find optimal (C, T) using raw logits if available ────────────
        if self._raw_logits:
            all_raw_logits = torch.cat(self._raw_logits)
            if len(all_raw_logits) != len(all_targets):
                all_raw_logits = None
        else:
            all_raw_logits = None
        opt_c = float("nan")
        count_threshold = self.count_threshold

        if all_raw_logits is not None and positive_pids and negative_pids:
            opt_c, count_threshold = self._find_optimal_ct(
                all_raw_logits, all_targets, pid_to_indices,
                positive_pids, negative_pids,
            )
            binary_preds = (all_raw_logits > opt_c).float()
        else:
            # Fallback: binarise shifted model outputs at 0.0
            binary_preds = (all_preds > 0.0).float()

        sensitivities: List[float] = []  # S_p for positive patients
        fp_counts: List[float] = []      # F_n for negative patients (raw count)
        fp_rates: List[float] = []       # F_n / total_patches for negative patients
        n_pos_patients = 0
        n_pos_correctly_diagnosed = 0
        n_neg_patients = 0
        n_neg_with_zero_fp = 0

        # For p/uL metrics
        detection_data: List[Tuple[float, bool]] = []  # (parasitemia, detected)
        volume_ratios: List[float] = []  # cV/V estimates

        for pid, indices in pid_to_indices.items():
            idx = torch.tensor(indices, dtype=torch.long)
            p_preds = binary_preds[idx]
            p_targets = all_targets[idx]

            is_positive_patient = p_targets.sum() > 0

            if is_positive_patient:
                tp = ((p_preds == 1) & (p_targets == 1)).sum().item()
                fn = ((p_preds == 0) & (p_targets == 1)).sum().item()
                denom = tp + fn
                s_p = tp / denom if denom > 0 else 0.0
                sensitivities.append(s_p)
                n_pos_patients += 1
                # Patient diagnosed if positive-predicted count >= count_threshold
                pos_count = int((p_preds == 1).sum().item())
                detected = pos_count >= count_threshold
                if detected:
                    n_pos_correctly_diagnosed += 1

                # Collect data for p/uL metrics
                parasitemia_val = pid_to_parasitemia.get(pid)
                if parasitemia_val is not None and parasitemia_val > 0:
                    detection_data.append((parasitemia_val, detected))
                    n_true_pos = p_targets.sum().item()
                    if n_true_pos > 0:
                        volume_ratios.append(parasitemia_val / n_true_pos)
            else:
                fp = (p_preds == 1).sum().item()
                total_patches = len(indices)
                fp_counts.append(float(fp))
                fp_rates.append(fp / total_patches if total_patches > 0 else 0.0)
                n_neg_patients += 1
                if fp < count_threshold:
                    n_neg_with_zero_fp += 1

        results: Dict[str, float] = {}

        # Per-positive-patient sensitivity stats
        if sensitivities:
            s_tensor = torch.tensor(sensitivities)
            mu_s = s_tensor.mean().item()
            sigma_s = s_tensor.std(correction=1).item() if len(sensitivities) > 1 else 0.0
            results["patient_mu_S"] = mu_s
            results["patient_sigma_S"] = sigma_s
            results["patient_sensitivity"] = n_pos_correctly_diagnosed / n_pos_patients
        else:
            mu_s = 0.0
            sigma_s = 0.0
            results["patient_mu_S"] = float("nan")
            results["patient_sigma_S"] = float("nan")
            results["patient_sensitivity"] = float("nan")

        # Per-negative-patient FP count stats
        if fp_counts:
            f_tensor = torch.tensor(fp_counts)
            mu_f = f_tensor.mean().item()
            sigma_f = f_tensor.std(correction=1).item() if len(fp_counts) > 1 else 0.0
            results["patient_mu_F"] = mu_f
            results["patient_sigma_F"] = sigma_f
            results["patient_specificity"] = n_neg_with_zero_fp / n_neg_patients
            # FP rate (FP / total_patches per patient)
            fr_tensor = torch.tensor(fp_rates)
            results["patient_mu_F_rate"] = fr_tensor.mean().item()
            results["patient_sigma_F_rate"] = (
                fr_tensor.std(correction=1).item() if len(fp_rates) > 1 else 0.0
            )
        else:
            mu_f = 0.0
            sigma_f = 0.0
            results["patient_mu_F"] = float("nan")
            results["patient_sigma_F"] = float("nan")
            results["patient_specificity"] = float("nan")
            results["patient_mu_F_rate"] = float("nan")
            results["patient_sigma_F_rate"] = float("nan")

        # Patient-level precision and F1
        n_neg_with_fp = n_neg_patients - n_neg_with_zero_fp
        n_called_positive = n_pos_correctly_diagnosed + n_neg_with_fp
        if n_called_positive > 0 and n_pos_patients > 0:
            precision = n_pos_correctly_diagnosed / n_called_positive
            recall = results.get("patient_sensitivity", 0.0)
            results["patient_precision"] = precision
            results["patient_f1"] = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
        else:
            results["patient_precision"] = float("nan")
            results["patient_f1"] = float("nan")

        # LoD in examined-volume units
        lod_patches = (3.3 * sigma_f + count_threshold) / max(mu_s, 1e-6)
        results["patient_lod"] = lod_patches
        results["patient_opt_C"] = opt_c
        results["patient_opt_T"] = float(count_threshold)

        # ── LoD in p/uL: Empirical (95% detection threshold) ────────────
        if len(detection_data) >= 5:
            # Sort by parasitemia ascending
            detection_data.sort(key=lambda x: x[0])
            parasitemias_sorted = [d[0] for d in detection_data]
            detected_sorted = [d[1] for d in detection_data]

            # Reverse-cumulative detection rate: for each parasitemia level,
            # what fraction of patients AT OR ABOVE it are detected?
            n = len(detection_data)
            cum_detected = 0
            cum_total = 0
            detection_rates = [0.0] * n
            for i in range(n - 1, -1, -1):
                cum_total += 1
                if detected_sorted[i]:
                    cum_detected += 1
                detection_rates[i] = cum_detected / cum_total

            # Find lowest parasitemia where cumulative detection rate >= 95%
            lod_puL = float("nan")
            for i in range(n):
                if detection_rates[i] >= 0.95:
                    lod_puL = parasitemias_sorted[i]
                    break
            results["patient_lod_puL"] = lod_puL
        else:
            results["patient_lod_puL"] = float("nan")
        results["patient_lod_n_patients"] = float(len(detection_data))

        # ── LoD in p/uL: Volume-calibrated ──────────────────────────────
        # Back-solve puL_per_patch = parasitemia / n_true_positive_patches,
        # then LOD_puL = LOD_patches * mean(puL_per_patch).
        if volume_ratios:
            mean_puL_per_patch = sum(volume_ratios) / len(volume_ratios)
            results["patient_lod_puL_calibrated"] = lod_patches * mean_puL_per_patch
            results["patient_puL_per_patch"] = mean_puL_per_patch
        else:
            results["patient_lod_puL_calibrated"] = float("nan")
            results["patient_puL_per_patch"] = float("nan")

        # ── FROC curve ────────────────────────────────────────────────────
        if all_raw_logits is not None and positive_pids and negative_pids:
            froc_points = self._compute_froc(
                all_raw_logits, all_targets, pid_to_indices,
                positive_pids, negative_pids,
            )
            results["patient_froc"] = self._make_froc_figure(froc_points)

        return results
