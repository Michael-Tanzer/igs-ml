"""Patient-level metrics for malaria patch classification.

These metrics require accumulating predictions across a full epoch (val or test)
before computing, since patient-level statistics require seeing all objects
from each patient together.

Metrics computed:
    mu_S, sigma_S    -- mean/std of per-positive-patient object sensitivity
    mu_F, sigma_F    -- mean/std of per-negative-patient false positive count
    lod              -- Limit of Detection in examined-volume units
    lod_puL          -- Empirical LoD in p/uL (95% detection threshold)
    lod_puL_calibrated -- Volume-calibrated LoD in p/uL
    patient_sens     -- fraction of positive patients correctly diagnosed
    patient_spec     -- fraction of negative patients with zero FPs
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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

    Predictions are binarised at ``> 0.0`` — the model's ``threshold_logit``
    buffer shifts outputs so that ``> 0`` is always the optimal decision
    boundary.

    This class uses ``add_to_modules = False`` so that ``BaseLitModule``
    does NOT register the internal torchmetric sub-modules -- accumulation
    is managed manually here, not via torchmetrics state.
    """

    name = "malaria_patient"
    add_to_modules = False

    def __init__(self):
        super().__init__()
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._pids: List[List[str]] = []
        self._parasitemias: List[List[str]] = []

    def reset(self):
        """Clear accumulated state (call at the start of each epoch)."""
        self._preds.clear()
        self._targets.clear()
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

        parasitemias = batch.malaria.parasitemia
        if not isinstance(parasitemias, (list, tuple)):
            parasitemias = [parasitemias]
        self._parasitemias.append(list(parasitemias))

    def compute(self) -> Dict[str, float]:
        """Compute patient-level metrics from accumulated batches.

        Returns:
            Dictionary of metric name -> scalar float value. Keys:
            ``patient_mu_S``, ``patient_sigma_S``, ``patient_mu_F``,
            ``patient_sigma_F``, ``patient_lod``, ``patient_sensitivity``,
            ``patient_specificity``.
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

        # Binarise at 0.0 — model outputs are already shifted.
        binary_preds = (all_preds > 0.0).float()

        # Group by PID
        pid_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, pid in enumerate(all_pids):
            pid_to_indices[str(pid)].append(i)

        # Per-patient parasitemia (all patches from one patient share the same value)
        pid_to_parasitemia: Dict[str, Optional[float]] = {}
        for pid, indices in pid_to_indices.items():
            pid_to_parasitemia[pid] = _parse_parasitemia(all_parasitemias[indices[0]])

        sensitivities: List[float] = []  # S_p for positive patients
        fp_counts: List[float] = []      # F_n for negative patients
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
                if tp >= 1:
                    n_pos_correctly_diagnosed += 1

                # Collect data for p/uL metrics
                parasitemia_val = pid_to_parasitemia.get(pid)
                if parasitemia_val is not None and parasitemia_val > 0:
                    detection_data.append((parasitemia_val, tp >= 1))
                    # puL_per_patch = parasitemia / n_true_positive_patches
                    # so LOD_puL = LOD_patches * mean(puL_per_patch)
                    n_true_pos = p_targets.sum().item()
                    if n_true_pos > 0:
                        volume_ratios.append(parasitemia_val / n_true_pos)
            else:
                fp = (p_preds == 1).sum().item()
                fp_counts.append(float(fp))
                n_neg_patients += 1
                if fp == 0:
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

        # Per-negative-patient FPR stats
        if fp_counts:
            f_tensor = torch.tensor(fp_counts)
            mu_f = f_tensor.mean().item()
            sigma_f = f_tensor.std(correction=1).item() if len(fp_counts) > 1 else 0.0
            results["patient_mu_F"] = mu_f
            results["patient_sigma_F"] = sigma_f
            results["patient_specificity"] = n_neg_with_zero_fp / n_neg_patients
        else:
            mu_f = 0.0
            sigma_f = 0.0
            results["patient_mu_F"] = float("nan")
            results["patient_sigma_F"] = float("nan")
            results["patient_specificity"] = float("nan")

        # LoD in examined-volume units (existing metric)
        lod_patches = (3.3 * sigma_f + 1) / max(mu_s, 1e-6)
        results["patient_lod"] = lod_patches

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

        return results
