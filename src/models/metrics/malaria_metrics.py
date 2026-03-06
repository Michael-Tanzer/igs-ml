"""Patient-level metrics for malaria patch classification.

These metrics require accumulating predictions across a full epoch (val or test)
before computing, since patient-level statistics require seeing all objects
from each patient together.

Metrics computed:
    mu_S, sigma_S    -- mean/std of per-positive-patient object sensitivity
    mu_F, sigma_F    -- mean/std of per-negative-patient false positive count
    lod              -- Limit of Detection estimate (p/uL proxy)
    patient_sens     -- fraction of positive patients correctly diagnosed
    patient_spec     -- fraction of negative patients with zero FPs
"""

from collections import defaultdict
from typing import Dict, List

import torch

from src.utils.data_objects import DataObject


class MalariaPatientMetrics(torch.nn.Module):
    """Epoch-level patient metrics for malaria binary classification.

    Accumulates per-batch predictions and groups them by patient_id (blood sample id) at
    epoch end to compute clinically meaningful metrics.

    A patient is considered **positive** if any of their objects has
    ``target == 1`` (y_binary_strict). All other patients are **negative**.

    This class uses ``add_to_modules = False`` so that ``BaseLitModule``
    does NOT register the internal torchmetric sub-modules -- accumulation
    is managed manually here, not via torchmetrics state.
    """

    name = "malaria_patient"
    add_to_modules = False

    def __init__(self, threshold: float = 0.0):
        """
        Args:
            threshold: Logit threshold for binarising predictions (default 0.0,
                       corresponding to sigmoid probability = 0.5).
        """
        super().__init__()
        self.threshold = threshold
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._pids: List[List[str]] = []

    def reset(self):
        """Clear accumulated state (call at the start of each epoch)."""
        self._preds.clear()
        self._targets.clear()
        self._pids.clear()

    def update(self, batch: DataObject):
        """Accumulate one batch of predictions.

        Args:
            batch: Computed DataObject with .output (logits), .target (binary),
                   and .malaria.patient_id (list of int per sample).
        """
        if batch.malaria is None:
            return

        patient_ids = batch.malaria.patient_id
        if not isinstance(patient_ids, (list, tuple)):
            patient_ids = [patient_ids]

        self._preds.append(batch.output.detach().cpu().float())
        self._targets.append(batch.target.detach().cpu().float())
        self._pids.append(list(patient_ids))

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

        # Binarise predictions at the chosen threshold
        binary_preds = (all_preds > self.threshold).float()

        # Group by PID
        pid_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, pid in enumerate(all_pids):
            pid_to_indices[str(pid)].append(i)

        sensitivities: List[float] = []  # S_p for positive patients
        fp_counts: List[float] = []      # F_n for negative patients
        n_pos_patients = 0
        n_pos_correctly_diagnosed = 0
        n_neg_patients = 0
        n_neg_with_zero_fp = 0

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

        # LoD = (3.3 * sigma_F + 1) / max(mu_S, eps)
        results["patient_lod"] = (3.3 * sigma_f + 1) / max(mu_s, 1e-6)

        return results
