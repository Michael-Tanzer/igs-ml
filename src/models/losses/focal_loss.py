"""Focal loss for binary classification with class imbalance.

Focal loss down-weights easy examples and focuses training on hard negatives,
which is particularly useful for imbalanced datasets like malaria patch
classification at low parasitemia levels.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Sigmoid focal loss for binary classification.

    Applies focal modulation ``(1 - p_t)^gamma`` to the standard BCE loss,
    reducing the contribution of well-classified examples.

    Args:
        alpha: Weighting factor for the positive class (analogous to pos_weight).
            When alpha > 0.5, positive examples are up-weighted. Set to -1 to
            disable alpha weighting entirely.
        gamma: Focusing parameter. gamma=0 recovers standard BCE.
            gamma=2 is the default from the original paper.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha: float = 0.66, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            input: Raw logits of shape (N,) or (N, 1).
            target: Binary targets of the same shape, values in {0, 1}.

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-sample loss.
        """
        # Flatten to 1-D
        input = input.view(-1)
        target = target.view(-1).float()

        # Numerically stable computation using logsigmoid
        # BCE = -[t * log(sigma(x)) + (1-t) * log(1-sigma(x))]
        # For focal: multiply by (1 - p_t)^gamma
        p = torch.sigmoid(input)
        p_t = p * target + (1 - p) * (1 - target)

        # Focal modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma

        # Use the numerically stable BCE formulation
        bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")

        # Alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(alpha={self.alpha}, "
            f"gamma={self.gamma}, reduction='{self.reduction}')"
        )


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """BCE with logits loss with label smoothing.

    Smooths hard labels [0, 1] to [epsilon, 1-epsilon] before computing
    BCE loss. This acts as a regularizer and can improve calibration.

    Args:
        epsilon: Smoothing factor. Labels become [epsilon, 1-epsilon].
        pos_weight: Optional weight for positive examples (same as BCEWithLogitsLoss).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, epsilon: float = 0.1, pos_weight: torch.Tensor = None,
                 reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed BCE loss.

        Args:
            input: Raw logits of shape (N,) or (N, 1).
            target: Binary targets of the same shape, values in {0, 1}.

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-sample loss.
        """
        # Smooth the targets
        smoothed = target * (1.0 - self.epsilon) + (1.0 - target) * self.epsilon

        return F.binary_cross_entropy_with_logits(
            input, smoothed,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(epsilon={self.epsilon}, "
            f"pos_weight={self.pos_weight}, reduction='{self.reduction}')"
        )
