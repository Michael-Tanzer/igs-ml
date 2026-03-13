"""Task-agnostic image classifier backed by timm.

Wraps ``timm.create_model`` so that every architectural knob (backbone name,
input channels, number of classes, dropout, pretrained weights, ...) is
driven purely by config.
"""

import torch
import timm
import torch.nn as nn


class TimmClassifier(nn.Module):
    """Thin wrapper around ``timm.create_model``.

    All constructor arguments map directly to ``timm.create_model`` kwargs,
    making the model fully configurable from a Hydra YAML without any
    task-specific code.

    When ``num_classes=1`` the final dimension is squeezed so the output
    shape is ``(B,)`` rather than ``(B, 1)`` -- this keeps binary
    classification pipelines consistent (output matches target shape).
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        in_chans=3,
        num_classes=1,
        pretrained=False,
        drop_rate=0.0,
        **kwargs,
    ):
        """Initialise a timm-backed classifier.

        Args:
            model_name: Any model name recognised by ``timm.list_models()``.
            in_chans: Number of input channels.
            num_classes: Number of output classes (1 for binary logits).
            pretrained: Load ImageNet-pretrained weights when available.
            drop_rate: Dropout rate applied before the classifier head.
            **kwargs: Additional kwargs forwarded to ``timm.create_model``
                (e.g. ``img_size=144`` for vision transformers).
        """
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_rate=drop_rate,
            **kwargs,
        )

        # Zero-initialize the classifier bias so initial logits are centered at 0.
        # timm's default init can produce a positive bias, causing all predictions
        # to be positive (recall=1, specificity=0) before any learning occurs.
        classifier = self.model.get_classifier()
        if classifier is not None and hasattr(classifier, "bias") and classifier.bias is not None:
            nn.init.zeros_(classifier.bias)

        # Optimal logit threshold — persisted with model weights via state_dict.
        # Logits are always shifted by this value so that the decision
        # boundary is always ``> 0``.
        self.register_buffer("threshold_logit", torch.tensor(0.0))
        # Count threshold T for patient-level diagnosis: a patient is positive
        # if it has >= T positive-predicted patches.  Default 1.0 preserves
        # the original "any positive patch" rule.
        self.register_buffer("count_threshold", torch.tensor(1.0))

    def set_threshold(self, value: float):
        """Update the optimal logit threshold."""
        self.threshold_logit.fill_(value)

    def set_count_threshold(self, value: float):
        """Update the count threshold for patient-level diagnosis."""
        self.count_threshold.fill_(value)

    def forward(self, x):
        """Run the backbone + classifier head.

        Args:
            x: Input tensor ``(B, C, H, W)``.

        Returns:
            Logits tensor ``(B, num_classes)`` or ``(B,)`` when
            ``num_classes == 1``.
        """
        out = self.model(x)
        if self.num_classes == 1:
            out = out.squeeze(-1)
        # Cast to float32 before threshold shift — under bf16-mixed autocast,
        # the backbone output is bf16. The subtraction bf16 - f32 stays in bf16
        # on CUDA, causing catastrophic cancellation when reconstructing
        # raw_logits = (backbone_out - T) + T in the wrapper.
        out = out.float()
        out = out - self.threshold_logit
        return out
