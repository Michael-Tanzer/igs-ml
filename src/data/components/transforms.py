"""Custom torchvision-compatible transforms for the malaria patch pipeline."""

import torch
import torch.nn as nn


class RepeatNormalize(nn.Module):
    """Normalize a multi-channel tensor by tiling 3-channel RGB mean/std.

    Designed for z-stack images where ``C = 3 * n_z``: each group of 3
    channels is a standard RGB z-slice, so the same per-channel RGB
    statistics apply to every group.

    If the tensor already has exactly 3 channels the behaviour is identical
    to ``torchvision.transforms.Normalize``.

    Args:
        mean: 3-element sequence (R, G, B) mean computed on the dataset.
        std:  3-element sequence (R, G, B) std  computed on the dataset.

    Example config::

        - _target_: src.data.components.transforms.RepeatNormalize
          mean: [0.485, 0.456, 0.406]
          std:  [0.229, 0.224, 0.225]
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = list(mean)
        self.std  = list(std)
        assert len(mean) == 3 and len(std) == 3, "mean/std must be 3-element RGB sequences"

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor channels by tiling the 3-ch mean/std.

        Args:
            tensor: Float tensor of shape ``(C, H, W)`` where C is a
                multiple of 3.

        Returns:
            Normalized tensor of the same shape.
        """
        C = tensor.shape[0]
        assert C % 3 == 0, f"Channel count {C} is not a multiple of 3"
        n_tiles = C // 3
        mean = torch.tensor(self.mean * n_tiles, dtype=tensor.dtype).view(C, 1, 1)
        std  = torch.tensor(self.std  * n_tiles, dtype=tensor.dtype).view(C, 1, 1)
        return (tensor - mean) / std

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
