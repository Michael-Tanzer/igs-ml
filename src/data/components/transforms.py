"""Custom torchvision-compatible transforms for the malaria patch pipeline."""

import torch
import torch.nn as nn
from torchvision.transforms import ColorJitter


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


class MultiplaneColorJitter(nn.Module):
    """Apply ColorJitter independently to each 3-channel z-plane.

    For z-stack images with ``C = 3 * n_z`` channels, standard ColorJitter
    fails because it only supports 1 or 3 channels.  This wrapper splits the
    tensor into 3-channel planes, applies a *fresh* random jitter to each
    plane independently, and re-concatenates.

    Args:
        brightness: How much to jitter brightness (see ``ColorJitter``).
        contrast: How much to jitter contrast.
        saturation: How much to jitter saturation.
        hue: How much to jitter hue.

    Example config::

        - _target_: src.data.components.transforms.MultiplaneColorJitter
          brightness: 0.2
          contrast: 0.2
          saturation: 0.3
          hue: 0.05
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply per-plane color jitter.

        Args:
            tensor: Float tensor of shape ``(C, H, W)`` where C is a
                multiple of 3.

        Returns:
            Jittered tensor of the same shape.
        """
        C = tensor.shape[0]
        assert C % 3 == 0, f"Channel count {C} is not a multiple of 3"
        jitter = ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )
        planes = tensor.split(3, dim=0)
        jittered = [jitter(plane) for plane in planes]
        return torch.cat(jittered, dim=0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue})"
        )
