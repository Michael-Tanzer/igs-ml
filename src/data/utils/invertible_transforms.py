import abc
import math
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from torchvision import transforms
from torchvision.transforms import functional as F

from src.utils.utils import get_relative_file_path


class CustomRandomTransform(metaclass=abc.ABCMeta):
    """Abstract base class for random transforms that can return random values."""
    
    def forward(self, sample: torch.Tensor, random_value: float = None):
        raise NotImplementedError


class CustomCompose(transforms.Compose):
    """Compose transforms with support for random value tracking.
    
    Extends torchvision.Compose to:
    - Track random values used in random transforms
    - Support transforms with forward_kwargs
    - Return random values for reproducibility
    """
    
    def __init__(self, transforms: Iterable[Dict[str, Any]], return_random_values: bool = False):
        super().__init__(transforms)
        self.return_random_values = return_random_values

    def __call__(self, sample: torch.Tensor, random_transforms_parameters: List[float] = np.nan, **kwargs):
        """Apply transforms with optional random value tracking.
        
        Args:
            sample: Input tensor
            random_transforms_parameters: List of random values to use (or np.nan to generate new)
            **kwargs: Additional arguments for transforms with forward_kwargs
            
        Returns:
            Transformed sample, optionally with random values list
        """
        if not isinstance(random_transforms_parameters, list) and np.isnan(random_transforms_parameters):
            random_transforms_parameters = [np.nan for _ in range(len(self.transforms))]

        random_values = []
        for t, rtp in zip(self.transforms, random_transforms_parameters):
            if isinstance(t, CustomRandomTransform):
                sample, random_value = t(sample, random_value=rtp)
                random_values.append(random_value)
            elif hasattr(t, "forward_kwargs"):
                sample = t(sample, **{k: v for k, v in kwargs.items() if k in t.forward_kwargs})
                random_values.append(np.nan)
            else:
                sample = t(sample)
                random_values.append(np.nan)

        if self.return_random_values:
            return sample, random_values
        
        return sample


class CustomRandomVerticalFlip(transforms.RandomVerticalFlip, CustomRandomTransform):
    """Random vertical flip with random value tracking."""
    
    def forward(self, sample: torch.Tensor, random_value: float = np.nan):
        if np.isnan(random_value):
            random_value = torch.rand(1).item()

        if random_value < self.p:
            return F.vflip(sample), random_value

        return sample, random_value


class CustomRandomHorizontalFlip(transforms.RandomHorizontalFlip, CustomRandomTransform):
    """Random horizontal flip with random value tracking."""
    
    def forward(self, sample: torch.Tensor, random_value: float = np.nan):
        if np.isnan(random_value):
            random_value = torch.rand(1).item()

        if random_value < self.p:
            return F.hflip(sample), random_value

        return sample, random_value


class CustomRandomRotation(transforms.RandomRotation, CustomRandomTransform):
    """Random rotation with random value tracking."""
    
    def forward(self, sample: torch.Tensor, random_value: float = np.nan):
        fill = self.fill
        channels, _, _ = F.get_dimensions(sample)
        if isinstance(sample, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        if np.isnan(random_value):
            random_value = self.get_params(self.degrees)

        return F.rotate(sample, random_value, self.interpolation, self.expand, self.center, fill), random_value


class InvertibleTransform:
    """Protocol/interface for transforms that can be inverted.
    
    Transforms implementing this interface should provide an `inverse` attribute
    containing a config dict that can be instantiated to create the inverse transform.
    """
    make_inverse: bool
    inverse: Optional[Dict[str, Any]]


class Scaler(torch.nn.Module, InvertibleTransform):
    """Generic scaling and shifting transform (formerly DICOMScaler).
    
    Supports per-channel scaling/shifting and can handle phase/magnitude data.
    """
    forward_kwargs = {"chunk_shapes"}

    def __init__(
        self,
        scale: Iterable[float] | float,
        shift: Iterable[float] | float,
        phase: bool = False,
        complex: bool = False,
        per_channel: bool = False,
        make_inverse: bool = True,
    ):
        """Initialize scaler transform.
        
        Args:
            scale: Scaling factor(s)
            shift: Shifting factor(s)
            phase: Whether data is phase/magnitude format (interleaved channels)
            complex: Whether data is complex format
            per_channel: Whether to apply different scale/shift per channel
            make_inverse: Whether to compute inverse transform config
        """
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.phase = phase
        self.complex = complex
        self.per_channel = per_channel
        self.inverse = None

        self.mag_phs = self.phase and not self.complex

        # if per-channel is used, we expect a list of scales and shifts
        if self.per_channel and not isinstance(self.scale, Iterable):
            raise ValueError("If per_channel is True, scale must be Iterable[float]")

        if self.per_channel:
            self.scale = torch.tensor(scale)
            self.shift = torch.tensor(shift)
            n_dims = self.scale.dim()
            self.scale = self.scale.view(1, -1, 1, 1)
            self.shift = self.shift.view(1, -1, 1, 1)

            if n_dims == 1 and self.mag_phs:
                self.scale = (self.scale, self.scale)
                self.shift = (self.shift, self.shift)
        else:  # here we expect either a scalar or a list of 2 scalars
            if self.mag_phs and not isinstance(self.scale, Iterable):
                self.scale = (self.scale, self.scale)

            if self.mag_phs and not isinstance(self.shift, Iterable):
                self.shift = (self.shift, self.shift)

        if make_inverse:
            self.inverse = self.get_inverse()

    def forward(self, sample: torch.Tensor, chunk_shapes: Optional[Iterable[int]] = None):
        """Apply scaling and shifting.
        
        Args:
            sample: Input tensor
            chunk_shapes: Optional list of chunk sizes for per-channel processing
            
        Returns:
            Scaled and shifted tensor
        """
        channel_dim = 1 if sample.dim() == 4 else 0

        if self.per_channel and sample.dim() < self.scale.dim():
            self.scale = self.scale.view(-1, 1, 1)
            self.shift = self.shift.view(-1, 1, 1)
        elif self.per_channel and sample.dim() > self.scale.dim():
            self.scale = self.scale.view(1, -1, 1, 1)
            self.shift = self.shift.view(1, -1, 1, 1)

        if self.per_channel:
            assert sample.dim() == self.scale.dim(), (
                f"Sample has {sample.dim()} dimensions, "
                f"but scale has {self.scale.dim()} dimensions"
            )

        if self.per_channel and sample.shape[channel_dim] != self.scale.shape[channel_dim] and chunk_shapes is None:
            number_chunks = math.ceil(sample.shape[channel_dim] / self.scale.shape[channel_dim])
            chunk_size = self.scale.shape[channel_dim]
            slicing = [
                (slice(i * chunk_size, (i + 1) * chunk_size),) for i in range(number_chunks)
            ]
            slicing = [(slice(None), *s) for s in slicing] if channel_dim == 1 else slicing
            chunks = [sample[s] for s in slicing]
        elif chunk_shapes is not None:
            slicing = [(slice(sum(chunk_shapes[:i]), sum(chunk_shapes[:i + 1])),) for i in range(len(chunk_shapes))]
            slicing = [(slice(None), *s) for s in slicing] if channel_dim == 1 else slicing
            chunks = [sample[s] for s in slicing]
        else:
            chunks = [sample]

        results = []
        for chunk in chunks:
            if self.per_channel and chunk.shape[channel_dim] != self.scale.shape[channel_dim]:
                results.append(chunk)
                continue

            if self.phase and not self.complex:
                chunk[0::2] *= self.scale[0]
                chunk[1::2] *= self.scale[1]
                chunk[0::2] += self.shift[0]
                chunk[1::2] += self.shift[1]
            elif self.complex:
                mag, phs = chunk.abs(), chunk.angle()
                mag *= self.scale
                mag += self.shift
                chunk = mag * torch.exp(1j * phs)
            else:
                chunk *= self.scale
                chunk += self.shift

            results.append(chunk)

        sample = torch.cat(results, dim=channel_dim)

        return sample

    def get_inverse(self):
        """Get inverse transform configuration."""
        if isinstance(self.scale, (tuple, list)):
            inverse_scale = [1 / s for s in self.scale]
        elif isinstance(self.scale, torch.Tensor):
            inverse_scale = 1 / self.scale
            inverse_scale = list([s.item() for s in inverse_scale.squeeze()])
        elif isinstance(self.scale, (float, int)):
            inverse_scale = 1 / self.scale
        else:
            raise TypeError(f"Scale must be float or Iterable[float], not {type(self.scale)}")

        if isinstance(self.shift, (tuple, list)):
            inverse_shift = [-sh / sc for sh, sc in zip(self.shift, self.scale)]
        elif isinstance(self.shift, torch.Tensor):
            inverse_shift = -self.shift / self.scale
            inverse_shift = list([s.item() for s in inverse_shift.squeeze()])
        elif isinstance(self.shift, (float, int)):
            inverse_shift = -self.shift / self.scale
        else:
            raise TypeError(f"Shift must be float or Iterable[float], not {type(self.shift)}")

        return {
            "_target_": f"{get_relative_file_path(__file__)}.{Scaler.__name__}",
            "_partial_": False,
            "_convert_": "all",
            "scale": inverse_scale,
            "shift": inverse_shift,
            "phase": self.phase,
            "complex": self.complex,
            "per_channel": self.per_channel,
            "make_inverse": False,
        }


class Pad(transforms.Pad, InvertibleTransform):
    """Invertible padding transform."""
    
    def __init__(self, padding, fill=0, padding_mode="constant", make_inverse=True):
        super().__init__(padding, fill, padding_mode)
        self.inverse = None
        if make_inverse:
            self.inverse = self.get_inverse()

    def get_inverse(self):
        """Get inverse transform configuration (crop)."""
        if isinstance(self.padding, int):
            inverse_padding = -self.padding
        elif isinstance(self.padding, Iterable):
            inverse_padding = tuple(-p for p in self.padding)
        else:
            raise TypeError(f"Padding must be int or Iterable[int], not {type(self.padding)}")

        return {
            "_target_": f"{get_relative_file_path(__file__)}.{Pad.__name__}",
            "_partial_": False,
            "_convert_": "all",
            "padding": inverse_padding,
            "fill": 0,
            "padding_mode": "constant",
            "make_inverse": False,
        }


class CenterCrop(transforms.CenterCrop, InvertibleTransform):
    """Invertible center crop transform."""
    
    def __init__(self, size, make_inverse=True):
        super().__init__(size)
        self.make_inverse = make_inverse
        self.inverse = None

    def forward(self, img):
        output = super().forward(img)
        # based on the strong assumption that all images have the same size
        if self.make_inverse and self.inverse is None:
            self.inverse = self.get_inverse(img.shape[-2:])
        return output

    def get_inverse(self, original_sizes: tuple[int, int]):
        """Get inverse transform configuration (resize/pad to original size)."""
        return {
            "_target_": f"{get_relative_file_path(__file__)}.{CenterCrop.__name__}",
            "_partial_": False,
            "_convert_": "all",
            "size": original_sizes,
            "make_inverse": False,
        }
