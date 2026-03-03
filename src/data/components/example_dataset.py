from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.data_objects import DataObject


@dataclass
class DenoisingProperties:
    """Example task-specific properties for image denoising."""
    noise_level: float
    noise_type: str = "gaussian"


class ExampleDataset(Dataset):
    """Example dataset demonstrating DataObject usage with optional sub-objects.
    
    This is a simple image denoising dataset that shows how to:
    - Create DataObjects with core fields
    - Add optional task-specific properties
    - Use transforms
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        noise_level: float = 0.1,
        use_properties: bool = True,
        transform=None,
    ):
        """Initialize example dataset.
        
        Args:
            data: Input noisy images tensor [N, C, H, W]
            target: Target clean images tensor [N, C, H, W]
            noise_level: Noise level for denoising properties
            use_properties: Whether to include DenoisingProperties sub-object
            transform: Optional transform to apply
        """
        self.data = data
        self.target = target
        self.noise_level = noise_level
        self.use_properties = use_properties
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single DataObject sample."""
        data = self.data[idx]
        target = self.target[idx]

        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
            target = self.transform(target)

        # Create DataObject with core fields
        data_obj = DataObject(
            index=idx,
            data=data,
            target=target,
            data_filepath=f"sample_{idx}.png",
            target_filepath=f"target_{idx}.png",
        )

        # Optionally add task-specific properties
        # DataObject supports dynamic attributes for sub-objects
        if self.use_properties:
            data_obj.denoising = DenoisingProperties(
                noise_level=self.noise_level,
                noise_type="gaussian",
            )

        return data_obj
