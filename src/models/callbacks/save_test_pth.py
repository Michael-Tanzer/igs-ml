import os
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from hydra.utils import instantiate
from lightning import Callback

from src.data.components.invertible_transforms import CustomCompose, InvertibleTransform
from src.utils.data_objects import DataObject

PTH_FILENAME = "test_output.pth"


def identity(x):
    """Identity function for default transform."""
    return x


class SaveTestPTHsCallback(Callback):
    """Callback to save test outputs as .pth files with inverse transforms.
    
    Accumulates test results across batches and saves them with inverse transforms
    for post-processing.
    """
    
    def __init__(self, output_dir: str):
        """Initialize callback.
        
        Args:
            output_dir: Directory to save .pth file
        """
        super().__init__()
        self.output_dir = output_dir
        self.invertible_transforms = identity
        self.data_filepath = os.path.join(self.output_dir, PTH_FILENAME)

        os.makedirs(self.output_dir, exist_ok=True)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[torch.Tensor | Dict[str, Any]],
        batch: DataObject,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save test batch to .pth file."""
        if not trainer.testing:
            return

        # Extract inverse transforms from dataset on first batch
        if self.invertible_transforms == identity:
            if hasattr(trainer.test_dataloaders, "dataset"):
                dataset = trainer.test_dataloaders.dataset
                if hasattr(dataset, "transform_base") and hasattr(dataset.transform_base, "transforms"):
                    original_transforms = dataset.transform_base.transforms
                    inverse_transforms = []
                    for transform in original_transforms:
                        if isinstance(transform, InvertibleTransform) and transform.inverse is not None:
                            inverse_transforms.append(transform.inverse)

                    if inverse_transforms:
                        self.invertible_transforms = [
                            instantiate(transform) for transform in inverse_transforms
                        ]
                        self.invertible_transforms.reverse()
                        self.invertible_transforms = CustomCompose(self.invertible_transforms)

        assert batch.computed, "Batch must be computed before saving"
        assert batch.data_filepath, "Batch must have a data_filepath"
        assert batch.target_filepath, "Batch must have a target_filepath"

        # Move to CPU before saving
        batch = batch.to("cpu")

        # Load existing data or create new
        if os.path.exists(self.data_filepath):
            data = torch.load(self.data_filepath)["data"]
            # Append new data
            new_data = self.merge_dicts(data.to_dict(), batch.to_dict())
            data = DataObject(**new_data)
        else:
            data = batch

        pth_data = {
            "data": data,
            "invertible_transforms": self.invertible_transforms,
        }

        torch.save(pth_data, self.data_filepath)

    @staticmethod
    def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
        """Merge two dictionaries, concatenating tensors/arrays and handling sub-objects.
        
        Args:
            dict1: First dictionary (existing data)
            dict2: Second dictionary (new data)
            
        Returns:
            Merged dictionary
        """
        return_dict = dict1.copy()
        for key, value in dict1.items():
            if key not in dict2:
                continue
                
            new_value = dict2[key]
            
            # Handle sub-objects (nested dicts)
            if isinstance(value, dict) and not isinstance(value, DataObject):
                return_dict[key] = SaveTestPTHsCallback.merge_dicts(value, new_value)
            # Handle chunk_shapes specially
            elif key == "chunk_shapes":
                if isinstance(value, list) and isinstance(new_value, list):
                    return_dict[key] = [
                        torch.cat([v, nv], dim=0) if isinstance(v, torch.Tensor) else v + nv
                        for v, nv in zip(value, new_value)
                    ]
            # Handle random_transforms_parameters
            elif key == "random_transforms_parameters":
                if isinstance(value, torch.Tensor):
                    if np.isnan(value).all():
                        return_dict[key] = torch.cat([value, new_value], dim=0)
                    else:
                        return_dict[key] = torch.stack([value, new_value], dim=0)
                elif isinstance(value, list):
                    return_dict[key] = value + new_value
            # Handle lists/tuples
            elif isinstance(value, (list, tuple)):
                return_dict[key] = list(value) + list(new_value)
            # Handle tensors
            elif isinstance(value, torch.Tensor):
                if value.ndim > 0:
                    return_dict[key] = torch.cat([value, new_value], dim=0)
                else:
                    return_dict[key] = value + new_value
            # Handle numpy arrays
            elif isinstance(value, np.ndarray):
                return_dict[key] = np.concatenate([value, new_value], axis=0)
            # Handle scalars/immutables
            else:
                assert value == new_value, f"Data mismatch for key {key}: {value} != {new_value}"
                return_dict[key] = value

        return return_dict
