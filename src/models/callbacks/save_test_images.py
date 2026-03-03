import os
from copy import deepcopy
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule

from src.utils.data_objects import DataObject
from src.utils.image_utils import center_image, get_binary_mask, resize_with_padding

# Try to import colorcet for better colormaps (optional dependency)
try:
    import colorcet as cc
    COLORCET_AVAILABLE = True
except ImportError:
    COLORCET_AVAILABLE = False
    cc = None

# Try to import colorcet for better colormaps
try:
    import colorcet as cc
    COLORCET_AVAILABLE = True
except ImportError:
    COLORCET_AVAILABLE = False


class SaveTestImagesCallback(Callback):
    """Callback to save test images with optional error visualization.
    
    Works with DataObject and supports sub-objects for mask extraction.
    """
    
    def __init__(
        self,
        output_dir: str,
        save_data: bool = False,
        save_target: bool = False,
        save_mask: bool = False,
        save_data_mae: bool = True,
        save_output_mae: bool = True,
        errors_vmax_factor: float = 0.25,
        better_automatic_mask: bool = False,
        mask_attr: str = None,  # e.g., "sms.myocardial_mask" for sub-objects
    ):
        """Initialize callback.
        
        Args:
            output_dir: Directory to save images
            save_data: Whether to save input data images
            save_target: Whether to save target images
            save_mask: Whether to save mask images
            save_data_mae: Whether to save data-target error images
            save_output_mae: Whether to save output-target error images
            errors_vmax_factor: Factor for error image vmax (relative to max error)
            better_automatic_mask: Whether to compute mask automatically from target
            mask_attr: Attribute path for mask (e.g., "sms.myocardial_mask" or "mask")
        """
        super().__init__()
        self.output_dir = output_dir
        self.save_data = save_data
        self.save_target = save_target
        self.save_mask = save_mask
        self.save_data_mae = save_data_mae
        self.save_output_mae = save_output_mae
        self.errors_vmax_factor = errors_vmax_factor
        self.better_automatic_mask = better_automatic_mask
        self.mask_attr = mask_attr

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "output"), exist_ok=True)

        if self.save_data:
            os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)
        if self.save_target:
            os.makedirs(os.path.join(self.output_dir, "target"), exist_ok=True)
        if self.save_mask:
            os.makedirs(os.path.join(self.output_dir, "mask"), exist_ok=True)
        if self.save_data_mae:
            os.makedirs(os.path.join(self.output_dir, "error_data"), exist_ok=True)
        if self.save_output_mae:
            os.makedirs(os.path.join(self.output_dir, "error_output"), exist_ok=True)

    def _get_mask(self, batch: DataObject) -> np.ndarray:
        """Extract mask from DataObject, supporting sub-objects."""
        if self.better_automatic_mask:
            return get_binary_mask(batch.target.cpu().numpy())
        
        if self.mask_attr:
            # Support nested attributes like "sms.myocardial_mask"
            parts = self.mask_attr.split(".")
            obj = batch
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    return None
            if obj is not None:
                mask = obj.cpu().numpy().astype(bool) if isinstance(obj, torch.Tensor) else obj
                return mask
        
        # Try common mask locations
        if batch.sms is not None and hasattr(batch.sms, "myocardial_mask"):
            return batch.sms.myocardial_mask.cpu().numpy().astype(bool)
        
        return None

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[torch.Tensor | Dict[str, Any]],
        batch: DataObject,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save images at end of test batch."""
        if not trainer.testing:
            return

        assert batch.computed, "Batch must be computed before saving images"

        # Get normalization stats if available from dataset
        mean = 0
        std = 1
        if hasattr(trainer.test_dataloaders, "dataset"):
            dataset = trainer.test_dataloaders.dataset
            if hasattr(dataset, "means") and len(dataset.means) > 0:
                mean = dataset.means[0][0] if isinstance(dataset.means[0], (list, tuple)) else dataset.means[0]
            if hasattr(dataset, "stds") and len(dataset.stds) > 0:
                std = dataset.stds[0][0] if isinstance(dataset.stds[0], (list, tuple)) else dataset.stds[0]

        batch_output = batch.output.cpu().numpy().copy()
        batch_data = batch.data.cpu().numpy().copy()
        batch_target = batch.target.cpu().numpy().copy()
        batch_mask = self._get_mask(batch)
        batch_indices = batch.index

        # Handle phase/magnitude data if needed (check in sub-objects)
        is_mag_phase = False
        if batch.sms is not None:
            if hasattr(batch.sms, "phase_data") and hasattr(batch.sms, "complex_data"):
                is_mag_phase = batch.sms.phase_data and not batch.sms.complex_data
        
        if is_mag_phase:
            # Filter phase data (keep magnitude channels)
            batch_output = batch_output[:, 0::2, :, :] if batch_output.ndim == 4 else batch_output[::2, :, :]
            batch_data = batch_data[:, 0::2, :, :] if batch_data.ndim == 4 else batch_data[::2, :, :]
            batch_target = batch_target[:, 0::2, :, :] if batch_target.ndim == 4 else batch_target[::2, :, :]
            if batch_mask is not None:
                batch_mask = batch_mask[:, 0::2, :, :] if batch_mask.ndim == 4 else batch_mask[::2, :, :]

        if batch_data.shape[1] > batch_target.shape[1]:
            batch_data = batch_data[:, : batch_target.shape[1]]

        if batch_mask is None:
            batch_mask = np.ones_like(batch_target, dtype=bool)
        elif batch_mask.shape[1] == 1 and batch_target.shape[1] > 1:
            batch_mask = batch_mask.repeat(batch_target.shape[1], axis=1)

        for i, data, output, target, mask in zip(
            batch_indices, batch_data, batch_output, batch_target, batch_mask
        ):
            data = data * std + mean
            output = output * std + mean
            target = target * std + mean

            data[~mask] = np.nan
            target[~mask] = np.nan
            output[~mask] = np.nan

            for channel in range(data.shape[0]):
                filename = f"{i}_{channel}.png"

                if self.save_data:
                    image_path = os.path.join(self.output_dir, "data", filename)
                    self.save_image(data[channel], image_path)

                if self.save_target:
                    image_path = os.path.join(self.output_dir, "target", filename)
                    self.save_image(target[channel], image_path)

                if self.save_mask:
                    image_path = os.path.join(self.output_dir, "mask", filename)
                    self.save_image(mask[channel].astype(float), image_path)

                if self.save_output_mae:
                    error = np.abs(target[channel] - output[channel])
                    image_path = os.path.join(self.output_dir, "error_output", filename)
                    self.save_image(error, image_path, error_image=True)

                if self.save_data_mae:
                    error = np.abs(target[channel] - data[channel])
                    image_path = os.path.join(self.output_dir, "error_data", filename)
                    self.save_image(error, image_path, error_image=True)

                image_path = os.path.join(self.output_dir, "output", filename)
                self.save_image(output[channel], image_path)

    def save_image(self, image, image_path, error_image=False):
        """Save a single image."""
        resized_image = resize_with_padding(image, (100, 100), padding_mode="edge")
        centered_image = center_image(resized_image)

        plt.figure(
            figsize=(
                centered_image.shape[1] / 25,
                centered_image.shape[0] / 25,
            )
        )

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        if np.isnan(image).all():
            cmap = deepcopy(plt.get_cmap("gray"))
            vmin = 0
            vmax = 0
        elif error_image:
            if COLORCET_AVAILABLE and cc is not None:
                cmap = deepcopy(cc.m_CET_L3)
            else:
                cmap = deepcopy(plt.get_cmap("hot"))
            vmin = 0
            vmax = np.nanmax(centered_image) * self.errors_vmax_factor
        else:
            cmap = deepcopy(plt.get_cmap("gray"))
            vmin = np.nanmin(centered_image)
            vmax = np.nanmax(centered_image)

        cmap.set_bad("k", 0.0)

        plt.imshow(centered_image, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.savefig(image_path, transparent=True)
        plt.close()
