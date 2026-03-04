from typing import Type

import torch
from torchmetrics import Metric

from src.utils.data_objects import DataObject
from utils.image_utils import filter_phase_data, get_binary_mask


def loss_wrapper(metric: Type[Metric], name: str) -> Type[Metric]:
    _name = name

    class WrappedLoss(metric):
        def __init__(
            self,
            name: str = _name,
            weight: float = 1.0,
            heart_only: bool = False,
            use_weight: bool = False,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.name = name
            self.weight = weight
            self.heart_only = heart_only
            self.use_weight = use_weight

        def forward(self, data_object: DataObject) -> torch.Tensor:
            mask = data_object.use_for_loss

            output = data_object.output[mask]
            target = data_object.target[mask]

            if data_object.is_mag_phase():
                output = filter_phase_data(output)
                target = filter_phase_data(target)

            mask_exists = (
                data_object.myocardial_mask != 0
            ).any() and data_object.myocardial_mask.numel() > 0
            if self.heart_only and mask_exists:
                mask = data_object.myocardial_mask
            else:
                mask = get_binary_mask(target.detach().cpu())
            
            if mask.device != output.device:
                mask = mask.to(output.device)

            output = output * mask
            target = target * mask

            if self.use_weight and data_object.weight is not None:
                try:
                    return (
                        super().forward(
                            output,
                            target,
                            weight=data_object.weight,
                            is_polar=data_object.is_polar,
                        )
                        * self.weight
                    )
                except TypeError:
                    try:
                        return (
                            super().forward(output, target, is_polar=data_object.is_polar)
                            * data_object.weight
                            * self.weight
                        )
                    except TypeError:
                        return super().forward(output, target) * data_object.weight * self.weight

            try:
                return super().forward(output, target, is_polar=data_object.is_polar) * self.weight
            except TypeError:
                return super().forward(output, target) * self.weight

    return WrappedLoss