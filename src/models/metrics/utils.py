from typing import Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL.Image import Image
from torchmetrics import Metric

from src.utils.data_objects import DataObject
from src.utils.enums import METRIC_TYPE, TRAINING_STAGE, MetricType, TrainingStage

METRIC_OUTPUT = torch.Tensor | np.ndarray | plt.Figure | Image


class WrappedMetric(torch.nn.Module):
    """Wrapper for torchmetrics that supports DataObject and multiple training stages."""
    
    def __init__(
        self,
        metric: Type[Metric],
        name: str,
        add_to_modules: bool = True,
        eval_on_data: bool = False,
        **kwargs
    ):
        """Initialize wrapped metric.
        
        Args:
            metric: Torchmetrics Metric class
            name: Name of the metric
            add_to_modules: Whether to add metric modules to parent module
            eval_on_data: Whether to evaluate on input data instead of output
            **kwargs: Additional arguments for metric initialization
        """
        super().__init__()

        self.name = name
        self.add_to_modules = add_to_modules
        self.eval_on_data = eval_on_data
        self.train_metric = metric(**kwargs)
        self.val_metric = metric(**kwargs)
        self.test_metric = metric(**kwargs)

    def reset(self):
        """Reset all stage metrics."""
        self.train_metric.reset()
        self.val_metric.reset()
        self.test_metric.reset()

    def log_train(self, x: DataObject) -> METRIC_OUTPUT:
        """Log metric for training stage."""
        raise NotImplementedError

    def log_val(self, x: DataObject) -> METRIC_OUTPUT:
        """Log metric for validation stage."""
        raise NotImplementedError

    def log_test(self, x: DataObject) -> METRIC_OUTPUT:
        """Log metric for test stage."""
        raise NotImplementedError

    def log(self, x: DataObject, stage: TRAINING_STAGE) -> METRIC_OUTPUT:
        """Log metric for given stage."""
        if stage == TrainingStage.TRAIN:
            return self.log_train(x)
        elif stage == TrainingStage.VAL:
            return self.log_val(x)
        elif stage == TrainingStage.TEST:
            return self.log_test(x)
        else:
            raise ValueError("stage must be either 'train', 'val' or 'test'")

    def get(self, stage: TRAINING_STAGE) -> Metric:
        """Get metric instance for given stage."""
        if stage == TrainingStage.TRAIN:
            return self.train_metric
        elif stage == TrainingStage.VAL:
            return self.val_metric
        elif stage == TrainingStage.TEST:
            return self.test_metric
        else:
            raise ValueError("stage must be either 'train', 'val' or 'test'")

    def set(self, stage: TRAINING_STAGE, metric: Metric):
        """Set metric instance for given stage."""
        if stage == TrainingStage.TRAIN:
            self.train_metric = metric
        elif stage == TrainingStage.VAL:
            self.val_metric = metric
        elif stage == TrainingStage.TEST:
            self.test_metric = metric
        else:
            raise ValueError("stage must be either 'train', 'val' or 'test'")

    def __getitem__(self, item):
        return self.get(item)


class PairedMetric(WrappedMetric):
    """Metric that compares output with target (or data with target if eval_on_data=True)."""
    
    def log_step(self, x: DataObject, step: str) -> torch.Tensor:
        """Log metric for a specific step."""
        if step == "train":
            metric = self.train_metric
        elif step == "val":
            metric = self.val_metric
        elif step == "test":
            metric = self.test_metric
        else:
            raise ValueError(f"step must be 'train', 'val', or 'test', got {step}")

        data, output, target = x.get_data_output_target_tuple()

        if self.eval_on_data:
            if data.shape[1] > target.shape[1]:
                data = data[:, : target.shape[1]]
            return metric(data, target)
        else:
            return metric(output, target)

    def log_train(self, x: DataObject) -> torch.Tensor:
        return self.log_step(x, "train")

    def log_val(self, x: DataObject) -> torch.Tensor:
        return self.log_step(x, "val")

    def log_test(self, x: DataObject) -> torch.Tensor:
        return self.log_step(x, "test")


class SingleMetric(WrappedMetric):
    """Metric that operates on a single tensor."""
    
    def log_train(self, x: torch.Tensor) -> torch.Tensor:
        return self.train_metric(x)

    def log_val(self, x: torch.Tensor) -> torch.Tensor:
        return self.val_metric(x)

    def log_test(self, x: torch.Tensor) -> torch.Tensor:
        return self.test_metric(x)


class ImageMetric(WrappedMetric):
    """Metric that returns image/logging output."""
    
    def log_train(self, x: DataObject) -> METRIC_OUTPUT:
        return self.train_metric(x)

    def log_val(self, x: DataObject) -> METRIC_OUTPUT:
        return self.val_metric(x)

    def log_test(self, x: DataObject) -> METRIC_OUTPUT:
        return self.test_metric(x)


class ImageLoggingMetric:
    """Protocol for metrics that can return images for logging."""
    
    def get_image(self, force: bool = False) -> Optional[METRIC_OUTPUT]:
        """Get image representation of metric.
        
        Args:
            force: Force generation even if conditions not met
            
        Returns:
            Image representation or None
        """
        raise NotImplementedError


def metric_wrapper(metric: Type[Metric], name: str, mode: METRIC_TYPE) -> Type[WrappedMetric]:
    """Factory function to create wrapped metric of specific type.
    
    Args:
        metric: Torchmetrics Metric class
        name: Name of the metric
        mode: Type of metric (PAIRED, SINGLE, or IMAGE)
        
    Returns:
        WrappedMetric subclass
    """
    _name = name
    if mode == MetricType.PAIRED:
        parent = PairedMetric
    elif mode == MetricType.SINGLE:
        parent = SingleMetric
    elif mode == MetricType.IMAGE:
        parent = ImageMetric
    else:
        raise ValueError("mode must be either 'paired', 'single' or 'image'")

    class SpecificWrappedMetric(parent):
        def __init__(
            self,
            name: str = _name,
            add_to_modules: bool = True,
            eval_on_data: bool = False,
            **kwargs
        ):
            super().__init__(
                metric, name, add_to_modules=add_to_modules, eval_on_data=eval_on_data, **kwargs
            )

    return SpecificWrappedMetric
