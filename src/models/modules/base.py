from typing import Any, Callable, List, Optional

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.types import _METRIC as METRIC_PL
from lightning_utilities.core.rank_zero import rank_zero_warn
from torchmetrics import Metric

from src.models.components.model_wrapper import DataObjectModelWrapper
from src.models.metrics.utils import ImageLoggingMetric, WrappedMetric
from src.utils.data_objects import DataObject
from src.utils.enums import TRAINING_STAGE, TrainingStage
from src.utils.utils import remove_best_from_string, warn_once

# Optional logger backends -- only needed when the corresponding logger is active
try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    wandb = None
    WandbLogger = None

try:
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    TensorBoardLogger = None

try:
    from lightning.pytorch.loggers import MLFlowLogger
except ImportError:
    MLFlowLogger = None

try:
    from lightning.pytorch.loggers import NeptuneLogger
except ImportError:
    NeptuneLogger = None

try:
    from lightning.pytorch.loggers import CometLogger
except ImportError:
    CometLogger = None

try:
    import aim
    from aim.pytorch_lightning import AimLogger
except ImportError:
    aim = None
    AimLogger = None


class BaseLitModule(LightningModule):
    """Base Lightning module that works with DataObject.
    
    Provides a foundation for models that use DataObject for data handling.
    Supports:
    - DataObject input/output
    - Multiple loss functions
    - Wrapped metrics
    - Image logging to multiple loggers
    """

    def __init__(
        self,
        net: DataObjectModelWrapper,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        criteria: List[Metric] = None,
        metrics: List[WrappedMetric] = None,
        name: str = "Base",
        log_metrics_every_n_steps: int = 1,
    ):
        """Initialize base module.

        Args:
            net: Model wrapped in DataObjectModelWrapper
            optimizer: Optimizer class (will be instantiated with model parameters)
            scheduler: Optional learning rate scheduler class
            criteria: List of loss/metric criteria
            metrics: List of wrapped metrics
            name: Module name
            log_metrics_every_n_steps: Log and update metrics every N steps (1 = every step).
        """
        super().__init__()

        # Save hyperparameters (excluding non-serializable objects)
        self.save_hyperparameters(logger=False, ignore=["net", "criteria", "metrics"])

        self.net = net
        self.criteria = torch.nn.ModuleList(criteria or [])
        self.metrics = metrics or []

        self.add_metrics_modules()

    def forward(self, x: DataObject) -> DataObject:
        """Forward pass.
        
        Args:
            x: Input DataObject
            
        Returns:
            DataObject with output stored
        """
        x.epoch = self.current_epoch
        return self.net(x)

    def on_train_start(self):
        """Reset metrics at start of training."""
        for metric in self.metrics:
            metric.reset()

    def model_step(self, batch: DataObject) -> DataObject:
        """Perform a model step (forward + loss computation).
        
        Args:
            batch: Input DataObject
            
        Returns:
            DataObject with output and losses computed
        """
        computed_batch = self.forward(batch)

        # Compute losses
        losses = {}
        for loss in self.criteria:
            losses[loss.name] = loss(computed_batch)

        # Merge losses from computed_batch and criteria
        computed_batch.losses = {**computed_batch.losses, **losses}

        return computed_batch

    def training_step(self, batch: DataObject, batch_idx: int):
        """Training step."""
        computed_batch = self.model_step(batch)
        self.on_step_log(computed_batch, stage=TrainingStage.TRAIN, batch_idx=batch_idx)

        # Return dict with loss information
        return_batch = {
            k: v for k, v in computed_batch.to_dict().items() if k in ["loss", "losses"]
        }

        if return_batch.get("loss", 0.0) == 0.0:
            rank_zero_warn("Loss is 0.0, skipping step...")
            return None

        return return_batch

    def on_train_epoch_end(self):
        """Called at end of training epoch."""
        self.on_epoch_end(stage=TrainingStage.TRAIN)

    def validation_step(self, batch: DataObject, batch_idx: int):
        """Validation step."""
        computed_batch = self.model_step(batch)
        self.on_step_log(computed_batch, stage=TrainingStage.VAL, batch_idx=batch_idx)
        return computed_batch.to_dict()

    def on_validation_epoch_end(self):
        """Called at end of validation epoch."""
        self.on_epoch_end(stage=TrainingStage.VAL)

    def test_step(self, batch: DataObject, batch_idx: int):
        """Test step."""
        computed_batch = self.model_step(batch)
        self.on_step_log(computed_batch, stage=TrainingStage.TEST, batch_idx=batch_idx)
        return computed_batch.to_dict()

    def on_test_epoch_end(self):
        """Called at end of test epoch."""
        self.on_epoch_end(stage=TrainingStage.TEST)

    def on_step_log(self, computed_batch: DataObject, stage: TRAINING_STAGE, batch_idx: int = None):
        """Log metrics and losses for a step.

        When log_metrics_every_n_steps > 1, metrics are only updated and logged
        every N steps (epoch aggregates then use those steps only).

        Args:
            computed_batch: DataObject with computed outputs
            stage: Training stage (train/val/test)
            batch_idx: Current batch index (required when log_metrics_every_n_steps > 1).
        """
        on_step = stage == TrainingStage.TRAIN
        n = self.hparams.log_metrics_every_n_steps
        do_metrics_this_step = (
            batch_idx is None or n <= 1 or (batch_idx % n == 0)
        )

        # Log losses
        for loss in self.criteria:
            self.log(
                f"{stage}/{loss.name}",
                computed_batch.losses[loss.name],
                on_step=on_step,
                on_epoch=True,
                prog_bar=True,
                batch_size=computed_batch.output.shape[0],
            )

        if computed_batch.losses.get("model_loss", None) is not None:
            self.log(
                f"{stage}/model_loss",
                computed_batch.losses["model_loss"],
                on_step=on_step,
                on_epoch=True,
                prog_bar=True,
                batch_size=computed_batch.output.shape[0],
            )

        # Update and log metrics (every step if n==1, else every n-th step)
        if do_metrics_this_step:
            for metric in self.metrics:
                if "best" in metric.name.lower():
                    continue

                metric.log(computed_batch, stage)
                self.log(
                    f"{stage}/{metric.name}",
                    metric.get(stage),
                    on_step=on_step,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=computed_batch.output.shape[0],
                )

        # Sum non-None losses
        loss = sum(filter(None, computed_batch.losses.values()))
        computed_batch.loss = loss
        self.log(
            f"{stage}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=computed_batch.output.shape[0],
        )

    def on_epoch_end(self, stage: TRAINING_STAGE):
        """Called at end of epoch for best metric tracking."""
        for metric in self.metrics:
            if "best" in metric.name.lower():
                continue

            value = metric.get(stage).compute()
            best_metric = [
                m
                for m in self.metrics
                if "best" in m.name.lower() and remove_best_from_string(m.name) == metric.name
            ]

            if len(best_metric) == 0:
                continue

            best_metric = best_metric[0]
            best_metric.log(value, stage=stage)
            self.log(
                f"{stage}/{metric.name}_best",
                best_metric.get(stage).compute(),
                prog_bar=True,
            )

        # Reset metrics
        for metric in self.metrics:
            try:
                self.get_submodule(f"{metric.name} {stage}")
            except AttributeError:
                metric.get(stage).reset()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        try:
            params = self.net.ensemble_parameters()
        except AttributeError:
            params = self.parameters()

        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                interval = "step"
            else:
                interval = "epoch"

            ret = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": interval,
                    "frequency": 1,
                    "strict": False,
                    "name": "train/lr",
                },
            }
        else:
            ret = {"optimizer": optimizer}

        if hasattr(self.net, "ensemble") and self.net.ensemble:
            self.net.model.ensemble_optimizer = optimizer
            self.net.model.ensemble_scheduler = scheduler

        return ret

    def add_metrics_modules(self):
        """Add metric modules to this module for proper device placement."""
        for i, metric in enumerate(self.metrics):
            for stage in ["train", "val", "test"]:
                if not metric.add_to_modules:
                    continue

                self.add_module(f"{metric.name} {stage}", getattr(metric, f"{stage}_metric"))
                self.metrics[i].set(stage, metric.get(stage).to(self.device))

    def log(
        self,
        name: str,
        value: METRIC_PL,
        prog_bar: bool = False,
        logger: Optional[bool] = None,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: str | Callable = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
        is_image: bool = False,
    ) -> None:
        """Override log to handle image logging."""
        if isinstance(value, ImageLoggingMetric) or is_image:
            # Handle image logging
            if isinstance(value, ImageLoggingMetric) and value is not None:
                value = value.get_image()

            if value is None:
                return

            for logger in self.trainer.loggers:
                if WandbLogger is not None and isinstance(logger, WandbLogger):
                    image = wandb.Image(value, caption="Data, output, target")
                    logger.experiment.log({name: image}, commit=True)
                elif TensorBoardLogger is not None and isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_image(name, value, global_step=self.global_step)
                elif MLFlowLogger is not None and isinstance(logger, MLFlowLogger):
                    logger.experiment.log_artifact(value, name)
                elif NeptuneLogger is not None and isinstance(logger, NeptuneLogger):
                    logger.experiment[name].log(value)
                elif CometLogger is not None and isinstance(logger, CometLogger):
                    logger.experiment.log_image(name, value, step=self.global_step)
                elif AimLogger is not None and isinstance(logger, AimLogger):
                    if value.ndim == 3 and value.shape[2] > 1:  # color
                        image = aim.Image(
                            (value * 255).astype(np.uint8),
                            caption="Data, output, target",
                            quality=100,
                        )
                    else:  # grayscale
                        value = value / value.max() * 255
                        value = value.numpy() if isinstance(value, torch.Tensor) else value
                        value = value.astype(np.uint8)
                        image = aim.Image(value, caption="Data, output, target", quality=100)
                    logger.experiment.track(
                        name=name, value=image, step=self.global_step, epoch=self.current_epoch
                    )
                elif not isinstance(logger, CSVLogger):
                    warn_once(
                        f"Warning: logger of class {logger.__class__.__qualname__} "
                        f"not supported for logging images."
                    )
        else:
            # Standard logging
            super().log(
                name=name,
                value=value,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                enable_graph=enable_graph,
                sync_dist=sync_dist,
                sync_dist_group=sync_dist_group,
                add_dataloader_idx=add_dataloader_idx,
                batch_size=batch_size,
                metric_attribute=metric_attribute,
                rank_zero_only=rank_zero_only,
            )
