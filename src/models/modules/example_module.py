import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError

from src.models.components.model_wrapper import DataObjectModelWrapper
from src.models.metrics.utils import PairedMetric, metric_wrapper
from src.models.modules.base import BaseLitModule
from src.utils.data_objects import DataObject


class SimpleDenoisingNet(nn.Module):
    """Simple CNN for image denoising."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class MSELoss(nn.Module):
    """MSE Loss that works with DataObject."""
    
    def __init__(self):
        super().__init__()
        self.name = "mse"
        self.mse = nn.MSELoss()
    
    def forward(self, data_object: DataObject):
        """Compute MSE loss between output and target."""
        return self.mse(data_object.output, data_object.target)


class ExampleModule(BaseLitModule):
    """Example Lightning module demonstrating BaseLitModule usage.
    
    Shows how to:
    - Inherit from BaseLitModule
    - Use DataObjectModelWrapper
    - Define losses and metrics
    - Work with DataObject
    """
    
    def __init__(
        self,
        net: DataObjectModelWrapper,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        lr: float = 1e-3,
    ):
        """Initialize example module.
        
        Args:
            net: Model wrapped in DataObjectModelWrapper
            optimizer: Optimizer class
            scheduler: Optional scheduler class
            lr: Learning rate
        """
        # Define loss function (must work with DataObject)
        mse_loss = MSELoss()
        
        # Define metrics
        mse_metric = metric_wrapper(MeanSquaredError, "mse", "paired")(
            name="mse",
            add_to_modules=True,
        )

        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            criteria=[mse_loss],
            metrics=[mse_metric],
            name="Example",
        )
        
        self.lr = lr

    def configure_optimizers(self):
        """Configure optimizer with learning rate."""
        # Use lr from hyperparameters or instance variable
        lr = getattr(self.hparams, "lr", self.lr) if hasattr(self, "hparams") else self.lr
        optimizer = self.hparams.optimizer(params=self.parameters(), lr=lr)
        
        if hasattr(self.hparams, "scheduler") and self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}
