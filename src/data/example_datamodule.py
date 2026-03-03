from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.data.components.example_dataset import ExampleDataset
from src.utils.data_objects import custom_object_collate_fn


class ExampleDataModule(LightningDataModule):
    """Example DataModule demonstrating DataObject usage.
    
    Shows how to:
    - Use DataObject in datasets
    - Use custom collate function
    - Handle train/val/test splits
    """
    
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: tuple = (0.8, 0.1, 0.1),
        noise_level: float = 0.1,
        use_properties: bool = True,
    ):
        """Initialize example datamodule.
        
        Args:
            data_dir: Directory containing data (not used in this example)
            batch_size: Batch size
            num_workers: Number of data loader workers
            pin_memory: Whether to pin memory
            train_val_test_split: Split ratios (train, val, test)
            noise_level: Noise level for synthetic data
            use_properties: Whether to use DenoisingProperties
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.noise_level = noise_level
        self.use_properties = use_properties

        # Generate synthetic data for example
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        """Download or prepare data (called only on main process)."""
        # In a real scenario, you would download/prepare data here
        pass

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        # Generate synthetic data for demonstration
        # In practice, you would load real data here
        num_samples = 1000
        image_size = (3, 64, 64)
        
        # Generate clean images (random for example)
        clean_images = torch.randn(num_samples, *image_size)
        
        # Add noise
        noisy_images = clean_images + self.noise_level * torch.randn_like(clean_images)

        # Create full dataset
        full_dataset = ExampleDataset(
            data=noisy_images,
            target=clean_images,
            noise_level=self.noise_level,
            use_properties=self.use_properties,
        )

        # Split dataset
        train_size = int(self.train_val_test_split[0] * len(full_dataset))
        val_size = int(self.train_val_test_split[1] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_data, self.val_data, self.test_data = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=custom_object_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=custom_object_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=custom_object_collate_fn,
        )
