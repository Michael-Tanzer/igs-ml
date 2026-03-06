"""Compute per-channel mean and std of the malaria patch dataset.

Uses Welford's online algorithm to accumulate statistics incrementally,
so the full dataset never needs to fit in RAM.

For z_mode=all (33-channel) images, each z-slice is treated as an
independent 3-channel RGB observation — yielding a single 3-element
mean/std that can be reused for any z_mode:
  - z_mode=best  (3 ch):  use mean/std directly
  - z_mode=all  (33 ch):  repeat each value 11 times in the config

Results are saved to data/dataset_stats.json and printed for copy-paste.

Usage::

    uv run python scripts/compute_dataset_stats.py

The script reads from the pre-built metadata cache (samples.parquet,
z_stack_file_map.json) so the DB is not required.
"""

import json
import os
import sys

import numpy as np
import rootutils
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.malaria_patch_dataset import MalariaPatchDataset
from src.data.malaria_patch_datamodule import _split_by_patient_id
from src.utils.data_objects import custom_object_collate_fn

# ---------------------------------------------------------------------------
# Configuration — edit these or pass via env vars
# ---------------------------------------------------------------------------
CACHE_DIR = os.environ.get(
    "MALARIA_CACHE_DIR",
    "/data/mtanzer/malaria_cache",
)
IMAGE_ROOT = os.environ.get(
    "MALARIA_IMAGE_ROOT",
    "/data/mtanzer/autoscope_images",
)
PATCH_CACHE_VERSION = int(os.environ.get("PATCH_CACHE_VERSION", "1"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "144"))
MAX_Z = int(os.environ.get("MAX_Z", "11"))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.7"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
OUTPUT_PATH = os.environ.get(
    "STATS_OUTPUT",
    os.path.join(os.path.dirname(__file__), "..", "data", "dataset_stats.json"),
)


def get_patch_cache_dir():
    from src.data.components.patch_cache import get_cache_dir
    return get_cache_dir(CACHE_DIR, PATCH_CACHE_VERSION, PATCH_SIZE, MAX_Z)


def load_train_samples():
    """Load train samples from the pre-built metadata cache."""
    import pandas as pd

    samples_path = os.path.join(CACHE_DIR, "samples.parquet")
    zmap_path = os.path.join(CACHE_DIR, "z_stack_file_map.json")

    if not os.path.isfile(samples_path):
        sys.exit(f"ERROR: samples.parquet not found at {samples_path}\n"
                 "Run training once with prepare_data=True or run the full datamodule first.")

    samples_df = pd.read_parquet(samples_path)
    with open(zmap_path) as f:
        z_stack_file_map = json.load(f)

    z_stack_file_map = {
        zstack: {int(k): v for k, v in zmap.items()}
        for zstack, zmap in z_stack_file_map.items()
    }

    samples = samples_df.to_dict(orient="records")
    train_samples, _, _ = _split_by_patient_id(
        samples,
        [TRAIN_RATIO, (1 - TRAIN_RATIO) / 2, (1 - TRAIN_RATIO) / 2],
        SPLIT_SEED,
    )
    print(f"Train samples: {len(train_samples):,}")
    return train_samples, z_stack_file_map


def build_dataloader(train_samples, z_stack_file_map):
    """Build a DataLoader with z_mode=all and no normalization."""
    patch_cache_dir = get_patch_cache_dir()

    transform = Compose([
        ToTensor(),
        Resize([224, 224], antialias=True),
    ])

    dataset = MalariaPatchDataset(
        samples=train_samples,
        z_stack_file_map=z_stack_file_map,
        image_root=IMAGE_ROOT,
        patch_size=PATCH_SIZE,
        z_mode="all",
        max_z=MAX_Z,
        transform=transform,
        cache_dir=patch_cache_dir,
        cache_mmap=False,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=custom_object_collate_fn,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS > 0,
    )


def compute_stats(loader):
    """Welford online mean/std over all z-slices treated as independent RGB images.

    Each 33-ch image is viewed as 11 independent 3-ch slices.
    Accumulates per-channel mean/variance over all (sample, z-slice, pixel).
    """
    # Welford state: 3 channels
    n = 0
    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)

    total_batches = len(loader)
    for batch in tqdm(loader, total=total_batches, desc="Computing stats"):
        # batch.data: (B, 33, H, W) float32 in [0, 1]
        imgs = batch.data  # tensor (B, 33, H, W)
        B, C, H, W = imgs.shape
        assert C == MAX_Z * 3, f"Expected {MAX_Z * 3} channels, got {C}"

        # Reshape to (B * MAX_Z, 3, H, W) — each z-slice is an independent RGB image
        imgs = imgs.view(B * MAX_Z, 3, H, W)  # (B*11, 3, H, W)

        # Per-image, per-channel spatial mean: (B*11, 3)
        img_means = imgs.mean(dim=[2, 3]).numpy().astype(np.float64)  # (N, 3)

        # Welford update (one observation = one z-slice image)
        for ch_means in img_means:
            n += 1
            delta = ch_means - mean
            mean += delta / n
            delta2 = ch_means - mean
            M2 += delta * delta2

    std = np.sqrt(M2 / max(n - 1, 1))
    return mean, std, n


def main():
    print("=" * 60)
    print("Malaria patch dataset statistics computation")
    print("=" * 60)

    train_samples, z_stack_file_map = load_train_samples()
    loader = build_dataloader(train_samples, z_stack_file_map)

    print(f"\nStreaming {len(loader):,} batches (batch_size={BATCH_SIZE}) "
          f"with {NUM_WORKERS} workers...")
    mean, std, n_obs = compute_stats(loader)

    print(f"\nDone. Computed over {n_obs:,} z-slice images "
          f"({n_obs // MAX_Z:,} samples × {MAX_Z} z-slices)\n")

    print("Per-channel RGB statistics (apply to each z-slice independently):")
    print(f"  mean: {mean.tolist()}")
    print(f"  std:  {std.tolist()}")

    # Build 33-element lists for z_mode=all
    mean_33 = (mean.tolist() * MAX_Z)
    std_33 = (std.tolist() * MAX_Z)

    print()
    print("For z_mode=best (3 channels) — paste into transforms_base Normalize:")
    print(f"  mean: {mean.tolist()}")
    print(f"  std:  {std.tolist()}")

    print()
    print(f"For z_mode=all ({MAX_Z * 3} channels) — paste into transforms_base Normalize:")
    print(f"  mean: {mean_33}")
    print(f"  std:  {std_33}")

    output = {
        "mean_3ch": mean.tolist(),
        "std_3ch": std.tolist(),
        "mean_33ch": mean_33,
        "std_33ch": std_33,
        "n_z_slice_images": int(n_obs),
        "n_samples": int(n_obs // MAX_Z),
        "max_z": MAX_Z,
        "patch_size": PATCH_SIZE,
        "resize": 224,
    }

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
