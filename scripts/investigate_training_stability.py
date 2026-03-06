"""Diagnostic script: confirm FP16 overflow and normalization effects.

Loads one batch from the dataset, runs forward+backward passes in both
fp32 and fp16 for nfnet_f1 and efficientnet_b4 (pretrained=True), and
reports logit ranges, NaN/Inf presence, and gradient norms.

Demonstrates the root causes of:
  1. NaN loss (FP16 overflow without normalization)
  2. recall=1.0 (pretrained efficientnet with unnormalized input)

Usage::

    uv run python scripts/investigate_training_stability.py

Optional env vars:
    MALARIA_CACHE_DIR  — path to malaria_cache directory
    MALARIA_IMAGE_ROOT — path to autoscope_images
    PATCH_CACHE_VERSION, PATCH_SIZE, MAX_Z, SPLIT_SEED
"""

import json
import os
import sys
from contextlib import contextmanager

import numpy as np
import rootutils
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.malaria_patch_dataset import MalariaPatchDataset
from src.data.malaria_patch_datamodule import _split_by_patient_id
from src.models.components.timm_classifier import TimmClassifier
from src.utils.data_objects import custom_object_collate_fn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = os.environ.get("MALARIA_CACHE_DIR", "/data/mtanzer/malaria_cache")
IMAGE_ROOT = os.environ.get("MALARIA_IMAGE_ROOT", "/data/mtanzer/autoscope_images")
PATCH_CACHE_VERSION = int(os.environ.get("PATCH_CACHE_VERSION", "1"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "144"))
MAX_Z = int(os.environ.get("MAX_Z", "11"))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODELS_TO_CHECK = [
    ("dm_nfnet_f1",    3, False),   # (model_name, in_chans, pretrained)
    ("efficientnet_b4", 3, True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_one_batch(normalize=False):
    """Return a single batch (B=4) from the training set."""
    import pandas as pd

    samples_path = os.path.join(CACHE_DIR, "samples.parquet")
    zmap_path = os.path.join(CACHE_DIR, "z_stack_file_map.json")
    if not os.path.isfile(samples_path):
        sys.exit(f"ERROR: {samples_path} not found. Run prepare_data first.")

    samples_df = pd.read_parquet(samples_path)
    with open(zmap_path) as f:
        z_stack_file_map = json.load(f)
    z_stack_file_map = {
        k: {int(zi): v for zi, v in zmap.items()}
        for k, zmap in z_stack_file_map.items()
    }
    samples = samples_df.to_dict(orient="records")
    train_samples, _, _ = _split_by_patient_id(samples, [0.7, 0.15, 0.15], SPLIT_SEED)

    from src.data.components.patch_cache import get_cache_dir
    patch_cache_dir = get_cache_dir(CACHE_DIR, PATCH_CACHE_VERSION, PATCH_SIZE, MAX_Z)

    tfms = [ToTensor(), Resize([224, 224], antialias=True)]
    if normalize:
        tfms.append(Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    dataset = MalariaPatchDataset(
        samples=train_samples[:8],
        z_stack_file_map=z_stack_file_map,
        image_root=IMAGE_ROOT,
        patch_size=PATCH_SIZE,
        z_mode="best",
        max_z=MAX_Z,
        transform=Compose(tfms),
        cache_dir=patch_cache_dir,
        cache_mmap=False,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=custom_object_collate_fn)
    return next(iter(loader))


def tensor_stats(t, name=""):
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    return (f"{name:30s}  min={t.float().min():.4f}  max={t.float().max():.4f}  "
            f"mean={t.float().mean():.4f}  std={t.float().std():.4f}  "
            f"NaN={has_nan}  Inf={has_inf}")


def check_model(model_name, in_chans, pretrained, batch, dtype_str):
    print(f"\n  --- {model_name}  pretrained={pretrained}  dtype={dtype_str} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dtype_str == "fp16" else torch.float32

    model = TimmClassifier(
        model_name=model_name,
        in_chans=in_chans,
        num_classes=1,
        pretrained=pretrained,
        drop_rate=0.0,
    ).to(device).to(dtype)
    model.train()

    imgs = batch.data.to(device=device, dtype=dtype)
    tgt  = batch.target.to(device=device, dtype=dtype)

    print("  " + tensor_stats(imgs, "input"))

    try:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype_str == "fp16")):
            logits = model(imgs)

        print("  " + tensor_stats(logits, "logits"))

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, tgt)
        print(f"  {'loss':30s}  value={loss.item():.6f}  NaN={torch.isnan(loss).item()}")

        loss.backward()

        # Check gradient norms
        total_grad_norm = 0.0
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    has_nan_grad = True
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  {'grad norm':30s}  {total_grad_norm:.4f}  NaN_grad={has_nan_grad}")

    except Exception as e:
        print(f"  EXCEPTION: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Training stability diagnostic")
    print("=" * 70)

    for normalize in [False, True]:
        label = "WITH normalization (ImageNet)" if normalize else "WITHOUT normalization"
        print(f"\n{'='*70}")
        print(f"Input: {label}")
        print("=" * 70)

        try:
            batch = load_one_batch(normalize=normalize)
        except Exception as e:
            print(f"Could not load batch: {e}")
            continue

        print(tensor_stats(batch.data, "raw batch"))

        for model_name, in_chans, pretrained in MODELS_TO_CHECK:
            for dtype_str in ["fp32", "fp16"]:
                check_model(model_name, in_chans, pretrained, batch, dtype_str)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("Expected findings:")
    print("  WITHOUT normalization + fp16  -> NaN loss or NaN gradients")
    print("  WITHOUT normalization + fp32  -> large logit range, no NaN (fp32 headroom)")
    print("  pretrained efficientnet, no norm -> logits systematically positive")
    print("  WITH normalization + fp16    -> stable loss, no NaN")


if __name__ == "__main__":
    main()
