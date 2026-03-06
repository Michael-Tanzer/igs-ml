"""LightningDataModule for malaria patch binary classification.

Queries the autoscope DB once (caching results to disk), groups rows into
per-patch samples across the z-stack, splits by patient ID, and exposes
streaming DataLoaders that load tile images lazily.  Optionally pre-crops
patches to local SSD via the per-sample ``.npy`` cache.
"""

import glob as _glob
import hashlib
import json
import os
import warnings

import hydra
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.data.components.malaria_patch_dataset import MalariaPatchDataset
from src.data.components.patch_cache import (
    SENTINEL_ALL_CACHED,
    check_all_cached,
    get_cache_dir,
    precompute_all,
)
from src.data.utils.db_client import get_connection, run_query_from_file
from src.utils.data_objects import custom_object_collate_fn


def _query_hash(query_path: str) -> str:
    """Return an MD5 hex digest of all .sql files in the same directory as query_path."""
    sql_dir = os.path.dirname(os.path.abspath(query_path))
    content = b""
    for sql_file in sorted(_glob.glob(os.path.join(sql_dir, "*.sql"))):
        with open(sql_file, "rb") as f:
            content += f.read()
    return hashlib.md5(content).hexdigest()


def _build_z_stack_file_map(df):
    """Build ``{z_stack_filename: {z_index: file_location}}`` from the raw query DataFrame.

    Args:
        df: pandas DataFrame with at least ``z_stack_filename``, ``z_index``,
            ``file_location`` columns.

    Returns:
        Dict mapping z-stack name to a dict of z-index -> normalised file path.
    """
    z_map = {}
    for zstack, z_idx, floc in zip(
        df["z_stack_filename"], df["z_index"], df["file_location"]
    ):
        floc_norm = str(floc).replace("\\", "/")
        z_map.setdefault(zstack, {})[int(z_idx)] = floc_norm
    return z_map


def _group_samples(df, label_column):
    """Group rows by (z_stack_filename, x, y) into patch samples.

    Args:
        df: DataFrame filtered to non-null labels.
        label_column: Name of the label column (e.g. ``y_binary_strict``).

    Returns:
        List of sample dicts.
    """
    df = df.copy()
    df["x_int"] = df["x"].round().astype(int)
    df["y_int"] = df["y"].round().astype(int)

    grouped = df.groupby(["z_stack_filename", "x_int", "y_int"])

    samples = []
    for (zstack, xi, yi), grp in grouped:
        label_mode = grp[label_column].mode()
        if label_mode.empty:
            continue
        label = float(label_mode.iloc[0])

        first = grp.iloc[0]
        samples.append(
            {
                "z_stack_filename": zstack,
                "x": float(xi),
                "y": float(yi),
                "label": label,
                "object_ids": grp["object_id"].tolist(),
                "PID": str(first.get("PID", "")),
                "species": str(first.get("species", "")),
                "stage": str(first.get("stage", "")),
                "smear_type": str(first.get("smear_type", "")),
                "parasitemia": str(first.get("parasitemia", "")),
                "id_image_set": int(first.get("id_image_set", -1)),
                "z_stack_height": int(first.get("z_stack_height", 0)),
                "pixels_per_micron": float(first.get("pixels_per_micron", 0.0)),
            }
        )

    return samples


def _split_by_pid(samples, ratios, seed):
    """Deterministically split samples by PID.

    Args:
        samples: List of sample dicts (each has ``PID`` key).
        ratios: 3-element list ``[train, val, test]`` that sums to 1.
        seed: Random seed for reproducibility.

    Returns:
        Tuple ``(train_samples, val_samples, test_samples)``.
    """
    pids = sorted(set(s["PID"] for s in samples))
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)

    n = len(pids)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))

    train_pids = set(pids[:n_train])
    val_pids = set(pids[n_train : n_train + n_val])
    test_pids = set(pids[n_train + n_val :])

    train = [s for s in samples if s["PID"] in train_pids]
    val = [s for s in samples if s["PID"] in val_pids]
    test = [s for s in samples if s["PID"] in test_pids]

    return train, val, test


def _instantiate_transforms(cfg_list):
    """Hydra-instantiate a list of transform configs.

    Args:
        cfg_list: OmegaConf list of dicts, each with ``_target_`` and params,
            or a list of already-instantiated transform objects.

    Returns:
        List of instantiated transform objects.
    """
    if cfg_list is None:
        return []
    out = []
    for t in cfg_list:
        if isinstance(t, (dict, DictConfig)):
            out.append(hydra.utils.instantiate(t))
        else:
            out.append(t)
    return out


class MalariaPatchDataModule(LightningDataModule):
    """LightningDataModule for malaria patch binary classification.

    Handles DB querying, caching, PID-based splitting, and streaming
    DataLoaders.  See ``configs/data/malaria_patch.yaml`` for the full
    set of configurable parameters.
    """

    def __init__(
        self,
        db,
        query_path,
        image_root,
        patch_size,
        label_column,
        z_mode,
        max_z,
        train_val_test_split,
        split_seed,
        batch_size,
        num_workers,
        prefetch_factor,
        persistent_workers,
        pin_memory,
        cache_dir,
        io_threads=11,
        patch_cache="auto",
        patch_cache_version=1,
        patch_cache_mmap=True,
        precompute_workers=1,
        transforms_base=None,
        transforms_augment=None,
    ):
        """Initialise the data module.

        Args:
            db: Database connection config (host, port, database, user,
                password) -- typically ``${db}`` from Hydra.
            query_path: Path to the SQL query file (relative to project root).
            image_root: Base directory for tile images.
            patch_size: Side length of the square crop.
            label_column: DB column used as label (``y_binary_strict`` or
                ``y_binary_inclusive``).
            z_mode: ``"all"`` or ``"best"`` (Brenner focus selection).
            max_z: Fixed number of z-slices for ``z_mode="all"`` (pads/truncates).
            train_val_test_split: 3-element list of ratios.
            split_seed: Seed for the PID split RNG.
            batch_size: Batch size.
            num_workers: DataLoader workers.
            prefetch_factor: Batches prefetched per worker.
            persistent_workers: Keep workers alive between epochs.
            pin_memory: Pin memory for GPU transfer.
            cache_dir: Directory for cached metadata.
            io_threads: Number of threads per DataLoader worker for parallel
                z-slice I/O reads (default matches ``max_z``).
            patch_cache: ``"auto"`` (precompute missing, use cache),
                ``"require"`` (error if missing), or ``"off"`` (NAS only).
            patch_cache_version: Manual version integer -- bump to invalidate
                the precomputed patch cache.
            patch_cache_mmap: If True, load cached .npy with mmap for faster
                access. No effect when patch_cache is ``"off"``.
            precompute_workers: Number of parallel processes for patch
                precomputation (1 = single-process).
            transforms_base: Hydra-instantiable list applied to all splits.
            transforms_augment: Hydra-instantiable list applied to train only.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = batch_size

    def _load_and_split(self):
        """Load cached metadata and split samples by PID.

        Returns:
            Tuple of ``(all_samples, z_stack_file_map,
            train_samples, val_samples, test_samples)``.
        """
        cache_dir = self.hparams.cache_dir
        samples_df = pd.read_parquet(os.path.join(cache_dir, "samples.parquet"))
        with open(os.path.join(cache_dir, "z_stack_file_map.json")) as f:
            z_stack_file_map = json.load(f)

        z_stack_file_map = {
            zstack: {int(k): v for k, v in zmap.items()}
            for zstack, zmap in z_stack_file_map.items()
        }

        # Filter to only samples produced by the current SQL query.
        hash_ids_path = os.path.join(
            cache_dir, f"query_{_query_hash(self.hparams.query_path)}_ids.parquet"
        )
        if os.path.isfile(hash_ids_path):
            keys = pd.read_parquet(hash_ids_path)
            samples_df = samples_df.copy()
            samples_df["x_int"] = samples_df["x"].round().astype(int)
            samples_df["y_int"] = samples_df["y"].round().astype(int)
            samples_df = samples_df.merge(
                keys[["z_stack_filename", "x_int", "y_int"]],
                on=["z_stack_filename", "x_int", "y_int"],
                how="inner",
            ).drop(columns=["x_int", "y_int"])

        samples = samples_df.to_dict(orient="records")

        train_samples, val_samples, test_samples = _split_by_pid(
            samples,
            list(self.hparams.train_val_test_split),
            self.hparams.split_seed,
        )

        return samples, z_stack_file_map, train_samples, val_samples, test_samples

    def _get_patch_cache_dir(self):
        """Return the versioned patch cache directory, or ``None`` if caching is off."""
        if self.hparams.patch_cache == "off":
            return None
        return get_cache_dir(
            self.hparams.cache_dir,
            self.hparams.patch_cache_version,
            self.hparams.patch_size,
            self.hparams.max_z,
        )

    def prepare_data(self):
        """Query DB, group samples, cache to disk, and precompute patches.

        Runs on rank-0 only.  The patch precompute step (controlled by
        ``patch_cache``) skips samples that already have a cached ``.npy``.
        """
        cache_dir = self.hparams.cache_dir
        samples_path = os.path.join(cache_dir, "samples.parquet")
        zmap_path = os.path.join(cache_dir, "z_stack_file_map.json")

        query_hash = _query_hash(self.hparams.query_path)
        hash_ids_path = os.path.join(cache_dir, f"query_{query_hash}_ids.parquet")

        if not (os.path.isfile(samples_path) and os.path.isfile(zmap_path)):
            os.makedirs(cache_dir, exist_ok=True)

            db_cfg = OmegaConf.to_container(self.hparams.db, resolve=True)
            with get_connection(db_cfg) as conn:
                rows = run_query_from_file(conn, self.hparams.query_path)

            df = pd.DataFrame(rows)
            if df.empty:
                warnings.warn("DB query returned 0 rows -- dataset will be empty.")
                pd.DataFrame().to_parquet(samples_path)
                with open(zmap_path, "w") as f:
                    json.dump({}, f)
                return

            z_stack_file_map = _build_z_stack_file_map(df)

            label_col = self.hparams.label_column
            df_filtered = df[df[label_col].notna()].copy()
            samples = _group_samples(df_filtered, label_col)

            pd.DataFrame(samples).to_parquet(samples_path, index=False)
            with open(zmap_path, "w") as f:
                json.dump(z_stack_file_map, f)

            # Write hash-keyed ID file so _load_and_split can filter correctly.
            pd.DataFrame(
                [{"z_stack_filename": s["z_stack_filename"],
                  "x_int": int(round(s["x"])),
                  "y_int": int(round(s["y"]))}
                 for s in samples]
            ).to_parquet(hash_ids_path, index=False)

        elif not os.path.isfile(hash_ids_path):
            # samples.parquet exists but was built from a different query; re-run
            # the current query to record which samples it produces.
            os.makedirs(cache_dir, exist_ok=True)
            db_cfg = OmegaConf.to_container(self.hparams.db, resolve=True)
            with get_connection(db_cfg) as conn:
                rows = run_query_from_file(conn, self.hparams.query_path)

            df = pd.DataFrame(rows)
            if not df.empty:
                label_col = self.hparams.label_column
                df_filtered = df[df[label_col].notna()].copy()
                samples = _group_samples(df_filtered, label_col)
            else:
                samples = []

            pd.DataFrame(
                [{"z_stack_filename": s["z_stack_filename"],
                  "x_int": int(round(s["x"])),
                  "y_int": int(round(s["y"]))}
                 for s in samples]
            ).to_parquet(hash_ids_path, index=False)

        patch_cache_dir = self._get_patch_cache_dir()
        if patch_cache_dir is None:
            return

        all_samples, z_stack_file_map, *_ = self._load_and_split()

        if self.hparams.patch_cache == "require":
            missing = check_all_cached(
                all_samples,
                patch_cache_dir,
                write_sentinel_if_complete=True,
            )
            if missing:
                raise FileNotFoundError(
                    f"patch_cache='require' but {len(missing)} samples are "
                    f"missing from {patch_cache_dir}. Run the precompute "
                    f"script first or set patch_cache='auto'."
                )
        elif self.hparams.patch_cache == "auto":
            precompute_all(
                all_samples,
                z_stack_file_map,
                self.hparams.image_root,
                self.hparams.patch_size,
                self.hparams.max_z,
                patch_cache_dir,
                io_threads=self.hparams.io_threads,
                workers=self.hparams.precompute_workers,
            )
            # Write sentinel so next run can skip check_all_cached walk
            with open(os.path.join(patch_cache_dir, SENTINEL_ALL_CACHED), "w"):
                pass

    def setup(self, stage=None):
        """Load cache, split by PID, build transforms, create Datasets."""
        if self.data_train is not None:
            return

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible "
                    f"by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        _, z_stack_file_map, train_samples, val_samples, test_samples = (
            self._load_and_split()
        )

        base_tfms = _instantiate_transforms(self.hparams.transforms_base)
        aug_tfms = _instantiate_transforms(self.hparams.transforms_augment)

        train_transform = Compose(base_tfms + aug_tfms) if (base_tfms or aug_tfms) else None
        eval_transform = Compose(base_tfms) if base_tfms else None

        patch_cache_dir = self._get_patch_cache_dir()

        common = dict(
            z_stack_file_map=z_stack_file_map,
            image_root=self.hparams.image_root,
            patch_size=self.hparams.patch_size,
            z_mode=self.hparams.z_mode,
            max_z=self.hparams.max_z,
            io_threads=self.hparams.io_threads,
            cache_dir=patch_cache_dir,
            cache_mmap=self.hparams.patch_cache_mmap,
        )

        self.data_train = MalariaPatchDataset(
            samples=train_samples, transform=train_transform, **common
        )
        self.data_val = MalariaPatchDataset(
            samples=val_samples, transform=eval_transform, **common
        )
        self.data_test = MalariaPatchDataset(
            samples=test_samples, transform=eval_transform, **common
        )

    def _dataloader(self, dataset, shuffle):
        """Build a DataLoader with shared settings.

        Args:
            dataset: The MalariaPatchDataset instance.
            shuffle: Whether to shuffle.

        Returns:
            DataLoader.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=custom_object_collate_fn,
        )

    def train_dataloader(self):
        """Return the training DataLoader."""
        return self._dataloader(self.data_train, shuffle=True)

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return self._dataloader(self.data_val, shuffle=False)

    def test_dataloader(self):
        """Return the test DataLoader."""
        return self._dataloader(self.data_test, shuffle=False)

    def teardown(self, stage=None):
        """Clean up (no-op)."""
        pass

    def state_dict(self):
        """Return empty state dict (all state is on disk)."""
        return {}

    def load_state_dict(self, state_dict):
        """Load state dict (no-op)."""
        pass
