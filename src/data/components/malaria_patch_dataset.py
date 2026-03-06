"""Backup/reference copy of the cache-aware MalariaPatchDataset.

If malaria_patch_dataset.py is overwritten (e.g. by a revert), copy this file
over it to restore the version that uses load_or_generate, io_threads,
cache_dir, and cache_mmap.  See malaria_patch_dataset.py for the full
module docstring.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.data_objects import DataObject, MalariaProperties

def brenner_score(patch):
    """Brenner gradient focus measure -- higher means sharper.

    Args:
        patch: numpy array with shape (H, W, C) or (H, W).

    Returns:
        Scalar float score.
    """
    gray = patch.mean(axis=2) if patch.ndim == 3 else patch
    return float(((gray[2:, :] - gray[:-2, :]) ** 2).sum())


class MalariaPatchDataset(Dataset):
    """Dataset of malaria microscopy patches with optional local caching.

    For each sample the dataset loads tile images for every (or the best)
    z-index, crops a ``patch_size x patch_size`` window around the annotated
    (x, y) coordinate and returns a :class:`DataObject`.

    When ``cache_dir`` is set, patches are read from pre-cached ``.npy`` files
    on local SSD (see :mod:`src.data.components.patch_cache`).  On a cache
    miss the sample is loaded from the NAS and saved to the cache for next
    time.

    Args:
        samples: List of dicts, each with keys ``z_stack_filename``, ``x``,
            ``y``, ``label``, and DB metadata (``PID``, ``species``, etc.).
        z_stack_file_map: ``{z_stack_filename: {z_index: rel_path}}`` built
            during data preparation.
        image_root: Absolute base directory prepended to every relative
            ``file_location`` from the DB.
        patch_size: Side length (pixels) of the square crop around (x, y).
        z_mode: ``"all"`` to stack every z-index (3N channels) or ``"best"``
            to pick the sharpest via Brenner score (3 channels).
        max_z: Fixed number of z-slices when ``z_mode="all"``. Samples with
            fewer slices are zero-padded; samples with more are truncated.
            Ignored when ``z_mode="best"``. Required when ``z_mode="all"``
            so that every sample produces the same channel count (``3 * max_z``).
        transform: An optional ``torchvision.transforms.Compose`` (or similar
            callable) applied to the stacked tensor **after** it is converted
            to a torch Tensor.
        io_threads: Number of threads used to read z-slice images in parallel
            within each DataLoader worker. Defaults to 11 (matching ``max_z``).
        cache_dir: Versioned patch cache directory. When set, patches are
            loaded from / saved to ``.npy`` files here.  ``None`` disables
            caching (NAS-only mode).
        cache_mmap: If True, load cached ``.npy`` with mmap for faster reads.
            No effect when ``cache_dir`` is ``None``.
    """

    def __init__(
        self,
        samples,
        z_stack_file_map,
        image_root,
        patch_size,
        z_mode="all",
        max_z=11,
        transform=None,
        io_threads=11,
        cache_dir=None,
        cache_mmap=True,
    ):
        """Initialise dataset -- see class docstring for arg details."""
        self.samples = samples
        self.z_stack_file_map = z_stack_file_map
        self.image_root = image_root
        self.patch_size = patch_size
        self.z_mode = z_mode
        self.max_z = max_z
        self.transform = transform
        self.cache_dir = cache_dir
        self.cache_mmap = cache_mmap
        self._io_pool = ThreadPoolExecutor(max_workers=io_threads)

    def __len__(self):
        """Number of patch samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load patch images, optionally select best z, return DataObject."""
        from src.data.components.patch_cache import load_or_generate

        sample = self.samples[idx]
        zstack = sample["z_stack_filename"]
        label = sample["label"]

        # (max_z, patch_size, patch_size, 3) uint8 -- from cache or NAS
        all_patches = load_or_generate(
            sample,
            self.z_stack_file_map,
            self.image_root,
            self.patch_size,
            self.max_z,
            self.cache_dir,
            self._io_pool,
            use_mmap=self.cache_mmap,
        )

        # Reconstruct z-index / filepath metadata
        z_map = self.z_stack_file_map.get(zstack, {})
        sorted_z = sorted(z_map.keys())
        z_indices_used = list(sorted_z)
        filepaths = [
            os.path.join(self.image_root, z_map[z].replace("\\", "/"))
            for z in sorted_z
        ]
        while len(z_indices_used) < self.max_z:
            z_indices_used.append(-1)
            filepaths.append("")
        z_indices_used = z_indices_used[: self.max_z]
        filepaths = filepaths[: self.max_z]

        if self.z_mode == "best":
            scores = [
                brenner_score(all_patches[i])
                for i in range(all_patches.shape[0])
            ]
            best_idx = int(np.argmax(scores))
            stacked = all_patches[best_idx]
            z_indices_used = [z_indices_used[best_idx]]
            filepaths = [filepaths[best_idx]]
        else:
            # (max_z, H, W, 3) -> (H, W, max_z, 3) -> (H, W, 3*max_z)
            # Channels: R0G0B0 R1G1B1 ... (interleaved per z-slice).
            stacked = np.transpose(all_patches, (1, 2, 0, 3)).reshape(
                self.patch_size, self.patch_size, 3 * self.max_z
            )

        if not stacked.flags.writeable:
            stacked = stacked.copy()

        if self.transform is not None:
            tensor = self.transform(stacked)
        else:
            tensor = torch.from_numpy(
                stacked.astype(np.float32) / 255.0
            ).permute(2, 0, 1)

        target = torch.tensor(label, dtype=torch.float32)

        malaria = MalariaProperties(
            object_ids=sample.get("object_ids", []),
            z_stack_filename=zstack,
            z_indices_used=z_indices_used,
            id_image_set=sample.get("id_image_set", -1),
            PID=sample.get("PID", ""),
            species=sample.get("species", ""),
            stage=sample.get("stage", ""),
            smear_type=sample.get("smear_type", ""),
            parasitemia=sample.get("parasitemia", ""),
            z_stack_height=sample.get("z_stack_height", 0),
            pixels_per_micron=sample.get("pixels_per_micron", 0.0),
        )

        return DataObject(
            index=idx,
            data=tensor,
            target=target,
            data_filepath=filepaths,
            malaria=malaria,
        )