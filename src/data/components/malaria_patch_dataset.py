"""PyTorch Dataset for malaria microscopy patch classification.

Each sample corresponds to a unique (z_stack_filename, x, y) annotation group
from the autoscope database.  Images are loaded lazily -- only metadata lives
in memory.
"""

import os
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.data_objects import DataObject, MalariaProperties


_MISSING_Z_WARNED = False


def brenner_score(patch):
    """Brenner gradient focus measure -- higher means sharper.

    Args:
        patch: numpy array with shape (H, W, C) or (H, W).

    Returns:
        Scalar float score.
    """
    gray = patch.mean(axis=2) if patch.ndim == 3 else patch
    return float(((gray[2:, :] - gray[:-2, :]) ** 2).sum())


def _crop_patch(image, cx, cy, patch_size):
    """Crop a square patch centred at (cx, cy), zero-padding at edges.

    Args:
        image: numpy array (H, W, C).
        cx: centre x (column).
        cy: centre y (row).
        patch_size: side length of the square patch.

    Returns:
        numpy array (patch_size, patch_size, C).
    """
    h, w = image.shape[:2]
    half = patch_size // 2

    y1, y2 = cy - half, cy + (patch_size - half)
    x1, x2 = cx - half, cx + (patch_size - half)

    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2 - h)
    pad_right = max(0, x2 - w)

    y1_c = max(0, y1)
    x1_c = max(0, x1)
    y2_c = min(h, y2)
    x2_c = min(w, x2)

    crop = image[y1_c:y2_c, x1_c:x2_c]

    if pad_top or pad_bottom or pad_left or pad_right:
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return crop


class MalariaPatchDataset(Dataset):
    """Lazy-loading dataset of malaria microscopy patches.

    For each sample the dataset loads tile images for every (or the best)
    z-index, crops a ``patch_size x patch_size`` window around the annotated
    (x, y) coordinate and returns a :class:`DataObject`.

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
    ):
        """Initialise dataset -- see class docstring for arg details."""
        self.samples = samples
        self.z_stack_file_map = z_stack_file_map
        self.image_root = image_root
        self.patch_size = patch_size
        self.z_mode = z_mode
        self.max_z = max_z
        self.transform = transform

    def __len__(self):
        """Number of patch samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load patch images, optionally select best z, return DataObject."""
        global _MISSING_Z_WARNED
        sample = self.samples[idx]

        zstack = sample["z_stack_filename"]
        cx = int(round(sample["x"]))
        cy = int(round(sample["y"]))
        label = sample["label"]

        z_map = self.z_stack_file_map.get(zstack, {})
        sorted_z = sorted(z_map.keys())

        patches = []
        z_indices_used = []
        filepaths = []

        for z_idx in sorted_z:
            rel_path = z_map[z_idx].replace("\\", "/")
            full_path = os.path.join(self.image_root, rel_path)

            if not os.path.isfile(full_path):
                if not _MISSING_Z_WARNED:
                    warnings.warn(
                        f"Missing z-index image (padding with zeros): {full_path}. "
                        "Further missing-file warnings are suppressed.",
                        stacklevel=2,
                    )
                    _MISSING_Z_WARNED = True
                patch = np.zeros(
                    (self.patch_size, self.patch_size, 3), dtype=np.uint8
                )
            else:
                img = cv2.imread(full_path, cv2.IMREAD_COLOR)
                if img is None:
                    patch = np.zeros(
                        (self.patch_size, self.patch_size, 3), dtype=np.uint8
                    )
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    patch = _crop_patch(img, cx, cy, self.patch_size)

            patches.append(patch)
            z_indices_used.append(z_idx)
            filepaths.append(full_path)

        if not patches:
            patches = [
                np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            ]
            z_indices_used = [-1]
            filepaths = [""]

        if self.z_mode == "best":
            scores = [brenner_score(p) for p in patches]
            best_idx = int(np.argmax(scores))
            stacked = patches[best_idx]
            z_indices_used = [z_indices_used[best_idx]]
            filepaths = [filepaths[best_idx]]
        else:
            # Pad or truncate to exactly max_z slices for consistent channel count
            zero_patch = np.zeros(
                (self.patch_size, self.patch_size, 3), dtype=np.uint8
            )
            while len(patches) < self.max_z:
                patches.append(zero_patch)
                z_indices_used.append(-1)
                filepaths.append("")
            patches = patches[: self.max_z]
            z_indices_used = z_indices_used[: self.max_z]
            filepaths = filepaths[: self.max_z]

            stacked = np.concatenate(patches, axis=2)

        # stacked is uint8 numpy (H, W, C). Transforms (e.g. ToTensor) handle
        # the uint8->float /255 normalisation and HWC->CHW permutation.
        if self.transform is not None:
            tensor = self.transform(stacked)
        else:
            tensor = torch.from_numpy(stacked.astype(np.float32) / 255.0).permute(2, 0, 1)

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
