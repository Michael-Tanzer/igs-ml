"""I/O helpers for loading and cropping z-slice patches.

Used by :mod:`src.data.components.patch_cache` to avoid circular imports
with :mod:`src.data.components.malaria_patch_dataset`.
"""

import cv2

cv2.setLogLevel(2)  # ERROR level; named constants not available in all builds

import numpy as np


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


def load_z_slice(full_path, cx, cy, patch_size):
    """Read a single z-slice image from disk and crop the patch around (cx, cy).

    Args:
        full_path: Absolute path to the tile image.
        cx: Centre x (column) of the crop.
        cy: Centre y (row) of the crop.
        patch_size: Side length of the square crop.

    Returns:
        numpy array (patch_size, patch_size, 3) uint8.
    """
    img = cv2.imread(full_path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _crop_patch(img, cx, cy, patch_size)
