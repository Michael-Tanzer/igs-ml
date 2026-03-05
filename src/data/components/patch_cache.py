"""Per-sample patch caching utilities for malaria microscopy datasets.

Provides on-disk caching of cropped z-slice patches as individual ``.npy``
files on local SSD.  A versioned directory structure invalidates the cache
when format-affecting parameters (``patch_size``, ``max_z``, manual version)
change, while allowing query changes to reuse already-cached patches.
"""

import hashlib
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

from src.data.components.patch_io import load_z_slice


def get_cache_dir(base_cache_dir, patch_cache_version, patch_size, max_z):
    """Construct the versioned patch cache directory path.

    Args:
        base_cache_dir: Root cache directory (e.g. ``/data/.../malaria_cache``).
        patch_cache_version: Manual version integer for cache invalidation.
        patch_size: Patch crop size in pixels.
        max_z: Number of z-slices stored per sample.

    Returns:
        Absolute path to the versioned directory.
    """
    dirname = f"patches_v{patch_cache_version}_ps{patch_size}_mz{max_z}"
    return os.path.join(base_cache_dir, dirname)


SENTINEL_ALL_CACHED = ".all_cached"


def _existing_cache_paths_set(cache_dir):
    """Build set of all existing .npy paths under cache_dir with a single walk.

    Args:
        cache_dir: Versioned patch cache directory.

    Returns:
        Set of absolute paths to .npy files.
    """
    if not os.path.isdir(cache_dir):
        return set()
    out = set()
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            if name.endswith(".npy"):
                out.add(os.path.join(root, name))
    return out


def sample_cache_path(cache_dir, sample):
    """Compute the ``.npy`` file path for a single sample.

    Uses a SHA-256 hash of the sample identity ``(z_stack_filename, x, y)``
    with a 2-character subdirectory prefix to keep directories manageable.

    Args:
        cache_dir: Versioned patch cache directory.
        sample: Sample dict with ``z_stack_filename``, ``x``, ``y`` keys.

    Returns:
        Absolute path to the ``.npy`` file.
    """
    key = f"{sample['z_stack_filename']}|{int(round(sample['x']))}|{int(round(sample['y']))}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, h[:2], f"{h}.npy")


def _load_sample_from_nas(sample, z_stack_file_map, image_root, patch_size, max_z, io_pool=None):
    """Load all z-slice patches for a sample from NAS.

    Reads up to ``max_z`` z-slices from the NAS, cropping each around the
    sample's (x, y) coordinate.  Pads with zeros if fewer than ``max_z``
    slices are available, truncates if more.

    Args:
        sample: Sample dict.
        z_stack_file_map: ``{z_stack_filename: {z_index: rel_path}}``.
        image_root: Base directory for tile images on NAS.
        patch_size: Side length of the square crop.
        max_z: Fixed number of z-slices (pads/truncates).
        io_pool: Optional ``ThreadPoolExecutor`` for parallel reads.

    Returns:
        numpy array ``(max_z, patch_size, patch_size, 3)`` uint8.
    """
    zstack = sample["z_stack_filename"]
    cx = int(round(sample["x"]))
    cy = int(round(sample["y"]))

    z_map = z_stack_file_map.get(zstack, {})
    sorted_z = sorted(z_map.keys())
    full_paths = [
        os.path.join(image_root, z_map[z].replace("\\", "/"))
        for z in sorted_z
    ]

    load_fn = partial(load_z_slice, cx=cx, cy=cy, patch_size=patch_size)
    if io_pool is not None and full_paths:
        patches = list(io_pool.map(load_fn, full_paths))
    else:
        patches = [load_fn(fp) for fp in full_paths]

    expected_shape = (patch_size, patch_size, 3)
    zero = np.zeros(expected_shape, dtype=np.uint8)

    patches = [
        p if p is not None and p.shape == expected_shape else zero
        for p in patches
    ]

    if not patches:
        patches = [zero]

    while len(patches) < max_z:
        patches.append(zero)
    patches = patches[:max_z]

    return np.stack(patches, axis=0)


def load_or_generate(sample, z_stack_file_map, image_root, patch_size, max_z, cache_dir, io_pool=None, use_mmap=True):
    """Load a sample's patches from cache, or generate from NAS and save.

    When ``cache_dir`` is ``None``, always reads from NAS without caching
    (used for ``patch_cache: off`` mode).

    Args:
        sample: Sample dict.
        z_stack_file_map: ``{z_stack_filename: {z_index: rel_path}}``.
        image_root: Base directory for tile images on NAS.
        patch_size: Side length of the square crop.
        max_z: Fixed number of z-slices.
        cache_dir: Versioned patch cache directory, or ``None`` to skip cache.
        io_pool: Optional ``ThreadPoolExecutor`` for parallel NAS reads.
        use_mmap: If True, load cached .npy with mmap (read-only); can be
            faster when the OS page cache is warm and avoids a full copy.

    Returns:
        numpy array ``(max_z, patch_size, patch_size, 3)`` uint8.
    """
    if cache_dir is not None:
        path = sample_cache_path(cache_dir, sample)
        if os.path.isfile(path):
            if use_mmap:
                return np.load(path, mmap_mode="r")
            return np.load(path)

    arr = _load_sample_from_nas(
        sample, z_stack_file_map, image_root, patch_size, max_z, io_pool,
    )

    if cache_dir is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, arr)

    return arr


_WORKER_CTX = {}


def _init_worker(z_stack_file_map, image_root, patch_size, max_z, cache_dir, io_threads):
    """Per-process initializer -- avoids re-pickling large objects per task."""
    _WORKER_CTX.update(
        z_stack_file_map=z_stack_file_map,
        image_root=image_root,
        patch_size=patch_size,
        max_z=max_z,
        cache_dir=cache_dir,
        io_pool=ThreadPoolExecutor(max_workers=io_threads),
    )


def _process_one(sample):
    """Worker target for a single sample (reads from module-level context)."""
    load_or_generate(
        sample,
        _WORKER_CTX["z_stack_file_map"],
        _WORKER_CTX["image_root"],
        _WORKER_CTX["patch_size"],
        _WORKER_CTX["max_z"],
        _WORKER_CTX["cache_dir"],
        _WORKER_CTX["io_pool"],
    )


def precompute_all(
    samples, z_stack_file_map, image_root, patch_size, max_z,
    cache_dir, io_threads=11, workers=1,
):
    """Precompute and cache patches for all samples, skipping existing ones.

    When ``workers > 1`` uses a ``multiprocessing.Pool`` so that multiple
    samples are read from NAS concurrently (each process also gets its own
    thread pool for z-slice I/O).

    Args:
        samples: List of sample dicts.
        z_stack_file_map: ``{z_stack_filename: {z_index: rel_path}}``.
        image_root: Base directory for tile images on NAS.
        patch_size: Side length of the square crop.
        max_z: Fixed number of z-slices.
        cache_dir: Versioned patch cache directory.
        io_threads: Thread pool size for parallel z-slice reads per worker.
        workers: Number of parallel worker processes (1 = single-process).
    """
    os.makedirs(cache_dir, exist_ok=True)

    to_compute = [
        s for s in samples
        if not os.path.isfile(sample_cache_path(cache_dir, s))
    ]

    if not to_compute:
        return

    if workers <= 1:
        with ThreadPoolExecutor(max_workers=io_threads) as io_pool:
            for sample in tqdm(to_compute, desc="Precomputing patches", unit="sample"):
                load_or_generate(
                    sample, z_stack_file_map, image_root,
                    patch_size, max_z, cache_dir, io_pool,
                )
    else:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(z_stack_file_map, image_root, patch_size, max_z, cache_dir, io_threads),
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(_process_one, to_compute, chunksize=64),
                total=len(to_compute),
                desc=f"Precomputing patches ({workers} workers)",
                unit="sample",
            ):
                pass


def check_all_cached(samples, cache_dir, write_sentinel_if_complete=False):
    """Check that every sample has a cached ``.npy`` file.

    Uses a single directory walk to build the set of existing paths (instead of
    one stat per sample). If a sentinel file exists, skips the check and
    returns [] (assumes cache is complete). Optionally writes the sentinel when
    no samples are missing so the next run can skip the walk.

    Args:
        samples: List of sample dicts.
        cache_dir: Versioned patch cache directory.
        write_sentinel_if_complete: If True and no samples are missing, write
            a sentinel file so the next call can skip the check.

    Returns:
        List of samples that are missing from the cache (empty if all cached).
    """
    sentinel_path = os.path.join(cache_dir, SENTINEL_ALL_CACHED)
    if os.path.isfile(sentinel_path):
        return []

    existing = _existing_cache_paths_set(cache_dir)
    missing = [
        s for s in samples
        if sample_cache_path(cache_dir, s) not in existing
    ]
    if write_sentinel_if_complete and not missing:
        with open(sentinel_path, "w"):
            pass
    return missing


def expected_cache_shape(max_z, patch_size):
    """Return the expected numpy shape for a single cached patch file.

    Args:
        max_z: Number of z-slices per sample.
        patch_size: Side length of the square crop.

    Returns:
        Tuple (max_z, patch_size, patch_size, 3).
    """
    return (max_z, patch_size, patch_size, 3)


def validate_cache_dir(cache_dir, expected_shape, show_progress=True):
    """Find cached .npy files that are missing, unreadable, or have wrong shape.

    Walks the cache directory for all .npy files, loads each, and checks that
    the array shape matches expected_shape (max_z, patch_size, patch_size, 3).

    Args:
        cache_dir: Versioned patch cache directory (e.g. patches_v1_ps144_mz11).
        expected_shape: Tuple (max_z, patch_size, patch_size, 3).
        show_progress: If True, show a tqdm progress bar while checking files.

    Yields:
        Tuples (file_path, error_message) for each invalid file. error_message
        is a short string (e.g. "wrong shape (10,144,144,3), expected (11,144,144,3)"
        or "load failed: ...").
    """
    if not os.path.isdir(cache_dir):
        return

    paths = []
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            if name.endswith(".npy"):
                paths.append(os.path.join(root, name))

    iterator = tqdm(paths, desc="Checking cache", unit="file") if show_progress else paths
    for path in iterator:
        try:
            arr = np.load(path)
            if getattr(arr, "shape", None) != expected_shape:
                got = getattr(arr, "shape", "no shape")
                yield (path, f"wrong shape {got}, expected {expected_shape}")
        except Exception as e:
            yield (path, f"load failed: {e}")
