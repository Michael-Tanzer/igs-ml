"""Find corrupted or shape-mismatched patch cache files.

Walks the versioned patch cache directory, loads each .npy file, and checks
that its shape matches the expected (max_z, patch_size, patch_size, 3).
Reports any file that fails to load or has the wrong shape.

Use the same Hydra config as training so the cache path and expected shape
match (e.g. experiment=malaria_patch_baseline). Optionally delete bad files
so precompute can regenerate them.

Usage::

    uv run scripts/check_patch_cache.py experiment=malaria_patch_baseline
    uv run scripts/check_patch_cache.py experiment=malaria_patch_baseline delete=true
"""

import os

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.patch_cache import (
    expected_cache_shape,
    get_cache_dir,
    validate_cache_dir,
)

from src.utils import fill, math_eval, machine_name

OmegaConf.register_new_resolver("fill", fill, replace=True)
OmegaConf.register_new_resolver("math_eval", math_eval, replace=True)
OmegaConf.register_new_resolver("machine_name", machine_name, replace=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Resolve cache dir and expected shape from config, then validate cache files."""
    OmegaConf.resolve(cfg)

    cache_dir = get_cache_dir(
        cfg.data.cache_dir,
        cfg.data.patch_cache_version,
        cfg.data.patch_size,
        cfg.data.max_z,
    )
    expected_shape = expected_cache_shape(cfg.data.max_z, cfg.data.patch_size)

    if not os.path.isdir(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}", flush=True)
        raise SystemExit(1)

    print(f"Cache dir: {cache_dir}")
    print(f"Expected shape: {expected_shape}")
    print()

    bad = list(validate_cache_dir(cache_dir, expected_shape, show_progress=True))
    if not bad:
        print("All cached files are valid.")
        return

    print(f"Found {len(bad)} invalid file(s):")
    for path, msg in bad:
        print(f"  {path}: {msg}")
    if cfg.get("delete", False):
        for path, _ in bad:
            os.remove(path)
            print(f"Removed: {path}")
    else:
        print("Re-run with delete=true to remove them and then re-run precompute.")


if __name__ == "__main__":
    main()
