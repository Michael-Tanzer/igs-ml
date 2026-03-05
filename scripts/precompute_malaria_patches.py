"""Standalone script to precompute malaria patch cache.

Runs the same DB-query + patch-cropping pipeline as training, but only
the ``prepare_data()`` step.  Useful for pre-populating the cache as a
batch job before training starts.

Usage::

    uv run python scripts/precompute_malaria_patches.py experiment=malaria_patch_baseline

Any Hydra override accepted by ``train.py`` works here too (e.g.
``data.patch_size=200 data.max_z=7``).
"""

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import fill, math_eval, machine_name

OmegaConf.register_new_resolver("fill", fill, replace=True)
OmegaConf.register_new_resolver("math_eval", math_eval, replace=True)
OmegaConf.register_new_resolver("machine_name", machine_name, replace=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Instantiate the datamodule and run prepare_data (precompute patches)."""
    overrides = {}
    if OmegaConf.select(cfg.data, "patch_cache") is not None:
        overrides["patch_cache"] = "auto"
    else:
        raise ValueError(
            "The configured data module does not support patch_cache. "
            "Did you forget to set the experiment? E.g.: "
            "experiment=malaria_patch_baseline"
        )

    datamodule = hydra.utils.instantiate(cfg.data, **overrides)
    datamodule.prepare_data()


if __name__ == "__main__":
    main()
