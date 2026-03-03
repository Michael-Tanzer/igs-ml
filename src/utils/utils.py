import functools
import re
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from lightning_utilities.core.rank_zero import rank_zero_warn
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def remove_best_from_string(string: str) -> str:
    """Remove 'best' substring from a string (case-insensitive).
    
    Args:
        string: Input string
        
    Returns:
        String with 'best' removed and cleaned up
    """
    pattern_best = re.compile(r"(best)", re.IGNORECASE)
    pattern_whitespaces = re.compile(r"\s+", re.IGNORECASE)
    pattern_underscores = re.compile(r"_+", re.IGNORECASE)
    pattern_trailing = re.compile(r"^\s|^_|\s$|_$", re.IGNORECASE)

    string = pattern_best.sub("", string)
    string = pattern_whitespaces.sub(" ", string)
    string = pattern_underscores.sub("_", string)
    string = pattern_trailing.sub("", string)

    return string


@functools.lru_cache(maxsize=None)
def warn_once(msg: str) -> int:
    """Print a warning message only once (using functools.lru_cache).
    
    Args:
        msg: Warning message
        
    Returns:
        Always returns 0 (for compatibility)
    """
    rank_zero_warn(msg)
    return 0


def get_relative_file_path(file_path: str) -> str:
    """Get relative file path from absolute path for use in configs.
    
    Converts absolute file paths to relative paths that can be used in Hydra configs.
    
    Args:
        file_path: Absolute file path (can use __file__)
        
    Returns:
        Relative path string suitable for use in configs (e.g., "src.data.components.invertible_transforms")
    """
    import os
    from pathlib import Path
    
    # Get the project root (assuming .project-root file exists)
    current = Path(file_path).resolve()
    project_root = current
    while project_root != project_root.parent:
        if (project_root / ".project-root").exists():
            break
        project_root = project_root.parent
    
    # Get relative path
    try:
        rel_path = current.relative_to(project_root)
        # Convert to module path format
        rel_path_str = str(rel_path).replace(os.sep, ".").replace(".py", "")
        return rel_path_str
    except ValueError:
        # If not relative to project root, return as-is
        return str(current).replace(os.sep, ".").replace(".py", "")
