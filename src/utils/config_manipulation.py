import os
import socket
from typing import Tuple

import numpy as np
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf
from omegaconf._utils import split_key
from omegaconf.errors import InterpolationKeyError
from regex import regex


def fill(x: str, default=None, *, _parent_: DictConfig, _root_: DictConfig):
    """Generic config value lookup resolver.
    
    Searches the entire config tree for a key matching 'x' and returns the first match.
    Useful for referencing config values from anywhere in the hierarchy.
    
    Args:
        x: Key to search for (substring match)
        default: Default value if key not found
        _parent_: Parent config node (provided by OmegaConf)
        _root_: Root config node (provided by OmegaConf)
        
    Returns:
        First matching value found in config tree
        
    Raises:
        InterpolationKeyError: If key not found and no default provided
        
    Example:
        In config:
            model:
                in_channels: 3
            data:
                channels: ${fill:in_channels}
        Will resolve to 3.
    """
    moses = "."
    all_values = {}
    queue = [(k, getattr(_root_, k)) for k in dir(_root_) if not k.startswith("_")]
    while queue:
        attr, val = queue[0]
        assert attr not in all_values, f"Attribute {attr} already in all_values"
        if OmegaConf.is_config(val):
            viable_options = []
            for k in dir(val):
                if k.startswith("__"):
                    continue

                if k.isdigit():
                    k = int(k)

                if OmegaConf.is_interpolation(val, k):
                    continue

                try:
                    viable_options.append((f"{attr}{moses}{k}", getattr(val, str(k))))
                except MissingMandatoryValue:
                    ...

            queue.extend(viable_options)

        all_values[attr] = val
        queue.pop(0)

    options = [(k, v) for k, v in all_values.items() if x in k]

    options = [
        (path, value)
        for path, value in options
        if not regex.findall(rf"\{moses}(0|[1-9][0-9]*)($|\{moses})", path)
    ]

    if options:
        return options[0][1]
    elif default is not None:
        return default
    else:
        raise InterpolationKeyError(
            f'the key "{x}" was not found when interpolating the configuration'
        )


def math_eval(x: str, *, _parent_: DictConfig, _root_: DictConfig):
    """Evaluate a mathematical expression using numpy.
    
    Args:
        x: Mathematical expression string (e.g., "2 * 3", "np.sqrt(16)")
        _parent_: Parent config node (provided by OmegaConf)
        _root_: Root config node (provided by OmegaConf)
        
    Returns:
        Result of the mathematical expression
        
    Example:
        In config:
            model:
                hidden_dim: ${math_eval:32 * 2}  # Will resolve to 64
    """
    return eval(x, {"__builtins__": None}, {"np": np})  # nosec


def machine_name(*, _parent_: DictConfig, _root_: DictConfig):
    """Get the current machine hostname.
    
    Args:
        _parent_: Parent config node (provided by OmegaConf)
        _root_: Root config node (provided by OmegaConf)
        
    Returns:
        Hostname string
        
    Example:
        In config:
            paths:
                data_dir: /data/${machine_name}/datasets
    """
    return socket.gethostname()
