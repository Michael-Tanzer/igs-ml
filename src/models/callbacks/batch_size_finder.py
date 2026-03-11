"""BatchSizeFinder that works with PyTorch >= 2.6 weights_only defaults.

Lightning's built-in BatchSizeFinder restores its internal checkpoint with
``weights_only=None``, which since PyTorch 2.6 defaults to ``True`` and
rejects common objects (``functools.partial``, optimiser classes, etc.).

This subclass monkey-patches the single ``restore()`` call inside
``_scale_batch_size`` so that it always passes ``weights_only=False``.
"""

from __future__ import annotations

from functools import wraps
from unittest.mock import patch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import BatchSizeFinder as _BatchSizeFinder


class BatchSizeFinder(_BatchSizeFinder):
    """Drop-in replacement that avoids ``weights_only`` checkpoint errors."""

    def scale_batch_size(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        _original_restore = trainer._checkpoint_connector.restore

        @wraps(_original_restore)
        def _restore_unsafe(checkpoint_path=None, weights_only=None):
            return _original_restore(checkpoint_path, weights_only=False)

        with patch.object(
            trainer._checkpoint_connector, "restore", _restore_unsafe
        ):
            super().scale_batch_size(trainer, pl_module)
