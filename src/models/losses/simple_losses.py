"""Simple, task-agnostic loss wrappers for DataObject-based pipelines."""

import torch.nn as nn

from src.utils.data_objects import DataObject


class DataObjectLoss(nn.Module):
    """Wraps any ``(prediction, target) -> scalar`` loss to accept a DataObject.

    Extracts ``.output`` and ``.target`` from the DataObject and delegates to
    the inner ``loss_fn``.  Satisfies the interface expected by
    :class:`~src.models.modules.base.BaseLitModule` (callable with a DataObject,
    has a ``.name`` attribute).

    Example config::

        criteria:
          - _target_: src.models.losses.simple_losses.DataObjectLoss
            loss_fn:
              _target_: torch.nn.BCEWithLogitsLoss
            name: bce
    """

    def __init__(self, loss_fn, name="loss", weight=1.0):
        """Initialise the loss wrapper.

        Args:
            loss_fn: Any PyTorch loss module whose ``forward`` signature is
                ``(input, target) -> scalar``.
            name: Human-readable name used for logging.
            weight: Multiplicative factor applied to the loss value.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.name = name
        self.weight = weight

    def forward(self, data_object):
        """Compute loss from a DataObject.

        Args:
            data_object: A :class:`~src.utils.data_objects.DataObject` with
                ``.output`` and ``.target`` populated.

        Returns:
            Weighted scalar loss tensor.
        """
        return self.loss_fn(data_object.output, data_object.target) * self.weight
