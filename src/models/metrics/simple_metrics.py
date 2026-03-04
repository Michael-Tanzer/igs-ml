"""Pre-wrapped binary classification metrics ready for Hydra instantiation.

Each constant below is a *class* (not an instance) created by
:func:`~src.models.metrics.utils.metric_wrapper`.  Hydra configs can
reference them directly, e.g.::

    metrics:
      - _target_: src.models.metrics.simple_metrics.BinaryAccuracyMetric
        threshold: 0.0
      - _target_: src.models.metrics.simple_metrics.BinaryAUROCMetric

Extra ``**kwargs`` in the config are forwarded to the underlying
``torchmetrics`` constructor (one instance per train/val/test stage).
"""

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score

from src.models.metrics.utils import metric_wrapper
from src.utils.enums import MetricType

BinaryAccuracyMetric = metric_wrapper(BinaryAccuracy, "accuracy", MetricType.PAIRED)
BinaryAUROCMetric = metric_wrapper(BinaryAUROC, "auroc", MetricType.PAIRED)
BinaryF1Metric = metric_wrapper(BinaryF1Score, "f1", MetricType.PAIRED)
