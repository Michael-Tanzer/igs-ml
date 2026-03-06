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

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from src.models.metrics.utils import metric_wrapper
from src.utils.enums import MetricType

BinaryAccuracyMetric = metric_wrapper(BinaryAccuracy, "accuracy", MetricType.PAIRED, cast_target_to_long=True)
BinaryAUROCMetric = metric_wrapper(BinaryAUROC, "auroc", MetricType.PAIRED, cast_target_to_long=True)
BinaryF1Metric = metric_wrapper(BinaryF1Score, "f1", MetricType.PAIRED, cast_target_to_long=True)
BinaryPrecisionMetric = metric_wrapper(BinaryPrecision, "precision", MetricType.PAIRED, cast_target_to_long=True)
BinaryRecallMetric = metric_wrapper(BinaryRecall, "recall", MetricType.PAIRED, cast_target_to_long=True)
BinarySpecificityMetric = metric_wrapper(BinarySpecificity, "specificity", MetricType.PAIRED, cast_target_to_long=True)
BinaryAveragePrecisionMetric = metric_wrapper(BinaryAveragePrecision, "avg_precision", MetricType.PAIRED, cast_target_to_long=True)
