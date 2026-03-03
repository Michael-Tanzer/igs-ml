from enum import Enum
from typing import Literal


class ComparableMultiEnum(Enum):
    """Base enum class that supports comparison with strings and multiple value aliases.
    
    Allows enum values to be compared with strings (case-insensitive) and supports
    multiple string aliases for the same enum value.
    """
    
    def __eq__(self, other):
        if isinstance(other, ComparableMultiEnum):
            return self.value == other.value
        else:
            if isinstance(self.value, (list, tuple)):
                return other.lower() in [v.lower() for v in self.value]

            return self.value == other

    def __str__(self):
        if isinstance(self.value, (list, tuple)):
            return self.value[0]
        else:
            return self.value

    def __repr__(self):
        return self.__str__()


class TrainingStage(ComparableMultiEnum):
    """Enum for training stage."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


TS = TrainingStage
TS_TYPE = TRAINING_STAGE = TrainingStage | Literal["train", "val", "test"]


class ImageLoggingMode(ComparableMultiEnum):
    """Enum for image logging mode."""
    ALL = "all"
    FIRST = "first"
    RANDOM = "random"


IMLM = ImageLoggingMode
IMLM_TYPE = IMAGE_LOGGING_MODE = ImageLoggingMode | Literal["all", "first", "random"]


class MetricType(ComparableMultiEnum):
    """Enum for metric type."""
    PAIRED = "paired"
    SINGLE = "single"
    IMAGE = "image"


MT = MetricType
MT_TYPE = METRIC_TYPE = MetricType | Literal["paired", "single", "image"]
