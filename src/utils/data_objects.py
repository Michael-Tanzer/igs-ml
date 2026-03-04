from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import default_collate


@dataclass
class SMSProperties:
    """SMS-specific properties for medical imaging tasks.
    
    This is provided as a reference example of how to create task-specific
    property classes. Users can create their own property classes for their
    specific tasks.
    """
    heart_mask: torch.Tensor
    myocardial_mask: torch.Tensor
    b_value: torch.Tensor
    diffusion_direction: torch.Tensor
    image_orientation_patient: torch.Tensor
    existing_myocardial_mask: bool | Iterable[bool]
    use_for_loss: bool | Iterable[bool]
    use_for_metrics: bool | Iterable[bool]
    complex_data: bool | Iterable[bool]
    phase_data: bool | Iterable[bool]
    cardiac_phase: str = "N/A"
    health_status: str = "N/A"
    pathology_name: str = "N/A"
    slice_position: int = -1
    repetition: int = -1
    image_index: int = -1
    is_polar: bool = False
    joint_images_tensors: bool = False

    def to(self, device):
        """Move all tensors in this properties object to device."""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


@dataclass
class MalariaProperties:
    """Malaria-specific metadata for patch samples.

    Carries per-sample DB metadata through the pipeline so that downstream
    code (logging, analysis, stratified metrics) can inspect it without a
    second DB round-trip.
    """
    object_ids: list
    z_stack_filename: str
    z_indices_used: list
    id_image_set: int
    PID: str
    species: str
    stage: str
    smear_type: str
    parasitemia: str
    z_stack_height: int
    pixels_per_micron: float

    def to(self, device):
        """No-op -- all fields are plain Python objects."""
        return self


def get_repr(dict_obj, name="DataObject"):
    """Helper function to generate string representation."""
    s = f"{name}("
    for k, v in dict_obj.items():
        if hasattr(v, "shape"):
            s += f"{k}={list(v.shape)}, "
        elif hasattr(v, "__dict__"):  # Handle sub-objects
            s += f"{k}={type(v).__name__}(...), "
        else:
            try:
                s += f"{k}=[{len(v)}], "
            except TypeError:
                s += f"{k}={v}, "
    return s + ")"


@dataclass
class DataObject:
    """Core data container with task-specific sub-objects.
    
    This class provides a flexible structure for machine learning data:
    - Core fields (data, target, output, loss) are always present
    - Task-specific properties can be added via sub-objects (e.g., sms, medical, etc.)
    
    Example:
        >>> # Simple usage without sub-objects
        >>> obj = DataObject(
        ...     index=0,
        ...     data=torch.randn(1, 3, 32, 32),
        ...     target=torch.randn(1, 3, 32, 32)
        ... )
        >>> 
        >>> # Usage with SMS properties
        >>> obj = DataObject(
        ...     index=0,
        ...     data=image_tensor,
        ...     target=target_tensor,
        ...     sms=SMSProperties(heart_mask=mask, b_value=bval, ...)
        ... )
        >>> mask = obj.sms.heart_mask
    """
    # Core fields (always present)
    index: int | Iterable[int]
    data: torch.Tensor
    target: torch.Tensor
    output: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    variance: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    loss: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    computed: bool = False
    epoch: int = -1
    
    # File paths
    data_filepath: str | List[str] = field(default_factory=list)
    target_filepath: str | List[str] = field(default_factory=list)
    weight: Optional[torch.Tensor] = None
    
    # Transform tracking
    chunk_shapes: List[int] = field(default_factory=list)
    random_transforms_parameters: List[float] = field(default_factory=list)
    
    # Task-specific properties (optional sub-objects)
    sms: Optional[SMSProperties] = None
    malaria: Optional[MalariaProperties] = None

    def __post_init__(self):
        """Normalize filepath fields to consistent format."""
        # Normalize data_filepath
        if isinstance(self.data_filepath, (str, Path)):
            self.data_filepath = [str(self.data_filepath)]
        elif isinstance(self.data_filepath, np.ndarray):
            self.data_filepath = [
                str(filepath)
                for filepath in self.data_filepath
                if str(filepath) not in ("0", "", "None")
            ]
        elif isinstance(self.data_filepath, (list, tuple)):
            self.data_filepath = [
                str(fp) if not isinstance(fp, (list, tuple)) else [str(f) for f in fp]
                for fp in self.data_filepath
            ]

        # Normalize target_filepath
        if isinstance(self.target_filepath, (str, Path)):
            self.target_filepath = [str(self.target_filepath)]
        elif isinstance(self.target_filepath, np.ndarray):
            self.target_filepath = [
                str(filepath)
                for filepath in self.target_filepath
                if str(filepath) not in ("0", "", "None")
            ]
        elif isinstance(self.target_filepath, (list, tuple)):
            self.target_filepath = [
                str(fp) if not isinstance(fp, (list, tuple)) else [str(f) for f in fp]
                for fp in self.target_filepath
            ]

    def __repr__(self):
        return get_repr(self.__dict__, "DataObject")

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        """Allow dictionary-like access: obj['key']"""
        return getattr(self, item)

    def get_input_data_as_dict(self):
        """Get input data as dictionary, excluding target."""
        result = {}
        for k, v in self.__dict__.items():
            if k != "target":
                if hasattr(v, "__dict__"):  # Handle sub-objects
                    result[k] = v.__dict__
                else:
                    result[k] = v
        return result

    def get_target_data_as_dict(self):
        """Get target data as dictionary, with target renamed to 'data'."""
        result = {"data": self.target}
        for k, v in self.__dict__.items():
            if k != "data":
                if hasattr(v, "__dict__"):  # Handle sub-objects
                    result[k] = v.__dict__
                else:
                    result[k] = v
        return result

    def to(self, device):
        """Move all tensors to device, including sub-objects.
        
        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device)
            
        Returns:
            self (for chaining)
        """
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
            elif hasattr(v, "__dict__") and hasattr(v, "to"):  # Handle sub-objects (dataclasses)
                setattr(self, k, v.to(device))
        return self

    def __iter__(self):
        """For back-compatibility: iterate as (index, input_dict, target_dict)."""
        return iter((self.index, self.get_input_data_as_dict(), self.get_target_data_as_dict()))

    def set_output(
        self,
        output: torch.Tensor,
        variance: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
    ):
        """Set output, variance, and loss tensors.
        
        Args:
            output: Model output tensor
            variance: Optional variance tensor
            loss: Optional loss tensor
        """
        self.output = output
        if variance is not None:
            self.variance = variance
        if loss is not None:
            self.loss = loss
        self.computed = True

    def to_dict(self):
        """Convert to dictionary, including sub-objects."""
        result = {}
        for k, v in self.__dict__.items():
            if hasattr(v, "__dict__"):  # Handle sub-objects
                result[k] = v.__dict__
            else:
                result[k] = v
        return result

    def get_data_output_target_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get (data, output, target) tuple, optionally filtered by use_for_metrics.
        
        Returns:
            Tuple of (data, output, target) tensors
        """
        # Check if we have use_for_metrics in sub-objects
        use_for_metrics = None
        if self.sms is not None and hasattr(self.sms, "use_for_metrics"):
            use_for_metrics = self.sms.use_for_metrics

        if use_for_metrics is not None:
            if isinstance(use_for_metrics, torch.Tensor) or isinstance(use_for_metrics, np.ndarray):
                mask = use_for_metrics
                data = self.data[mask] if len(self.data.shape) > len(mask.shape) else self.data[mask]
                output = self.output[mask] if len(self.output.shape) > len(mask.shape) else self.output[mask]
                target = self.target[mask] if len(self.target.shape) > len(mask.shape) else self.target[mask]
            else:
                data, output, target = self.data, self.output, self.target
        else:
            data, output, target = self.data, self.output, self.target

        return data, output, target


def collate_dataclass(dataclass_type, items: List):
    """Collate a list of dataclass instances into a single instance.
    
    Args:
        dataclass_type: The dataclass type to instantiate
        items: List of dataclass instances to collate
        
    Returns:
        A single dataclass instance with batched fields
    """
    if not items:
        return None
    
    collated_dict = {}
    for field_name in dataclass_type.__dataclass_fields__:
        values = [getattr(item, field_name) for item in items]
        # Check if all values are None
        if all(v is None for v in values):
            collated_dict[field_name] = None
        # Check if all values are not None
        elif all(v is not None for v in values):
            try:
                collated_dict[field_name] = default_collate(values)
            except (TypeError, RuntimeError):
                # If collation fails, keep as list
                collated_dict[field_name] = values
        else:
            # Mixed None/not-None: keep as list
            collated_dict[field_name] = values
    
    return dataclass_type(**collated_dict)


def custom_object_collate_fn(data_points: List[DataObject]) -> DataObject:
    """Collate DataObjects, handling nested dataclasses.
    
    Args:
        data_points: List of DataObject instances
        
    Returns:
        A single DataObject with batched fields
    """
    if not data_points:
        raise ValueError("Cannot collate empty list of DataObjects")
    
    # Get core fields (excluding sub-objects)
    core_fields = {
        k: v
        for k, v in data_points[0].__dict__.items()
        if not (hasattr(v, "__dict__") and not isinstance(v, (list, dict, tuple)))
    }
    
    # Collate core fields
    collated_dict = {}
    for k in core_fields:
        values = [getattr(p, k) for p in data_points]
        if all(v is None for v in values):
            collated_dict[k] = None
        elif all(v is not None for v in values):
            try:
                collated_dict[k] = default_collate(values)
            except (TypeError, RuntimeError):
                # If collation fails, keep as list
                collated_dict[k] = values
        else:
            # Mixed None/not-None: keep as list
            collated_dict[k] = values
    
    # Collate sub-objects recursively (check for any dataclass sub-objects)
    # Check all attributes that might be sub-objects
    for attr_name in data_points[0].__dict__:
        if attr_name in core_fields:
            continue  # Already handled
        attr_value = getattr(data_points[0], attr_name, None)
        if attr_value is not None and hasattr(attr_value, "__dict__") and is_dataclass(attr_value):
            # This is a dataclass sub-object
            sub_objs = [getattr(p, attr_name, None) for p in data_points]
            if all(obj is None for obj in sub_objs):
                collated_dict[attr_name] = None
            elif all(obj is not None for obj in sub_objs):
                collated_dict[attr_name] = collate_dataclass(type(sub_objs[0]), sub_objs)
            else:
                # Mixed None/not-None: keep as list
                collated_dict[attr_name] = sub_objs
        elif attr_name not in collated_dict:
            # Regular attribute, collate normally
            values = [getattr(p, attr_name, None) for p in data_points]
            if all(v is None for v in values):
                collated_dict[attr_name] = None
            elif all(v is not None for v in values):
                try:
                    collated_dict[attr_name] = default_collate(values)
                except (TypeError, RuntimeError):
                    collated_dict[attr_name] = values
            else:
                collated_dict[attr_name] = values
    
    return DataObject(**collated_dict)
