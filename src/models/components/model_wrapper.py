from typing import Dict, List, Type

import torch

from src.utils.data_objects import DataObject
from src.utils.image_utils import get_binary_mask


class DataObjectModelWrapper(torch.nn.Module):
    """Wrapper that adapts standard PyTorch models to work with DataObject.
    
    Extracts inputs from DataObject, passes them to the model, and stores outputs
    back in the DataObject.
    """
    
    def __init__(
        self,
        model: Type[torch.nn.Module],
        *args,
        dataobject_input_args: List[str] = None,
        dataobject_input_kwargs: Dict[str, str] = None,
        override_background: bool = False,
        **kwargs
    ):
        """Initialize model wrapper.
        
        Args:
            model: PyTorch model class to wrap
            *args: Positional arguments for model initialization
            dataobject_input_args: List of DataObject attribute names to pass as positional args
            dataobject_input_kwargs: Dict mapping DataObject attr names to model kwarg names
            override_background: Whether to override background with mask (if available)
            **kwargs: Keyword arguments for model initialization
        """
        super().__init__()
        self.model = model(*args, **kwargs)

        if dataobject_input_args is None:
            dataobject_input_args = ["data"]

        if dataobject_input_kwargs is None:
            dataobject_input_kwargs = {}

        self.override_background = override_background
        self.dataobject_input_args = dataobject_input_args
        self.dataobject_input_kwargs = dataobject_input_kwargs

    def forward(self, data_object: DataObject) -> DataObject:
        """Forward pass through model.
        
        Args:
            data_object: Input DataObject
            
        Returns:
            DataObject with output stored in .output attribute
        """
        # Extract inputs from DataObject
        args = [data_object[k] for k in self.dataobject_input_args]
        # dataobject_input_kwargs maps DataObject attr names to model kwarg names
        kwargs = {v: data_object[k] for k, v in self.dataobject_input_kwargs.items()}

        # Forward through model
        model_output = self.model(*args, **kwargs)

        # Store output in DataObject
        if isinstance(model_output, tuple):
            data_object.output = model_output[0]
            if len(model_output) > 1:
                data_object.variance = model_output[1]
        elif isinstance(model_output, torch.Tensor):
            data_object.output = model_output
        elif isinstance(model_output, dict):
            data_object.output = model_output["output"]
            data_object.variance = model_output.get("variance", None)
            data_object.losses["model_loss"] = model_output.get("loss", None)
        else:
            raise ValueError("Model output must be tuple, torch.Tensor, or dict")

        # Optional: override background with mask if available
        if self.override_background:
            # Check for mask in sub-objects (e.g., sms properties)
            mask = None
            if data_object.sms is not None and hasattr(data_object.sms, "myocardial_mask"):
                mask_tensor = data_object.sms.myocardial_mask
                if (mask_tensor != 0).any() and mask_tensor.numel() > 0:
                    mask = mask_tensor
                    data_object.output = data_object.output * mask
            else:
                # Fallback: use binary mask from target
                mask = get_binary_mask(data_object.target.detach().cpu())
                if isinstance(mask, torch.Tensor):
                    mask = mask.to(data_object.output.device)
                output = data_object.output
                new_output = torch.zeros_like(output)
                new_output[mask] = output[mask]
                new_output[~mask] = data_object.target[~mask].detach()
                data_object.output = new_output

        data_object.computed = True

        return data_object

    def __getattr__(self, item):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.model, item)

    def __repr__(self):
        return self.model.__repr__()

    def __str__(self):
        return self.model.__str__()


def model_wrapper(
    model: Type[torch.nn.Module],
    dataobject_input_args: List[str] = None,
    dataobject_input_kwargs: Dict[str, str] = None,
) -> Type[DataObjectModelWrapper]:
    """Factory function to create a model wrapper class.
    
    Args:
        model: PyTorch model class
        dataobject_input_args: List of DataObject attribute names for positional args
        dataobject_input_kwargs: Dict mapping DataObject attr names to model kwarg names
        
    Returns:
        DataObjectModelWrapper subclass
    """
    class SpecificDataObjectModelWrapper(DataObjectModelWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(
                model,
                *args,
                dataobject_input_args=dataobject_input_args,
                dataobject_input_kwargs=dataobject_input_kwargs,
                **kwargs
            )

    return SpecificDataObjectModelWrapper
