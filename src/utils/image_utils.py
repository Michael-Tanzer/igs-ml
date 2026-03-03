from typing import Tuple

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision.transforms.functional as func_transforms
from skimage.filters.thresholding import threshold_otsu


def remove_non_largest_components(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Remove all connected components except the largest one.
    
    Args:
        image: Binary image tensor or array
        
    Returns:
        Image with only the largest connected component
    """
    image = image.astype("uint8")
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=4
    )
    sizes = stats[:, -1]

    max_label = 1

    try:
        max_size = sizes[1]
    except IndexError:
        return image

    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    new_mask = np.zeros(output.shape)
    new_mask[output == max_label] = 1

    return new_mask


def get_binary_mask(image_data: torch.Tensor | np.ndarray) -> torch.BoolTensor | np.ndarray:
    """Get binary mask of the image data using Otsu thresholding.
    
    Args:
        image_data: Image tensor or array (can be 2D, 3D, or 4D)
        
    Returns:
        Binary mask with same shape as input (excluding channel dimension)
    """
    originally_torch = False
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.numpy()
        originally_torch = True

    if image_data.ndim == 2:
        image_data = image_data[None, ...]

    if image_data.ndim == 3:
        image_data = image_data[None, ...]

    mask = np.zeros_like(image_data, dtype=bool)
    for slice in range(image_data.shape[0]):
        for channel in range(image_data.shape[1]):
            image = image_data[slice, channel]
            new_mask = image > threshold_otsu(image, nbins=100)
            new_mask = remove_non_largest_components(new_mask)
            mask[slice, channel] = new_mask

    if image_data.ndim == 2:
        mask = mask[0, 0]

    if image_data.ndim == 3:
        mask = mask[0]

    if originally_torch:
        mask = torch.tensor(mask, dtype=torch.bool)

    return mask


def center_image(image):
    """Center an image by computing the centroid and rolling.
    
    Args:
        image: 2D numpy array (may contain NaN values)
        
    Returns:
        Centered image
    """
    if np.isnan(image).all():
        return image

    indices = np.indices((image.shape[0], image.shape[1])).astype(float)  # [2, x, y]
    indices -= np.array((image.shape[0] / 2, image.shape[1] / 2)).reshape((2, 1, 1))
    mask = 1 - np.isnan(image).astype(float)
    mask = mask.reshape((1, mask.shape[0], mask.shape[1]))

    weighted_indices = mask * indices
    weighted_indices[mask.repeat(2, 0) == 0] = np.nan
    mean = np.nanmean(weighted_indices, axis=(1, 2))

    image = np.roll(image, [-int(x) if not np.isnan(x) else 0 for x in mean], axis=(0, 1))
    return image


def get_rgba(image, cmap, clim):
    """Convert image to RGBA format with colormap and color limits.
    
    Args:
        image: 2D numpy array
        cmap: Matplotlib colormap
        clim: Color limits tuple (vmin, vmax) or None
        
    Returns:
        RGBA array with shape (*image.shape, 4)
    """
    if clim is not None:
        vmin, vmax = clim
    else:
        vmin, vmax = np.nanmin(image), np.nanmax(image)

    if np.all(np.isnan(image)):
        return np.zeros((*image.shape, 4))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)

    return m.to_rgba(image)


def resize_with_padding(
    image_to_resize: torch.Tensor | np.ndarray,
    target_size: Tuple[int, int],
    fill: int | float = 0,
    padding_mode: str = "constant",
) -> torch.Tensor | np.ndarray:
    """Resize image with padding to maintain aspect ratio.
    
    Args:
        image_to_resize: Image tensor or array
        target_size: Target (height, width)
        fill: Fill value for padding
        padding_mode: Padding mode ('constant', 'reflect', etc.)
        
    Returns:
        Resized and padded image
    """
    was_numpy = False
    if isinstance(image_to_resize, np.ndarray):
        was_numpy = True
        image_to_resize = torch.from_numpy(image_to_resize)

    pad_top = (target_size[0] - image_to_resize.shape[-2]) // 2
    pad_bottom = target_size[0] - image_to_resize.shape[-2] - pad_top
    pad_left = (target_size[1] - image_to_resize.shape[-1]) // 2
    pad_right = target_size[1] - image_to_resize.shape[-1] - pad_left

    padded_image = func_transforms.pad(
        image_to_resize,
        [pad_left, pad_top, pad_right, pad_bottom],
        fill=fill,
        padding_mode=padding_mode,
    )

    if was_numpy:
        padded_image = padded_image.numpy()

    return padded_image


def cart2pol(
    x: torch.Tensor, dim: int, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert cartesian coordinates to polar coordinates.
    
    Args:
        x: Tensor with shape (..., 3, ...) containing cartesian coordinates
        dim: Dimension containing the 3 cartesian coordinates
        eps: Small value to avoid division by zero
        
    Returns:
        Tuple of (rho, theta, phi) tensors with shape (..., 1, ...)
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    if x.shape[dim] != 3:
        raise ValueError(f"Expected tensor with shape (..., 3, ...), got {x.shape}")

    if dim < 0:
        dim = len(x.shape) + dim

    # Slice the tensor to select the correct axis for slicing
    slice_x = tuple(slice(None) if i != dim else slice(0, 1) for i in range(len(x.shape)))
    slice_y = tuple(slice(None) if i != dim else slice(1, 2) for i in range(len(x.shape)))
    slice_z = tuple(slice(None) if i != dim else slice(2, 3) for i in range(len(x.shape)))
    rho = torch.norm(x, dim=dim, keepdim=True, p=2)
    theta = torch.acos(x[slice_z] / (rho + eps)) * 2 - torch.pi  # between -pi and pi
    phi = torch.atan2(x[slice_y], x[slice_x])  # between -pi and pi
    return rho, theta, phi


def pol2cart(rho: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert polar coordinates to cartesian coordinates.
    
    Args:
        rho: Tensor with shape (..., 1, ...) containing radius
        theta: Tensor with shape (..., 1, ...) containing polar angle
        phi: Tensor with shape (..., 1, ...) containing azimuthal angle
        dim: Dimension to concatenate along
        
    Returns:
        Tensor with shape (..., 3, ...) containing cartesian coordinates
    """
    if not isinstance(rho, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(rho)}")

    if rho.shape != theta.shape or rho.shape != phi.shape:
        raise ValueError(
            f"Expected tensors with same shape, got {rho.shape}, {theta.shape}, {phi.shape}"
        )

    if rho.shape[dim] != 1:
        raise ValueError(f"Expected tensor with shape (..., 1, ...), got {rho.shape}")

    if dim < 0:
        dim = len(rho.shape) + dim

    # Convert to cartesian
    x = rho * torch.sin(theta) * torch.cos(phi)
    y = rho * torch.sin(theta) * torch.sin(phi)
    z = rho * torch.cos(theta)
    return torch.cat((x, y, z), dim=dim)
