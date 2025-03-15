import matplotlib
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F


def resize_max_res(img: Image.Image, max_edge_resolution: int, resample=Image.BICUBIC) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
    Returns:
        `Image.Image`: Resized image.
    """
    
    original_width, original_height = img.size
    
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    new_width = int(np.round(new_width) / 64) * 64
    new_height = int(np.round(new_height) / 64) * 64

    resized_img = img.resize((new_width, new_height), resample=resample)
    return resized_img


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc
