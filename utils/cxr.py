from pathlib import Path
from typing import Union

import cv2
import numpy as np

from utils.misc import min_max_normalization


def windowing(image: np.ndarray, use_median: bool = False, width_param: float = 4.0) -> np.ndarray:
    """
    Windowing function that clips the values based on the given params.
    Args:
        image (str): the image to do the windowing
        use_median (bool): use median as center if True, mean otherwise
        width_param (float): the width of the value range for windowing.
        brightness (float) : brightness_rate. a value between 0 and 1 and indicates the amount to subtract.

    Returns:
        image that was windowed.
    """
    center = np.median(image) if use_median else image.mean()


    range_width_half = (image.std() * width_param) / 2.0
    low = center - range_width_half
    high = center + range_width_half
    image = np.clip(image, low, high)
    return image

def load_cxr_image(path_image: Union[Path, str]) -> np.ndarray:
    """
    Load CXR image
    Args:
        path_image (Union[Path, str]): path to image

    Returns:
        image (np.ndarray): normalized image array (H, W) and dtype is np.uint8
    """
    image: np.ndarray = cv2.imread(path_image, cv2.IMREAD_UNCHANGED)
    image = image.astype(np.float32)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    image = min_max_normalization(image)
    image = image * 255.0
    image = image.astype(np.uint8)
    return np.ascontiguousarray(image)  # H W
