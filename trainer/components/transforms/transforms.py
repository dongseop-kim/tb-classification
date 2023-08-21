from typing import Dict, Any, List
import albumentations as A
from albumentations.core.serialization import Serializable
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import DictConfig, ListConfig

# pixel-level and spatial-level transforms
import .pixel as t_p
import .spatial as t_s


available_transforms__ = {"resize": A.Resize,
                          "random_flip": A.HorizontalFlip,
                          # pixel augmentations
                          "random_gamma": t_p.RandomGamma,
                          "random_blur": t_p.RandomBlur,
                          "random_clahe": t_p.RandomClahe,
                          "random_brightness": t_p.RandomBrightness,
                          "random_contrast": t_p.RandomContrast,
                          "random_hist_equal": t_p.RandomHistEqual,
                          "random_compression": t_p.RandomCompression,
                          "random_noise": t_p.RandomNoise,
                          "random_bilateral_filter": t_p.RandomBilateralFilter,

                          # spatial augmentations
                          "random_ratio": t_s.RandomRatio,
                          "random_resize_crop": t_s.RandomResizeCrop,
                          "to_tensor_v2": ToTensorV2,
                          # random augmentations
                          "rand_aug_pixel": t_p.RandAugmentPixel,
                          }


def build_transforms(transforms: Dict[str, Any]) -> List[Serializable]:

    def check(transform: dict):
        tmp = dict()
        for key, val in transform.items():
            # print(key, val)
            if isinstance(val, ListConfig):
                tmp[key] = tuple(val)
            elif isinstance(val, DictConfig):
                tmp[key] = check(val)
            else:
                tmp[key] = val
        return tmp

    keys = list(transforms.keys())

    transform_list = []
    for key, val in transforms.items():
        new_key = str(key)
        val = {} if val is None else dict(val)
        new_val = check(val)
        transform_list.append(available_transforms__[new_key](**new_val))
        keys.remove(key)
    assert not keys

    return transform_list
