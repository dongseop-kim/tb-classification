from typing import Any, Dict, List

import albumentations as A
from albumentations.core.serialization import Serializable
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import DictConfig, ListConfig

# pixel-level and spatial-level transforms
import trainer.transforms.pixel as t_p
import trainer.transforms.spatial as t_s

_AVAILABLE_TRANSFORMS = {'resize': A.Resize,
                         'random_flip': A.HorizontalFlip,
                         # pixel augmentations
                         'random_gamma': t_p.RandomGamma,
                         'random_blur': t_p.RandomBlur,
                         'random_clahe': t_p.RandomClahe,
                         'random_brightness': t_p.RandomBrightness,
                         'random_contrast': t_p.RandomContrast,
                         'random_hist_equal': t_p.RandomHistEqual,
                         'random_compression': t_p.RandomCompression,
                         'random_noise': t_p.RandomNoise,
                         'random_bilateral_filter': t_p.RandomBilateralFilter,
                         # spatial augmentations
                         'random_ratio': t_s.RandomRatio,
                         'random_resize_crop': t_s.RandomResizeCrop,
                         
                         # spatial augmentations -custom ()
                        'random_translate_v1': t_s.random_translate_v1,
                        'random_translate_v2': t_s.random_translate_v2,
                        'random_rotate_v1': t_s.random_rotate_v1,
                        'random_rotate_v2': t_s.random_rotate_v2,
                        'random_shear_v1': t_s.random_shear_v1,
                        'random_shear_v2': t_s.random_shear_v2,
                        'random_spatial_augment_v1': t_s.random_spatial_augment_v1,
                        'random_spatial_augment_v2': t_s.random_spatial_augment_v2,


                         # random augmentations
                         'rand_aug_pixel': t_p.RandAugmentPixel, 

                         # etc
                         'to_tensor_v2': ToTensorV2,
                         }


def parse_transforms(transforms: Dict[str, Any]) -> List[Serializable]:

    def check(transform: Dict[str, Any]):
        tmp = {key: tuple(val) if isinstance(val, ListConfig) else
               check(val) if isinstance(val, DictConfig) else val
               for key, val in transform.items()}
        return tmp

    return [_AVAILABLE_TRANSFORMS[str(key)](**check(dict(val) if val else {}))
                      for key, val in transforms.items()]
