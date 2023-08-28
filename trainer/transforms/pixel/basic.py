import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from utils.cxr import windowing


class RandomWindowing(ImageOnlyTransform):
    def __init__(self, 
                 width_param: float = 4.0,
                 width_range:float = 1.0,
                 use_median:bool = True,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.use_median = use_median
        self.width_param = width_param
        self.width_range = width_range
        self.use_median = use_median

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        width_param = (self.width_param- (self.width_range/2)) + \
        (np.random.rand(1) * (self.width_range))
        return windowing(img, use_median=self.use_median, width_param=width_param)

    
class RandomGamma(A.RandomGamma):
    def __init__(self, gamma_limit: int = 20, eps=None, always_apply=False, p=0.5):
        # gamma_limit = (100 - gamma_limit, 100 + gamma_limit)
        self.gamma_limit = (int(100 - gamma_limit), int(100 + gamma_limit))
        super().__init__(self.gamma_limit, eps, always_apply, p)


class RandomBlur(ImageOnlyTransform):
    def __init__(self, blur_limit=15, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        # random kernel size (3, blur_limit)
        basic = A.Blur(blur_limit=blur_limit, p=1.0)
        gaussian = A.GaussianBlur(blur_limit=(3, blur_limit), p=1.0)
        median = A.MedianBlur(blur_limit=blur_limit, p=1.0)
        motion = A.MotionBlur(blur_limit=blur_limit, p=1.0)
        self.blur = A.OneOf([basic, gaussian, median, motion], p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return self.blur(image=img)["image"]

    def get_transform_init_args_names(self) -> Tuple[str,]:
        return ("blur_limit", )


class RandomClahe(A.CLAHE):
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8),
                 always_apply=False, p=0.5):
        # clip_limit = (1, clip_limit)
        super().__init__(clip_limit=clip_limit, tile_grid_size=tile_grid_size,
                         always_apply=always_apply, p=p)


# NOTE: for avoiding deprecated warning
class RandomBrightness(A.RandomBrightnessContrast):
    def __init__(self, brightness_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5):
        # brightness_limit = (1 - brightness_limit, 1 + brightness_limit)
        super().__init__(brightness_limit=brightness_limit, contrast_limit=0.0,
                         brightness_by_max=brightness_by_max, always_apply=always_apply,
                         p=p)


# NOTE: for avoiding deprecated warning
class RandomContrast(A.RandomBrightnessContrast):
    def __init__(self, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5):
        # contrast_limit = (1 - contrast_limit, 1 + contrast_limit)
        super().__init__(brightness_limit=0.0, contrast_limit=contrast_limit,
                         brightness_by_max=brightness_by_max, always_apply=always_apply,
                         p=p)


class RandomHistEqual(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        t1 = A.Equalize(mode='cv')
        t2 = A.Equalize(mode='pil')
        self.t = A.OneOf([t1, t2], p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return self.t(image=img)["image"]


class RandomCompression(ImageOnlyTransform):
    def __init__(self, quality_lower=70, quality_upper=100, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        t1 = A.ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper,
                                compression_type=A.ImageCompression.ImageCompressionType.JPEG)
        self.t = t1

        r'''
        NOTE : it is not working. 이유는 모르겠습니다.
        thatValueError: cannot reshape array of size
        @preserve_shape
            def image_compression(img, quality, image_type):
            ~~~
        '''
        # t2 = A.ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper,
        #                         compression_type=A.ImageCompression.ImageCompressionType.WEBP)
        # self.t = A.OneOf([t1, t2], p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        out = self.t(image=img)["image"]
        # print(img.shape, out.shape)
        return out


class RandomNoise(ImageOnlyTransform):
    def __init__(self, noise_limit=0.1, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        t1 = A.GaussNoise(var_limit=(10.0, 50.0), mean=0)
        t2 = A.MultiplicativeNoise(multiplier=(0.9, 1.1))
        self.t = A.OneOf([t1, t2], p=p)

        # this is working only for 3 channels
        # t3 = A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5))
        # self.t = A.OneOf([t1, t2, t3], p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return self.t(image=img)["image"]


class RandomBilateralFilter(ImageOnlyTransform):
    def __init__(self, max_d=11, sigma_color=20, sigma_space=20, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        bisis = 100
        self.d = [d_ for d_ in range(3, max_d+1, 2)]
        self.sigma_color = (bisis-sigma_color, bisis+sigma_color)
        self.sigma_space = (bisis-sigma_space, bisis+sigma_space)

    # NOTE: if d is -1 , it is tooooooo slow. so, I don't suggest to use d=-1.
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        d = int(random.uniform(self.d[0], self.d[1]))
        sigma_color = random.uniform(self.sigma_color[0], self.sigma_color[1])
        sigma_space = random.uniform(self.sigma_space[0], self.sigma_space[1])
        out = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        return out
