import random
from typing import Any, Dict

import numpy as np
from albumentations import SomeOf
from albumentations.core.transforms_interface import ImageOnlyTransform

from .basic import (RandomBilateralFilter, RandomBlur, RandomBrightness,
                    RandomClahe, RandomCompression, RandomContrast,
                    RandomGamma, RandomHistEqual, RandomNoise)

available_t = {'random_blur': RandomBlur,
               'random_gamma': RandomGamma,
               'random_clahe': RandomClahe,
               'random_brightness': RandomBrightness,
               'random_contrast': RandomContrast,
               'random_hist_equal': RandomHistEqual,
               'random_compression': RandomCompression,
               'random_noise': RandomNoise,
               'random_bilateral_filter': RandomBilateralFilter
               }


class RandAugmentPixel(ImageOnlyTransform):
    '''RandAugment for pixel-level transforms
    Args:
        max_n (int): maximum number of transforms to apply
        transforms (Dict[str, Dict]): dictionary of transforms to apply
        channel_stacking (bool): whether to stack channels or not
        num_channels (int): number of channels to stack
        p (float): probability of applying the transform. (default: 0.5) 
                   the p of each transform is normalized.
    '''

    def __init__(self, max_n: int = 2,
                 transforms: Dict[str, Dict] = None,
                 channel_stacking: bool = True,
                 replicate=False,
                 num_channels: int = 4,
                 train: bool = True,
                 p: float = 1.0):
        super(ImageOnlyTransform, self).__init__(always_apply=False, p=p)
        self.max_n = max_n
        self.channel_stacking = channel_stacking
        self.replicate = replicate
        self.num_channels = num_channels
        self.train = train

        if transforms is None and train:
            raise ValueError('transforms must be specified for training')

        if not train:
            self.transforms = None
            return

        t = [available_t[key](**val) for key, val in transforms.items()]
        self.transforms = [SomeOf(t, n=i, p=p) for i in range(1, max_n+1)]

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if not self.train:
            tmp = np.concatenate([img for _ in range(self.num_channels)], axis=2)
            return tmp

        if not self.channel_stacking:
            return random.choice(self.transforms)(image=img)['image']

        if self.replicate:
            data = random.choice(self.transforms)(image=img)['image']
            return np.stack([data for _ in range(self.num_channels)], axis=2)
        else:
            t = random.choice(self.transforms)
            data = [t(image=img)['image'] if random.random() < 0.5
                    else img for _ in range(self.num_channels)]
            return np.stack(data, axis=2)
