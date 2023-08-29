from typing import List, Union

import torch.nn as nn


class BaseDecoder(nn.Module):
    def __init__(self, in_channels: Union[int, List[int]], in_strides: Union[int, List[int]]):
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides
        self.out_channels: Union[int, List[int]] = -1
        self.out_strides: Union[int, List[int]] = -1

