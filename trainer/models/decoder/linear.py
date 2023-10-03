from typing import List, Union

import torch
from trainer.models.decoder.base import BaseDecoder


class LinearDecoder(BaseDecoder):
    """
    Linear decoder that does not change the input.

    Args:
        in_channels (Union[int, List[int]]): number of channels for each feature map that is passed to the module
        in_strides (Union[int, List[int]]): stride of each feature map that is passed to the module
    """

    def __init__(self, in_channels: Union[int, List[int]], in_strides: Union[int, List[int]]):
        super().__init__(in_channels, in_strides)
        self.out_channels = in_channels
        self.out_strides = in_strides

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        return x
