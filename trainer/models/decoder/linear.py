from typing import List

import torch

from .base import BaseDecoder


class LinearDecoder(BaseDecoder):
    """
    Linear decoder that does not change the input.

    Args:
        in_channels (int): Number of input channels.
        in_strides (int): List of input strides.

    """

    def __init__(self, in_channels: int, in_strides: int):
        if isinstance(in_channels, List):
            raise ValueError("Linear decoder only supports single input channel.")
        if isinstance(in_strides, List):
            raise ValueError("Linear decoder only supports single input stride.")
        super().__init__(in_channels, in_strides)
        self.out_channels = in_channels
        self.out_strides = in_strides

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
