from typing import List, Optional

import torch
import torch.nn as nn

from trainer.models.layers import ConvBNReLU
from .base import BaseDecoder


class UpsampleCat(BaseDecoder):
    """
    Upsample and concatenate all multiple features. 

    Args:
        in_channels (List[int]): Number of input channels.
        in_strides (List[int]): List of input strides.
        out_channels (Optional[int]): Number of output channels. 
                                      if None, it will be set to the sum of all input channels.
        out_strides (Optional[int]): Output stride. if None, it will be set to the first input stride.


    Returns:
        torch.Tensor: Output tensor that has the shape of (N, out_channels, H, W).
    """

    def __init__(self,
                 in_channels: List[int],
                 in_strides: List[int],
                 out_channels: Optional[int] = None,
                 out_strides: Optional[int] = None):
        if len(in_channels) != len(in_strides):
            raise ValueError("in_channels and in_strides should have the same length.")
        super().__init__(in_channels, in_strides)

        self.out_channels = sum(self.in_channels) if out_channels is None else out_channels
        self.out_strides = self.in_strides[0] if out_strides is None else out_strides
        self.upsampler = nn.ModuleDict({f"decoder_{i}": nn.Identity() if (s // self.in_strides[0]) == 1
                                        else nn.Upsample(scale_factor=(s // self.in_strides[0]), mode="bilinear", align_corners=True)
                                        for i, (c, s) in enumerate(zip(self.in_channels, self.in_strides))})
        self.conv = ConvBNReLU(sum(self.in_channels), self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for i, feature in enumerate(features):
            out = self.upsampler[f"decoder_{i}"](feature)
            outs.append(out)
        out: torch.Tensor = torch.cat(outs, dim=1)
        out: torch.Tensor = self.conv(out)
        return out
