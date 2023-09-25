import logging
from typing import List, Tuple, Union

import torch
from segmentation_models_pytorch.decoders.deeplabv3.decoder import (
    DeepLabV3Decoder, DeepLabV3PlusDecoder)


class DeepLabV3(DeepLabV3Decoder):
    """
    DeepLabV3 decoder from 'Rethinking Atrous Convolution for Semantic Image Segmentation,' https://arxiv.org/abs/1706.05587.

    Args:
        in_channels (int): Number of input channels.
        in_strides (int): Strides of the input features.
        out_channels (int): Number of output channels. Default is 256.
        atrous_rates (Tuple[int]): Atrous rates for atrous spatial pyramid pooling (ASPP). default is (12, 24, 36).

    Returns:
        torch.Tensor: Resulted tensor. (shape: (batch, out_channels, h, w))
    """

    def __init__(self,
                 in_channels: int,
                 in_strides: int,
                 out_channels: int = 256,
                 atrous_rates: Tuple[int] = (12, 24, 36)):
        if isinstance(in_channels, List):
            in_channels = in_channels[-1]
            logging.warning("in_channels is a list, so in_channels[-1] is used as in_channels")

        if isinstance(in_strides, List):
            in_strides = in_strides[-1]
            logging.warning("in_strides is a list, so in_strides[-1] is used as in_strides")

        self.in_channels = in_channels
        self.in_strides = in_strides

        DeepLabV3Decoder.__init__(self, in_channels=in_channels, out_channels=out_channels,
                                  atrous_rates=atrous_rates)
        self.out_strides = in_strides

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, List):
            x = x[-1]
            logging.warning("x is a list, so x[-1] is used as x")
        return super(DeepLabV3Decoder, self).forward(x)


class DeepLabV3Plus(DeepLabV3PlusDecoder):
    """
    DeepLabV3+ decoder from 'Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation,' https://arxiv.org/abs/1802.02611.

    NOTE:
        This module expects at least 4 list tensors, and the tensors at positions -4 and -1 must have a stride difference of either 4 or 8."

    Args:
        in_channels (List[int]): Number of input channels.
        in_strides (List[int]): Strides of the input features.
        out_channels (int): Number of output channels. Default is 256.
        atrous_rates (Tuple[int]): Atrous rates for atrous spatial pyramid pooling (ASPP). default is (12, 24, 36).

    Returns:
        torch.Tensor: Resulted tensor. (shape: (batch, out_channels, h, w)). 

    """

    def __init__(self,
                 in_channels: List[int],
                 in_strides: List[int],
                 out_channels: int = 256,
                 atrous_rates: Tuple[int] = (12, 24, 36)):
        if len(in_channels) < 4:
            raise ValueError("len(in_channels) should be greater than 4, got {}".format(len(in_channels)))
        self.in_channels = in_channels
        self.in_strides = in_strides

        if in_strides[-1] // in_strides[-4] == 2:
            output_stride = 8
        elif in_strides[-1] // in_strides[-4] == 4:
            output_stride = 16
        else:
            raise ValueError("output stride should be 8 or 16, got {}".format(in_strides[-1] // in_strides[-4]))

        DeepLabV3PlusDecoder.__init__(self, encoder_channels=in_channels, out_channels=out_channels,
                                      atrous_rates=atrous_rates, output_stride=output_stride)

        self.out_strides = in_strides[-4]
        self.out_channels = out_channels

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(*x)
