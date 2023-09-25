from typing import List, Optional, Union

import torch
import torch.nn as nn

from models.layers import ConvBNReLU, UpsampleAdd
from models.decoder.base import BaseDecoder
from models.decoder.upsamplecat import UpsampleCat


class FeaturePyramidNetwork(BaseDecoder):
    """
    Feature Pyramid Network (FPN) from 'Feature Pyramid Networks for Object Detection,' https://arxiv.org/abs/1612.03144.

    Args:
        in_channels (List[int]): Number of input channels.
        in_strides (List[int]): Strides of the input features.
        out_channels (int): Number of output channels. Default is 256.
        feature_aggregation (bool): Whether to use feature aggregation. if True, use UpsampleCat, else return list of features.
        out_strides (Optional[int]): Output stride. if None, it will be set to the first input stride. 
                                     it is only used when feature_aggregation is True.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: If feature_aggregation is True, return aggregated features, 
                                                 else return list of features.
    """

    def __init__(self,
                 in_channels: List[int],
                 in_strides: List[int],
                 out_channels: int = 256,
                 feature_aggregation: bool = False,
                 out_strides: Optional[int] = None):
        super().__init__(in_channels, in_strides)
        self.inner_layers = nn.ModuleList([ConvBNReLU(in_channel, out_channels, 1, 1, 0)
                                           for in_channel in in_channels])

        self.mid_layers = nn.ModuleList([UpsampleAdd(scale_factor=in_strides[i + 1] // in_strides[i])
                                         for i in range(len(in_strides) - 1)])[::-1]

        self.outer_layers = nn.ModuleList([ConvBNReLU(out_channels, out_channels, 3, 1, 1)
                                          for _ in range(len(in_channels))])

        self.feature_aggregation = feature_aggregation
        self.out_channels = out_channels
        self.out_strides = in_strides

        if self.feature_aggregation:
            self.aggregation = UpsampleCat(in_channels=[out_channels] * len(in_strides),
                                           in_strides=in_strides,
                                           out_strides=in_strides[0] if out_strides is None else out_strides)
            self.out_channels = self.aggregation.out_channels
            self.out_strides = self.aggregation.out_strides

    def forward(self, features: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        # every feature maps are passed through inner layer
        inner_outputs: List[torch.Tensor] = [inner(feature) for feature, inner in zip(features, self.inner_layers)]

        inner_outputs_inverted = inner_outputs[-2::-1]  # reverse except last feature map

        mid_outputs = [inner_outputs[-1]]  # last feature map is same
        for inner_output, mid_layer in zip(inner_outputs_inverted, self.mid_layers):
            # Upsample by a factor of 2 and add
            mid_outputs.insert(0, mid_layer(mid_outputs[0], inner_output))

        # Apply outer layers to the upsampled features
        outputs = [outer_layer(feature) for feature, outer_layer in zip(mid_outputs, self.outer_layers)]
        return self.aggregation(outputs) if self.feature_aggregation else outputs
