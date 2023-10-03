from typing import List, Tuple

import torch
import torch.nn as nn

from trainer.models.header.base import BaseHeader
from trainer.models.layers import ConvBNSwish


class ClassifierWith2Head(BaseHeader):
    """
    Basic classifier header that returns two outputs.

    features -> conv -> GAP -> linear -> sigmoid(optional)
                            ∟> linear -> sigmoid(optional)
    Args:
        num_classes (int): number of classes.
        num_classes2 (int): number of classes for the second output.
        in_channels (int): number of input channels.
        mid_channels (int): number of channels for the conv layer.
        in_strides (int): input stride.
        return_logits (bool): whether to return logits or probabilities.

    Returns:
        torch.Tensor: output tensor. (N, num_classes)
    """

    def __init__(self,
                 num_classes: int,
                 num_classes2: int,
                 in_channels: int,
                 in_strides: int,
                 mid_channels: int = 1024,
                 return_logits: bool = False,
                 initial_prob: float = 0.01):
        if isinstance(in_channels, List):
            raise ValueError("BasicClassifier only supports single input.")
        if isinstance(in_strides, List):
            raise ValueError("BasicClassifier only supports single input.")
        super().__init__(num_classes, in_channels, in_strides)

        self.num_classes2 = num_classes2
        self.mid_channels = mid_channels

        self.conv = nn.Sequential(*[ConvBNSwish(in_channels, mid_channels, 3, 1, 1, bias=False),
                                    nn.AdaptiveAvgPool2d((1, 1))])
        self.linear1 = nn.Sequential(*[nn.Conv2d(mid_channels, num_classes, 1, bias=True),
                                       nn.Flatten(start_dim=1)])
        self.linear2 = nn.Sequential(*[nn.Conv2d(mid_channels, num_classes2, 1, bias=True),
                                       nn.Flatten(start_dim=1)])

        self.return_logits = return_logits
        self.initial_prob = torch.tensor((1.0 - initial_prob) / initial_prob)

        # from focal loss official repo
        nn.init.constant_(self.linear1[0].bias, -torch.log(self.initial_prob))
        nn.init.constant_(self.linear2[0].bias, -torch.log(self.initial_prob))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.conv(x)  # N x mid_channels x 1 x 1
        out1: torch.Tensor = self.linear1(x)  # N x num_classes
        out2: torch.Tensor = self.linear2(x)  # N x num_classes2
        if not self.return_logits:
            out1 = torch.clamp(torch.sigmoid(out1), min=self.eps, max=1-self.eps)
            out2 = torch.clamp(torch.sigmoid(out2), min=self.eps, max=1-self.eps)
        return (out1, out2)
