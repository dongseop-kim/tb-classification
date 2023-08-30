from typing import List
import torch
import torch.nn as nn

from trainer.models.header.base import BaseHeader


class BasicClassifier(BaseHeader):
    """
    Basic classifier header.
    features -> global average pooling -> linear -> sigmoid(optional)

    Args:
        num_classes (int): number of classes.
        in_channels (int): number of input channels.
        in_strides (int): input stride.
        return_logits (bool): whether to return logits or probabilities.

    Returns:
        torch.Tensor: output tensor. (N, num_classes)
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 in_strides: int,
                 return_logits: bool = False,
                 initial_prob: float = 0.01,):
        if isinstance(in_channels, List):
            raise ValueError("BasicClassifier only supports single input.")
        if isinstance(in_strides, List):
            raise ValueError("BasicClassifier only supports single input.")
        super().__init__(num_classes, in_channels, in_strides)

        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Conv2d(in_channels, num_classes, 1, bias=True)

        self.return_logits = return_logits
        self.initial_prob = torch.tensor((1.0 - initial_prob) / initial_prob)
        # from focal loss official repo
        nn.init.constant_(self.linear.bias, -torch.log(self.initial_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.pooler(x)
        x: torch.Tensor = self.linear(x)
        x: torch.Tensor = x.flatten(1)
        if not self.return_logits:
            x = torch.sigmoid(x)
            x = torch.clamp(x, min=self.eps, max=1-self.eps)
        return x
