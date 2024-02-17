import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor as T


from trainer.models.layers import ConvBNReLU


def loss_tb_with_aux(logit_tb: T, logit_aux: T, target: dict[str, T],
                     criterion: Callable[[T, T], T]) -> T:
    # label
    #   'normal': 0
    #   'activetb': 1
    #   'inactivetb': 2
    #   'both': 3
    #   'others': 4
    labels = target['label']  # N
    label_tb = T([1 if l in [1, 2, 3] else 0 for l in labels])
    label_tb = label_tb.to(device=logit_tb.device, dtype=torch.float)
    loss_tb = criterion(logit_tb, label_tb)
    output = {'loss': loss_tb, 'loss_tb': loss_tb}

    # it works only when dataset is TBX11k dataset &  label is in [normal ,others]
    loss_aux = []
    for dataset, label, logit in zip(target['dataset'], labels, logit_aux):
        if dataset == 'tbx11k' and label not in [1, 2, 3]:
            label_aux = T([1 if label == 4 else 0])
            label_aux = label_aux.to(device=logit.device, dtype=torch.float)
            loss_aux.append(criterion(logit.unsqueeze(0), label_aux))

    if loss_aux:
        loss_aux = torch.stack(loss_aux).mean()
        output['loss_aux'] = loss_aux
        output['loss'] = torch.add(output['loss'], loss_aux)
    return output


class TBClassifierV1(nn.Module):
    def __init__(self,  num_classes: int, in_channels: int, in_strides: int, prior_prob: float = 0.01):
        super().__init__()
        assert isinstance(in_channels, int), "in_channels should be an integer."
        assert isinstance(in_strides, int), "in_strides should be an integer."

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.in_strides = in_strides

        self.conv1 = nn.Sequential(ConvBNReLU(in_channels, 512, 3, 1, 1, bias=False),
                                   ConvBNReLU(512, 1024, 3, 2, 1, bias=False))

        self.classifier = nn.Sequential(ConvBNReLU(1024, 1024, 3, 1, 1, bias=False),
                                        nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Conv2d(1024, 1, 1, 1, 0))

        self.aux_classifier = nn.Sequential(ConvBNReLU(1024, 1024, 3, 1, 1, bias=False),
                                            nn.AdaptiveAvgPool2d((1, 1)),
                                            nn.Conv2d(1024, 1, 1, 1, 0))

        # weight initialization with prior probability. this setting is used in Focal Loss.
        self.prior_prob = math.log((1 - prior_prob) / prior_prob)
        nn.init.normal_(self.classifier[-1].weight, std=0.01)
        nn.init.constant_(self.classifier[-1].bias, -1 * self.prior_prob)
        nn.init.normal_(self.aux_classifier[-1].weight, std=0.01)
        nn.init.constant_(self.aux_classifier[-1].bias, -1 * self.prior_prob)

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x: T, target: Optional[dict[str, Any]] = None) -> T:
        '''
        target: dict[str, T] must have 'dataset' key.
        '''
        x: T = self.conv1(x)  # N x 1024 x H x W
        logit_tb: T = self.classifier(x)  # N x 1 x 1 x 1
        logit_aux: T = self.aux_classifier(x)  # N x 1 x 1 x 1
        logit_tb = logit_tb.squeeze()  # N
        logit_aux = logit_aux.squeeze()  # N
        output = {'logit_tb': logit_tb, 'logit_aux': logit_aux}
        if target is not None:
            loss: dict[str, T] = loss_tb_with_aux(logit_tb, logit_aux, target, self.criterion)
            output.update(loss)
        return output
