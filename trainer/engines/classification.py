from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torchmetrics

from trainer.engines.base import BaseEngine
from trainer.models.base import BaseModel


# def tmp_criterion(pred, target):
#     target: torch.Tensor = target['labels']
#     return nn.functional.binary_cross_entropy(pred, target.long(), reduction='mean')


class ClassificationEngine(BaseEngine):
    def __init__(self,
                 model: BaseModel,
                 threshold=0.5,
                 optimizer: Optional[Any] = None,
                 scheduler: Optional[Any] = None,
                 criterion: Optional[Callable] = None,
                 ** kwargs):
        super().__init__(model, optimizer, scheduler)

        self.threshold = threshold

        # temporal
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # temporal
        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(threshold=self.threshold, task='binary'),
                                                 torchmetrics.Precision(threshold=self.threshold, task='binary'),
                                                 torchmetrics.Recall(threshold=self.threshold, task='binary'),
                                                 torchmetrics.F1Score(threshold=self.threshold, task='binary'),])
        self.meter_train, self.meter_valid = metrics.clone().cuda(), metrics.clone().cuda()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        pred, target = self._step(batch)
        losses = self.criterion(pred, target['labels'].float())
        self.meter_train(pred, target['labels'])
        return {'loss': losses}

    def training_step_end(self, step_output: Dict[str, torch.Tensor]):
        losses = {f'train/{key}': value for key, value in step_output.items()}
        scores = {f'train/{key}': value for key, value in self.meter_train.compute().items()}

        self.log_dict(losses, prog_bar=True, on_step=True, on_epoch=False)
        self.log_dict(scores, prog_bar=True, on_step=True, on_epoch=False)

        return step_output

    def training_epoch_end(self, step_outputs: List[Dict[str, torch.Tensor]]):
        self.meter_train.reset()

    def validation_step(self, batch: dict, batch_idx: int):
        pred, target = self._step(batch)
        losses = self.criterion(pred, target['labels'].float())
        self.meter_valid(pred, target['labels'])
        return {'loss': losses}

    def validation_epoch_end(self, step_outputs: List[Dict[str, torch.Tensor]]):
        losses = {f'valid/{key}': torch.stack([step_output[key] for step_output in step_outputs]).mean()
                  for key in step_outputs[0].keys()}
        scores = {f'valid/{key}': value for key, value in self.meter_valid.compute().items()}

        self.log_dict(losses, prog_bar=True,  on_step=False, on_epoch=True)
        self.log_dict(scores, prog_bar=True, on_step=False, on_epoch=True)

        self.meter_valid.reset()
