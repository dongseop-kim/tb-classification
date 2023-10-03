from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchmetrics

from trainer.engines.base import BaseEngine
from trainer.models.base import BaseModel


class LossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.2, calc_sick: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.calc_sick = calc_sick
        self.criteria = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self,
                logits: Tuple[torch.Tensor, torch.Tensor],
                target: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pred_tb, pred_sick = logits  # (batch_size, 1), (batch_size, 1)
        target_tb: torch.Tensor = target['labels']  # (batch_size, 1)

        # TODO: label smoothing 추가하기
        target_tb = torch.where(target_tb > 0, 1, target_tb).to(torch.float32)
        loss_tb: torch.Tensor = self.criteria(pred_tb, target_tb)

        # calculate normal&sick loss
        if self.calc_sick:
            filtered_pred_sick = []
            target_sick = []
            for path, pn in zip(target['path'], pred_sick):
                if 'sick' in path or 'health' in path:
                    filtered_pred_sick.append(pn)
                    target_sick.append(0 if 'health' in path else 1)

            if len(filtered_pred_sick) == 0:
                loss_sick = torch.tensor(0).to(pred_tb.device).to(torch.float32)
                loss = loss_tb + self.alpha * loss_sick
                return {'loss': loss, 'loss_tb': loss_tb, 'loss_sick': loss_sick}

            filtered_pred_sick = torch.stack(filtered_pred_sick).to(pred_tb.device)
            target_sick = torch.tensor(target_sick).to(torch.float32).unsqueeze(1).to(pred_tb.device)

            loss_sick = self.criteria(filtered_pred_sick, target_sick)
        else:
            loss_sick = torch.tensor(0).to(pred_tb.device).to(torch.float32)
        loss = loss_tb + self.alpha * loss_sick
        return {'loss': loss, 'loss_tb': loss_tb, 'loss_sick': loss_sick}


class TBClassification(BaseEngine):
    def __init__(self,
                 model: BaseModel,
                 threshold=0.5,
                 optimizer: Optional[Any] = None,
                 scheduler: Optional[Any] = None,
                 **kwargs):
        super().__init__(model, optimizer, scheduler)

        self.threshold = threshold

        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(threshold=self.threshold, task='binary'),
                                                 torchmetrics.Precision(threshold=self.threshold, task='binary'),
                                                 torchmetrics.Recall(threshold=self.threshold, task='binary'),
                                                 torchmetrics.F1Score(threshold=self.threshold, task='binary'),])
        self.meter_train, self.meter_valid = metrics.clone().cuda(), metrics.clone().cuda()

        self.criterion = LossWithLogits(alpha=0.2, calc_sick=True)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        pred, target = self._step(batch)
        losses: Dict[str, torch.Tensor] = self.criterion(pred, target)
        self.meter_train(pred[0], target['labels'])
        return losses

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
        losses: Dict[str, torch.Tensor] = self.criterion(pred, target)
        self.meter_valid(pred[0], target['labels'])
        return losses

    def validation_epoch_end(self, step_outputs: List[Dict[str, torch.Tensor]]):
        losses = {f'valid/{key}': torch.stack([step_output[key] for step_output in step_outputs]).mean()
                  for key in step_outputs[0].keys()}
        scores = {f'valid/{key}': value for key, value in self.meter_valid.compute().items()}

        self.log_dict(losses, prog_bar=True,  on_step=False, on_epoch=True)
        self.log_dict(scores, prog_bar=True, on_step=False, on_epoch=True)

        self.meter_valid.reset()
