from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix

from trainer.engine.base import BaseEngine
from trainer.models import Model


class TBClsEngine(BaseEngine):
    def __init__(self, model: Model, optimizer=None, scheduler=None):
        super().__init__(model, optimizer, scheduler)

        # each step outputs
        self.train_step_outputs: list[dict[str, torch.Tensor]] = []
        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []

        # metrics
        self.meter_train = ConfusionMatrix(task='binary')
        self.meter_val = ConfusionMatrix(task='binary')

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        x: torch.Tensor = batch['image']
        target = {'label': batch['label'], 'dataset': batch['dataset']}
        output = self.model(x, target)
        return output

    ''' == TRAINING == '''

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        pred_tb = torch.sigmoid(outputs['logit_tb'])  # N
        pred_aux = torch.sigmoid(outputs['logit_aux'])  # N
        label_tb = [1 if label in [1, 2, 3] else 0 for label in batch['label']]
        label_tb = torch.Tensor(label_tb).to(pred_tb.device).to(torch.long)
        self.meter_train.update(pred_tb, label_tb)

        self.log('train/loss', outputs['loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.train_step_outputs.append(outputs)  # save outputs

    def on_train_epoch_end(self):
        loss_tb_per_epoch = torch.stack([x['loss_tb'] for x in self.train_step_outputs]).mean()
        self.log('train/loss_tb', loss_tb_per_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.train_step_outputs.clear()
        self.meter_train.reset()

    ''' == VALIDATION == '''

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int):
        pred_tb = torch.sigmoid(outputs['logit_tb'])  # N
        pred_aux = torch.sigmoid(outputs['logit_aux'])  # N
        label_tb = [1 if label in [1, 2, 3] else 0 for label in batch['label']]
        label_tb = torch.Tensor(label_tb).to(pred_tb.device).to(torch.long)
        self.meter_val.update(pred_tb, label_tb)

        self.validation_step_outputs.append(outputs)  # save outputs

    def on_validation_epoch_end(self):
        loss_per_epoch = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        self.log('val/loss', loss_per_epoch, on_step=False, on_epoch=True, prog_bar=True)
        loss_tb_per_epoch = torch.stack([x['loss_tb'] for x in self.validation_step_outputs]).mean()
        self.log('val/loss_tb', loss_tb_per_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
        self.meter_val.reset()
