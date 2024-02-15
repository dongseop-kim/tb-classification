from typing import Any

import torch
from pytorch_lightning import LightningModule

from trainer.models import Model


class TBClsEngine(LightningModule):
    def __init__(self, model: Model, optimizer=None, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        x: torch.Tensor = batch['image']
        target = {'label': batch['label'], 'dataset': batch['dataset']}
        output = self.model(x, target)
        return output

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        # logging
        self.log('train/loss', outputs['loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss-tb', outputs['loss_tb'], on_step=True, on_epoch=False, prog_bar=False)
        if 'loss_aux' in outputs:
            self.log('train/loss-aux', outputs['loss_aux'], on_step=True, on_epoch=False, prog_bar=False)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        x: torch.Tensor = batch['image']
        target = {'label': batch['label'], 'dataset': batch['dataset']}
        output = self.model(x, target)
        return output

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int):
        # logging
        self.log('val/loss', outputs['loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.log('val/loss-tb', outputs['loss_tb'], on_step=True, on_epoch=False, prog_bar=False)
        if 'loss_aux' in outputs:
            self.log('val/loss-aux', outputs['loss_aux'], on_step=True, on_epoch=False, prog_bar=False)
