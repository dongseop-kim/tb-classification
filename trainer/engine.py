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
        # logging
        self.log('train/loss', output['loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss-tb', output['loss_tb'], on_step=True, on_epoch=False, prog_bar=False)
        return output

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        x: torch.Tensor = batch['image']
        target = {'label': batch['label'], 'dataset': batch['dataset']}
        output = self.model(x, target)
        # logging
        self.log('val/loss', output['loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.log('val/loss-tb', output['loss_tb'], on_step=True, on_epoch=False, prog_bar=False)
        return output
