import torch
from pytorch_lightning import LightningModule

from trainer.models import Model


class BaseEngine(LightningModule):
    def __init__(self, model: Model, optimizer=None, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}
