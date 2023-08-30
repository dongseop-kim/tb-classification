from typing import Any, Dict

from pytorch_lightning import LightningModule

from trainer.models.base import BaseModel


class BaseEngine(LightningModule):
    def __init__(self,
                 model: BaseModel,
                 optimizer=None,
                 scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = self.model.num_classes

    def configure_optimizers(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.scheduler}

    def _step(self, batch: Dict[str, Any]):
        x, target = batch['image'], batch['target']
        pred = self.model(x)
        return pred, target
