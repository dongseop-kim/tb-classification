from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix

from trainer.engine.base import BaseEngine
from trainer.models import Model

_eps = 1e-7


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

    def update_confusion_matrix(self, outputs: dict[str, torch.Tensor], batch: Any, meter: ConfusionMatrix):
        '''Update confusion matrix for each batch'''
        logit_tb = outputs['logit_tb']  # N
        pred_tb = torch.sigmoid(logit_tb)  # N
        label_tb = [1 if label in [1, 2, 3] else 0 for label in batch['label']]
        label_tb = torch.Tensor(label_tb).to(pred_tb.device).to(torch.long)
        meter.update(pred_tb, label_tb)

    def compute_confusion_matrix(self, meter: ConfusionMatrix):
        '''Compute confusion matrix'''
        confusion_matrix: torch.Tensor = meter.compute()
        tn, fp, fn, tp = confusion_matrix.view(-1)
        accuracy = (tp + tn) / (tp + tn + fp + fn + _eps)
        precision = tp / (tp + fp + _eps)
        sensitivity = tp / (tp + fn + _eps)
        specificity = tn / (tn + fp + _eps)
        f1score = 2 * (precision * sensitivity) / (precision + sensitivity + _eps)
        return {'accuracy': accuracy, 'precision': precision, 'sensitivity': sensitivity,
                'specificity': specificity, 'f1score': f1score}

    ''' ====================== '''
    ''' ===== TRAINING ===== '''
    ''' ====================== '''

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        self.log('train/loss', outputs['loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.train_step_outputs.append(outputs)  # save outputs
        self.update_confusion_matrix(outputs, batch, self.meter_train)
        scores: dict[str, torch.Tensor] = self.compute_confusion_matrix(self.meter_train)
        scores = {f'train/{k}': v for k, v in scores.items()}
        self.log_dict(scores, on_step=True, on_epoch=False, prog_bar=True)

    def on_train_epoch_end(self):
        self.aggregate_and_logging(self.train_step_outputs, 'loss_tb', prefix='train', is_step=False)
        self.train_step_outputs.clear()
        self.meter_train.reset()

    ''' ====================== '''
    ''' ===== VALIDATION ===== '''
    ''' ====================== '''

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self.step(batch)

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int):
        self.validation_step_outputs.append(outputs)
        self.update_confusion_matrix(outputs, batch, self.meter_val)

    def on_validation_epoch_end(self):
        scores = self.compute_confusion_matrix(self.meter_val)
        scores = {f'val/{k}': v for k, v in scores.items()}
        self.log_dict(scores, on_step=False, on_epoch=True, prog_bar=True)
        self.meter_val.reset()

        self.aggregate_and_logging(self.validation_step_outputs, 'loss', prefix='val', is_step=False)
        self.aggregate_and_logging(self.validation_step_outputs, 'loss_tb', prefix='val', is_step=False)
        self.validation_step_outputs.clear()

    ''' ====================== '''
    ''' ====== PREDICT  ====== '''
    ''' ====================== '''

    def predict_step(self, batch: dict[str, Any], batch_idx: int):
        output = self.model(batch['image'].to(self.device))
        logit_tb = output['logit_tb']
        logit_aux = output['logit_aux']
        pred_tb = torch.sigmoid(logit_tb)
        pred_aux = torch.sigmoid(logit_aux)
        return {'logit_tb': logit_tb, 'logit_aux': logit_aux, 
                'pred_tb': pred_tb, 'pred_aux': pred_aux}