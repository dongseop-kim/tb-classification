import argparse
from typing import Literal

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics import AUROC, ConfusionMatrix
from tqdm import tqdm
from univdt.components import PublicTuberculosis

import trainer.utils.common as uc
from trainer.datamodule import (DEFAULT_TEST_TRANSFORMS,
                                DEFAULT_VALID_TRANSFORMS)


@torch.no_grad()
def main(config: DictConfig, checkpoint: str, data_dir: str, dataset: str,
         batch_size: int, device: int, split: Literal['val', 'test']):

    # config.config_datamodule 에 transforms_val 이 있을때만 가져오기
    if 'transforms_val' in config.config_datamodule:
        transforms = config.config_datamodule.transforms_val
    else:
        transforms = DEFAULT_VALID_TRANSFORMS
    dataset = PublicTuberculosis(root_dir=data_dir, split=split, dataset=dataset, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    device = f'cuda:{device}'
    model = uc.instantiate_model(config, num_classes=1)
    engine = uc.instantiate_engine(config, model, checkpoint=checkpoint)
    engine = engine.eval().to(device)

    auroc = AUROC(task='binary').to(device)
    cm = ConfusionMatrix(task='binary').to(device)

    scores, labels = [], []
    for i, batch in enumerate(tqdm(loader)):
        output = engine.predict_step(batch, i)
        pred_tb = output['pred_tb']

        label_tb = [1 if label in [1, 2, 3] else 0 for label in batch['label']]
        label_tb = torch.Tensor(label_tb).to(device).to(torch.long)

        scores += pred_tb
        labels += label_tb
        cm.update(pred_tb, label_tb)

    scores = torch.Tensor(scores).to(device)
    labels = torch.Tensor(labels).to(device)
    auroc.update(scores, labels)

    auc_score = auroc.compute()
    tn, fp, fn, tp = cm.compute().view(-1)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    sensitivity = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    f1score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-7)

    print(f'AUC: {auc_score:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'F1 Score: {f1score:.4f}')
    print(f'fn: {fn}, fp: {fp}, tn: {tn}, tp: {tp}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model.')
    parser.add_argument('--config', type=str, help='Configuration file path.')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file path.')
    parser.add_argument('--data-dir', type=str, help='Data directory path.')
    parser.add_argument('--dataset', type=str, help='Dataset name.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=int, default=0, help='Device number')
    parser.add_argument('--split', type=str, default='val', help='Split name')
    args = parser.parse_args()

    config = uc.load_config(args.config)
    main(config, args.checkpoint, args.data_dir, args.dataset, args.batch_size, args.device, args.split)
