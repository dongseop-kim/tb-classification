from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import torch
import hydra
import yaml
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from trainer.datamodules.base import BaseDataModule
from trainer.models.base import BaseModel
from trainer.engines.base import BaseEngine


def parse_yaml(filepath: str):
    """Reads a yaml file."""
    with open(filepath, "r") as f:
        return DictConfig(yaml.load(f, Loader=yaml.FullLoader))


def parse_yaml_with_key(filepath: Union[str, DictConfig], key: str):
    cfg = parse_yaml(filepath) if isinstance(filepath, str) else filepath
    return cfg[key]


def get_datamodule(cfg_file: Union[str, DictConfig]):
    config = parse_yaml_with_key(cfg_file, 'datamodule')
    return hydra.utils.instantiate(config)


def get_model(cfg_file: str, num_classes: int = 1):
    config = parse_yaml_with_key(cfg_file, 'model')
    return hydra.utils.instantiate(config, num_classes=num_classes)


def get_engine(cfg_file: str, model: BaseModel, optimizer=None, scheduler=None):
    config = parse_yaml_with_key(cfg_file, 'engine')
    return hydra.utils.instantiate(config, model=model, optimizer=optimizer, scheduler=scheduler)


@torch.no_grad()
def main():
    path_cfg: str = './configs/experiment/tb-tbx11k.yaml'

    datamodule: BaseDataModule = get_datamodule(path_cfg)
    dataloader_test: DataLoader = datamodule.test_dataloader()
    model: BaseModel = get_model(path_cfg, num_classes=dataloader_test.dataset.num_classes)

    print(len(dataloader_test))
    print(dataloader_test.dataset.num_classes)

    path_weights = '/data1/dongseopkim/experiment_logs/tb-base/tuberculosis-tbx11k/tb-x11k-base-01-e100-1e-5/2023-08-31_23-00-59/checkpoints/epoch_060.ckpt'
    print(model.load_weights(path_weights))

    device = f'cuda:{3}'
    # engine
    engine: BaseEngine = get_engine(path_cfg, model)
    engine.eval().cuda(device)

    for batch in dataloader_test:
        batch['image'] = batch['image'].cuda(device)
        pred, target = engine._step(batch)
        pred = torch.sigmoid(pred)
        print(pred)
        break


if __name__ == '__main__':
    main()
