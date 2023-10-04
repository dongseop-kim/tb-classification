from pathlib import Path
from typing import Union

import hydra
import yaml
from omegaconf import DictConfig

from trainer.datamodules.base import BaseDataModule
from trainer.models.base import BaseModel


def parse_yaml(path: Union[str, Path]):
    """Reads a yaml file."""
    with open(str(path), "r") as f:
        return DictConfig(yaml.load(f, Loader=yaml.FullLoader))


def parse_datamodule(cfg:  Union[str, Path], verbose: bool = False) -> BaseDataModule:
    cfg = parse_yaml(cfg)['datamodule']
    if verbose:
        print(cfg, end="\n\n")
    return hydra.utils.instantiate(cfg)


def parse_model(cfg:  Union[str, Path], num_classes: int = 1, verbose: bool = False) -> BaseModel:
    cfg = parse_yaml(cfg)['model']
    if verbose:
        print(cfg, end="\n\n")
    return hydra.utils.instantiate(cfg, num_classes=num_classes)
