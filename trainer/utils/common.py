import hydra
import torch
import yaml
from omegaconf import DictConfig


def load_config(config_path: str) -> DictConfig:
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return DictConfig(config)


def instantiate_model(config: DictConfig, num_classes: int):
    model = hydra.utils.instantiate(config.config_model, num_classes=num_classes)
    return model


def instantiate_engine(config: DictConfig, model, optimizer=None, scheduler=None, checkpoint=None):
    engine = hydra.utils.instantiate(config.config_engine, model=model, optimizer=optimizer, scheduler=scheduler)
    if not checkpoint:
        return engine
    checkpoint = torch.load(checkpoint, map_location='cpu')
    checkpoint = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    engine.load_state_dict(checkpoint)
    return engine
