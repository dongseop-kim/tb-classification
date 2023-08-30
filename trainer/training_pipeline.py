
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import hydra
import pytorch_lightning as lightning
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

import utils.logger_rank_zero as rank_zero_logger
from trainer.datamodules.base import BaseDataModule
from trainer.engines.base import BaseEngine
from trainer.models.base import BaseModel

logger = rank_zero_logger.get_logger(__name__)


@dataclass
class TrainerPackage:
    trainer: lightning.Trainer
    datamodule: lightning.LightningDataModule
    engine: lightning.LightningModule
    ckpt_path: Path
    pl_loggers: List[LightningLoggerBase]
    pl_callbacks: List[lightning.Callback]
    config: DictConfig


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    package = make_trainer_package(config)
    return train_with_trainer_package(package)


def train_with_trainer_package(package: TrainerPackage) -> Optional[float]:
    trainer = package.trainer
    datamodule = package.datamodule
    engine = package.engine
    ckpt_path = package.ckpt_path
    pl_loggers = package.pl_loggers
    config = package.config

    # Train the model
    logger.info("Starting training!")
    trainer.fit(model=engine, datamodule=datamodule, ckpt_path=str(ckpt_path) if ckpt_path else None)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception("Metric for hyperparameter optimization not found! "
                        "Make sure the `optimized_metric` in `hparams_search` config is correct!")

    score = trainer.callback_metrics.get(optimized_metric)  # get the last score as default

    # # Test the model
    # if config.get("test"):
    #     ckpt_path = "best"
    #     if not config.get("train") or config.trainer.get("fast_dev_run"):
    #         ckpt_path = None
    #     logger.info("Starting testing!")
    #     trainer.test(model=engine, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    logger.info("Finalizing!")
    rank_zero_logger.finish(logger=pl_loggers)

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        logger.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    return score


def make_trainer_package(config):
    """ Instantiates all components of the training pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if config.get('seed'):
        seed_everything(config.seed, workers=True)

    # initialize LightningDataModule datamodule
    logger.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: BaseDataModule = hydra.utils.instantiate(config.datamodule)

    # initailize LightningModule model
    logger.info(f"Instantiating model <{config.model._target_}>")
    model: BaseModel = hydra.utils.instantiate(config.model,
                                               num_classes=datamodule.num_classes)

    # initailze optimizer and scheduler
    logger.info(f"Instantiating optimizer <{config.optimizer._target_}>")
    optimizer: Any = hydra.utils.instantiate(config.optimizer, params=model.parameters())
    logger.info(f"Instantiating scheduler <{config.scheduler._target_}>")
    scheduler: Any = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)

    # initialize base engine
    logger.info(f"Instantiating lightning engine <{config.engine._target_}>")
    engine: BaseEngine = hydra.utils.instantiate(config.engine,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler)

    """Init lightning callbacks"""
    pl_callbacks: List[lightning.Callback] = list()
    if "callbacks" in config:
        logger.info(f"Instantiating pl_callbacks: {[v.get('_target_') for v in config.callbacks.values()]}")
        pl_callbacks = [hydra.utils.instantiate(_conf) for _conf in config.callbacks.values()]

    """Init lightning loggers"""
    pl_loggers: List[LightningLoggerBase] = list()
    if "logger" in config:
        # logger.info(f"Instantiating pl_loggers: {[v.get('_target_') for v in config.logger.values()]}")
        # pl_loggers = [hydra.utils.instantiate(_conf) for _conf in config.logger.values()]
        pl_loggers = hydra.utils.instantiate(config.logger)

    """Init lightning trainer"""
    # https://github.com/PyTorchLightning/pytorch-lightning/discussions/6761
    logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: lightning.Trainer = hydra.utils.instantiate(config.trainer,
                                                         callbacks=pl_callbacks,
                                                         logger=pl_loggers,
                                                         _convert_="partial")

    return TrainerPackage(trainer=trainer, datamodule=datamodule, engine=engine, ckpt_path=None,
                          pl_loggers=pl_loggers, pl_callbacks=pl_callbacks, config=config)
