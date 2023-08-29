from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, List, Optional

import hydra
import pytorch_lightning as lightning
import pytorch_lightning.callbacks.model_checkpoint as plcb_modelckpt
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning.loggers import LightningLoggerBase

import utils.logger_rank_zero as rank_zero_logger

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

    # Send some parameters from config to all lightning loggers
    logger.info("Logging hyperparameters!")
    rank_zero_logger.log_hyperparameters(config=config,
                                         model=engine,
                                         trainer=trainer)

    # Train the model
    if config.get("train"):
        logger.info("Starting training!")
        trainer.fit(model=engine, datamodule=datamodule, ckpt_path=str(ckpt_path) if ckpt_path else None)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception("Metric for hyperparameter optimization not found! "
                        "Make sure the `optimized_metric` in `hparams_search` config is correct!")

    score = trainer.callback_metrics.get(optimized_metric)  # get the last score as default

    # Try finding best score from model checkpoint callback
    model_checkpoint_callback = _get_model_checkpoint_callback(trainer)
    if model_checkpoint_callback:
        best_score = model_checkpoint_callback.best_model_score
        if best_score:
            score = best_score
        else:
            logger.warning("Cannot get best score from existing ModelCheckpoint callback. "
                           "Last score will be returned instead.")
    else:
        logger.warning("Cannot find ModelCheckpoint callback from trainer. Last score will be returned instead.")

    # Test the model
    if config.get("test"):
        ckpt_path = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        logger.info("Starting testing!")
        trainer.test(model=engine, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    logger.info("Finalizing!")
    rank_zero_logger.finish(logger=pl_loggers)

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        logger.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    if torch.isnan(score):
        logger.warning(f"Final Score is Nan!")
    return score


def make_trainer_package(config, suppress_logger=False):
    if suppress_logger:
        logger.info(f"In `make_trainer_package`: {suppress_logger=}")
        logger.setLevel(logging.ERROR)

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get('seed'):
        lightning.seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    resume_from_checkpoint = config.trainer.get('resume_from_checkpoint')
    if resume_from_checkpoint:
        config.trainer.resume_from_checkpoint = Path(hydra.utils.get_original_cwd()) / resume_from_checkpoint

    # `resume_from_checkpoint` is deprecated. Use `ckpt_path` instead
    ckpt_path = config.get('ckpt_path')
    if ckpt_path:
        ckpt_path = Path(hydra.utils.get_original_cwd()) / ckpt_path

    # Init lightning datamodule
    logger.info(f"Instantiating datamodule <{config.datamodule._target_}>")

    datamodule: lightning.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    if 'steps_per_epoch' in config.scheduler.keys() and config.get('override_steps_per_epoch'):
        config.scheduler.steps_per_epoch = len(datamodule.train_dataloader())

    # Init pytorch model
    logger.info(f"Instantiating model <{config.model._target_}>")
    model: nn.Module = hydra.utils.instantiate(config.model, num_classes=datamodule.data_train.num_classes)

    # `model_ckpt_path` is custom load option for model-only state dicts without resuming other lightning status
    model_ckpt_path = config.get('model_ckpt_path')
    if model_ckpt_path:
        model_ckpt_path = Path(hydra.utils.get_original_cwd()) / model_ckpt_path

        logger.info(f"Loading custom model-only ckpt without other lightning status: {model_ckpt_path=}")
        _temp_device = next(model.parameters()).device
        ckpt = torch.load(model_ckpt_path, map_location=_temp_device)
        ckpt = ckpt['state_dict']

        model_ckpt_cut_prefix = config.get('model_ckpt_cut_prefix')
        if model_ckpt_cut_prefix:
            ckpt = {re.sub('^model\.', '', k): v for k, v in ckpt.items()}

        model.load_state_dict(ckpt)

    # Init optimizer
    logger.info(f"Instantiating optimizer <{config.optimizer._target_}>")
    optimizer: Any = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    # Init scheduler
    logger.info(f"Instantiating scheduler <{config.scheduler._target_}>")
    scheduler: Any = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)

    # Init lightning model
    logger.info(f"Instantiating lightning engine <{config.engine._target_}>")
    engine: lightning.LightningModule = hydra.utils.instantiate(config.engine,
                                                                model=model,
                                                                optimizer=optimizer,
                                                                scheduler=scheduler,
                                                                _recursive_=False)

    """Init lightning callbacks"""
    pl_callbacks: List[lightning.Callback] = list()
    if "callbacks" in config:
        logger.info(f"Instantiating pl_callbacks: {[v.get('_target_') for v in config.callbacks.values()]}")
        pl_callbacks = [hydra.utils.instantiate(_conf) for _conf in config.callbacks.values()]

    """Init lightning loggers"""
    pl_loggers: List[LightningLoggerBase] = list()
    if "logger" in config:
        logger.info(f"Instantiating pl_loggers: {[v.get('_target_') for v in config.logger.values()]}")
        pl_loggers = [hydra.utils.instantiate(_conf) for _conf in config.logger.values()]

    """Init lightning trainer"""
    # https://github.com/PyTorchLightning/pytorch-lightning/discussions/6761
    logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: lightning.Trainer = hydra.utils.instantiate(config.trainer,
                                                         callbacks=pl_callbacks,
                                                         logger=pl_loggers,
                                                         _convert_="partial")

    return TrainerPackage(trainer=trainer, datamodule=datamodule, engine=engine, ckpt_path=ckpt_path,
                          pl_loggers=pl_loggers, pl_callbacks=pl_callbacks, config=config)


def _get_model_checkpoint_callback(trainer: lightning.Trainer) -> Optional[plcb_modelckpt.ModelCheckpoint]:
    for cb in trainer.callbacks:
        if isinstance(cb, plcb_modelckpt.ModelCheckpoint):
            return cb
