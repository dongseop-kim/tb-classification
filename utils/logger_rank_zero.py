import logging
from typing import List

import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def log_hyperparameters(config: DictConfig,
                        model: pl.LightningModule,
                        trainer: pl.Trainer) -> None:
    """ Log all hyperparameters defined in config. 
        Additionally, logs the number of model parameters for total, trainable, and non-trainable.
    """
    if not trainer.logger:
        return

    hparams = omegaconf.OmegaConf.to_container(config, resolve=True)

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    trainer.logger.log_hyperparams(hparams)


def finish(logger: List[pl.loggers.LightningLoggerBase]) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
