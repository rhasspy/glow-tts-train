"""Methods for saving/loading checkpoints"""
import logging
import typing
from dataclasses import dataclass
from pathlib import Path

import torch

from glow_tts_train.config import TrainingConfig
from glow_tts_train.models import (
    ModelType,
    setup_model,
    setup_optimizer,
    setup_scheduler,
)
from glow_tts_train.optimize import OptimizerType, SchedulerType

_LOGGER = logging.getLogger("glow_tts_train.checkpoint")

# -----------------------------------------------------------------------------


@dataclass
class Checkpoint:
    model: ModelType
    global_step: int
    epoch: int
    version: int
    best_loss: typing.Optional[float] = None
    optimizer: typing.Optional[OptimizerType] = None
    scheduler: typing.Optional[SchedulerType] = None


def save_checkpoint(checkpoint: Checkpoint, checkpoint_path: Path):
    """Save model/optimizer/training state to a Torch checkpoint"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = checkpoint.model
    optimizer = checkpoint.optimizer
    scheduler = checkpoint.scheduler

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()  # type: ignore
    else:
        state_dict = model.state_dict()

    checkpoint_dict = {
        "model": state_dict,
        "global_step": checkpoint.global_step,
        "epoch": checkpoint.epoch,
        "version": checkpoint.version,
        "best_loss": checkpoint.best_loss,
    }

    if optimizer is not None:
        checkpoint_dict["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint_dict["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint_dict, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    config: TrainingConfig,
    model: typing.Optional[ModelType] = None,
    optimizer: typing.Optional[OptimizerType] = None,
    scheduler: typing.Optional[SchedulerType] = None,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    use_cuda: bool = True,
) -> Checkpoint:
    """Load model/optimizer/training state from a Torch checkpoint"""
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    version = int(checkpoint_dict.get("version", 1))
    global_step = int(checkpoint_dict.get("global_step", 1))
    epoch = int(checkpoint_dict.get("epoch", 1))

    best_loss = checkpoint_dict.get("best_loss")
    if best_loss is not None:
        best_loss = float(best_loss)

    # Create model/optimizer if necessary
    if model is None:
        model = setup_model(config, use_cuda=use_cuda)

    # Load scheduler state
    if load_scheduler and ("scheduler" in checkpoint_dict):
        if scheduler is None:
            if optimizer is None:
                optimizer = setup_optimizer(config, model)

            scheduler = setup_scheduler(config, optimizer)

        scheduler.load_state_dict(checkpoint_dict["scheduler"])

    # Load optimizer state
    if load_optimizer and ("optimizer" in checkpoint_dict):
        if optimizer is None:
            optimizer = setup_optimizer(config, model)

        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()  # type: ignore
    else:
        state_dict = model.state_dict()

    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.warning("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)  # type: ignore
    else:
        model.load_state_dict(new_state_dict)

    return Checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch=epoch,
        version=version,
        best_loss=best_loss,
    )
