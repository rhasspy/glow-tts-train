"""Methods for saving/loading checkpoints"""
import logging
import typing
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import TrainingConfig
from .models import ModelType, setup_model
from .optimize import OptimizerType

_LOGGER = logging.getLogger("glow-tts-train.checkpoint")

# -----------------------------------------------------------------------------


@dataclass
class Checkpoint:
    model: ModelType
    optimizer: OptimizerType
    learning_rate: float
    global_step: int
    version: int


def save_checkpoint(checkpoint: Checkpoint, checkpoint_path: Path):
    """Save model/optimizer/training state to a Torch checkpoint"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = checkpoint.model
    optimizer = checkpoint.optimizer

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()  # type: ignore
    else:
        state_dict = model.state_dict()

    torch.save(
        {
            "model": state_dict,
            "global_step": checkpoint.global_step,
            "optimizer": optimizer.state_dict(),
            "learning_rate": checkpoint.learning_rate,
            "version": checkpoint.version,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    config: TrainingConfig,
    model: typing.Optional[ModelType] = None,
    optimizer: typing.Optional[OptimizerType] = None,
) -> Checkpoint:
    """Load model/optimizer/training state from a Torch checkpoint"""
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    version = int(checkpoint_dict["version"])
    global_step = int(checkpoint_dict["global_step"])
    learning_rate = float(checkpoint_dict["learning_rate"])

    # Create model/optimizer if necessary
    model, optimizer = setup_model(config, model=model, optimizer=optimizer)

    # Load optimizer state
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
        learning_rate=learning_rate,
        global_step=global_step,
        version=version,
    )
