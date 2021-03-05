import logging
import time
import typing
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .checkpoint import Checkpoint, save_checkpoint
from .config import TrainingConfig
from .models import ModelType, setup_model
from .optimize import OptimizerType
from .utils import clip_grad_value_, duration_loss, mle_loss, to_gpu

_LOGGER = logging.getLogger("glow_tts_train")

# -----------------------------------------------------------------------------


def train(
    train_loader: DataLoader,
    config: TrainingConfig,
    model_dir: Path,
    model: typing.Optional[ModelType] = None,
    optimizer: typing.Optional[OptimizerType] = None,
    global_step: int = 1,
    checkpoint_epochs: int = 1,
    rank: int = 0,
):
    """Run training for the specified number of epochs"""
    torch.manual_seed(config.seed)

    model, optimizer = setup_model(config, model=model, optimizer=optimizer)
    assert optimizer is not None
    assert model is not None

    # Gradient scaler
    scaler = GradScaler() if config.fp16_run else None
    if scaler:
        _LOGGER.info("Using fp16 scaler")

    # Begin training
    for epoch in range(1, config.epochs + 1):
        _LOGGER.debug(
            "Begin epoch %s/%s (global step=%s)", epoch, config.epochs, global_step
        )
        epoch_start_time = time.perf_counter()
        global_step = train_step(
            global_step=global_step,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            config=config,
            train_loader=train_loader,
            fp16_run=config.fp16_run,
            scaler=scaler,
        )

        if ((epoch % checkpoint_epochs) == 0) and (rank == 0):
            # Save checkpoint
            checkpoint_path = model_dir / f"checkpoint_{global_step}.pth"
            _LOGGER.debug("Saving checkpoint to %s", checkpoint_path)
            save_checkpoint(
                Checkpoint(
                    model=model,
                    optimizer=optimizer,
                    learning_rate=optimizer.cur_lr,
                    global_step=global_step,
                    version=config.version,
                ),
                checkpoint_path,
            )

            # Save checkpoint config
            config_path = model_dir / f"config_{global_step}.json"
            with open(config_path, "w") as config_file:
                config.save(config_file)

            _LOGGER.info("Saved checkpoint to %s", checkpoint_path)

        epoch_end_time = time.perf_counter()
        _LOGGER.debug(
            "Epoch %s complete in %s second(s) (global step=%s)",
            epoch,
            epoch_end_time - epoch_start_time,
            global_step,
        )


def train_step(
    global_step: int,
    epoch: int,
    model: ModelType,
    optimizer: OptimizerType,
    config: TrainingConfig,
    train_loader: DataLoader,
    fp16_run: bool,
    scaler: typing.Optional[GradScaler] = None,
):
    # train_loader.sampler.set_epoch(epoch)
    steps_per_epoch = len(train_loader)
    all_loss_g: typing.List[float] = []

    model.train()
    for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = (to_gpu(x), to_gpu(x_lengths))
        y, y_lengths = (to_gpu(y), to_gpu(y_lengths))

        # Train model
        optimizer.zero_grad()

        with autocast(enabled=fp16_run):
            (
                (z, z_m, z_logs, logdet, z_mask),
                (_x_m, _x_logs, _x_mask),
                (_attn, logw, logw_),
            ) = model(x, x_lengths, y, y_lengths)

            # Compute loss
            l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
            l_length = duration_loss(logw, logw_, x_lengths)

            # TODO: Weighted loss
            # loss_gs = [l_mle, l_length]
            loss_g = l_mle + l_length

        all_loss_g.append(loss_g.item())

        if fp16_run:
            # Float16
            assert scaler is not None
            scaler.scale(loss_g).backward()
            scaler.unscale_(optimizer._optim)  # pylint: disable=protected-access
            clip_grad_value_(model.parameters(), config.grad_clip)

            scaler.step(optimizer._optim)  # pylint: disable=protected-access
            scaler.update()
        else:
            # Float32
            loss_g.backward()
            clip_grad_value_(model.parameters(), config.grad_clip)
            optimizer.step()

        _LOGGER.debug(
            "Loss: %s (step=%s/%s)", loss_g.item(), batch_idx + 1, steps_per_epoch
        )
        global_step += 1

    if all_loss_g:
        avg_loss_g = sum(all_loss_g) / len(all_loss_g)
        _LOGGER.info(
            "Avg. Loss for epoch %s: %s (global step=%s)",
            epoch,
            avg_loss_g,
            global_step,
        )

    return global_step
