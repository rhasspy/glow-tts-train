import logging
import time
import typing
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from glow_tts_train.checkpoint import Checkpoint, save_checkpoint
from glow_tts_train.config import TrainingConfig
from glow_tts_train.dataset import Batch
from glow_tts_train.models import ModelType, setup_model, setup_optimizer
from glow_tts_train.utils import duration_loss, mle_loss, to_gpu

_LOGGER = logging.getLogger("glow_tts_train")

# -----------------------------------------------------------------------------


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_dir: Path,
    model: typing.Optional[ModelType] = None,
    optimizer: typing.Optional[torch.optim.Optimizer] = None,
    global_step: int = 1,
    checkpoint_epochs: int = 100,
    val_epochs: int = 1,
    rank: int = 0,
):
    """Run training for the specified number of epochs"""
    torch.manual_seed(config.seed)

    if model is None:
        model = setup_model(config)

    if optimizer is None:
        optimizer = setup_optimizer(config, model)

    assert model is not None
    assert optimizer is not None

    # Gradient scaler
    scaler = GradScaler() if config.fp16_run else None
    if scaler:
        _LOGGER.info("Using fp16 scaler")

    # Begin training
    best_val_loss = None
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

        if ((epoch % val_epochs) == 0) and (rank == 0):
            _LOGGER.debug("Running validation")
            val_loss = val_step(
                model=model,
                config=config,
                val_loader=val_loader,
                fp16_run=config.fp16_run,
            )

            if (best_val_loss is None) or (val_loss < best_val_loss):
                best_path = model_dir / "best_model.pth"
                _LOGGER.debug("Saving best model to %s", best_path)
                save_checkpoint(
                    Checkpoint(
                        model=model,
                        optimizer=optimizer,
                        global_step=global_step,
                        version=config.version,
                    ),
                    best_path,
                )

                best_val_loss = val_loss

        if ((epoch % checkpoint_epochs) == 0) and (rank == 0):
            # Save checkpoint
            checkpoint_path = model_dir / f"checkpoint_{global_step}.pth"
            _LOGGER.debug("Saving checkpoint to %s", checkpoint_path)
            save_checkpoint(
                Checkpoint(
                    model=model,
                    optimizer=optimizer,
                    global_step=global_step,
                    version=config.version,
                ),
                checkpoint_path,
            )

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
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    train_loader: DataLoader,
    fp16_run: bool,
    scaler: typing.Optional[GradScaler] = None,
):
    steps_per_epoch = len(train_loader)
    all_loss_g: typing.List[float] = []

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = typing.cast(Batch, batch)
        x, x_lengths, y, y_lengths, speaker_ids = (
            to_gpu(batch.phoneme_ids),
            to_gpu(batch.phoneme_lengths),
            to_gpu(batch.spectrograms),
            to_gpu(batch.spectrogram_lengths),
            to_gpu(batch.speaker_ids) if batch.speaker_ids is not None else None,
        )

        # Train model
        optimizer.zero_grad()

        with autocast(enabled=fp16_run):
            (
                (z, z_m, z_logs, logdet, z_mask),
                (_x_m, _x_logs, _x_mask),
                (_attn, logw, logw_),
            ) = model(x, x_lengths, y, y_lengths, g=speaker_ids)

            # Compute loss
            l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
            l_length = duration_loss(logw, logw_, x_lengths)

            loss_g = l_mle + l_length

        all_loss_g.append(loss_g.item())

        if fp16_run:
            # Float16
            assert scaler is not None
            scaler.scale(loss_g).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Float32
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
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


def val_step(
    model: ModelType, config: TrainingConfig, val_loader: DataLoader, fp16_run: bool,
):
    all_loss_g: typing.List[float] = []

    model.eval()
    for batch in val_loader:
        batch = typing.cast(Batch, batch)
        x, x_lengths, y, y_lengths, speaker_ids = (
            to_gpu(batch.phoneme_ids),
            to_gpu(batch.phoneme_lengths),
            to_gpu(batch.spectrograms),
            to_gpu(batch.spectrogram_lengths),
            to_gpu(batch.speaker_ids) if batch.speaker_ids is not None else None,
        )

        with autocast(enabled=fp16_run):
            (
                (z, z_m, z_logs, logdet, z_mask),
                (_x_m, _x_logs, _x_mask),
                (_attn, logw, logw_),
            ) = model(x, x_lengths, y, y_lengths, g=speaker_ids)

            # Compute loss
            l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
            l_length = duration_loss(logw, logw_, x_lengths)

            loss_g = l_mle + l_length

        all_loss_g.append(loss_g.item())

    if all_loss_g:
        avg_loss_g = sum(all_loss_g) / len(all_loss_g)
        _LOGGER.debug("Average validation loss: %s", avg_loss_g)

        return avg_loss_g

    return 0.0
