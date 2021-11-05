import logging
import time
import typing
from collections import Counter
from pathlib import Path

import torch
import torch.distributed
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from glow_tts_train.checkpoint import Checkpoint, save_checkpoint
from glow_tts_train.config import TrainingConfig
from glow_tts_train.dataset import Batch
from glow_tts_train.models import (
    ModelType,
    setup_model,
    setup_optimizer,
    setup_scheduler,
)
from glow_tts_train.optimize import OptimizerType, SchedulerType
from glow_tts_train.utils import duration_loss, mle_loss, to_gpu

_LOGGER = logging.getLogger("glow_tts_train")

# -----------------------------------------------------------------------------


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_dir: Path,
    model: typing.Optional[ModelType] = None,
    optimizer: typing.Optional[OptimizerType] = None,
    scheduler: typing.Optional[SchedulerType] = None,
    checkpoint_epochs: int = 100,
    val_epochs: int = 1,
    rank: int = 0,
):
    """Run training for the specified number of epochs"""
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.seed)

    if model is None:
        model = setup_model(config)

    if optimizer is None:
        optimizer = setup_optimizer(config, model)

    if (scheduler is None) and config.scheduler:
        scheduler = setup_scheduler(config, optimizer)

    assert model is not None
    assert optimizer is not None

    # Gradient scaler
    scaler = GradScaler() if config.fp16_run else None
    if scaler:
        _LOGGER.info("Using fp16 scaler")

    # Begin training
    best_val_loss = config.best_loss
    global_step = config.global_step

    bad_utterance_counts: typing.Counter[str] = Counter()
    bad_utterances_path = model_dir / "bad_utterances.txt"

    if bad_utterances_path.is_file():
        # Load bad counts
        with open(bad_utterances_path, "r", encoding="utf-8") as bad_utterances_file:
            for line in bad_utterances_file:
                line = line.strip()
                if not line:
                    continue

                utt_id, count_str = line.split(maxsplit=1)
                bad_utterance_counts[utt_id] = int(count_str)

    for epoch in range(config.last_epoch, config.epochs + 1):
        _LOGGER.debug(
            "Begin epoch %s/%s (global step=%s, learning_rate=%s)",
            epoch,
            config.epochs,
            global_step,
            optimizer._optim.param_groups[0]["lr"],  # pylint: disable=protected-access
        )
        epoch_start_time = time.perf_counter()
        global_step = train_step(
            rank=rank,
            global_step=global_step,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            config=config,
            train_loader=train_loader,
            fp16_run=config.fp16_run,
            scaler=scaler,
            bad_utterance_counts=bad_utterance_counts,
        )

        if ((epoch % val_epochs) == 0) and (rank == 0):
            _LOGGER.debug("Running validation")
            val_loss = val_step(
                model=model,
                config=config,
                val_loader=val_loader,
                fp16_run=config.fp16_run,
            )

            _LOGGER.debug("Validation loss: %s (best=%s)", val_loss, best_val_loss)

            if scheduler is not None:
                scheduler.step()

            if (best_val_loss is None) or (val_loss < best_val_loss):
                best_path = model_dir / "best_model.pth"
                _LOGGER.debug("Saving best model to %s", best_path)
                save_checkpoint(
                    Checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=epoch,
                        version=config.version,
                        best_loss=best_val_loss,
                    ),
                    best_path,
                )

                best_val_loss = val_loss

            with open(
                bad_utterances_path, "w", encoding="utf-8"
            ) as bad_utterances_file:
                for utt_id, bad_count in bad_utterance_counts.most_common():
                    print(utt_id, bad_count, file=bad_utterances_file)

        if ((epoch % checkpoint_epochs) == 0) and (rank == 0):
            # Save checkpoint
            checkpoint_path = model_dir / f"checkpoint_{global_step}.pth"
            _LOGGER.debug("Saving checkpoint to %s", checkpoint_path)
            save_checkpoint(
                Checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    version=config.version,
                    best_loss=best_val_loss,
                ),
                checkpoint_path,
            )

        epoch_end_time = time.perf_counter()
        _LOGGER.debug(
            "[%s] epoch %s complete in %s second(s) (global step=%s)",
            rank,
            epoch,
            epoch_end_time - epoch_start_time,
            global_step,
        )


def train_step(
    rank: int,
    global_step: int,
    epoch: int,
    model: ModelType,
    optimizer: OptimizerType,
    config: TrainingConfig,
    train_loader: DataLoader,
    fp16_run: bool,
    scaler: typing.Optional[GradScaler] = None,
    bad_utterance_counts: typing.Optional[typing.Counter[str]] = None,
):
    steps_per_epoch = len(train_loader)
    all_loss_g: typing.List[float] = []
    last_loss_g = None

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

        loss_g_num = loss_g.item()
        all_loss_g.append(loss_g_num)

        if fp16_run:
            # Float16
            assert scaler is not None
            scaler.scale(loss_g).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer._optim)  # pylint: disable=protected-access
            scaler.update()
        else:
            # Float32
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        _LOGGER.debug(
            "[%s] loss: %s (step=%s/%s, lr=%s)",
            rank,
            loss_g.item(),
            batch_idx + 1,
            steps_per_epoch,
            optimizer._optim.param_groups[0]["lr"],  # pylint: disable=protected-access
        )
        global_step += 1

        if (
            (bad_utterance_counts is not None)
            and (last_loss_g is not None)
            and (loss_g_num > last_loss_g)
        ):
            for utt_id in batch.utterance_ids:
                bad_utterance_counts[utt_id] += 1

        last_loss_g = loss_g_num

    if all_loss_g:
        avg_loss_g = sum(all_loss_g) / len(all_loss_g)
        _LOGGER.info(
            "[%s] avg. Loss for epoch %s: %s (global step=%s)",
            rank,
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
    with torch.no_grad():
        for batch in val_loader:
            batch = typing.cast(Batch, batch)
            x, x_lengths, y, y_lengths, speaker_ids = (
                to_gpu(batch.phoneme_ids),
                to_gpu(batch.phoneme_lengths),
                to_gpu(batch.spectrograms),
                to_gpu(batch.spectrogram_lengths),
                to_gpu(batch.speaker_ids) if batch.speaker_ids is not None else None,
            )

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
        return avg_loss_g

    return 0.0
