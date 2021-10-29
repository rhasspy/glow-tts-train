#!/usr/bin/env python3
import argparse
import logging
import random
import typing
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from glow_tts_train.checkpoint import load_checkpoint
from glow_tts_train.config import TrainingConfig
from glow_tts_train.dataset import (
    PhonemeIdsAndMelsDataset,
    UtteranceCollate,
    load_dataset,
)
from glow_tts_train.ddi import initialize_model
from glow_tts_train.models import ModelType
from glow_tts_train.optimize import OptimizerType
from glow_tts_train.train import train

_LOGGER = logging.getLogger("glow_tts_train")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-train")
    parser.add_argument(
        "--output", required=True, help="Directory to store model artifacts"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        nargs=3,
        action="append",
        default=[],
        metavar=("dataset_name", "metadata_dir", "audio_dir"),
        help="Speaker id, phonemes CSV, and directory with audio files",
    )
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: use config)"
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to run (default: use config)"
    )
    parser.add_argument("--checkpoint", help="Path to restore checkpoint")
    parser.add_argument("--git-commit", help="Git commit to store in config")
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=100,
        help="Number of epochs between checkpoints",
    )
    parser.add_argument(
        "--cache",
        help="Directory to store cached spectrograms (default: <output>/cache",
    )
    parser.add_argument(
        "--local_rank", type=int, help="Rank passed from torch.distributed.launch"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    assert torch.cuda.is_available(), "GPU is required for training"

    is_distributed = args.local_rank is not None

    if is_distributed:
        _LOGGER.info("Setting up distributed run (rank=%s)", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # -------------------------------------------------------------------------

    # Convert to paths
    args.output = Path(args.output)
    args.dataset = [
        (dataset_name, Path(phonemes_path), Path(audio_dir))
        for dataset_name, phonemes_path, audio_dir in args.dataset
    ]

    if args.config:
        args.config = [Path(p) for p in args.config]
    else:
        output_config_path = args.output / "config.json"
        assert (
            output_config_path.is_file()
        ), f"No config file found at {output_config_path}"

        args.config = [output_config_path]

    if args.checkpoint:
        args.checkpoint = Path(args.checkpoint)

    if args.cache:
        args.cache = Path(args.cache)
    else:
        args.cache = args.output / "cache"

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    config.git_commit = args.git_commit

    _LOGGER.debug(config)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    _LOGGER.debug("Setting random seed to %s", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    if args.epochs is not None:
        # Use command-line option
        config.epochs = args.epochs

    num_speakers = config.model.n_speakers
    if num_speakers > 1:
        assert (
            config.model.gin_channels > 0
        ), "Multispeaker model must have gin_channels > 0"

    assert (
        len(args.dataset) <= num_speakers
    ), "More datasets than speakers in model config"

    if len(args.dataset) < num_speakers:
        _LOGGER.warning(
            "Model has %s speaker(s), but only %s dataset(s) were provided",
            num_speakers,
            len(args.dataset),
        )

    datasets = []
    for dataset_name, metadata_dir, audio_dir in args.dataset:
        metadata_dir = Path(metadata_dir)
        audio_dir = Path(audio_dir)

        datasets.append(
            load_dataset(
                config=config,
                dataset_name=dataset_name,
                metadata_dir=metadata_dir,
                audio_dir=audio_dir,
            )
        )

    # Create data loader
    batch_size = config.batch_size if args.batch_size is None else args.batch_size
    train_dataset = PhonemeIdsAndMelsDataset(
        config, datasets, split="train", cache_dir=args.cache
    )
    val_dataset = PhonemeIdsAndMelsDataset(
        config, datasets, split="val", cache_dir=args.cache
    )
    collate_fn = UtteranceCollate()

    train_loader = DataLoader(
        train_dataset,
        shuffle=(not is_distributed),
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=DistributedSampler(train_dataset) if is_distributed else None,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model: typing.Optional[ModelType] = None
    optimizer: typing.Optional[OptimizerType] = None
    global_step: int = 1

    if args.checkpoint:
        _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
        checkpoint = load_checkpoint(args.checkpoint, config)
        model, optimizer = checkpoint.model, checkpoint.optimizer
        global_step = checkpoint.global_step
        _LOGGER.info(
            "Loaded checkpoint from %s (global step=%s, learning rate=%s)",
            args.checkpoint,
            global_step,
            config.learning_rate,
        )
    else:
        # Data-dependent initialization
        _LOGGER.info("Doing data-dependent initialization...")
        model = initialize_model(train_loader, config)

    if is_distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Train
    _LOGGER.info(
        "Training started (batch size=%s, epochs=%s)", batch_size, config.epochs
    )

    try:
        train(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            model_dir=args.output,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            checkpoint_epochs=args.checkpoint_epochs,
            rank=(args.local_rank if is_distributed else 0),
        )
        _LOGGER.info("Training finished")
    except KeyboardInterrupt:
        _LOGGER.info("Training stopped")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
