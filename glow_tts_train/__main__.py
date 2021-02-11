#!/usr/bin/env python3
import argparse
import logging
import random
import typing
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .config import TrainingConfig
from .dataset import PhonemeMelCollate, PhonemeMelLoader, load_mels, load_phonemes
from .ddi import initialize_model
from .models import ModelType
from .optimize import OptimizerType
from .train import train

_LOGGER = logging.getLogger("glow_tts_train")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-train")
    parser.add_argument("model_dir", help="Directory to store model artifacts")
    parser.add_argument(
        "phonemes_csv", help="CSV file with utterance id|phoneme ids lines"
    )
    parser.add_argument("mels_jsonl", help="JSONL file with mel spectrograms")
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: use config)"
    )
    parser.add_argument("--checkpoint", help="Path to restore checkpoint")
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=1,
        help="Number of epochs between checkpoints",
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

    # -------------------------------------------------------------------------

    # Convert to paths
    args.model_dir = Path(args.model_dir)
    args.phonemes_csv = Path(args.phonemes_csv)
    args.mels_jsonl = Path(args.mels_jsonl)

    if args.config:
        args.config = [Path(p) for p in args.config]

    if args.checkpoint:
        args.checkpoint = Path(args.checkpoint)

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    _LOGGER.debug(config)

    # Create output directory
    args.model_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.debug("Setting random seed to %s", config.seed)
    random.seed(config.seed)

    # Load phonemes
    _LOGGER.debug("Loading phonemes from %s", args.phonemes_csv)
    with open(args.phonemes_csv, "r") as phonemes_file:
        id_phonemes = load_phonemes(phonemes_file)

    _LOGGER.info("Loaded phonemes for %s utterances", len(id_phonemes))

    # Load mels
    # TODO: Verify audio configuration
    _LOGGER.debug("Loading mels from %s", args.mels_jsonl)
    with open(args.mels_jsonl, "r") as mels_file:
        id_mels = load_mels(mels_file)

    _LOGGER.info("Loaded mels for %s utterances", len(id_mels))

    # Set num_symbols
    if config.model.num_symbols < 1:
        config.model.num_symbols = max(max(p_ids) for p_ids in id_phonemes.values()) + 1

    assert config.model.num_symbols > 0, "No symbols"

    # Save config
    if not args.config.is_file():
        with open(args.config, "w") as config_file:
            config.save(config_file)

        _LOGGER.debug("Saved config to %s", args.config)

    # Create data loader
    dataset = PhonemeMelLoader(id_phonemes, id_mels)
    collate_fn = PhonemeMelCollate()

    batch_size = config.batch_size if args.batch_size is None else args.batch_size
    train_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model: typing.Optional[ModelType] = None
    optimizer: typing.Optional[OptimizerType] = None
    global_step: int = 1

    if args.checkpoint:
        _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
        checkpoint = load_checkpoint(args.checkpoint, config)
        model, optimizer = checkpoint.model, checkpoint.optimizer
        config.learning_rate = checkpoint.learning_rate
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

    # Train
    _LOGGER.info("Training started (batch size=%s)", batch_size)

    try:
        train(
            train_loader,
            config,
            args.model_dir,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            checkpoint_epochs=args.checkpoint_epochs,
        )
        _LOGGER.info("Training finished")
    except KeyboardInterrupt:
        _LOGGER.info("Training stopped")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
