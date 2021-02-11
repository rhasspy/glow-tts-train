#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.export")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-train.export")
    parser.add_argument("config", help="Path to JSON configuration file")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument("output", help="Path to output model (.pth)")

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

    # Convert to paths
    args.config = Path(args.config)
    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)

    # Load config
    with open(args.config, "r") as config_file:
        config = TrainingConfig.load(config_file)

    # Load checkpoint
    _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, config, use_cuda=False)
    model, _ = checkpoint.model, checkpoint.optimizer
    _LOGGER.info(
        "Loaded checkpoint from %s (global step=%s)",
        args.checkpoint,
        checkpoint.global_step,
    )

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Do not calcuate jacobians for fast decoding
    model.decoder.store_inverse()
    model.eval()

    jitted_model = torch.jit.script(model)
    torch.jit.save(jitted_model, str(args.output))

    _LOGGER.info("Saved TorchScript model to %s", args.output)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
