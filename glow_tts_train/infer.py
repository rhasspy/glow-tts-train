#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

import jsonlines
import torch

from .checkpoint import load_checkpoint
from .config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.infer")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-train.infer")
    parser.add_argument("config", help="Path to JSON configuration file")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--csv", action="store_true", help="Input format is id|p1 p2 p3..."
    )
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")
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

    # Load config
    with open(args.config, "r") as config_file:
        config = TrainingConfig.load(config_file)

    output_obj = {
        "id": "",
        "audio": {
            "filter_length": config.audio.filter_length,
            "hop_length": config.audio.hop_length,
            "win_length": config.audio.win_length,
            "mel_channels": config.audio.n_mel_channels,
            "sample_rate": config.audio.sampling_rate,
            "sample_bytes": config.audio.sample_bytes,
            "channels": config.audio.channels,
            "mel_fmin": config.audio.mel_fmin,
            "mel_fmax": config.audio.mel_fmax,
            "normalized": config.audio.normalized,
        },
        "mel": [],
    }

    # Load checkpoint
    _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, config, use_cuda=args.cuda)
    model, _ = checkpoint.model, checkpoint.optimizer
    _LOGGER.info(
        "Loaded checkpoint from %s (global step=%s)",
        args.checkpoint,
        checkpoint.global_step,
    )

    # Do not calcuate jacobians for fast decoding
    model.decoder.store_inverse()
    model.eval()

    if os.isatty(sys.stdin.fileno()):
        print("Reading whitespace-separated phoneme ids from stdin...", file=sys.stderr)

    # Read phoneme ids from standard input.
    # Phoneme ids are separated by whitespace (<p1> <p2> ...)
    writer = jsonlines.Writer(sys.stdout, flush=True)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            utt_id = ""
            if args.csv:
                # Input format is id | p1 p2 p3...
                utt_id, line = line.split("|", maxsplit=1)

            # Phoneme ids as p1 p2 p3...
            phoneme_ids = [int(p) for p in line.split()]
            _LOGGER.debug("%s (id=%s)", phoneme_ids, utt_id)

            # Convert to tensors
            # TODO: Allow batches
            text = torch.autograd.Variable(torch.LongTensor(phoneme_ids).unsqueeze(0))
            text_lengths = torch.LongTensor([text.shape[1]])

            if args.cuda:
                text.cuda()
                text_lengths.cuda()

            # Infer mel spectrograms
            with torch.no_grad():
                (mel, *_), *_, (_attn_gen, *_) = model(
                    text,
                    text_lengths,
                    gen=True,
                    noise_scale=args.noise_scale,
                    length_scale=args.length_scale,
                )

                # Write mel spectrogram and settings as a JSON object on one line
                mel_list = mel.squeeze(0).cpu().float().numpy().tolist()
                output_obj["id"] = utt_id
                output_obj["mel"] = mel_list

                writer.write(output_obj)
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
