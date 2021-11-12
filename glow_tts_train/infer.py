#!/usr/bin/env python3
import argparse
import csv
import dataclasses
import io
import logging
import os
import sys
import time
from pathlib import Path

import jsonlines
import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.infer")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-train.infer")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--numpy-dir", help="Output numpy files to a directory instead of JSONL"
    )
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--num-symbols", type=int, help="Number of symbols in the model"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Input format is id|p1 p2 p3..."
    )
    parser.add_argument("--noise-scale", type=float, default=0.333)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")
    parser.add_argument("--jit", action="store_true", help="Load TorchScript model")
    parser.add_argument(
        "--speaker", type=int, help="Speaker id number (multispeaker model only)"
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

    # Convert to paths
    if args.config:
        args.config = [Path(p) for p in args.config]

    args.checkpoint = Path(args.checkpoint)

    if args.numpy_dir:
        args.numpy_dir = Path(args.numpy_dir)
        args.numpy_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    if args.num_symbols is not None:
        config.model.num_symbols = args.num_symbols

    _LOGGER.debug(config)

    assert (
        config.model.num_symbols > 0
    ), "Number of symbols not set (did you forget --config or --num-symbols?)"

    # Default mel settings
    output_obj = {"id": "", "audio": dataclasses.asdict(config.audio), "mel": []}

    # Load checkpoint
    start_time = time.perf_counter()

    if args.jit:
        # TorchScript model
        _LOGGER.debug("Loading TorchScript from %s", args.checkpoint)
        model = torch.jit.load(str(args.checkpoint))
        end_time = time.perf_counter()

        _LOGGER.info(
            "Loaded TorchScript model from %s in %s second(s)",
            args.checkpoint,
            end_time - start_time,
        )

        model.eval()
    else:
        # Checkpoint
        _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
        checkpoint = load_checkpoint(args.checkpoint, config, use_cuda=args.cuda)
        end_time = time.perf_counter()

        model, _ = checkpoint.model, checkpoint.optimizer
        _LOGGER.info(
            "Loaded checkpoint from %s in %s second(s) (global step=%s)",
            args.checkpoint,
            end_time - start_time,
            checkpoint.global_step,
        )

        # Do not calcuate jacobians for fast decoding
        model.decoder.store_inverse()
        model.eval()

    # Multispeaker
    speaker_id = None
    if args.speaker or config.model.n_speakers > 1:
        if args.speaker is None:
            args.speaker = 0

        speaker_id = torch.LongTensor([args.speaker])

    if args.cuda and (speaker_id is not None):
        speaker_id = speaker_id.cuda()

    # -------------------------------------------------------------------------

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
                with io.StringIO(line) as csv_line:
                    row = next(csv.reader(csv_line, delimiter="|"))
                    utt_id, line = row[0], row[-1]

            # Phoneme ids as p1 p2 p3...
            phoneme_ids = [float(p) for p in line.split()]
            _LOGGER.debug("%s (id=%s)", phoneme_ids, utt_id)

            # Convert to tensors
            text = torch.autograd.Variable(torch.FloatTensor(phoneme_ids).unsqueeze(0))
            text_lengths = torch.LongTensor([text.shape[1]])

            if args.cuda:
                text.contiguous().cuda()
                text_lengths.contiguous().cuda()

            # Infer mel spectrograms
            with torch.no_grad():
                start_time = time.perf_counter()
                (mel, *_), _, _ = model(
                    text,
                    text_lengths,
                    noise_scale=args.noise_scale,
                    length_scale=args.length_scale,
                    gen=True,
                    g=speaker_id,
                )
                end_time = time.perf_counter()

                # Write mel spectrogram and settings as a JSON object on one line
                mel = mel.squeeze(0).cpu().float().numpy()

                if args.numpy_dir:
                    if not utt_id:
                        # Use timestamp for file name
                        utt_id = str(time.time())

                    # Save as numpy file
                    mel_path = args.numpy_dir / (utt_id + ".npy")
                    np.save(str(mel_path), mel, allow_pickle=True)

                    _LOGGER.debug("Wrote %s", mel_path)
                else:
                    # Write line of JSON
                    mel_list = mel.tolist()
                    output_obj["id"] = utt_id
                    output_obj["mel"] = mel_list

                    writer.write(output_obj)

                _LOGGER.debug(
                    "Generated mel in %s second(s) (%s, shape=%s)",
                    end_time - start_time,
                    utt_id,
                    list(mel.shape),
                )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
