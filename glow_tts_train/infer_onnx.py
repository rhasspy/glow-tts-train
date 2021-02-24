#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import jsonlines
import numpy as np
import onnxruntime

from .config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.infer_onnx")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow_tts-train.infer_onnx")
    parser.add_argument("model", help="Path to onnx model")
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Input format is id|p1 p2 p3..."
    )
    parser.add_argument(
        "--no-optimizations", action="store_true", help="Disable Onnx optimizations"
    )
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Convert to paths
    if args.config:
        args.config = [Path(p) for p in args.config]

    args.model = Path(args.model)

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    # Load model
    sess_options = onnxruntime.SessionOptions()
    if args.no_optimizations:
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

    _LOGGER.debug("Loading model from %s", args.model)
    model = onnxruntime.InferenceSession(str(args.model), sess_options=sess_options)
    _LOGGER.info("Loaded model from %s", args.model)

    # Process input phonemes
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

    start_time = time.perf_counter()

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
            text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
            text_lengths = np.array([text.shape[1]], dtype=np.int64)
            scales = np.array([args.noise_scale, args.length_scale], dtype=np.float32)

            # Infer mel spectrograms
            start_time = time.perf_counter()
            mel = model.run(
                None, {"input": text, "input_lengths": text_lengths, "scales": scales}
            )[0]
            end_time = time.perf_counter()

            # Write mel spectrogram and settings as a JSON object on one line
            mel_list = mel.squeeze(0).tolist()
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
