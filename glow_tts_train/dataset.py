"""Classes and methods for loading phonemes and mel spectrograms"""
import csv
import json
import logging
import random
import typing
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from .config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.dataset")

# -----------------------------------------------------------------------------


class PhonemeMelLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        id_phonemes: typing.Dict[typing.Tuple[int, str], torch.IntTensor],
        id_mels: typing.Dict[typing.Tuple[int, str], torch.FloatTensor],
        mel_dirs: typing.Optional[typing.Dict[int, Path]] = None,
        multispeaker: bool = False,
    ):
        self.id_phonemes = id_phonemes
        self.id_mels = id_mels
        self.mel_dirs = mel_dirs
        self.multispeaker = multispeaker

        if self.id_mels:
            self.ids = list(
                set.intersection(set(id_phonemes.keys()), set(id_mels.keys()))
            )
            assert self.ids, "No shared utterance ids between phonemes and mels"
        else:
            # Assume all ids will be present in mels_dir
            self.ids = list(id_phonemes.keys())

        random.shuffle(self.ids)

    def __getitem__(self, index):
        utt_key = self.ids[index]
        speaker_idx, utt_id = utt_key
        text = self.id_phonemes[utt_key]
        mel = self.id_mels.get(utt_key)

        if mel is None:
            mels_dir = self.mel_dirs.get(speaker_idx)
            assert mels_dir, f"Missing mel for id {utt_id}, but no mels_dir"
            mel_path = mels_dir / (utt_id + ".npy")

            # TODO: Verify shape
            mel = torch.from_numpy(np.load(mel_path, allow_pickle=True))

            # Cache mel
            self.id_mels[utt_key] = mel

        if self.multispeaker:
            # phonemes, mels, length, speaker
            return (text, mel, len(text), speaker_idx)

        # phonemes, mels, length
        return (text, mel, len(text))

    def __len__(self):
        return len(self.ids)


class PhonemeMelCollate:
    def __init__(self, n_frames_per_step: int = 1, multispeaker: bool = False):
        self.n_frames_per_step = n_frames_per_step
        self.multispeaker = multispeaker

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        speaker_ids = None
        if self.multispeaker:
            speaker_ids = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

            if speaker_ids is not None:
                speaker_ids[i] = batch[ids_sorted_decreasing[i]][3]

        return text_padded, input_lengths, mel_padded, output_lengths, speaker_ids


# -----------------------------------------------------------------------------


def load_phonemes(
    csv_file: typing.TextIO, config: TrainingConfig
) -> typing.Dict[str, torch.IntTensor]:
    phonemes = {}
    num_too_small = 0
    num_too_large = 0

    reader = csv.reader(csv_file, delimiter="|")
    for row in reader:
        utt_id, phoneme_str = row[0], row[1]
        phoneme_ids = [int(p) for p in phoneme_str.strip().split()]
        num_phonemes = len(phoneme_ids)

        if (config.min_seq_length is not None) and (
            num_phonemes < config.min_seq_length
        ):
            _LOGGER.debug(
                "Dropping %s (%s < %s)", utt_id, num_phonemes, config.min_seq_length
            )
            num_too_small += 1
            continue

        if (config.max_seq_length is not None) and (
            num_phonemes > config.max_seq_length
        ):
            _LOGGER.debug(
                "Dropping %s (%s > %s)", utt_id, num_phonemes, config.max_seq_length
            )
            num_too_large += 1
            continue

        phonemes[utt_id] = torch.IntTensor(phoneme_ids)

    if (num_too_small > 0) or (num_too_large > 0):
        _LOGGER.warning(
            "Dropped some utterance (%s too small, %s too large)",
            num_too_small,
            num_too_large,
        )

    return phonemes


def load_mels(jsonl_file: typing.TextIO) -> typing.Dict[str, torch.FloatTensor]:
    mels = {}
    for line in jsonl_file:
        line = line.strip()
        if not line:
            continue

        mel_obj = json.loads(line)
        utt_id = mel_obj["id"]
        mels[utt_id] = torch.FloatTensor(mel_obj["mel"])

    return mels
