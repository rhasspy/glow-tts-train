"""Classes and methods for loading phonemes and mel spectrograms"""
import csv
import json
import random
import typing

import torch
import torch.utils.data


class PhonemeMelLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        id_phonemes: typing.Dict[str, torch.IntTensor],
        id_mels: typing.Dict[str, torch.FloatTensor],
    ):
        self.id_phonemes = id_phonemes
        self.id_mels = id_mels

        self.ids = list(set.intersection(set(id_phonemes.keys()), set(id_mels.keys())))
        assert self.ids, "No shared utterance ids between phonemes and mels"
        random.shuffle(self.ids)

    def __getitem__(self, index):
        utt_id = self.ids[index]

        # phonemes, mels
        return (self.id_phonemes[utt_id], self.id_mels[utt_id])

    def __len__(self):
        return len(self.ids)


class PhonemeMelCollate:
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

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
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, output_lengths


# -----------------------------------------------------------------------------


def load_phonemes(csv_file: typing.TextIO) -> typing.Dict[str, torch.IntTensor]:
    phonemes = {}
    reader = csv.reader(csv_file, delimiter="|")
    for row in reader:
        utt_id, phoneme_str = row[0], row[1]
        phoneme_ids = [int(p) for p in phoneme_str.strip().split()]
        phonemes[utt_id] = torch.IntTensor(phoneme_ids)

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
