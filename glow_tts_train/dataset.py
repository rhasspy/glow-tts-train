"""Classes and methods for loading phonemes and mel spectrograms"""
import csv
import logging
import typing
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from glow_tts_train.config import MetadataFormat, TrainingConfig
from glow_tts_train.utils import StandardScaler

_LOGGER = logging.getLogger("glow_tts_train.dataset")

# -----------------------------------------------------------------------------


@dataclass
class Utterance:
    id: str
    phoneme_ids: typing.Sequence[int]
    spec_path: Path
    speaker_id: typing.Optional[int] = None


@dataclass
class UtteranceTensors:
    id: str
    phoneme_ids: torch.LongTensor
    spectrogram: torch.FloatTensor
    speaker_id: typing.Optional[torch.LongTensor] = None


@dataclass
class Batch:
    utterance_ids: typing.Sequence[str]
    phoneme_ids: torch.LongTensor
    phoneme_lengths: torch.LongTensor
    spectrograms: torch.FloatTensor
    spectrogram_lengths: torch.LongTensor
    speaker_ids: typing.Optional[torch.LongTensor] = None


UTTERANCE_PHONEME_IDS = typing.Dict[str, typing.List[int]]
UTTERANCE_SPEAKER_IDS = typing.Dict[str, int]
UTTERANCE_IDS = typing.Collection[str]


@dataclass
class DatasetInfo:
    name: str
    cache_dir: Path
    utt_phoneme_ids: UTTERANCE_PHONEME_IDS
    utt_speaker_ids: UTTERANCE_SPEAKER_IDS
    split_ids: typing.Mapping[str, UTTERANCE_IDS]


# -----------------------------------------------------------------------------


class PhonemeIdsAndMelsDataset(Dataset):
    def __init__(
        self,
        config: TrainingConfig,
        datasets: typing.Sequence[DatasetInfo],
        split: str,
        mel_scaler: typing.Optional[StandardScaler] = None,
    ):
        super().__init__()

        self.config = config
        self.utterances = []
        self.split = split
        self.mel_scaler = mel_scaler

        for dataset in datasets:
            for utt_id in dataset.split_ids.get(split, []):
                spec_path = dataset.cache_dir / f"{utt_id}.spec.pt"
                if spec_path.is_file():
                    self.utterances.append(
                        Utterance(
                            id=utt_id,
                            phoneme_ids=dataset.utt_phoneme_ids[utt_id],
                            spec_path=spec_path,
                            speaker_id=dataset.utt_speaker_ids.get(utt_id),
                        )
                    )
                else:
                    _LOGGER.warning("Missing spec file: %s", spec_path)

    def __getitem__(self, index):
        utterance = self.utterances[index]

        spectrogram_path = utterance.spec_path
        spectrogram = torch.load(str(spectrogram_path))

        if self.mel_scaler is not None:
            self.mel_scaler.transform(spectrogram)

        speaker_id = None
        if utterance.speaker_id is not None:
            speaker_id = torch.LongTensor([utterance.speaker_id])

        return UtteranceTensors(
            id=utterance.id,
            phoneme_ids=torch.LongTensor(utterance.phoneme_ids),
            spectrogram=spectrogram,
            speaker_id=speaker_id,
        )

    def __len__(self):
        return len(self.utterances)


class UtteranceCollate:
    def __call__(self, utterances: typing.Sequence[UtteranceTensors]) -> Batch:
        num_utterances = len(utterances)
        assert num_utterances > 0, "No utterances"

        max_phonemes_length = 0
        max_spec_length = 0

        num_mels = 0
        multispeaker = False

        # Determine lengths
        for utt_idx, utt in enumerate(utterances):
            assert utt.spectrogram is not None

            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)

            max_phonemes_length = max(max_phonemes_length, phoneme_length)
            max_spec_length = max(max_spec_length, spec_length)

            num_mels = utt.spectrogram.size(0)
            if utt.speaker_id is not None:
                multispeaker = True

        # Create padded tensors
        phonemes_padded = torch.LongTensor(num_utterances, max_phonemes_length)
        # phonemes_padded = torch.FloatTensor(num_utterances, max_phonemes_length)
        spec_padded = torch.FloatTensor(num_utterances, num_mels, max_spec_length)

        phonemes_padded.zero_()
        spec_padded.zero_()

        phoneme_lengths = torch.LongTensor(num_utterances)
        spec_lengths = torch.LongTensor(num_utterances)

        speaker_ids: typing.Optional[torch.LongTensor] = None
        if multispeaker:
            speaker_ids = torch.LongTensor(num_utterances)

        # Sort by decreasing spectrogram length
        sorted_utterances = sorted(
            utterances, key=lambda u: u.spectrogram.size(1), reverse=True
        )
        for utt_idx, utt in enumerate(sorted_utterances):
            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)

            phonemes_padded[utt_idx, :phoneme_length] = utt.phoneme_ids
            phoneme_lengths[utt_idx] = phoneme_length

            spec_padded[utt_idx, :, :spec_length] = utt.spectrogram
            spec_lengths[utt_idx] = spec_length

            if utt.speaker_id is not None:
                assert speaker_ids is not None
                speaker_ids[utt_idx] = utt.speaker_id

        return Batch(
            utterance_ids=[utt.id for utt in sorted_utterances],
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            speaker_ids=speaker_ids,
        )


# -----------------------------------------------------------------------------


def load_dataset(
    config: TrainingConfig,
    dataset_name: str,
    metadata_dir: typing.Union[str, Path],
    cache_dir: typing.Union[str, Path],
    splits=("train", "val"),
    speaker_id_map: typing.Optional[typing.Dict[str, int]] = None,
) -> DatasetInfo:
    metadata_dir = Path(metadata_dir)
    cache_dir = Path(cache_dir)

    multispeaker = config.model.n_speakers > 1
    if multispeaker:
        assert speaker_id_map, "Speaker id map required for multispeaker models"

    ids_format = MetadataFormat.PHONEME_IDS.value

    # Determine data paths
    data_paths: typing.Dict[str, typing.Dict[str, typing.Any]] = defaultdict(dict)

    for split in splits:
        csv_path = metadata_dir / f"{split}_{ids_format}.csv"
        assert csv_path.is_file(), f"Missing {csv_path}"

        data_paths[split]["csv_path"] = csv_path
        data_paths[split]["utt_ids"] = []

    # Load utterances
    utt_phoneme_ids = {}
    utt_speaker_ids = {}

    for split in splits:
        csv_path = data_paths[split]["csv_path"]
        if not csv_path.is_file():
            _LOGGER.debug("Skipping data for %s", split)
            continue

        utt_ids = data_paths[split]["utt_ids"]

        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_idx, row in enumerate(reader):
                assert len(row) > 1, f"{row} in {csv_path}:{row_idx+1}"
                utt_id, phonemes_or_ids = row[0], row[-1]

                if multispeaker:
                    assert speaker_id_map is not None

                    if len(row) > 2:
                        utt_speaker_ids[utt_id] = speaker_id_map[row[1]]
                    else:
                        utt_speaker_ids[utt_id] = speaker_id_map[dataset_name]

                phoneme_ids = [int(p_id) for p_id in phonemes_or_ids.split()]
                phoneme_ids = [
                    p_id for p_id in phoneme_ids if 0 <= p_id < config.model.num_symbols
                ]

                if phoneme_ids:
                    utt_phoneme_ids[utt_id] = phoneme_ids
                    utt_ids.append(utt_id)
                else:
                    _LOGGER.warning("No phoneme ids for %s (%s)", utt_id, csv_path)

        _LOGGER.debug(
            "Loaded %s utterance(s) for %s from %s", len(utt_ids), split, csv_path
        )

    # Filter utterances based on min/max settings in config
    _LOGGER.debug("Filtering data")
    drop_utt_ids: typing.Set[str] = set()

    num_phonemes_too_small = 0
    num_phonemes_too_large = 0
    num_spec_missing = 0

    for utt_id, phoneme_ids in utt_phoneme_ids.items():
        # Check phonemes length
        if (config.min_seq_length is not None) and (
            len(phoneme_ids) < config.min_seq_length
        ):
            drop_utt_ids.add(utt_id)
            num_phonemes_too_small += 1
            continue

        if (config.max_seq_length is not None) and (
            len(phoneme_ids) > config.max_seq_length
        ):
            drop_utt_ids.add(utt_id)
            num_phonemes_too_large += 1
            continue

        # Check if spec file is missing
        spec_path = cache_dir / f"{utt_id}.spec.pt"

        if not spec_path.is_file():
            drop_utt_ids.add(utt_id)
            _LOGGER.warning(
                "Dropped %s because spec file is missing: %s", utt_id, spec_path
            )
            num_spec_missing += 1
            continue

    # Filter out dropped utterances
    if drop_utt_ids:
        _LOGGER.info("Dropped %s utterance(s)", len(drop_utt_ids))

        if num_phonemes_too_small > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose phoneme length was smaller than %s",
                num_phonemes_too_small,
                config.min_seq_length,
            )

        if num_phonemes_too_large > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose phoneme length was larger than %s",
                num_phonemes_too_large,
                config.max_seq_length,
            )

        if num_spec_missing > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose spec file was missing",
                num_spec_missing,
            )

        utt_phoneme_ids = {
            utt_id: phoneme_ids
            for utt_id, phoneme_ids in utt_phoneme_ids.items()
            if utt_id not in drop_utt_ids
        }
    else:
        _LOGGER.info("Kept all %s utterances", len(utt_phoneme_ids))

    if not utt_phoneme_ids:
        _LOGGER.warning("No utterances after filtering")

    return DatasetInfo(
        name=dataset_name,
        cache_dir=cache_dir,
        utt_phoneme_ids=utt_phoneme_ids,
        utt_speaker_ids=utt_speaker_ids,
        split_ids={
            split: set(data_paths[split]["utt_ids"]) - drop_utt_ids for split in splits
        },
    )
