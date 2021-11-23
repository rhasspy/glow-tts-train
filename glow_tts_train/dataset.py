"""Classes and methods for loading phonemes and mel spectrograms"""
import csv
import logging
import re
import shutil
import tempfile
import typing
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import gruut_ipa
import librosa
import phonemes2ids
import torch
from torch.utils.data import Dataset

from glow_tts_train.config import Phonemizer, TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.dataset")

# -----------------------------------------------------------------------------


@dataclass
class Utterance:
    id: str
    phoneme_ids: typing.Sequence[int]
    audio_path: Path
    cache_path: typing.Optional[Path]
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


UTTERANCE_PHONEME_IDS = typing.Dict[str, typing.Sequence[int]]
UTTERANCE_SPEAKER_IDS = typing.Dict[str, str]
UTTERANCE_IDS = typing.Collection[str]


@dataclass
class DatasetInfo:
    name: str
    audio_dir: Path
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
        cache_dir: typing.Optional[typing.Union[str, Path]] = None,
    ):
        super().__init__()

        self.config = config
        self.utterances = []
        self.split = split

        self.temp_dir: typing.Optional[tempfile.TemporaryDirectory] = None

        if cache_dir is None:
            # pylint: disable=consider-using-with
            self.temp_dir = tempfile.TemporaryDirectory(prefix="glow_tts_train")
            self.cache_dir = Path(self.temp_dir.name)
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        for dataset in datasets:
            for utt_id in dataset.split_ids.get(split, []):
                audio_path = dataset.audio_dir / utt_id

                if not audio_path.is_file():
                    # Try WAV extension
                    audio_path = dataset.audio_dir / f"{utt_id}.wav"

                if audio_path.is_file():
                    cache_path = self.cache_dir / dataset.name / f"{utt_id}.spec.pt"
                    self.utterances.append(
                        Utterance(
                            id=utt_id,
                            phoneme_ids=dataset.utt_phoneme_ids[utt_id],
                            audio_path=audio_path,
                            cache_path=cache_path,
                            speaker_id=dataset.utt_speaker_ids.get(utt_id),
                        )
                    )
                else:
                    _LOGGER.warning("Missing audio file: %s", audio_path)

    def __getitem__(self, index):
        utterance = self.utterances[index]

        spectrogram_path = utterance.cache_path

        if (
            (spectrogram_path is not None)
            and spectrogram_path.is_file()
            and (spectrogram_path.stat().st_size > 0)
        ):
            spectrogram = torch.load(str(spectrogram_path))
        else:
            # Load audio and resample
            audio, _sample_rate = librosa.load(
                str(utterance.audio_path), sr=self.config.audio.sample_rate
            )

            spectrogram = torch.FloatTensor(self.config.audio.wav2mel(audio))

            if spectrogram_path is not None:
                # Save to cache.
                spectrogram_path.parent.mkdir(parents=True, exist_ok=True)

                # Use temporary file to avoid multiple processes writing at the same time.
                with tempfile.NamedTemporaryFile(mode="wb") as spec_file:
                    torch.save(spectrogram, spec_file.name)
                    shutil.copy(spec_file.name, spectrogram_path)

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


def make_dataset_phonemes(
    config: TrainingConfig,
    text_csv_path: typing.Union[str, Path],
    phonemes_csv_path: typing.Union[str, Path],
):
    assert config.text_language, "text_language is required in config"
    assert config.phonemizer, "phonemizer is required in config"

    if config.phonemizer == Phonemizer.SYMBOLS:

        def text_to_phonemes(text):
            for word_str in text.split(config.phonemes.word_separator):
                yield list(gruut_ipa.IPA.graphemes(word_str))

    else:
        raise ValueError(f"Unknown phonemizer: {config.phonemizer}")

    phonemes_csv_path = Path(phonemes_csv_path)
    phonemes_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(text_csv_path, "r", encoding="utf-8") as text_file, open(
        phonemes_csv_path, "w", encoding="utf-8"
    ) as phonemes_file:
        reader = csv.reader(text_file, delimiter="|")
        writer = csv.writer(phonemes_file, delimiter="|")

        for row in reader:
            text = row[-1]
            word_phonemes = text_to_phonemes(text)
            phonemes_str = config.phonemes.word_separator.join(
                config.phonemes.phoneme_separator.join(p for p in wp)
                for wp in word_phonemes
            )
            writer.writerow((*row, phonemes_str))


def learn_dataset_ids(
    config: TrainingConfig,
    phonemes_csv_paths: typing.Iterable[typing.Union[str, Path]],
    phoneme_map_path: typing.Union[str, Path],
):
    phoneme_map_path = Path(phoneme_map_path)
    phoneme_map_path.parent.mkdir(parents=True, exist_ok=True)

    all_phonemes = set()
    for phonemes_csv_path in phonemes_csv_paths:
        with open(phonemes_csv_path, "r", encoding="utf-8") as phonemes_file:
            reader = csv.reader(phonemes_file, delimiter="|")
            for row in reader:
                phonemes_str = row[-1]
                word_phonemes = [
                    word_str.split(config.phonemes.phoneme_separator)
                    for word_str in phonemes_str.split(config.phonemes.word_separator)
                ]

                phonemes2ids.learn_phoneme_ids(
                    word_phonemes=word_phonemes,
                    all_phonemes=all_phonemes,
                    simple_punctuation=config.phonemes.simple_punctuation,
                    punctuation_map=config.phonemes.punctuation_map,
                    separate=config.phonemes.separate,
                    separate_graphemes=config.phonemes.separate_graphemes,
                    separate_tones=config.phonemes.separate_tones,
                    phoneme_map=config.phonemes.phoneme_map,
                )

    phoneme_to_id = {}

    if config.phonemes.pad and (config.phonemes.pad not in phoneme_to_id):
        # Add pad symbol
        phoneme_to_id[config.phonemes.pad] = len(phoneme_to_id)

    if config.phonemes.bos and (config.phonemes.bos not in phoneme_to_id):
        # Add BOS symbol
        phoneme_to_id[config.phonemes.bos] = len(phoneme_to_id)

    if config.phonemes.eos and (config.phonemes.eos not in phoneme_to_id):
        # Add EOS symbol
        phoneme_to_id[config.phonemes.eos] = len(phoneme_to_id)

    if config.phonemes.blank and (config.phonemes.blank not in phoneme_to_id):
        # Add blank symbol
        phoneme_to_id[config.phonemes.blank] = len(phoneme_to_id)

    if config.phonemes.blank_word and (config.phonemes.blank_word not in phoneme_to_id):
        # Add blank symbol
        phoneme_to_id[config.phonemes.blank_word] = len(phoneme_to_id)

    for phoneme in sorted(all_phonemes):
        if phoneme not in phoneme_to_id:
            phoneme_to_id[phoneme] = len(phoneme_to_id)

    # Write id<space>phoneme
    with open(phoneme_map_path, "w", encoding="utf-8") as map_file:
        for phoneme, phoneme_id in phoneme_to_id.items():
            print(phoneme_id, phoneme, file=map_file)


def make_dataset_ids(
    config: TrainingConfig,
    phonemes_csv_path: typing.Union[str, Path],
    ids_csv_path: typing.Union[str, Path],
):
    assert config.phonemes.phoneme_to_id, "Phoneme/id map is required"

    with open(phonemes_csv_path, "r", encoding="utf-8") as phonemes_file, open(
        ids_csv_path, "w", encoding="utf-8"
    ) as ids_file:
        reader = csv.reader(phonemes_file, delimiter="|")
        writer = csv.writer(ids_file, delimiter="|")

        for row in reader:
            phonemes_str = row[-1]
            word_phonemes = [
                word_str.split(config.phonemes.phoneme_separator)
                for word_str in phonemes_str.split(config.phonemes.word_separator)
            ]

            phoneme_ids = phonemes2ids.phonemes2ids(
                word_phonemes=word_phonemes,
                phoneme_to_id=config.phonemes.phoneme_to_id,
                pad=config.phonemes.pad,
                bos=config.phonemes.bos,
                eos=config.phonemes.eos,
                blank=config.phonemes.blank,
                blank_word=config.phonemes.blank_word,
                blank_between=config.phonemes.blank_between,
                blank_at_start=config.phonemes.blank_at_start,
                blank_at_end=config.phonemes.blank_at_end,
                tone_before=config.phonemes.tone_before,
                simple_punctuation=config.phonemes.simple_punctuation,
                punctuation_map=config.phonemes.punctuation_map,
                separate=config.phonemes.separate,
                separate_graphemes=config.phonemes.separate_graphemes,
                separate_tones=config.phonemes.separate_tones,
                phoneme_map=config.phonemes.phoneme_map,
                auto_bos_eos=config.phonemes.auto_bos_eos,
            )

            ids_str = " ".join(str(p_id) for p_id in phoneme_ids)

            writer.writerow((*row, ids_str))


def load_dataset(
    config: TrainingConfig,
    dataset_name: str,
    metadata_dir: typing.Union[str, Path],
    audio_dir: typing.Union[str, Path],
    splits=("train", "val"),
) -> DatasetInfo:
    metadata_dir = Path(metadata_dir)
    audio_dir = Path(audio_dir)
    multispeaker = config.model.n_speakers > 1

    # Determine data paths
    data_paths = defaultdict(dict)
    for split in splits:
        csv_path = metadata_dir / f"{split}_ids.csv"
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
                    if len(row) > 2:
                        utt_speaker_ids[utt_id] = row[1]
                    else:
                        utt_speaker_ids[utt_id] = dataset_name

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
    num_audio_missing = 0

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

        # Check if audio file is missing
        audio_path = audio_dir / utt_id
        if not audio_path.is_file():
            # Try WAV extension
            audio_path = audio_dir / f"{utt_id}.wav"

        if not audio_path.is_file():
            drop_utt_ids.add(utt_id)
            _LOGGER.warning(
                "Dropped %s because audio file is missing: %s", utt_id, audio_path
            )
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

        if num_audio_missing > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose audio file was missing",
                num_audio_missing,
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
        audio_dir=audio_dir,
        utt_phoneme_ids=utt_phoneme_ids,
        utt_speaker_ids=utt_speaker_ids,
        split_ids={
            split: set(data_paths[split]["utt_ids"]) - drop_utt_ids for split in splits
        },
    )
