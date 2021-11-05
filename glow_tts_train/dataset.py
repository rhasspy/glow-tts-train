"""Classes and methods for loading phonemes and mel spectrograms"""
import csv
import logging
import re
import shutil
import tempfile
import typing
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset

from glow_tts_train.config import TrainingConfig

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
    audio_dir: typing.Union[str, Path],
    splits=("train", "val"),
) -> DatasetInfo:
    metadata_dir = Path(metadata_dir)
    audio_dir = Path(audio_dir)
    multispeaker = config.model.n_speakers > 1

    # Determine data paths
    data_paths = defaultdict(dict)
    for split in splits:
        is_phonemes = False
        csv_path = metadata_dir / f"{split}_ids.csv"
        if not csv_path.is_file():
            csv_path = metadata_dir / f"{split}_phonemes.csv"
            is_phonemes = True

        data_paths[split]["is_phonemes"] = is_phonemes
        data_paths[split]["csv_path"] = csv_path
        data_paths[split]["utt_ids"] = []

    # train/val sets are required
    for split in splits:
        assert data_paths[split][
            "csv_path"
        ].is_file(), (
            f"Missing {split}_ids.csv or {split}_phonemes.csv in {metadata_dir}"
        )

    # Load phonemes
    phoneme_to_id = {}
    phonemes_path = metadata_dir / "phonemes.txt"

    _LOGGER.debug("Loading phonemes from %s", phonemes_path)
    with open(phonemes_path, "r", encoding="utf-8") as phonemes_file:
        for line in phonemes_file:
            line = line.strip("\r\n")
            if (not line) or line.startswith("#"):
                continue

            phoneme_id, phoneme = re.split(r"[ \t]", line, maxsplit=1)

            # Avoid overwriting duplicates
            if phoneme not in phoneme_to_id:
                phoneme_id = int(phoneme_id)
                phoneme_to_id[phoneme] = phoneme_id

    id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

    # Load utterances
    utt_phoneme_ids = {}
    utt_speaker_ids = {}

    for split in splits:
        csv_path = data_paths[split]["csv_path"]
        if not csv_path.is_file():
            _LOGGER.debug("Skipping data for %s", split)
            continue

        is_phonemes = data_paths[split]["is_phonemes"]
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

                if is_phonemes:
                    # TODO: Map phonemes with phonemes2ids
                    raise NotImplementedError(csv_path)
                    # phoneme_ids = [phoneme_to_id[p] for p in phonemes if p in phoneme_to_id]
                    # phoneme_ids = intersperse(phoneme_ids, 0)
                else:
                    phoneme_ids = [int(p_id) for p_id in phonemes_or_ids.split()]
                    phoneme_ids = [
                        p_id for p_id in phoneme_ids if p_id in id_to_phoneme
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
