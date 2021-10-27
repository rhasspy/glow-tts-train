"""Classes and methods for loading phonemes and mel spectrograms"""
import csv
import json
import logging
import random
import shutil
import tempfile
import typing
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from torch.utils.data import DataLoader, Dataset

from glow_tts_train.config import TrainingConfig

_LOGGER = logging.getLogger("glow_tts_train.dataset")

# -----------------------------------------------------------------------------


@dataclass
class Utterance:
    id: str
    phoneme_ids: typing.Sequence[int]
    audio_path: Path
    speaker_id: typing.Optional[int] = None


@dataclass
class UtteranceTensors:
    id: str
    phoneme_ids: torch.LongTensor
    spectrogram: torch.FloatTensor
    speaker_id: typing.Optional[torch.LongTensor] = None


@dataclass
class Batch:
    phoneme_ids: torch.LongTensor
    phoneme_lengths: torch.LongTensor
    spectrograms: torch.FloatTensor
    spectrogram_lengths: torch.LongTensor
    speaker_ids: typing.Optional[torch.LongTensor] = None


# -----------------------------------------------------------------------------


class PhonemeIdsAndMelsDataset(Dataset):
    def __init__(
        self,
        config: TrainingConfig,
        utt_phoneme_ids: typing.Mapping[str, typing.Sequence[int]],
        audio_dir: typing.Union[str, Path],
        utt_speaker_ids: typing.Optional[typing.Mapping[str, int]] = None,
        cache_dir: typing.Optional[typing.Union[str, Path]] = None,
    ):
        super().__init__()

        self.config = config
        self.audio_dir = Path(audio_dir)
        self.utterances: typing.List[Utterance] = []

        self.temp_dir: typing.Optional[tempfile.TemporaryDirectory] = None

        if cache_dir is None:
            # pylint: disable=consider-using-with
            self.temp_dir = tempfile.TemporaryDirectory(prefix="vits_train")
            self.cache_dir = Path(self.temp_dir.name)
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if utt_speaker_ids is None:
            utt_speaker_ids = {}

        for utt_id, phoneme_ids in utt_phoneme_ids.items():
            audio_path = self.audio_dir / utt_id
            if not audio_path.is_file():
                # Try WAV extension
                audio_path = self.audio_dir / f"{utt_id}.wav"

            if audio_path.is_file():
                self.utterances.append(
                    Utterance(
                        id=utt_id,
                        phoneme_ids=phoneme_ids,
                        audio_path=audio_path,
                        speaker_id=utt_speaker_ids.get(utt_id),
                    )
                )
            else:
                _LOGGER.warning("Missing audio file: %s", audio_path)

    def __getitem__(self, index):
        utterance = self.utterances[index]

        spectrogram_path = (
            self.cache_dir / utterance.audio_path.with_suffix(".spec.pt").name
        )

        if spectrogram_path.is_file() and (spectrogram_path.stat().st_size > 0):
            spectrogram = torch.load(str(spectrogram_path))
        else:
            # Load audio and resample
            audio, _sample_rate = librosa.load(
                str(utterance.audio_path), sr=self.config.audio.sample_rate
            )

            spectrogram = torch.FloatTensor(self.config.audio.wav2mel(audio))

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
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            speaker_ids=speaker_ids,
        )


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
