#!/usr/bin/env python3
import csv
import logging
import re
import typing
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from glow_tts_train.config import TrainingConfig
from glow_tts_train.dataset import Batch, PhonemeIdsAndMelsDataset, UtteranceCollate
from glow_tts_train.models import setup_model
from glow_tts_train.utils import duration_loss, intersperse, mle_loss

_LOGGER = logging.getLogger("glow_tts_train.train")

# -----------------------------------------------------------------------------


class GlowTTSTraining(pl.LightningModule):
    def __init__(
        self,
        config: TrainingConfig,
        utt_phoneme_ids: typing.Mapping[str, typing.Sequence[int]],
        audio_dir: typing.Union[str, Path],
        train_ids: typing.Iterable[str],
        val_ids: typing.Iterable[str],
        test_ids: typing.Iterable[str],
        utt_speaker_ids: typing.Optional[typing.Mapping[str, int]] = None,
        cache_dir: typing.Optional[typing.Union[str, Path]] = None,
    ):
        super().__init__()

        self.config = config
        self.cache_dir = cache_dir
        self.audio_dir = Path(audio_dir)
        self.utt_speaker_ids = utt_speaker_ids if utt_speaker_ids is not None else {}

        self.generator = setup_model(config)
        self.collate_fn = UtteranceCollate()

        # Filter utterances based on min/max settings in config
        drop_utt_ids: typing.Set[str] = set()

        num_phonemes_too_small = 0
        num_phonemes_too_large = 0
        num_audio_missing = 0

        for utt_id, phoneme_ids in utt_phoneme_ids.items():
            # Check phonemes length
            if (self.config.min_seq_length is not None) and (
                len(phoneme_ids) < self.config.min_seq_length
            ):
                drop_utt_ids.add(utt_id)
                num_phonemes_too_small += 1
                continue

            if (self.config.max_seq_length is not None) and (
                len(phoneme_ids) > self.config.max_seq_length
            ):
                drop_utt_ids.add(utt_id)
                num_phonemes_too_large += 1
                continue

            # Check if audio file is missing
            audio_path = self.audio_dir / utt_id
            if not audio_path.is_file():
                # Try WAV extension
                audio_path = self.audio_dir / f"{utt_id}.wav"

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
                    self.config.min_seq_length,
                )

            if num_phonemes_too_large > 0:
                _LOGGER.debug(
                    "%s utterance(s) dropped whose phoneme length was larger than %s",
                    num_phonemes_too_large,
                    self.config.max_seq_length,
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

        assert utt_phoneme_ids, "No utterances after filtering"

        train_ids = set(train_ids) - drop_utt_ids
        assert train_ids, "No training utterances after filtering"

        val_ids = set(val_ids) - drop_utt_ids
        assert val_ids, "No validation utterances after filtering"

        test_ids = set(test_ids) - drop_utt_ids
        # assert test_ids, "No testing utterances after filtering"

        self.train_dataset = PhonemeIdsAndMelsDataset(
            config=self.config,
            utt_phoneme_ids={utt_id: utt_phoneme_ids[utt_id] for utt_id in train_ids},
            audio_dir=self.audio_dir,
            utt_speaker_ids={
                utt_id: self.utt_speaker_ids[utt_id]
                for utt_id in train_ids
                if utt_id in self.utt_speaker_ids
            },
            cache_dir=cache_dir,
        )

        self.val_dataset = PhonemeIdsAndMelsDataset(
            config=self.config,
            utt_phoneme_ids={utt_id: utt_phoneme_ids[utt_id] for utt_id in val_ids},
            audio_dir=self.audio_dir,
            utt_speaker_ids={
                utt_id: self.utt_speaker_ids[utt_id]
                for utt_id in val_ids
                if utt_id in self.utt_speaker_ids
            },
            cache_dir=cache_dir,
        )

        self.test_dataset = PhonemeIdsAndMelsDataset(
            config=self.config,
            utt_phoneme_ids={utt_id: utt_phoneme_ids[utt_id] for utt_id in test_ids},
            audio_dir=self.audio_dir,
            utt_speaker_ids={
                utt_id: self.utt_speaker_ids[utt_id]
                for utt_id in test_ids
                if utt_id in self.utt_speaker_ids
            },
            cache_dir=cache_dir,
        )

    def forward(self, *args, **kwargs):
        return self.net_g(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.generator.parameters(),
            self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
        )

    def training_step(self, train_batch: Batch, batch_idx: int):
        x, x_lengths, y, y_lengths, g = (
            train_batch.phoneme_ids,
            train_batch.phoneme_lengths,
            train_batch.spectrograms,
            train_batch.spectrogram_lengths,
            train_batch.speaker_ids,
        )

        (
            (z, z_m, z_logs, logdet, z_mask),
            (_x_m, _x_logs, _x_mask),
            (_attn, logw, logw_),
        ) = self.generator(x, x_lengths, y, y_lengths, g=g)

        # Compute loss
        l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = duration_loss(logw, logw_, x_lengths)

        loss_g = l_mle + l_length

        return loss_g

    # def validation_step(self, val_batch: Batch, batch_idx):
    #     return self.training_step(val_batch, batch_idx, optimizer_idx=0)

    # def test_step(self, test_batch: Batch, batch_idx, optimizer_idx):
    #     y_hat, *_ = self.net_g.infer(
    #         test_batch.phoneme_ids,
    #         test_batch.phoneme_lengths,
    #         sid=test_batch.speaker_ids,
    #     )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         shuffle=False,
    #         batch_size=self.config.batch_size,
    #         # pin_memory=True,
    #         drop_last=False,
    #         collate_fn=self.collate_fn,
    #         num_workers=8,
    #     )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         shuffle=False,
    #         batch_size=self.config.batch_size,
    #         # pin_memory=True,
    #         drop_last=False,
    #         collate_fn=self.collate_fn,
    #         num_workers=8,
    #     )


# -----------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.DEBUG)

    model_dir = Path("local/ljspeech-lightning")
    audio_dir = Path("/home/hansenm/opt/vits-train/data/ljspeech/wavs")
    cache_dir = model_dir / "cache"

    train_path = model_dir / "train.csv"
    val_path = model_dir / "val.csv"
    test_path = model_dir / "test.csv"

    config_path = model_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = TrainingConfig.load(config_file)

    phoneme_to_id = {}
    phonemes_path = model_dir / "phonemes.txt"
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

    utt_phoneme_ids = {}
    train_ids = []
    val_ids = []
    test_ids = []

    for ids, csv_path in [
        (train_ids, train_path),
        (val_ids, val_path),
        # (test_ids, test_path),
    ]:
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_idx, row in enumerate(reader):
                # TODO: speaker
                assert len(row) > 1, f"{row} in {csv_path}:{row_idx+1}"
                utt_id, phonemes = row[0], row[1]
                phoneme_ids = [phoneme_to_id[p] for p in phonemes if p in phoneme_to_id]

                assert phoneme_ids, utt_id
                phoneme_ids = intersperse(phoneme_ids, 0)
                utt_phoneme_ids[utt_id] = phoneme_ids
                ids.append(utt_id)

    model = GlowTTSTraining(
        config=config,
        utt_phoneme_ids=utt_phoneme_ids,
        audio_dir=audio_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        cache_dir=cache_dir,
    )
    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.fit(model)


if __name__ == "__main__":
    main()
