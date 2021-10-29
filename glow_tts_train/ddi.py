"""Methods for data-dependent initialization of model"""
import typing

import torch
from torch.utils.data import DataLoader

from glow_tts_train.config import TrainingConfig
from glow_tts_train.dataset import Batch
from glow_tts_train.models import ModelType, setup_model
from glow_tts_train.utils import to_gpu


class FlowGeneratorDDI(ModelType):
    """A helper for data-dependent initialization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for f in self.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)


def initialize_model(train_loader: DataLoader, config: TrainingConfig) -> ModelType:
    """Do data-dependent model initialization"""
    torch.manual_seed(config.seed)
    model = setup_model(config, model_factory=FlowGeneratorDDI, use_cuda=True)
    model.train()

    for batch in train_loader:
        batch = typing.cast(Batch, batch)
        x, x_lengths, y, y_lengths, speaker_ids = (
            to_gpu(batch.phoneme_ids),
            to_gpu(batch.phoneme_lengths),
            to_gpu(batch.spectrograms),
            to_gpu(batch.spectrogram_lengths),
            to_gpu(batch.speaker_ids) if batch.speaker_ids is not None else None,
        )

        if speaker_ids is not None:
            speaker_ids = to_gpu(speaker_ids)

        _ = model(x, x_lengths, y, y_lengths, g=speaker_ids)
        break

    return model
