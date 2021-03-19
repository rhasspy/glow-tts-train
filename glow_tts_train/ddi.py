"""Methods for data-dependent initialization of model"""
import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .models import ModelType, setup_model
from .utils import to_gpu


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
    model, _ = setup_model(config, model_factory=FlowGeneratorDDI)
    model.cuda()

    model.train()
    for _batch_idx, (x, x_lengths, y, y_lengths, speaker_ids) in enumerate(
        train_loader
    ):
        x, x_lengths = to_gpu(x), to_gpu(x_lengths)
        y, y_lengths = to_gpu(y), to_gpu(y_lengths)

        if speaker_ids is not None:
            speaker_ids = to_gpu(speaker_ids)

        _ = model(x, x_lengths, y, y_lengths, g=speaker_ids)
        break

    return model
