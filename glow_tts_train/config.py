"""Configuration classes"""
import json
import typing
from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin


@dataclass
class AudioConfig(DataClassJsonMixin):
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    sampling_rate: int = 22050
    sample_bytes: int = 2
    channels: int = 1
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    normalized: bool = True


@dataclass
class ModelConfig(DataClassJsonMixin):
    num_symbols: int = 0
    hidden_channels: int = 192
    filter_channels: int = 768
    filter_channels_dp: int = 256
    kernel_size: int = 3
    p_dropout: float = 0.1
    n_blocks_dec: int = 12
    n_layers_enc: int = 6
    n_heads: int = 2
    p_dropout_dec: float = 0.05
    dilation_rate: int = 1
    kernel_size_dec: int = 5
    n_block_layers: int = 4
    n_sqz: int = 2
    prenet: bool = True
    mean_only: bool = True
    hidden_channels_enc: int = 192
    hidden_channels_dec: int = 192
    window_size: int = 4
    n_speakers: int = 1
    n_split: int = 4
    sigmoid_scale: bool = False
    block_length: typing.Optional[int] = None
    gin_channels: int = 0


@dataclass
class TrainingConfig(DataClassJsonMixin):
    seed: int = 1234
    epochs: int = 10000
    learning_rate: float = 1e0
    betas: typing.Tuple[float, float] = field(default=(0.9, 0.98))
    eps: float = 1e-9
    warmup_steps: int = 4000
    scheduler: str = "noam"
    batch_size: int = 32
    fp16_run: bool = True
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    version: int = 1

    def save(self, config_file: typing.TextIO):
        """Save config as JSON to a file"""
        json.dump(self.to_json(), config_file, indent=4)

    @staticmethod
    def load(config_file: typing.TextIO) -> "TrainingConfig":
        """Load config from a JSON file"""
        return TrainingConfig.from_json(config_file.read())
