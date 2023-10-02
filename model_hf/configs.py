import torch

from transformers import PretrainedConfig

from model.config import AcousticModelConfigType, PreprocessingConfig
from model.helpers.tools import get_device


class AcousticModelConfig(PretrainedConfig):
    model_type = "acoustic_model"

    def __init__(
        self,
        data_path: str,
        model_config: AcousticModelConfigType,
        preprocess_config: PreprocessingConfig,
        n_speakers: int = 5392,
        fine_tuning: bool = True,
        device: torch.device = get_device(),
        **kwargs
    ):
        self.data_path = data_path
        self.model_config = model_config
        self.preprocess_config = preprocess_config
        self.n_speakers = n_speakers
        self.fine_tuning = fine_tuning
        self.device = device

        super().__init__(**kwargs)
