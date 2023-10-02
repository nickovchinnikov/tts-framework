import torch
from transformers import PreTrainedModel

from model.acoustic_model import AcousticModel as AcousticModelBase

from .configs import AcousticModelConfig


# Very bad documented API, too risky to use
# Decided to switch to the PyTorch Lightning
class AcousticModel(PreTrainedModel):
    config_class = AcousticModelConfig
    base_model_prefix = "acoustic_model"

    def __init__(self, config: AcousticModelConfig):
        super().__init__(config)
        self.model = AcousticModelBase(
            config.data_path,
            config.preprocess_config,
            config.model_config,
            config.fine_tuning,
            config.n_speakers,
            device=config.device,
        )

    def forward(
        self,
        x: torch.Tensor,
        speakers: torch.Tensor,
        langs: torch.Tensor,
        p_control: float,
        d_control: float,
    ):
        return self.model(x, speakers, langs, p_control, d_control)
