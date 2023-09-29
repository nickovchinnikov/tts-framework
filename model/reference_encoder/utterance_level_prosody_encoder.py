import torch
import torch.nn as nn

from model.config import AcousticModelConfigType, PreprocessingConfig

from model.basenn import BaseNNModule

from helpers.tools import get_device

from .STL import STL
from .reference_encoder import ReferenceEncoder


class UtteranceLevelProsodyEncoder(BaseNNModule):
    r"""A class to define the utterance level prosody encoder.

    The encoder uses a Reference encoder class to convert input sequences into high-level features,
    followed by prosody embedding, self attention on the embeddings, and a feedforward transformation to generate the final output.Initializes the encoder with given specifications and creates necessary layers.

    Args:
        preprocess_config (PreprocessingConfig): Configuration object with preprocessing parameters.
        model_config (AcousticModelConfigType): Configuration object with acoustic model parameters.
        device (torch.device): The device to which the model should be moved. Defaults `get_device()`

    Returns:
        torch.Tensor: A 3-dimensional tensor sized `[N, seq_len, E]`.
    """

    def __init__(
        self,
        preprocess_config: PreprocessingConfig,
        model_config: AcousticModelConfigType,
        device: torch.device = get_device(),
    ):
        super().__init__(device=device)

        self.E = model_config.encoder.n_hidden
        self.d_q = self.d_k = model_config.encoder.n_hidden
        ref_enc_gru_size = model_config.reference_encoder.ref_enc_gru_size
        ref_attention_dropout = model_config.reference_encoder.ref_attention_dropout
        bottleneck_size = model_config.reference_encoder.bottleneck_size_u

        # Define important layers/modules for the encoder
        self.encoder = ReferenceEncoder(
            preprocess_config, model_config, device=self.device
        )
        self.encoder_prj = nn.Linear(ref_enc_gru_size, self.E // 2, device=self.device)
        self.stl = STL(model_config, device=self.device)
        self.encoder_bottleneck = nn.Linear(self.E, bottleneck_size, device=self.device)
        self.dropout = nn.Dropout(ref_attention_dropout)

    def forward(self, mels: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
        r"""Defines the forward pass of the utterance level prosody encoder.

        Args:
            mels (torch.Tensor): A 3-dimensional tensor containing input sequences. Size is `[N, Ty/r, n_mels*r]`.
            mel_lens (torch.Tensor): A 1-dimensional tensor containing the lengths of each sequence in mels. Length is N.

        Returns:
            torch.Tensor: A 3-dimensional tensor sized `[N, seq_len, E]`.
        """

        # Use the reference encoder to get prosody embeddings
        _, embedded_prosody, _ = self.encoder(mels, mel_lens)

        # Bottleneck
        # Use the linear projection layer on the prosody embeddings
        embedded_prosody = self.encoder_prj(embedded_prosody)

        # Apply the style token layer followed by the bottleneck layer
        out = self.encoder_bottleneck(self.stl(embedded_prosody))

        # Apply dropout for regularization
        out = self.dropout(out)

        # Reshape the output tensor before returning
        out = out.view((-1, 1, out.shape[3]))
        return out
