import torch
from torch import nn
from torch.nn import Module

from model.attention import ConformerMultiHeadedSelfAttention
from model.config import AcousticModelConfigType, PreprocessingConfig

from .reference_encoder import ReferenceEncoder


class PhonemeLevelProsodyEncoder(Module):
    r"""Phoneme Level Prosody Encoder Module

    This Class is used to encode the phoneme level prosody in the speech synthesis pipeline.

    Args:
        preprocess_config (PreprocessingConfig): Configuration for preprocessing.
        model_config (AcousticModelConfigType): Acoustic model configuration.

    Returns:
        torch.Tensor: The encoded tensor after applying masked fill.
    """

    def __init__(
        self,
        preprocess_config: PreprocessingConfig,
        model_config: AcousticModelConfigType,
    ):
        super().__init__()

        # Obtain the bottleneck size and reference encoder gru size from the model config.
        bottleneck_size = model_config.reference_encoder.bottleneck_size_p
        ref_enc_gru_size = model_config.reference_encoder.ref_enc_gru_size

        # Initialize ReferenceEncoder, Linear layer and ConformerMultiHeadedSelfAttention for attention mechanism.
        self.encoder = ReferenceEncoder(preprocess_config, model_config)
        self.encoder_prj = nn.Linear(ref_enc_gru_size, model_config.encoder.n_hidden)
        self.attention = ConformerMultiHeadedSelfAttention(
            d_model=model_config.encoder.n_hidden,
            num_heads=model_config.encoder.n_heads,
            dropout_p=model_config.encoder.p_dropout,
        )

        # Bottleneck layer to transform the output of the attention mechanism.
        self.encoder_bottleneck = nn.Linear(
            model_config.encoder.n_hidden, bottleneck_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        r"""The forward pass of the PhonemeLevelProsodyEncoder. Input tensors are passed through the reference encoder,
        attention mechanism, and a bottleneck.

        Args:
            x (torch.Tensor): Input tensor of shape [N, seq_len, encoder_embedding_dim].
            src_mask (torch.Tensor): The mask tensor which contains `True` at positions where the input x has been masked.
            mels (torch.Tensor): The mel-spectrogram with shape [N, Ty/r, n_mels*r], where r=1.
            mel_lens (torch.Tensor): The lengths of each sequence in mels.
            encoding (torch.Tensor): The relative positional encoding tensor.

        Returns:
            torch.Tensor: Output tensor of shape [N, seq_len, bottleneck_size].
        """
        # Use the reference encoder to embed prosody representation
        embedded_prosody, _, mel_masks = self.encoder(mels, mel_lens)

        # Pass the prosody representation through a bottleneck (dimension reduction)
        embedded_prosody = self.encoder_prj(embedded_prosody)

        # Flatten and apply attention mask
        attn_mask = mel_masks.view((mel_masks.shape[0], 1, 1, -1))
        x, _ = self.attention(
            query=x,
            key=embedded_prosody,
            value=embedded_prosody,
            mask=attn_mask,
            encoding=encoding,
        )

        # Apply the bottleneck to the output and mask out irrelevant parts
        x = self.encoder_bottleneck(x)
        return x.masked_fill(src_mask.unsqueeze(-1), 0.0)
