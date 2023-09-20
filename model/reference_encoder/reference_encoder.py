import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from model.conv_blocks import CoordConv1d

from helpers import tools

from config import AcousticModelConfigType, PreprocessingConfig

from model.constants import LEAKY_RELU_SLOPE


class ReferenceEncoder(nn.Module):
    r"""A class to define the reference encoder.
    Similar to Tacotron model, the reference encoder is used to extract the high-level features from the reference
    
    It consists of a number of convolutional blocks (`CoordConv1d` for the first one and `nn.Conv1d` for the rest), 
    then followed by instance normalization and GRU layers.
    The `CoordConv1d` at the first layer to better preserve positional information, paper:
    [Robust and fine-grained prosody control of end-to-end speech synthesis](https://arxiv.org/pdf/1811.02122.pdf)
    
    Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

    Args:
        preprocess_config (PreprocessingConfig): Configuration object with preprocessing parameters.
        model_config (AcousticModelConfigType): Configuration object with acoustic model parameters.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three tensors. _First_: The sequence tensor 
            produced by the last GRU layer after padding has been removed. _Second_: The GRU's final hidden state tensor.
            _Third_: The mask tensor, which has the same shape as x, and contains `True` at positions where the input x 
            has been masked.
    """
    def __init__(
        self,
        preprocess_config: PreprocessingConfig,
        model_config: AcousticModelConfigType,
    ):
        super().__init__()
        n_mel_channels = preprocess_config.stft.n_mel_channels
        ref_enc_filters = model_config.reference_encoder.ref_enc_filters
        ref_enc_size = model_config.reference_encoder.ref_enc_size
        ref_enc_strides = model_config.reference_encoder.ref_enc_strides
        ref_enc_gru_size = model_config.reference_encoder.ref_enc_gru_size

        self.n_mel_channels = n_mel_channels
        K = len(ref_enc_filters)
        filters = [self.n_mel_channels] + ref_enc_filters
        strides = [1] + ref_enc_strides

        # Use CoordConv1d at the first layer to better preserve positional information: https://arxiv.org/pdf/1811.02122.pdf
        convs = [
            CoordConv1d(
                in_channels=filters[0],
                out_channels=filters[0 + 1],
                kernel_size=ref_enc_size,
                stride=strides[0],
                padding=ref_enc_size // 2,
                with_r=True,
            ),
            *[nn.Conv1d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=ref_enc_size,
                stride=strides[i],
                padding=ref_enc_size // 2,
            )
            for i in range(1, K)]
        ]
        # Define convolution layers (ModuleList)
        self.convs = nn.ModuleList(convs)

        self.norms = nn.ModuleList(
            [
                nn.InstanceNorm1d(num_features=ref_enc_filters[i], affine=True)
                for i in range(K)
            ]
        )

        # Define GRU layer
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1],
            hidden_size=ref_enc_gru_size,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor,
        mel_lens: torch.Tensor, 
        leaky_relu_slope: float = LEAKY_RELU_SLOPE
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward pass of the ReferenceEncoder.

        Args:
            x (torch.Tensor): A 3-dimensional tensor containing the input sequences, its size is [N, n_mels, timesteps].
            mel_lens (torch.Tensor): A 1-dimensional tensor containing the lengths of each sequence in x. Its length is N.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three tensors. _First_: The sequence tensor 
                produced by the last GRU layer after padding has been removed. _Second_: The GRU's final hidden state tensor.
                _Third_: The mask tensor, which has the same shape as x, and contains `True` at positions where the input x 
                has been masked.
        """
        mel_masks = tools.get_mask_from_lengths(mel_lens).unsqueeze(1)
        x = x.masked_fill(mel_masks, 0)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = F.leaky_relu(x, leaky_relu_slope)  # [N, 128, Ty//2^K, n_mels//2^K]
            x = norm(x)

        for _ in range(2):
            mel_lens = tools.stride_lens_downsampling(mel_lens)

        mel_masks = tools.get_mask_from_lengths(mel_lens)

        x = x.masked_fill(mel_masks.unsqueeze(1), 0)
        x = x.permute((0, 2, 1))

        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            mel_lens.cpu().int(),
            batch_first=True,
            enforce_sorted=False
        )

        self.gru.flatten_parameters()
        # memory --- [N, Ty, E//2], out --- [1, N, E//2]
        out, memory = self.gru(packed_sequence)  
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        return out, memory, mel_masks

    def calculate_channels(
        self, L: int, kernel_size: int, stride: int, pad: int, n_convs: int
    ) -> int:
        r"""Calculate the number of channels after applying convolutions.

        Args:
            L (int): The original size.
            kernel_size (int): The kernel size used in the convolutions.
            stride (int): The stride used in the convolutions.
            pad (int): The padding used in the convolutions.
            n_convs (int): The number of convolutions.

        Returns:
            int: The size after the convolutions.
        """

        # Loop through each convolution
        for _ in range(n_convs):
            # Calculate the size after each convolution
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

