import torch
from torch import nn
from torch.nn import Module

from model.config import AcousticModelConfigType
from model.constants import LEAKY_RELU_SLOPE
from model.conv_blocks import ConvTransposed


class PhonemeProsodyPredictor(Module):
    r"""A class to define the Phoneme Prosody Predictor.

    In linguistics, prosody (/ˈprɒsədi, ˈprɒzədi/) is the study of elements of speech that are not individual phonetic segments (vowels and consonants) but which are properties of syllables and larger units of speech, including linguistic functions such as intonation, stress, and rhythm. Such elements are known as suprasegmentals.

    [Wikipedia Prosody (linguistics)](https://en.wikipedia.org/wiki/Prosody_(linguistics))

    This prosody predictor is non-parallel and is inspired by the **work of Du et al., 2021 ?**. It consists of
    multiple convolution transpose, Leaky ReLU activation, LayerNorm, and dropout layers, followed by a
    linear transformation to generate the final output.

    Args:
        model_config (AcousticModelConfigType): Configuration object with model parameters.
        phoneme_level (bool): A flag to decide whether to use phoneme level bottleneck size.
        leaky_relu_slope (float): The negative slope of LeakyReLU activation function.
    """

    def __init__(
        self,
        model_config: AcousticModelConfigType,
        phoneme_level: bool,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
    ):
        super().__init__()

        # Get the configuration
        self.d_model = model_config.encoder.n_hidden
        kernel_size = model_config.reference_encoder.predictor_kernel_size
        dropout = model_config.encoder.p_dropout

        # Decide on the bottleneck size based on phoneme level flag
        bottleneck_size = (
            model_config.reference_encoder.bottleneck_size_p
            if phoneme_level
            else model_config.reference_encoder.bottleneck_size_u
        )

        # Define the layers
        self.layers = nn.ModuleList(
            [
                ConvTransposed(
                    self.d_model,
                    self.d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(leaky_relu_slope),
                nn.LayerNorm(
                    self.d_model,
                ),
                nn.Dropout(dropout),
                ConvTransposed(
                    self.d_model,
                    self.d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(leaky_relu_slope),
                nn.LayerNorm(
                    self.d_model,
                ),
                nn.Dropout(dropout),
            ],
        )

        # Output bottleneck layer
        self.predictor_bottleneck = nn.Linear(
            self.d_model,
            bottleneck_size,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the prosody predictor.

        Args:
            x (torch.Tensor): A 3-dimensional tensor `[B, src_len, d_model]`.
            mask (torch.Tensor): A 2-dimensional tensor `[B, src_len]`.

        Returns:
            torch.Tensor: A 3-dimensional tensor `[B, src_len, 2 * d_model]`.
        """
        # Expand the mask tensor's dimensions from [B, src_len] to [B, src_len, 1]
        mask = mask.unsqueeze(2)

        # Pass the input through the layers
        for layer in self.layers:
            x = layer(x)

        # Apply mask
        x = x.masked_fill(mask, 0.0)

        # Final linear transformation
        return self.predictor_bottleneck(x)
