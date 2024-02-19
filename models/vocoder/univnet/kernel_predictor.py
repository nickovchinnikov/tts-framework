import torch
from torch import nn
from torch.nn import Module
from torch.nn.utils import parametrize


class KernelPredictor(Module):
    def __init__(
        self,
        cond_channels: int,
        conv_in_channels: int,
        conv_out_channels: int,
        conv_layers: int,
        conv_kernel_size: int = 3,
        kpnet_hidden_channels: int = 64,
        kpnet_conv_size: int = 3,
        kpnet_dropout: float = 0.0,
        lReLU_slope: float = 0.1,
    ):
        r"""Initializes a KernelPredictor object.
        KernelPredictor is a class that predicts the kernel size for the convolutional layers in the UnivNet model.
        The kernels of the LVC layers are predicted using a kernel predictor that takes the log-mel-spectrogram as the input.

        Args:
            cond_channels (int): The number of channels for the conditioning sequence.
            conv_in_channels (int): The number of channels for the input sequence.
            conv_out_channels (int): The number of channels for the output sequence.
            conv_layers (int): The number of layers in the model.
            conv_kernel_size (int, optional): The kernel size for the convolutional layers. Defaults to 3.
            kpnet_hidden_channels (int, optional): The number of hidden channels in the kernel predictor network. Defaults to 64.
            kpnet_conv_size (int, optional): The kernel size for the kernel predictor network. Defaults to 3.
            kpnet_dropout (float, optional): The dropout rate for the kernel predictor network. Defaults to 0.0.
            lReLU_slope (float, optional): The slope for the leaky ReLU activation function. Defaults to 0.1.
        """
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        kpnet_kernel_channels = (
            conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        )  # l_w

        kpnet_bias_channels = conv_out_channels * conv_layers  # l_b

        padding = (kpnet_conv_size - 1) // 2

        self.input_conv = nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    cond_channels,
                    kpnet_hidden_channels,
                    5,
                    padding=2,
                    bias=True,
                ),
            ),
            nn.LeakyReLU(lReLU_slope),
        )

        self.residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(kpnet_dropout),
                    nn.utils.parametrizations.weight_norm(
                        nn.Conv1d(
                            kpnet_hidden_channels,
                            kpnet_hidden_channels,
                            kpnet_conv_size,
                            padding=padding,
                            bias=True,
                        ),
                    ),
                    nn.LeakyReLU(lReLU_slope),
                    nn.utils.parametrizations.weight_norm(
                        nn.Conv1d(
                            kpnet_hidden_channels,
                            kpnet_hidden_channels,
                            kpnet_conv_size,
                            padding=padding,
                            bias=True,
                        ),
                    ),
                    nn.LeakyReLU(lReLU_slope),
                )
                for _ in range(3)
            ],
        )

        self.kernel_conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                kpnet_hidden_channels,
                kpnet_kernel_channels,
                kpnet_conv_size,
                padding=padding,
                bias=True,
            ),
        )
        self.bias_conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                kpnet_hidden_channels,
                kpnet_bias_channels,
                kpnet_conv_size,
                padding=padding,
                bias=True,
            ),
        )

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the forward pass of the model.

        Args:
            c (Tensor): The conditioning sequence (batch, cond_channels, cond_length).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the kernel and bias tensors.
        """
        batch, _, cond_length = c.shape
        c = self.input_conv(c.to(dtype=self.kernel_conv.weight.dtype))
        for residual_conv in self.residual_convs:
            c = c + residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = k.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            cond_length,
        )
        bias = b.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_out_channels,
            cond_length,
        )

        return kernels, bias

    def remove_weight_norm(self):
        r"""Removes weight normalization from the input, kernel, bias, and residual convolutional layers."""
        parametrize.remove_parametrizations(self.input_conv[0], "weight")
        parametrize.remove_parametrizations(self.kernel_conv, "weight")
        parametrize.remove_parametrizations(self.bias_conv, "weight")

        for block in self.residual_convs:
            parametrize.remove_parametrizations(block[1], "weight") # type: ignore
            parametrize.remove_parametrizations(block[3], "weight") # type: ignore
