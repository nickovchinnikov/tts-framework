from typing import List

from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kernel_predictor import KernelPredictor


class LVCBlock(LightningModule):
    r"""
    The location-variable convolutions block.

    To efficiently capture the local information of the condition, location-variable convolution (LVC)
    obtained better sound quality and speed while maintaining the model size.
    The kernels of the LVC layers are predicted using a kernel predictor that takes the log-mel-spectrogram
    as the input. The kernel predictor is connected to a residual stack. One kernel predictor simultaneously
    predicts the kernels of all LVC layers in one residual stack.

    Args:
        in_channels (int): The number of input channels.
        cond_channels (int): The number of conditioning channels.
        stride (int): The stride of the convolutional layers.
        dilations (List[int]): A list of dilation values for the convolutional layers.
        lReLU_slope (float): The slope of the LeakyReLU activation function.
        conv_kernel_size (int): The kernel size of the convolutional layers.
        cond_hop_length (int): The hop length of the conditioning sequence.
        kpnet_hidden_channels (int): The number of hidden channels in the kernel predictor network.
        kpnet_conv_size (int): The kernel size of the convolutional layers in the kernel predictor network.
        kpnet_dropout (float): The dropout rate for the kernel predictor network.

    Attributes:
        cond_hop_length (int): The hop length of the conditioning sequence.
        conv_layers (int): The number of convolutional layers.
        conv_kernel_size (int): The kernel size of the convolutional layers.
        kernel_predictor (KernelPredictor): The kernel predictor network.
        convt_pre (nn.Sequential): The convolutional transpose layer.
        conv_blocks (nn.ModuleList): The list of convolutional blocks.

    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        stride: int,
        dilations: List[int] = [1, 3, 9, 27],
        lReLU_slope: float = 0.2,
        conv_kernel_size: int = 3,
        cond_hop_length: int = 256,
        kpnet_hidden_channels: int = 64,
        kpnet_conv_size: int = 3,
        kpnet_dropout: float = 0.0,
    ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
            lReLU_slope=lReLU_slope,
        )

        self.convt_pre = nn.Sequential(
            nn.LeakyReLU(lReLU_slope),
            nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    in_channels,
                    in_channels,
                    2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=stride % 2,
                )
            ),
        )

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(lReLU_slope),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels,
                            in_channels,
                            conv_kernel_size,
                            padding=dilation * (conv_kernel_size - 1) // 2,
                            dilation=dilation,
                        )
                    ),
                    nn.LeakyReLU(lReLU_slope),
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        r"""Forward propagation of the location-variable convolutions.

        Args:
            x (Tensor): The input sequence (batch, in_channels, in_length).
            c (Tensor): The conditioning sequence (batch, cond_channels, cond_length).

        Returns:
            Tensor: The output sequence (batch, in_channels, in_length).
        """
        _, in_channels, _ = x.shape  # (B, c_g, L')

        x = self.convt_pre(x)  # (B, c_g, stride * L')
        kernels, bias = self.kernel_predictor(c)

        for i, conv in enumerate(self.conv_blocks):
            output = conv(x)  # (B, c_g, stride * L')

            k = kernels[:, i, :, :, :, :]  # (B, 2 * c_g, c_g, kernel_size, cond_length)
            b = bias[:, i, :, :]  # (B, 2 * c_g, cond_length)

            output = self.location_variable_convolution(
                output, k, b, hop_size=self.cond_hop_length
            )  # (B, 2 * c_g, stride * L'): LVC
            x = x + torch.sigmoid(output[:, :in_channels, :]) * torch.tanh(
                output[:, in_channels:, :]
            )  # (B, c_g, stride * L'): GAU

        return x

    def location_variable_convolution(
        self,
        x: torch.Tensor,
        kernel: torch.Tensor,
        bias: torch.Tensor,
        dilation: int = 1,
        hop_size: int = 256,
    ) -> torch.Tensor:
        r"""
        Perform location-variable convolution operation on the input sequence (x) using the local convolution kernel.
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.

        Args:
            x (Tensor): The input sequence (batch, in_channels, in_length).
            kernel (Tensor): The local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length).
            bias (Tensor): The bias for the local convolution (batch, out_channels, kernel_length).
            dilation (int): The dilation of convolution.
            hop_size (int): The hop_size of the conditioning sequence.

        Returns:
            (Tensor): The output sequence after performing local convolution. (batch, out_channels, in_length).
        """
        batch, _, in_length = x.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (
            kernel_length * hop_size
        ), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(
            x, (padding, padding), "constant", 0
        )  # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(
            2, hop_size + 2 * padding, hop_size
        )  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), "constant", 0)
        x = x.unfold(
            3, dilation, dilation
        )  # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(
            3, 4
        )  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        x = x.unfold(
            4, kernel_size, 1
        )  # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum("bildsk,biokl->bolsd", x, kernel)
        o = o.contiguous(memory_format=torch.channels_last_3d)

        bias = (
            bias.unsqueeze(-1)
            .unsqueeze(-1)
            .contiguous(memory_format=torch.channels_last_3d)
        )

        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)

        return o

    def remove_weight_norm(self) -> None:
        r"""
        Remove weight normalization from the convolutional layers in the LVCBlock.

        This method removes weight normalization from the kernel predictor and all convolutional layers in the LVCBlock.
        """
        self.kernel_predictor.remove_weight_norm()
        nn.utils.remove_weight_norm(self.convt_pre[1])
        for block in self.conv_blocks:
            nn.utils.remove_weight_norm(block[1])
