import torch
import torch.nn as nn

from model.constants import LEAKY_RELU_SLOPE
from .mas import b_mas


# @todo: refactor this class to use nn.Sequential
# Won't do this, to save time and because it's not a priority


class ConvLeakyReLU(nn.Module):
    r"""
    This class implements a Convolution followed by a Leaky ReLU activation layer.

    Attributes
    ----------
    layers : nn.Sequential
        Sequential container that holds the Convolution and LeakyReLU layers.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Passes the input through the Conv1d and LeakyReLU layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
    ):
        r"""
        Parameters
        ----------
        in_channels : int
            Number of channels in the input.
        out_channels : int
            Number of channels in the output.
        kernel_size : int
            Size of the convolving kernel.
        padding : int
            Zero-padding added to both sides of the input.
        leaky_relu_slope : float
            Controls the angle of the negative slope, default=LEAKY_RELU_SLOPE.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.LeakyReLU(leaky_relu_slope),
        )

    def forward(self, x):
        r"""
        Defines the forward pass of the ConvLeakyReLU.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after being passed through the Conv1d and LeakyReLU layers.
        """
        return self.layers(x)


class Aligner2(nn.Module):
    def __init__(
        self,
        d_enc_in: int,
        d_dec_in: int,
        d_hidden: int,
        kernel_size_enc: int = 3,
        kernel_size_dec: int = 7,
        temperature: float = 0.0005,
        softmax_dim: int = 3,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
    ):
        super().__init__()
        self.temperature = temperature

        self.softmax = torch.nn.Softmax(dim=softmax_dim)
        self.log_softmax = torch.nn.LogSoftmax(dim=softmax_dim)

        self.key_proj = nn.Sequential(
            ConvLeakyReLU(
                d_enc_in,
                d_hidden,
                kernel_size=kernel_size_enc,
                padding=kernel_size_enc // 2,
                leaky_relu_slope=leaky_relu_slope,
            ),
            ConvLeakyReLU(
                d_hidden,
                d_hidden,
                kernel_size=kernel_size_enc,
                padding=kernel_size_enc // 2,
                leaky_relu_slope=leaky_relu_slope,
            ),
        )

        self.query_proj = nn.Sequential(
            ConvLeakyReLU(
                d_dec_in,
                d_hidden,
                kernel_size=kernel_size_dec,
                padding=kernel_size_dec // 2,
                leaky_relu_slope=leaky_relu_slope,
            ),
            ConvLeakyReLU(
                d_hidden,
                d_hidden,
                kernel_size=kernel_size_dec,
                padding=kernel_size_dec // 2,
                leaky_relu_slope=leaky_relu_slope,
            ),
            ConvLeakyReLU(
                d_hidden,
                d_hidden,
                kernel_size=kernel_size_dec,
                padding=kernel_size_dec // 2,
                leaky_relu_slope=leaky_relu_slope,
            ),
        )

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(
                attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1
            )
        return torch.from_numpy(attn_out).to(attn.device)

    def forward(self, enc_in, dec_in, enc_len, dec_len, enc_mask, attn_prior):
        """
        :param enc_in: (B, C_1, T_1) Text encoder outputs.
        :param dec_in: (B, C_2, T_2) Data to align encoder outputs to.
        :speaker_emb: (B, C_3) Batch of speaker embeddings.
        """
        queries = dec_in
        keys = enc_in
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (
            queries_enc[:, :, :, None] - keys_enc[:, :, None]
        ) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            # print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape}
            # mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(
                attn_prior.permute((0, 2, 1))[:, None] + 1e-8
            )
            # print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")"""

        attn_logprob = attn.clone()

        if enc_mask is not None:
            attn.masked_fill(enc_mask.unsqueeze(1).unsqueeze(1), -float("inf"))

        attn_soft = self.softmax(attn)  # softmax along T2
        attn_hard = self.binarize_attention_parallel(attn_soft, enc_len, dec_len)
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        return attn_logprob, attn_soft, attn_hard, attn_hard_dur
